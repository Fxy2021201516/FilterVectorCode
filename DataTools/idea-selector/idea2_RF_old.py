import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 导入 ONNX 相关库 ---
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False

# --- 全局配置 ---
BASE_RESULTS_DIR = "/home/fengxiaoyao/1023FilterVector/FilterVectorResults"
DATASET_NAME = "Laion"
TOP_N_FEATURES_TO_SELECT = 6  # 在第二阶段中，选择最重要的特征数量

RECALL_METRIC_NAME = "Recall" 

RECALL_QUALITY_GATE = 0.9 # 如果两者 Recall 都 > 0.9，则比较时间；否则比较 Recall

# --- 路径配置 ---
DATASET_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, DATASET_NAME)
MODEL_OUTPUT_BASE_DIR = os.path.join(DATASET_RESULTS_DIR, "SelectModels/idea2")

# --- 定义从CSV中读取的、最原始的特征列 ---
RAW_FEATURES_FROM_CPP = [
    'NumEntries', 
    'NumDescendants', 
    'TotalCoverage'
]


def find_result_pairs(dir_ung, dir_acorn):
    """扫描结果目录，查找并匹配 UNG 和 ACORN 的成对实验结果。"""
    print(f"--- 正在扫描以下目录以查找成对的实验结果 (UNG vs ACORN) ---")
    print(f"       UNG 算法目录: '{dir_ung}'")
    print(f"       ACORN 算法目录: '{dir_acorn}'")
    
    pairs = []
    if not os.path.isdir(dir_ung) or not os.path.isdir(dir_acorn):
        print(f"❌ 警告: 必要的实验目录不存在，请检查路径。")
        return pairs

    for sub_dir_name in os.listdir(dir_ung):
        path_ung = os.path.join(dir_ung, sub_dir_name)
        if os.path.isdir(path_ung):
            path_acorn = os.path.join(dir_acorn, sub_dir_name)
            if os.path.isdir(path_acorn):
                pairs.append((path_ung, path_acorn))
                print(f"   [配对成功] -> {sub_dir_name}")
                
    if not pairs:
        print("❌ 警告: 未找到任何成对的 (UNG vs ACORN) 实验结果目录。")
    else:
        print(f"\n✅ 成功找到 {len(pairs)} 对实验结果。\n")
    return pairs

def load_performance_data_idea2(path_ung, path_acorn):
    """
    从成对的UNG和ACORN结果中加载并合并性能数据。
    """
    try:
        csv_ung_path_pattern = os.path.join(glob.escape(path_ung), "results", "query_details*.csv")
        csv_acorn_path_pattern = os.path.join(glob.escape(path_acorn), "results", "query_details*.csv")
        
        csv_ung_list = glob.glob(csv_ung_path_pattern)
        csv_acorn_list = glob.glob(csv_acorn_path_pattern)

        if not csv_ung_list or not csv_acorn_list:
            print(f"❌ 错误: 找不到 'query_details*.csv' 文件。")
            return pd.DataFrame()

        # 确保召回率指标列被包含在内
        if not RECALL_METRIC_NAME:
            print(f"❌ 错误: RECALL_METRIC_NAME 未设置。")
            return pd.DataFrame()
            
        cols_to_keep = ['Lsearch', 'QueryID', 'Time_ms', RECALL_METRIC_NAME]

        # 检查列是否存在
        try:
            temp_cols = pd.read_csv(csv_ung_list[0], nrows=1).columns
            if RECALL_METRIC_NAME not in temp_cols:
                print(f"❌ 致命错误: 召回率列 '{RECALL_METRIC_NAME}' 在 {csv_ung_list[0]} 中不存在。")
                print(f"   可用列: {temp_cols.tolist()}")
                return pd.DataFrame()
        except Exception as e:
            print(f"检查列时出错: {e}")
            return pd.DataFrame()

        df_ung = pd.read_csv(csv_ung_list[0], usecols=cols_to_keep)
        df_acorn = pd.read_csv(csv_acorn_list[0], usecols=cols_to_keep)
        
        # 合并以创建 Time_ms_U/A 和 RECALL_METRIC_NAME_U/A
        merged_df = pd.merge(df_ung, df_acorn, on=['Lsearch', 'QueryID'], suffixes=('_U', '_A'))
        
        return merged_df
        
    except Exception as e:
        print(f"加载性能数据时发生未知错误: {e}")
        return pd.DataFrame()

def create_comprehensive_features_idea2(df):
    """
    从Idea2最原始的指标中自动组合出丰富的衍生特征。
    """
    print("--- 正在从原始指标进行特征工程，创建丰富的衍生特征 ---")
    
    raw_df = df.copy()
    features = pd.DataFrame(index=raw_df.index)

    # 1. 对数特征 (处理数据倾斜)
    features['LogNumEntries'] = np.log1p(raw_df['NumEntries'])
    features['LogNumDescendants'] = np.log1p(raw_df['NumDescendants'])
    features['LogTotalCoverage'] = np.log1p(raw_df['TotalCoverage'])

    # 2. 多项式特征 (捕捉非线性关系)
    features['CoverageSquared'] = raw_df['TotalCoverage'] ** 2
    features['DescendantsSquared'] = raw_df['NumDescendants'] ** 2

    # 3. 交互特征 (捕捉特征间的协同效应)
    features['DescCovInteraction'] = raw_df['NumDescendants'] * raw_df['TotalCoverage']
    features['EntriesCovInteraction'] = raw_df['NumEntries'] * raw_df['TotalCoverage']
    features['EntriesDescInteraction'] = raw_df['NumEntries'] * raw_df['NumDescendants']

    # 4. 比率特征 (标准化或相对关系)
    features['DescPerEntry'] = raw_df['NumDescendants'] / (raw_df['NumEntries'] + 1e-9)
    features['CovPerEntry'] = raw_df['TotalCoverage'] / (raw_df['NumEntries'] + 1e-9)

    # 将原始特征也加入，让模型自己判断其重要性
    final_features = pd.concat([raw_df[RAW_FEATURES_FROM_CPP], features], axis=1)
    
    print(f"特征工程完成，共生成 {final_features.shape[1]} 个特征。")
    return final_features

def train_and_evaluate_model(X, y, model_description, feature_list):
    """
    通用函数：训练随机森林模型、评估其性能并返回特征重要性。
    
    返回:
        (model, importance_df, report_str, accuracy, training_time_sec)
    """
    print(f"\n--- 开始处理模型: '{model_description}' ---")
    print(f"使用 {X.shape[1]} 个特征进行训练。")
    
    # 检查 y 中是否只有一类
    if y.nunique() == 1:
        print(f"❌ 警告: 目标变量 y 中只存在一类 ({y.unique()})。无法进行分层抽样或训练。")
        print("   这通常意味着您的召回率或时间阈值过滤掉了所有反例。")
        return None, pd.DataFrame(), "目标变量 y 中只存在一类，无法训练。", 0.0, 0.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 模型训练
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, oob_score=True, class_weight='balanced')
    
    # --- 开始计时 ---
    print(f"   - 开始模型训练 (Timing)...")
    start_time = time.perf_counter()
    
    rf.fit(X_train, y_train)
    
    end_time = time.perf_counter()
    training_duration_sec = end_time - start_time
    print(f"   - 模型训练完成。耗时: {training_duration_sec:.4f} 秒")
    # --- 结束计时 ---

    # 性能评估
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # 修正标签顺序：0 = UNG, 1 = ACORN
    class_report_str = classification_report(y_test, y_pred, target_names=['应选方法 UNG', '应选方法 ACORN'])
    
    print(f"模型 '{model_description}' 训练完成。")
    print(f"袋外分数 (OOB Score): {rf.oob_score_:.4f}")
    print(f"测试集准确率 (Accuracy): {accuracy:.4f}")

    # 特征重要性
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_list,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return rf, feature_importance_df, class_report_str, accuracy, training_duration_sec

def run_two_stage_analysis_idea2(data, model_name_prefix, output_dir):
    """
    执行完整的两阶段分析流程
    """
    print(f"\n--- 开始为 '{model_name_prefix}' 执行两阶段分析 ---")
    
    # --- 0. 准备原始数据 ---
    original_count = len(data)
    print(f"--- 应用决策边界过滤 (共 {original_count} 条原始记录) ---")
    
    recall_col_A = f'{RECALL_METRIC_NAME}_A'
    recall_col_U = f'{RECALL_METRIC_NAME}_U'
    time_col_A = 'Time_ms_A'
    time_col_U = 'Time_ms_U'
    
    if recall_col_A not in data.columns or recall_col_U not in data.columns:
        print(f"❌ 致命错误: 召回率列 {recall_col_A} 或 {recall_col_U} 在合并数据后不存在。")
        return

    # --- 1. 决策逻辑定义 (质量门控) ---
    print(f"--- 步骤 1: 应用“质量门控”逻辑 (召回率门槛 > {RECALL_QUALITY_GATE}) ---")

    # 情况一：两者召回率均达标 -> "速度模式"
    both_recall_ok_mask = (data[recall_col_A] > RECALL_QUALITY_GATE) & (data[recall_col_U] > RECALL_QUALITY_GATE)
    
    # 情况二：至少有一方不达标 -> "质量模式"
    quality_comparison_mask = ~both_recall_ok_mask

    count_speed_mode = both_recall_ok_mask.sum()
    count_quality_mode = quality_comparison_mask.sum()

    print(f"     决策模式分析 (共 {original_count} 条):")
    print(f"     - {count_speed_mode:<6} 条: 进入“速度模式” (两者 Recall > {RECALL_QUALITY_GATE})")
    print(f"     - {count_quality_mode:<6} 条: 进入“质量模式” (至少一方 Recall <= {RECALL_QUALITY_GATE})")


    # ---  步骤 1.5: 详细的过滤前胜出情况统计 ---
    print("\n--- 步骤 1.5: 原始数据胜出情况 (过滤前) ---")
    
    # 1. 速度模式下的胜负
    speed_acorn_wins_mask = both_recall_ok_mask & (data[time_col_A] < data[time_col_U])
    speed_ung_wins_mask = both_recall_ok_mask & (data[time_col_U] < data[time_col_A])
    
    # 2. 质量模式下的胜负
    quality_acorn_wins_mask = quality_comparison_mask & (data[recall_col_A] > data[recall_col_U])
    quality_ung_wins_mask = quality_comparison_mask & (data[recall_col_U] > data[recall_col_A])
    
    # 3. 质量模式下的平局
    recall_tie_mask = quality_comparison_mask & (data[recall_col_A] == data[recall_col_U])
    tie_acorn_wins_mask = recall_tie_mask & (data[time_col_A] < data[time_col_U])
    tie_ung_wins_mask = recall_tie_mask & (data[time_col_U] < data[time_col_A])

    print(f"     [速度模式] (共 {count_speed_mode} 条):")
    print(f"       - ACORN 因[时间]胜出: {speed_acorn_wins_mask.sum():<6} 条")
    print(f"       - UNG   因[时间]胜出: {speed_ung_wins_mask.sum():<6} 条")
    
    print(f"\n     [质量模式] (共 {count_quality_mode} 条):")
    print(f"       - ACORN 因[召回]胜出: {quality_acorn_wins_mask.sum():<6} 条")
    print(f"       - UNG   因[召回]胜出: {quality_ung_wins_mask.sum():<6} 条")
    print(f"       - 召回率平局:       {recall_tie_mask.sum():<6} 条")
    print(f"         - (平局) ACORN 因[时间]胜出: {tie_acorn_wins_mask.sum():<6} 条")
    print(f"         - (平局) UNG   因[时间]胜出: {tie_ung_wins_mask.sum():<6} 条")

    # 打印 ACORN 胜出的前 5 条
    all_acorn_wins_mask = speed_acorn_wins_mask | quality_acorn_wins_mask | tie_acorn_wins_mask
    print("\n     --- ACORN 胜出的前 5 条样本 (过滤前) ---")
    cols_to_show = ['QueryID', 'Lsearch', recall_col_U, recall_col_A, time_col_U, time_col_A]
    print(data[all_acorn_wins_mask][cols_to_show].head(5).to_string())
    # --- [结束] ---


    # --- 2. 性能差异过滤 (Time Difference) ---
    PERCENTAGE_THRESHOLD = 0.5  
    print(f"\n--- 步骤 2: 应用性能差异过滤 (时间差异阈值 > {PERCENTAGE_THRESHOLD:.0%}) ---")
    
    data['time_diff_abs'] = np.abs(data[time_col_U] - data[time_col_A])
    data['min_time'] = np.minimum(data[time_col_U], data[time_col_A])
    data['time_diff_percent'] = data['time_diff_abs'] / (data['min_time'] + 1e-9)

    # 仅在比较时间时，才需要考虑时间差异是否显著
    time_diff_significant_mask = (data['time_diff_percent'] > PERCENTAGE_THRESHOLD)

    # --- 3. 识别要保留的样本 (过滤) ---
    # 召回率不相等
    recall_diff_mask = (data[recall_col_A] != data[recall_col_U])
    # 召回率相等
    recall_tie_mask = (data[recall_col_A] == data[recall_col_U]) 

    # 规则A: 质量模式 & 召回率不相等 (必须保留)
    keep_mask_A = quality_comparison_mask & recall_diff_mask
    
    # 规则B: 速度模式 & 时间差异显著 (保留)
    keep_mask_B = both_recall_ok_mask & time_diff_significant_mask
    
    # 规则C: 质量模式 & 召回率相等 & 时间差异显著 (平局决胜，保留)
    keep_mask_C = quality_comparison_mask & recall_tie_mask & time_diff_significant_mask

    final_keep_mask = (keep_mask_A | keep_mask_B | keep_mask_C)
    
    final_data = data[final_keep_mask].copy()

    # 统计被过滤的样本
    filtered_count = original_count - len(final_data)
    print(f"     过滤统计: {filtered_count} 条样本被移除 (因差异过小或平局)。")
    print(f"     最终用于训练的样本数: {len(final_data)} ({len(final_data) / original_count:.2%} 保留)")

    # --- 过滤后统计 ---
    if not final_data.empty:
        final_speed_mode_mask = both_recall_ok_mask.loc[final_data.index]
        final_quality_mode_mask = quality_comparison_mask.loc[final_data.index]
        print(f"     过滤后保留的样本中:")
        print(f"       - {final_speed_mode_mask.sum()} 条属于“速度模式”")
        print(f"       - {final_quality_mode_mask.sum()} 条属于“质量模式”")
    # --- [结束] ---


    if final_data.empty:
        print(f"❌ 错误：所有样本均被过滤，没有数据可用于训练。")
        return

    # --- 4. 定义目标变量 y ---
    print(f"--- 步骤 3: 根据新逻辑定义最终目标变量 (y) ---")
    
    # 重新在 final_data 上定位 Mask
    final_both_recall_ok = both_recall_ok_mask.loc[final_data.index]
    final_quality_compare = quality_comparison_mask.loc[final_data.index]

    # 定义 y 值的条件
    conditions = [
        # --- “速度模式” (两者 Recall > 0.9) ---
        final_both_recall_ok & (final_data[time_col_A] < final_data[time_col_U]), # ACORN 更快
        final_both_recall_ok & (final_data[time_col_U] < final_data[time_col_A]), # UNG 更快

        # --- “质量模式” (至少一方 Recall <= 0.9) ---
        final_quality_compare & (final_data[recall_col_A] > final_data[recall_col_U]), # ACORN 召回率更高
        final_quality_compare & (final_data[recall_col_U] > final_data[recall_col_A]), # UNG 召回率更高
        
        # --- “平局决胜” (召回率相等，且至少一方 <= 0.9) ---
        final_quality_compare & (final_data[recall_col_A] == final_data[recall_col_U]) & (final_data[time_col_A] < final_data[time_col_U]), # ACORN 更快
        final_quality_compare & (final_data[recall_col_A] == final_data[recall_col_U]) & (final_data[time_col_U] < final_data[time_col_A])  # UNG 更快
    ]
    
    # 1 = 选 ACORN, 0 = 选 UNG
    choices = [
        1,  # ACORN (速度)
        0,  # UNG (速度)
        1,  # ACORN (质量)
        0,  # UNG (质量)
        1,  # ACORN (平局时间)
        0   # UNG (平局时间)
    ]

    y_values = np.select(conditions, choices, default=-1) 
    y = pd.Series(y_values, index=final_data.index)
    
    if (y == -1).any():
        print(f"❌ 警告: 在定义 y 值时出现逻辑错误，存在 { (y == -1).sum() } 个未覆盖的样本。")
        
    print(f"     目标变量分布:\n{y.value_counts(normalize=True).to_string()}")


    # --- 5. 阶段一：使用所有组合特征进行探索性训练 ---
    print("\n" + "="*80 + "\n### 阶段一: 使用所有组合特征进行探索性训练 ###\n" + "="*80)
    
    X_full = create_comprehensive_features_idea2(final_data) 
    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    valid_indices = X_full.dropna().index
    X_full_clean = X_full.loc[valid_indices]
    y_full_clean = y.loc[valid_indices]
    
    if len(X_full_clean) != len(X_full):
        print(f"     特征工程中移除了 {len(X_full) - len(X_full_clean)} 行 (因 inf/nan)。")

    if y_full_clean.empty:
        print(f"❌ 错误: 特征工程 (dropna) 后没有剩余样本。")
        return
    if y_full_clean.nunique() < 2:
         print(f"❌ 错误: 特征工程后 y 中只剩一类，无法训练。")
         return

    # --- 捕获 train_time_phase1 ---
    _, full_feature_importance, _, _, train_time_phase1 = train_and_evaluate_model(
        X=X_full_clean, y=y_full_clean, model_description="探索性模型 (All Features)", feature_list=X_full_clean.columns.tolist()
    )
    
    if full_feature_importance.empty:
        print("--- 阶段一训练失败 (可能由于 y 中只有一类)，终止流程。 ---")
        return

    # --- 6. 阶段二：使用最重要的特征进行最终模型训练 ---
    print("\n" + "="*80 + f"\n### 阶段二: 使用Top-{TOP_N_FEATURES_TO_SELECT}特征训练最终的精简模型 ###\n" + "="*80)
    top_features_list = full_feature_importance['Feature'].head(TOP_N_FEATURES_TO_SELECT).tolist()
    
    print(f"根据阶段一的结果，选择最重要的 {TOP_N_FEATURES_TO_SELECT} 个特征:")
    print(top_features_list)
    
    X_top_n = X_full_clean[top_features_list]
    
    # --- 捕获 train_time_phase2 ---
    final_model, top_n_importance, final_report_str, final_accuracy, train_time_phase2 = train_and_evaluate_model(
        X=X_top_n, y=y_full_clean, model_description=f"最终模型 (Top {TOP_N_FEATURES_TO_SELECT} Features)", feature_list=top_features_list
    )
    
    if final_model is None:
        print("--- 最终模型训练失败，无法保存。 ---")
        return

    # --- 7. 保存最终的精简模型 ---
    print("\n--- 正在保存最终的精简模型 ---")
    model_filename_joblib = os.path.join(output_dir, "idea2_selector_model_final.joblib")
    joblib.dump(final_model, model_filename_joblib)
    print(f"✅ 模型 (joblib) 已成功保存到: {model_filename_joblib}")

    model_filename_onnx = ""
    if SKL2ONNX_AVAILABLE:
        try:
            model_filename_onnx = os.path.join(output_dir, "idea2_selector_model_final.onnx")
            initial_type = [('float_input', FloatTensorType([None, len(top_features_list)]))]
            onnx_model = convert_sklearn(final_model, initial_types=initial_type, target_opset=15)
            with open(model_filename_onnx, "wb") as f: f.write(onnx_model.SerializeToString())
            print(f"✅ 模型 (ONNX) 已成功导出到: {model_filename_onnx}\n")
        except Exception as e:
            print(f"❌ 转换为 ONNX 格式时出错: {e}")
    else:
        print("- [警告] skl2onnx 库未安装，跳过 ONNX 转换。")

    # --- 8. 生成最终的综合分析报告 ---
    report_filename = os.path.join(output_dir, f"analysis_report_{model_name_prefix}.txt")
    print(f"--- 正在生成综合分析报告: {report_filename} ---")
    
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"两阶段随机森林模型分析报告 - {model_name_prefix}\n" + "="*80 + "\n\n")
        
        f.write("### 决策逻辑与数据过滤 ###\n" + "-"*60 + "\n")
        f.write(f"1. 召回率质量门 (Recall Quality Gate): {RECALL_QUALITY_GATE:.2f} (使用列: '{RECALL_METRIC_NAME}')\n")
        f.write(f"   - 规则: 如果 (Recall_A > {RECALL_QUALITY_GATE}) 且 (Recall_U > {RECALL_QUALITY_GATE})，则比较时间。\n")
        f.write(f"   - 规则: 否则，比较召回率 (选择更高者)。\n")
        f.write(f"2. 时间差异阈值 (Time Diff Threshold): {PERCENTAGE_THRESHOLD:.0%}\n")
        f.write(f"   - 规则: 仅在比较时间时 (包括召回率平局时)，才过滤掉差异 < {PERCENTAGE_THRESHOLD:.0%} 的样本。\n\n")
        
        f.write(f"数据过滤统计:\n")
        f.write(f"   - 原始样本数: {original_count}\n")
        f.write(f"   - 进入“速度模式” (两者 Recall OK): {count_speed_mode}\n")
        f.write(f"   - 进入“质量模式” (至少一方 Recall Bad): {count_quality_mode}\n")
        f.write(f"   - 因[差异过小]被过滤: {filtered_count}\n")
        f.write(f"   - 最终训练样本数: {len(final_data)}\n\n")
        
        f.write("### 阶段一: 探索性模型分析 (使用全部衍生特征) ###\n" + "-"*60 + "\n")
        f.write("此阶段的目标是自动发现哪些特征对于区分两种方法的性能至关重要。\n\n")
        
        # --- 添加时间到报告 ---
        f.write(f"训练耗时 (Training Time): {train_time_phase1:.4f} 秒\n\n")
        
        f.write("完整特征重要性排名:\n" + full_feature_importance.to_string() + "\n\n")
        
        f.write(f"### 阶段二: 最终精简模型分析 (使用Top-{TOP_N_FEATURES_TO_SELECT}特征) ###\n" + "-"*60 + "\n")
        f.write(f"此阶段使用在阶段一中发现的最重要的 {TOP_N_FEATURES_TO_SELECT} 个特征来训练一个轻量、高效的最终模型。\n\n")
        f.write(f"最终选定的特征列表: {top_features_list}\n\n")
        f.write(f"最终模型测试集准确率 (Accuracy): {final_accuracy:.4f}\n\n")
        
        # --- 添加时间到报告 ---
        f.write(f"最终模型训练耗时 (Training Time): {train_time_phase2:.4f} 秒\n\n")
        
        f.write("最终模型详细分类报告:\n" + final_report_str + "\n\n")
        
        f.write("### 模型部署 ###\n" + "-"*60 + "\n")
        f.write("训练好的最终精简模型已保存到以下文件，可用于生产环境：\n")
        f.write(f"- Joblib 格式: {model_filename_joblib}\n")
        if model_filename_onnx: f.write(f"- ONNX 格式: {model_filename_onnx} (用于 C++ 推理)\n")

    print(f"✅ 报告已成功保存到 {report_filename}")
    print(f"\n=== 两阶段分析流程 '{model_name_prefix}' 全部完成 ===")

def main():
    """主执行函数 (已更新以匹配分散的 query_features.csv 文件)"""
    # --- 路径配置 ---
    results_ung_base_dir = os.path.join(DATASET_RESULTS_DIR, "Results", "UNG-nTfalse")
    results_acorn_base_dir = os.path.join(DATASET_RESULTS_DIR, "Results", "ACORN-gamma")
    
    model_output_dir = os.path.join(MODEL_OUTPUT_BASE_DIR)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"✅ 本次训练的模型及报告将保存到: {model_output_dir}\n")

    # --- 查找成对的实验目录 ---
    result_pairs = find_result_pairs(results_ung_base_dir, results_acorn_base_dir)
    if not result_pairs: return
        
    # --- 全新的数据加载和合并流程 ---
    all_merged_data = []
    print("--- 开始逐一加载、合并每个实验对的数据 ---")
    
    for path_u, path_a in result_pairs:
        sub_dir_name = os.path.basename(path_u)
        print(f"\n-> 正在处理实验: {sub_dir_name}")

        # 1. 加载当前实验对的性能数据 (Y)
        perf_data = load_performance_data_idea2(path_u, path_a)
        if perf_data.empty:
            print(f"   - 警告: 未能加载性能数据，跳过此实验。")
            continue
        
        # 2. 定位并加载与当前实验对应的特征文件 (X)
        features_csv_path = os.path.join(path_u, "results", "query_features.csv")
        
        if not os.path.exists(features_csv_path):
            print(f"   - ❌ 错误: 在 {features_csv_path} 中找不到对应的特征文件，跳过此实验。")
            continue
            
        cols_to_load = ['QueryID'] + RAW_FEATURES_FROM_CPP
        features_df = pd.read_csv(features_csv_path, usecols=lambda c: c in cols_to_load)
        
        # 3. 将性能数据和特征数据进行合并
        merged_df = pd.merge(perf_data, features_df, on='QueryID', how='inner')
        
        if merged_df.empty:
            print(f"   - 警告: 性能数据与特征数据合并后为空，请检查 QueryID 是否匹配。跳过此实验。")
            continue

        print(f"   - ✅ 成功合并 {len(merged_df)} 条记录。")
        all_merged_data.append(merged_df)
            
    # --- 步骤 3: 连接所有数据 ---
    if not all_merged_data:
        print("\n❌ 错误: 未能成功加载并合并任何数据，程序终止。")
        return
        
    final_dataframe = pd.concat(all_merged_data, ignore_index=True)
    print(f"\n✅ 所有数据加载和合并完成，共获得 {len(final_dataframe)} 条原始记录。")

    
    # ######################################################################
    # ### --- 按 QueryID 筛选最佳行 (修正逻辑) --- ###
    # ######################################################################
    
    print(f"\n--- 正在为每个 QueryID 筛选“最佳”参数行 (使用分级排序逻辑) ---")
    
    # 1. 定义召回率和时间列
    recall_col_U = f'{RECALL_METRIC_NAME}_U'
    recall_col_A = f'{RECALL_METRIC_NAME}_A'
    
    if recall_col_U not in final_dataframe.columns or recall_col_A not in final_dataframe.columns:
        print(f"❌ 致命错误: 召回率列在 {recall_col_U} 或 {recall_col_A} 在合并数据后不存在。")
        return

    # 2. 计算用于排序的“联合指标”
    final_dataframe['__joint_recall'] = final_dataframe[recall_col_U] + final_dataframe[recall_col_A]
    final_dataframe['__joint_time'] = final_dataframe['Time_ms_U'] + final_dataframe['Time_ms_A']

    # 3. 定义优先级 Tier
    #    Tier 1: 速度模式 (两者 Recall > 0.9) -> 排序逻辑：Time Asc (快者优先), Recall Desc (备用)
    #    Tier 2: 质量模式 (至少一方 Recall <= 0.9) -> 排序逻辑：Recall Desc (高质优先), Time Asc (备用)
    
    speed_mode_mask = (final_dataframe[recall_col_U] > RECALL_QUALITY_GATE) & \
                      (final_dataframe[recall_col_A] > RECALL_QUALITY_GATE)
    
    final_dataframe['__priority_tier'] = np.where(speed_mode_mask, 1, 2)
    
    print(f"     - Tier 1 (速度模式) 候选行: {(final_dataframe['__priority_tier'] == 1).sum()} 条")
    print(f"     - Tier 2 (质量模式) 候选行: {(final_dataframe['__priority_tier'] == 2).sum()} 条")


    # 4. ### --- 核心修正：分治排序 (Split-Sort-Merge) --- ###
    # 目的：Tier 1 和 Tier 2 需要完全不同的排序标准，不能混在一起排。
    
    # 分离 Tier 1 和 Tier 2 数据
    df_tier1 = final_dataframe[final_dataframe['__priority_tier'] == 1].copy()
    df_tier2 = final_dataframe[final_dataframe['__priority_tier'] == 2].copy()
    
    # Tier 1 排序：优先时间 (升序)，次要召回 (降序)
    if not df_tier1.empty:
        df_tier1 = df_tier1.sort_values(
            by=['QueryID', '__joint_time', '__joint_recall'],
            ascending=[True, True, False]
        )
        
    # Tier 2 排序：优先召回 (降序)，次要时间 (升序) <--- 修正点：确保低质量模式下选召回率最高的
    if not df_tier2.empty:
        df_tier2 = df_tier2.sort_values(
            by=['QueryID', '__joint_recall', '__joint_time'],
            ascending=[True, False, True]
        )
    
    # 合并：Tier 1 必须排在 Tier 2 前面 (因为 Tier 1 优先级更高)
    df_sorted = pd.concat([df_tier1, df_tier2], ignore_index=True)

    # 5. 保留每个 QueryID 的第一行
    #    - 如果 Query X 有 Tier 1 数据，它们在最上面，且按时间排好了，取第一个即为“达标且最快”。
    #    - 如果 Query Y 只有 Tier 2 数据，它们在下面，但该区块内按召回排好了，取第一个即为“不达标但质量最高”。
    df_filtered_best_rows = df_sorted.drop_duplicates(subset='QueryID', keep='first').copy()
    
    # 6. 清理临时排序列
    df_filtered_best_rows = df_filtered_best_rows.drop(
        columns=['__joint_recall', '__joint_time', '__priority_tier']
    )
    
    original_count_all_rows = len(final_dataframe)
    filtered_count_unique_queries = len(df_filtered_best_rows)
    
    print(f"✅ 筛选完成：从 {original_count_all_rows} 条原始记录中，为 {filtered_count_unique_queries} 个独立 QueryID 各自筛选出1条“理想”记录。")

    # --- 步骤 4: 执行分析流程 (使用筛选后的数据) ---
    run_two_stage_analysis_idea2(df_filtered_best_rows, "idea2_selector_best_params", model_output_dir)

if __name__ == "__main__":
    main()