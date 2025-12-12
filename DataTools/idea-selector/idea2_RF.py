import os
import glob
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
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
TOP_N_FEATURES_TO_SELECT = 6

RECALL_METRIC_NAME = "Recall" 
RECALL_QUALITY_GATE = 0.9 

# 时间显著性阈值 
TIME_DIFF_THRESHOLD = 0.2

# 只有目录名包含此字符串时才会被加载。若设为 "" 或 None 则不限制。
DIR_FILTER_KEYWORD = "GT[GT_query_C_D"

# --- 路径配置 ---
DATASET_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, DATASET_NAME)
MODEL_OUTPUT_BASE_DIR = os.path.join(DATASET_RESULTS_DIR, "SelectModels/idea2-time")

DIR_NAME_UNG = "UNG-nTfalse"
DIR_NAME_ACORN = "ACORN-gamma"
DIR_NAME_IMP = "ACORN-gamma-improved" 

# --- 原始特征列 ---
RAW_FEATURES_FROM_CPP = [
    'NumEntries', 
    'NumDescendants', 
    'TotalCoverage'
]

def find_result_triplets(dir_ung, dir_acorn, dir_imp):
    print(f"--- 正在扫描目录以查找三算法配对结果 ---")
    if DIR_FILTER_KEYWORD:
        print(f"   [过滤条件] 仅包含: {DIR_FILTER_KEYWORD}")

    triplets = []
    if not all(os.path.isdir(d) for d in [dir_ung, dir_acorn, dir_imp]):
        print(f"❌ 警告: 部分实验目录不存在。")
        return triplets

    for sub_dir_name in os.listdir(dir_ung):
        # [修改点] 目录名过滤逻辑
        if DIR_FILTER_KEYWORD and DIR_FILTER_KEYWORD not in sub_dir_name:
            continue

        p_u = os.path.join(dir_ung, sub_dir_name)
        p_a = os.path.join(dir_acorn, sub_dir_name)
        p_i = os.path.join(dir_imp, sub_dir_name)

        if os.path.isdir(p_u) and os.path.isdir(p_a) and os.path.isdir(p_i):
            triplets.append((p_u, p_a, p_i))
            print(f"   [匹配成功] -> {sub_dir_name}")
            
    if not triplets:
        print(f"⚠️ 警告: 未找到符合过滤条件的目录。")
        
    return triplets

def load_performance_data_three_algos(path_u, path_a, path_i):
    try:
        def get_csv(path):
            pattern = os.path.join(glob.escape(path), "results", "query_details*.csv")
            files = glob.glob(pattern)
            return files[0] if files else None

        csv_u, csv_a, csv_i = get_csv(path_u), get_csv(path_a), get_csv(path_i)
        if not all([csv_u, csv_a, csv_i]): return pd.DataFrame()

        # [修改点 1] 指定读取 'search_time_ms' (纯搜索时间) 而不是 'Time_ms' (总时间)
        cols = ['Lsearch', 'QueryID', 'search_time_ms', RECALL_METRIC_NAME]
        
        df_u = pd.read_csv(csv_u, usecols=cols)
        df_a = pd.read_csv(csv_a, usecols=cols)
        df_i = pd.read_csv(csv_i, usecols=cols)

        # [修改点 2] 立即将 'search_time_ms' 重命名为 'Time_ms'
        df_u.rename(columns={'search_time_ms': 'Time_ms'}, inplace=True)
        df_a.rename(columns={'search_time_ms': 'Time_ms'}, inplace=True)
        df_i.rename(columns={'search_time_ms': 'Time_ms'}, inplace=True)

        merged_ua = pd.merge(df_u, df_a, on=['Lsearch', 'QueryID'], suffixes=('_U', '_A'))
        df_i = df_i.rename(columns={'Time_ms': 'Time_ms_Imp', RECALL_METRIC_NAME: f'{RECALL_METRIC_NAME}_Imp'})
        
        final_merged = pd.merge(merged_ua, df_i, on=['Lsearch', 'QueryID'])
        return final_merged
    except Exception as e:
        print(f"加载数据出错: {e}")
        return pd.DataFrame()

def create_comprehensive_features_idea2(df):
    raw_df = df.copy()
    features = pd.DataFrame(index=raw_df.index)

    features['LogNumEntries'] = np.log1p(raw_df['NumEntries'])
    features['LogNumDescendants'] = np.log1p(raw_df['NumDescendants'])
    features['LogTotalCoverage'] = np.log1p(raw_df['TotalCoverage'])
    features['CoverageSquared'] = raw_df['TotalCoverage'] ** 2
    features['DescendantsSquared'] = raw_df['NumDescendants'] ** 2
    features['DescCovInteraction'] = raw_df['NumDescendants'] * raw_df['TotalCoverage']
    features['EntriesCovInteraction'] = raw_df['NumEntries'] * raw_df['TotalCoverage']
    features['EntriesDescInteraction'] = raw_df['NumEntries'] * raw_df['NumDescendants']
    features['DescPerEntry'] = raw_df['NumDescendants'] / (raw_df['NumEntries'] + 1e-9)
    features['CovPerEntry'] = raw_df['TotalCoverage'] / (raw_df['NumEntries'] + 1e-9)

    final_features = pd.concat([raw_df[RAW_FEATURES_FROM_CPP], features], axis=1)
    return final_features

def determine_label(row):
    """
    打标逻辑 (含阈值过滤)
    Returns: 0, 1, 2, or -1 (Noise)
    """
    r_u, r_a, r_i = row[f'{RECALL_METRIC_NAME}_U'], row[f'{RECALL_METRIC_NAME}_A'], row[f'{RECALL_METRIC_NAME}_Imp']
    t_u, t_a, t_i = row['Time_ms_U'], row['Time_ms_A'], row['Time_ms_Imp']
    
    recalls = [r_u, r_a, r_i]
    times = [t_u, t_a, t_i]
    labels = [0, 1, 2]
    
    # 1. 资格赛: 谁达标了? (>= 0.9)
    qualified_indices = [i for i, r in enumerate(recalls) if r >= RECALL_QUALITY_GATE]
    
    if qualified_indices:
        # --- 竞速模式 ---
        candidates = [(times[i], labels[i]) for i in qualified_indices]
        candidates.sort(key=lambda x: x[0]) # 按时间升序
        
        best_time, best_label = candidates[0]
        
        # 检查时间差异显著性
        if len(candidates) > 1:
            second_time, _ = candidates[1]
            diff_percent = (second_time - best_time) / (best_time + 1e-9)
            
            if diff_percent < TIME_DIFF_THRESHOLD:
                return -1 
                
        return best_label
    else:
        # --- 质量模式 (全员不及格) ---
        winner_idx = np.argmax(recalls)
        return labels[winner_idx]

def train_and_evaluate_model(X, y, model_desc, feature_list):
    print(f"\n--- 正在训练: {model_desc} ---")
    if y.nunique() < 2: return None, pd.DataFrame(), "No Report", 0.0, 0.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    
    start = time.perf_counter()
    rf.fit(X_train, y_train)
    duration = time.perf_counter() - start
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    target_names_map = {0: 'UNG', 1: 'ACORN', 2: 'Imp'}
    unique_labels = sorted(y_test.unique())
    present_names = [target_names_map[i] for i in unique_labels]
    report_str = classification_report(y_test, y_pred, labels=unique_labels, target_names=present_names)
    
    importance_df = pd.DataFrame({'Feature': feature_list, 'Importance': rf.feature_importances_})\
        .sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print(f"   -> 耗时: {duration:.4f}s | Acc: {acc:.4f}")
    return rf, importance_df, report_str, acc, duration

def generate_detailed_report_content(df, y, filtered_count, report_filename):
    """生成报告"""
    r_u = df[f'{RECALL_METRIC_NAME}_U']
    r_a = df[f'{RECALL_METRIC_NAME}_A']
    r_i = df[f'{RECALL_METRIC_NAME}_Imp']
    
    mask_tier1 = (r_u >= RECALL_QUALITY_GATE) & (r_a >= RECALL_QUALITY_GATE) & (r_i >= RECALL_QUALITY_GATE)
    df_tier1 = df[mask_tier1]
    y_tier1 = y[mask_tier1]
    
    df_tier2 = df[~mask_tier1]
    y_tier2 = y[~mask_tier1]
    
    target_map = {0: 'UNG', 1: 'ACORN', 2: 'Imp'}

    with open(report_filename, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write("### 详细数据分布统计 (Detailed Statistics) ###\n")
        f.write("="*80 + "\n\n")

        f.write(f"1. 阈值过滤统计 (Threshold: {TIME_DIFF_THRESHOLD:.0%}):\n")
        f.write(f"   - 有效样本 (Valid): {len(df)} 条\n")
        f.write(f"   - 丢弃噪声 (Dropped): {filtered_count} 条 (胜出优势不明显)\n\n")

        f.write("2. 模式分布 (Tier Distribution):\n")
        f.write(f"   - Tier 1 (全员达标/竞速): {len(df_tier1)} 条\n")
        f.write(f"   - Tier 2 (质量优先):       {len(df_tier2)} 条\n\n")

        f.write("3. Tier 1 胜负分布 (Both Recall >= 0.9):\n")
        if not y_tier1.empty:
            counts = y_tier1.value_counts().sort_index()
            for lbl in counts.index:
                f.write(f"   - {target_map.get(lbl, lbl)}: {counts[lbl]} ({counts[lbl]/len(y_tier1):.2%})\n")
        else: f.write("   (无数据)\n")

        f.write("\n4. Tier 2 胜负分布 (At least one Recall < 0.9):\n")
        if not y_tier2.empty:
            counts = y_tier2.value_counts().sort_index()
            for lbl in counts.index:
                f.write(f"   - {target_map.get(lbl, lbl)}: {counts[lbl]} ({counts[lbl]/len(y_tier2):.2%})\n")
        else: f.write("   (无数据)\n")
        
        # --- 打印案例 ---
        f.write("\n" + "="*80 + "\n### 真实案例采样 ###\n" + "="*80 + "\n")
        cols = ['QueryID', f'{RECALL_METRIC_NAME}_U', f'{RECALL_METRIC_NAME}_A', f'{RECALL_METRIC_NAME}_Imp',
                'Time_ms_U', 'Time_ms_A', 'Time_ms_Imp']
        
        def print_cases(sub_df, sub_y, name):
            f.write(f"--- {name} ---\n")
            for lbl in sorted(sub_y.unique()):
                idxs = sub_y[sub_y == lbl].index
                samples = sub_df.loc[idxs].head(3)
                f.write(f"  > 赢家: {target_map.get(lbl, lbl)}\n")
                if samples.empty: continue
                
                header = f"    {'QueryID':<8} | {'R_U':<6} {'R_A':<6} {'R_I':<6} | {'T_U':<6} {'T_A':<6} {'T_I':<6}"
                f.write(header + "\n    " + "-"*len(header) + "\n")
                for _, r in samples.iterrows():
                    f.write(f"    {r['QueryID']:<8} | {r[cols[1]]:.4f} {r[cols[2]]:.4f} {r[cols[3]]:.4f} | "
                            f"{r[cols[4]]:<6.1f} {r[cols[5]]:<6.1f} {r[cols[6]]:<6.1f}\n")
                f.write("\n")
        
        print_cases(df_tier1, y_tier1, "Tier 1 (竞速)")
        print_cases(df_tier2, y_tier2, "Tier 2 (质量)")

def run_analysis_pipeline(df, output_dir):
    print("\n" + "="*80 + "\n### 开始两阶段模型训练流程 ###\n" + "="*80)
    
    # 1. 生成标签 (含噪声标记 -1)
    print("--- 生成目标标签 (y) 并应用阈值过滤 ---")
    y_raw = df.apply(determine_label, axis=1)
    
    # 过滤掉 -1 的噪声行
    valid_mask = (y_raw != -1)
    dropped_count = (~valid_mask).sum()
    
    df_clean = df[valid_mask].copy()
    y = y_raw[valid_mask].copy()
    
    print(f"原始样本: {len(df)} -> 丢弃(噪声): {dropped_count} -> 有效样本: {len(df_clean)}")
    print(f"有效样本标签分布:\n{y.value_counts(normalize=True).to_string()}")

    # 2. 特征工程
    X_full = create_comprehensive_features_idea2(df_clean)
    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_idx = X_full.dropna().index
    
    X_train_ready = X_full.loc[valid_idx]
    y_train_ready = y.loc[valid_idx]
    
    # 准备用于报告的原始信息 (对应有效行)
    df_report_ready = df_clean.loc[valid_idx]

    if X_train_ready.empty:
        print("❌ 错误: 无有效样本。")
        return

    # 初始化报告
    report_filename = os.path.join(output_dir, "analysis_report_3algos.txt")
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"三算法选择器训练报告 (含阈值过滤 {TIME_DIFF_THRESHOLD:.0%})\n生成: {datetime.now()}\n{'='*80}\n\n")
        f.write(f"数据集: {DATASET_NAME}\n")
        f.write(f"目录过滤关键词: {DIR_FILTER_KEYWORD}\n\n")

    # 3. 阶段一
    rf_p1, imp_p1, report_p1, acc_p1, time_p1 = train_and_evaluate_model(
        X_train_ready, y_train_ready, "阶段一: 全特征", X_train_ready.columns.tolist()
    )
    if rf_p1 is None: return

    with open(report_filename, "a", encoding="utf-8") as f:
        # [修改点 1]在此处添加 Training Time 的输出
        f.write(f"### Phase 1 ###\nTraining Time: {time_p1:.4f}s\nAcc: {acc_p1:.4f}\n\n{report_p1}\n\nFeatures:\n{imp_p1.to_string()}\n\n")

    # 4. 阶段二
    top_feats = imp_p1['Feature'].head(TOP_N_FEATURES_TO_SELECT).tolist()
    print(f"\nTop Features: {top_feats}")
    
    X_top = X_train_ready[top_feats]
    rf_p2, imp_p2, report_p2, acc_p2, time_p2 = train_and_evaluate_model(
        X_top, y_train_ready, f"阶段二: Top-{TOP_N_FEATURES_TO_SELECT}", top_feats
    )

    # 保存
    joblib.dump(rf_p2, os.path.join(output_dir, "idea2_selector_model_final.joblib"))
    if SKL2ONNX_AVAILABLE:
        initial_type = [('float_input', FloatTensorType([None, len(top_feats)]))]
        onnx_model = convert_sklearn(rf_p2, initial_types=initial_type, target_opset=15)
        with open(os.path.join(output_dir, "idea2_selector_model_final.onnx"), "wb") as f: 
            f.write(onnx_model.SerializeToString())

    with open(report_filename, "a", encoding="utf-8") as f:
        # [修改点 2]在此处添加 Training Time 的输出
        f.write(f"### Phase 2 ###\nTraining Time: {time_p2:.4f}s\nAcc: {acc_p2:.4f}\n\n{report_p2}\n\nFeatures:\n{imp_p2.to_string()}\n")

    # 5. 生成详细统计 (传入丢弃数量)
    generate_detailed_report_content(df_report_ready, y_train_ready, dropped_count, report_filename)
    
    print(f"\n✅ 完成。报告: {report_filename}")

def get_best_performance_for_algo(df, algo_suffix):
    """
    解耦逻辑：为单个算法筛选最佳表现。
    逻辑：
    1. 优先找 Recall >= 0.9 的，按时间升序。
    2. 如果没有，找 Recall 最高的，按时间升序。
    """
    recall_col = f'{RECALL_METRIC_NAME}_{algo_suffix}'
    time_col = f'Time_ms_{algo_suffix}'
    
    # 只提取该算法相关的列 + QueryID
    sub_df = df[['QueryID', recall_col, time_col]].copy()
    
    # 1. 拆分达标组和未达标组
    mask_qualified = sub_df[recall_col] >= RECALL_QUALITY_GATE
    
    # Group A: 达标 (Recall >= 0.9) -> 越快越好
    df_qualified = sub_df[mask_qualified].sort_values(
        by=['QueryID', time_col], 
        ascending=[True, True]
    )
    
    # Group B: 未达标 -> Recall 越高越好，Recall 相同则时间越短
    df_unqualified = sub_df[~mask_qualified].sort_values(
        by=['QueryID', recall_col, time_col], 
        ascending=[True, False, True]
    )
    
    # 2. 合并：优先保留 Group A，然后是 Group B
    # 因为 drop_duplicates保留第一条，所以我们要把最好的排在前面
    df_sorted = pd.concat([df_qualified, df_unqualified], ignore_index=True)
    
    # 3. 去重：每个 QueryID 只留一条最好的
    df_best = df_sorted.drop_duplicates(subset='QueryID', keep='first')
    
    # 设置索引以便后续合并
    return df_best.set_index('QueryID')

def main():
    # --- 1. 路径定义 ---
    path_ung = os.path.join(DATASET_RESULTS_DIR, "Results", DIR_NAME_UNG)
    path_acorn = os.path.join(DATASET_RESULTS_DIR, "Results", DIR_NAME_ACORN)
    path_imp = os.path.join(DATASET_RESULTS_DIR, "Results", DIR_NAME_IMP)
    os.makedirs(MODEL_OUTPUT_BASE_DIR, exist_ok=True)
    
    # --- 2. 查找匹配的三元组目录 ---
    triplets = find_result_triplets(path_ung, path_acorn, path_imp)
    if not triplets: 
        print("未找到匹配的实验结果目录，程序退出。")
        return

    # --- 3. 加载并合并所有数据 ---
    all_data = []
    print("\n--- 加载数据 ---")
    for p_u, p_a, p_i in triplets:
        perf_df = load_performance_data_three_algos(p_u, p_a, p_i)
        if perf_df.empty: continue
        
        # 加载特征文件 (特征与 QueryID 绑定，与算法参数无关，取哪一行的都一样)
        feat_path = os.path.join(p_u, "results", "query_features.csv")
        if not os.path.exists(feat_path): 
            print(f"特征文件缺失: {feat_path}")
            continue
            
        cols_feat = ['QueryID'] + RAW_FEATURES_FROM_CPP
        feat_df = pd.read_csv(feat_path, usecols=lambda c: c in cols_feat)
        
        # 合并
        merged = pd.merge(perf_df, feat_df, on='QueryID', how='inner')
        all_data.append(merged)
        print(f"   -> {os.path.basename(p_u)}: {len(merged)} 条")
        
    if not all_data: 
        print("没有有效数据可供训练。")
        return
    full_df = pd.concat(all_data, ignore_index=True)
    
    # ==============================================================================
    # 核心修改：解耦筛选逻辑 (Decoupled Selection)
    # ==============================================================================
    print(f"\n--- 筛选最佳配置 (Decoupled: 各算法独立寻优) ---")
    
    # 1. 分别寻找三个算法的最佳表现
    best_u = get_best_performance_for_algo(full_df, 'U')
    best_a = get_best_performance_for_algo(full_df, 'A')
    best_i = get_best_performance_for_algo(full_df, 'Imp')
    
    # 2. 拼装成一行
    # 使用 inner join 确保只有三个算法都有数据的 QueryID 才会被保留
    df_best_perf = pd.concat([best_u, best_a, best_i], axis=1, join='inner').reset_index()
    
    # 3. 找回特征列
    # 只需要从原始 full_df 中对每个 QueryID 随便取一行特征即可
    df_features = full_df[['QueryID'] + RAW_FEATURES_FROM_CPP].drop_duplicates(subset='QueryID')
    
    # 4. 最终合并
    df_final = pd.merge(df_best_perf, df_features, on='QueryID', how='inner')
    
    print(f"✅ 筛选完成: {len(df_final)} 条 (已应用解耦拼装策略)")
    
    # --- 4. 运行训练流程 ---
    run_analysis_pipeline(df_final, MODEL_OUTPUT_BASE_DIR)

if __name__ == "__main__":
    main()