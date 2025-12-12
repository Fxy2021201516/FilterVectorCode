import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import time  # <--- 修改 1: 导入 time 库
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_text

# --- 全局配置 ---
BASE_RESULTS_DIR = "/home/fengxiaoyao/1023FilterVector/FilterVectorResults"
DATASET_NAME = "BookReviews"
TOP_N_FEATURES_TO_SELECT = 7  # 在第二阶段中，选择最重要的特征数量

# 根据上述配置自动生成路径
DATASET_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, DATASET_NAME)
MODEL_OUTPUT_BASE_DIR = os.path.join(DATASET_RESULTS_DIR, "SelectModels/idea1")


def find_result_pairs(dir_false, dir_true):
   """
   扫描新的结果目录结构，查找并匹配 nT=false 和 nT=true 的成对实验结果。
   """
   print(f"--- 正在扫描以下目录以查找成对的实验结果 ---")
   print(f"    方法 F (nT=false) 目录: '{dir_false}'")
   print(f"    方法 T (nT=true)  目录: '{dir_true}'")
   
   pairs = []
   if not os.path.isdir(dir_false) or not os.path.isdir(dir_true):
      print(f"警告: 必要的实验目录不存在，请检查路径。")
      return pairs

   # 遍历 nT=false 的结果目录
   for sub_dir_name in os.listdir(dir_false):
      path_false = os.path.join(dir_false, sub_dir_name)
      if os.path.isdir(path_false):
         # 在 nT=true 目录中构建对应的路径
         path_true = os.path.join(dir_true, sub_dir_name)
         # 检查对应的 nT=true 目录是否存在
         if os.path.isdir(path_true):
               pairs.append((path_false, path_true))
               print(f"  [配对成功] -> {sub_dir_name}")
               
   if not pairs:
      print("警告: 未找到任何成对的 (nT=true vs nT=false) 实验结果目录。")
   else:
      print(f"\n成功找到 {len(pairs)} 对实验结果。\n")
   return pairs

def load_performance_data(path_false, path_true):
   """
   数据加载和合并成对实验的性能指标。
   """
   try:
      csv_false_path_pattern = os.path.join(glob.escape(path_false), "results", "query_details*.csv")
      csv_true_path_pattern = os.path.join(glob.escape(path_true), "results", "query_details*.csv")
      
      csv_false_list = glob.glob(csv_false_path_pattern)
      csv_true_list = glob.glob(csv_true_path_pattern)

      if not csv_false_list or not csv_true_list:
         print(f"❌ 错误: 在目录中找不到 'query_details*.csv' 文件。")
         return pd.DataFrame()

      # 仅加载性能指标和ID列
      cols_to_keep = ['Lsearch', 'QueryID', 'MinSupersetT_ms']
      df_false = pd.read_csv(csv_false_list[0], usecols=cols_to_keep)
      df_true = pd.read_csv(csv_true_list[0], usecols=cols_to_keep)
      
      # 合并以获取 MinSupersetT_ms_F 和 MinSupersetT_ms_T
      perf_df = pd.merge(df_false, df_true, on=['Lsearch', 'QueryID'], suffixes=('_F', '_T'))
      
      return perf_df
      
   except Exception as e:
      print(f"加载性能数据时发生未知错误: {e}")
      return pd.DataFrame()
      
def create_comprehensive_features(df):
   """
   从最原始的指标(QuerySize, CandSize, Trie静态特征)中自动组合出丰富的衍生特征。
   (已更新以匹配 query_features.csv 的实际列名)
   """
   print("--- 正在从原始指标进行特征工程，创建丰富的衍生特征 ---")
   
   # 使用 .copy() 避免 SettingWithCopyWarning
   raw_df = df.copy()
   
   # --- 为了安全地计算，预处理原始数据 ---
   raw_df['Log_CandSize'] = np.log1p(raw_df['CandSize'])
   raw_df['Log_QuerySize'] = np.log1p(raw_df['QuerySize'])
   raw_df['Log_TrieTotalNodes'] = np.log1p(raw_df['TrieTotalNodes'])

   # --- 特征组合 ---
   features = pd.DataFrame(index=raw_df.index)
   
   # 1. 基础比率特征 (查询与Trie树结构的关系)
   features['Cand_Coverage_Ratio'] = raw_df['CandSize'] / (raw_df['TrieTotalNodes'] + 1e-9)
   features['Query_Cardinality_Ratio'] = raw_df['QuerySize'] / (raw_df['TrieLabelCardinality'] + 1e-9)

   # 2. 密度和选择性特征
   features['Query_Path_Density'] = raw_df['QuerySize'] * raw_df['TrieAvgBranchingFactor']
   avg_nodes_per_label = raw_df['TrieTotalNodes'] / (raw_df['TrieLabelCardinality'] + 1e-9)
   features['Cand_Selectivity'] = avg_nodes_per_label / (raw_df['CandSize'] + 1e-9)
   features['Query_Cand_Ratio'] = raw_df['QuerySize'] / (raw_df['CandSize'] + 1e-9)

   # 3. 交互特征
   features['Cand_x_Query_Interaction'] = raw_df['CandSize'] * raw_df['QuerySize']
   features['Branching_x_CandSize'] = raw_df['TrieAvgBranchingFactor'] * raw_df['CandSize']
   features['Branching_x_QuerySize'] = raw_df['TrieAvgBranchingFactor'] * raw_df['QuerySize']

   # 4. 多项式特征 (二次方)
   features['CandSize_sq'] = raw_df['CandSize'] ** 2
   features['QuerySize_sq'] = raw_df['QuerySize'] ** 2
   
   # 5. 对数交互特征
   features['LogCand_x_LogQuery'] = raw_df['Log_CandSize'] * raw_df['Log_QuerySize']

   # 最后，将最原始的特征也加进来，让模型自己判断它们的原始值是否重要
   final_features = pd.concat([raw_df[['QuerySize', 'CandSize']], features], axis=1)

   # base_feature_columns = ['QuerySize', 'CandSize', 'TrieTotalNodes', 'TrieLabelCardinality', 'TrieAvgBranchingFactor']
   # final_features = pd.concat([raw_df[base_feature_columns], features], axis=1)
   
   print(f"特征工程完成，共生成 {final_features.shape[1]} 个特征。")
   return final_features

def train_and_evaluate_model(X, y, model_description, feature_list):
   """
   一个通用的函数，用于训练随机森林模型、评估其性能并返回特征重要性。
   (已修改：增强鲁棒性，支持单类别数据训练)
   """
   print(f"\n--- 开始处理模型: '{model_description}' ---")
   print(f"使用 {X.shape[1]} 个特征进行训练。")

   # --- 鲁棒性修改 1: 处理单类别导致的 stratify 报错 ---
   # 如果只有一种类别（例如全是1），stratify会报错或无意义，因此需要动态调整
   unique_classes = y.unique()
   if len(unique_classes) < 2:
       print(f"⚠️ 注意: 数据集仅包含单一类别 {unique_classes}，禁用分层抽样 (stratify)。")
       stratify_param = None
   else:
       stratify_param = y

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_param)
   
   print(f"训练集样本: {len(X_train)}, 测试集样本: {len(X_test)}")
   
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
   
   # --- 鲁棒性修改 2: 强制指定 labels 和 zero_division ---
   # 即使 y_test 中只有一类，这里也会强制输出两列，缺失的一列全为 0
   class_report_str = classification_report(
       y_test, 
       y_pred, 
       labels=[0, 1], 
       target_names=['应选方法F', '应选方法T'],
       zero_division=0
   )
   
   print(f"模型 '{model_description}' 训练完成。")
   print(f"袋外分数 (OOB Score): {rf.oob_score_:.4f}")
   print(f"测试集准确率 (Accuracy): {accuracy:.4f}")
   print("详细分类报告:\n", class_report_str)

   # 特征重要性
   importances = rf.feature_importances_
   feature_importance_df = pd.DataFrame({
      'Feature': feature_list,
      'Importance': importances
   }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
   
   print("\n特征重要性排名:")
   print(feature_importance_df)
   
   # --- 返回训练时间 ---
   return rf, feature_importance_df, class_report_str, accuracy, training_duration_sec

   
def run_two_stage_analysis(data, model_name_prefix, output_dir):
   """
   执行完整的两阶段分析流程 (已修改：增加详细统计打印)
   """
   # --- 0. 数据准备和预处理 ---
   print(f"\n--- 开始为 '{model_name_prefix}' 执行两阶段分析 ---")

   # 过滤掉性能差异过小的模糊样本
   PERCENTAGE_THRESHOLD = 0.4
   data['time_diff_abs'] = np.abs(data['MinSupersetT_ms_T'] - data['MinSupersetT_ms_F'])
   data['min_time'] = np.minimum(data['MinSupersetT_ms_T'], data['MinSupersetT_ms_F'])
   data['time_diff_percent'] = data['time_diff_abs'] / (data['min_time'] + 1e-9)

   original_count = len(data)
   significant_diff_data = data[data['time_diff_percent'] > PERCENTAGE_THRESHOLD].copy()
   filtered_count = len(significant_diff_data)

   print(f"--- 应用决策边界过滤 ---")
   print(f"原始样本数: {original_count}")
   print(f"过滤阈值 (性能差异): > {PERCENTAGE_THRESHOLD:.0%}")
   print(f"过滤后剩余样本数: {filtered_count} ({filtered_count / original_count:.2%} 保留)")

   if significant_diff_data.empty:
      print(f"❌ 错误：阈值({PERCENTAGE_THRESHOLD:.0%})过高，所有样本被过滤！")
      return
      
   # 定义目标变量
   significant_diff_data['choose_method_T'] = (significant_diff_data['MinSupersetT_ms_T'] < significant_diff_data['MinSupersetT_ms_F']).astype(int)
   y = significant_diff_data['choose_method_T']
   
   # --- 新增：打印详细的类别统计 ---
   print("\n" + "="*40)
   print(">>> 类别分布统计 (Ground Truth) <<<")
   counts = y.value_counts().sort_index()
   count_0 = counts.get(0, 0)
   count_1 = counts.get(1, 0)
   print(f"  方法 F 更快 (Label 0): {count_0} 个")
   print(f"  方法 T 更快 (Label 1): {count_1} 个")
   
   if len(y.unique()) < 2:
       print(f"⚠️ 警告: 只有一种类别存在！模型将学习永远预测该类别。")
   print("="*40 + "\n")
   # ----------------------------------
   
   # --- 1. 阶段一：使用所有组合特征进行探索性训练 ---
   print("\n" + "="*80)
   print("### 阶段一: 使用所有组合特征进行探索性训练 ###")
   print("="*80)

   # 从过滤后的高质量数据中创建特征
   X_full = create_comprehensive_features(significant_diff_data)
   
   # 清理可能产生的NaN/Inf值
   X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
   valid_indices = X_full.dropna().index
   X_full = X_full.loc[valid_indices]
   y_full = y.loc[valid_indices]

   # --- 捕获 train_time_phase1 ---
   _, full_feature_importance, _, _, train_time_phase1 = train_and_evaluate_model(
      X=X_full, 
      y=y_full, 
      model_description="探索性模型 (All Features)", 
      feature_list=X_full.columns.tolist()
   )

   # --- 2. 阶段二：使用最重要的特征进行最终模型训练 ---
   print("\n" + "="*80)
   print(f"### 阶段二: 使用Top-{TOP_N_FEATURES_TO_SELECT}特征训练最终的精简模型 ###")
   print("="*80)
   
   top_features_list = full_feature_importance['Feature'].head(TOP_N_FEATURES_TO_SELECT).tolist()
   
   print(f"根据阶段一的结果，选择最重要的 {TOP_N_FEATURES_TO_SELECT} 个特征进行最终模型训练:")
   print(top_features_list)
   
   X_top_n = X_full[top_features_list]
   
   # --- 捕获 train_time_phase2 ---
   final_model, top_n_importance, final_report_str, final_accuracy, train_time_phase2 = train_and_evaluate_model(
      X=X_top_n, 
      y=y_full, 
      model_description=f"最终模型 (Top {TOP_N_FEATURES_TO_SELECT} Features)",
      feature_list=top_features_list,
   )

   # --- 3. 保存最终的精简模型 ---
   print("\n--- 正在保存最终的精简模型 ---")
   model_filename = os.path.join(output_dir, f"idea1_selector_model_final.joblib")
   joblib.dump(final_model, model_filename)
   print(f"模型已成功保存到: {model_filename}")

   try:
      from skl2onnx import convert_sklearn
      from skl2onnx.common.data_types import FloatTensorType
      onnx_model_filename = os.path.join(output_dir, f"idea1_selector_model_final.onnx")
      initial_type = [('float_input', FloatTensorType([None, len(top_features_list)]))]
      onnx_model = convert_sklearn(final_model, initial_types=initial_type, target_opset=15)
      with open(onnx_model_filename, "wb") as f:
         f.write(onnx_model.SerializeToString())
      print(f"模型已成功导出为 ONNX 格式: {onnx_model_filename}\n")
   except ImportError:
      onnx_model_filename = "N/A (skl2onnx not installed)"
      print(" skl2onnx 未安装，跳过 ONNX 转换。")


   # --- 4. 生成最终的综合分析报告 ---
   report_filename = os.path.join(output_dir, f"analysis_report_{model_name_prefix}.txt")
   print(f"--- 正在生成综合分析报告: {report_filename} ---")
   
   with open(report_filename, "w", encoding="utf-8") as f:
      f.write(f"两阶段随机森林模型分析报告 - {model_name_prefix}\n")
      f.write("="*80 + "\n\n")
      
      # --- 新增：报告中也记录分布 ---
      f.write("### 数据集类别分布 ###\n")
      f.write(f"  Label 0 (F更快): {count_0}\n")
      f.write(f"  Label 1 (T更快): {count_1}\n\n")

      f.write("### 阶段一: 探索性模型分析 (使用全部衍生特征) ###\n")
      f.write("-"*60 + "\n")
      f.write("此阶段的目标是自动发现哪些特征对于区分两种方法的性能至关重要。\n\n")
      
      # --- 添加时间到报告 ---
      f.write(f"训练耗时 (Training Time): {train_time_phase1:.4f} 秒\n\n")
      
      f.write("完整特征重要性排名:\n")
      f.write(full_feature_importance.to_string() + "\n\n")

      f.write(f"### 阶段二: 最终精简模型分析 (使用Top-{TOP_N_FEATURES_TO_SELECT}特征) ###\n")
      f.write("-"*60 + "\n")
      f.write(f"此阶段使用在阶段一中发现的最重要的 {TOP_N_FEATURES_TO_SELECT} 个特征来训练一个轻量、高效的最终模型。\n\n")
      f.write(f"最终选定的特征列表: {top_features_list}\n\n")
      f.write(f"最终模型测试集准确率 (Accuracy): {final_accuracy:.4f}\n\n")
      
      # --- 添加时间到报告 ---
      f.write(f"最终模型训练耗时 (Training Time): {train_time_phase2:.4f} 秒\n\n")
      
      f.write("最终模型详细分类报告:\n")
      f.write(final_report_str + "\n\n")
      f.write("最终模型特征重要性 (在所选特征内部的相对重要性):\n")
      f.write(top_n_importance.to_string() + "\n\n")

      f.write("### 模型部署 ###\n")
      f.write("-"*60 + "\n")
      f.write("训练好的最终精简模型已保存到以下文件，可用于生产环境：\n")
      f.write(f"- Joblib 格式: {model_filename}\n")
      f.write(f"- ONNX 格式: {onnx_model_filename}\n")
      f.write("可以使用 joblib.load() (Python) 或 ONNX Runtime (C++/Python/etc.) 加载模型。\n")

   print(f"报告已成功保存到 {report_filename}")
   print(f"\n=== 两阶段分析流程 '{model_name_prefix}' 全部完成 ===")
   
   

def main():
    """主执行函数 (已更新以匹配分散的 query_features.csv 文件)"""
    # --- 路径配置  ---
    results_false_base_dir = os.path.join(DATASET_RESULTS_DIR, "Results", "UNG-nTfalse")
    results_true_base_dir = os.path.join(DATASET_RESULTS_DIR, "Results", "UNG-nTtrue")
    
    model_output_dir = os.path.join(MODEL_OUTPUT_BASE_DIR)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"本次训练的模型及分析报告将保存到: {model_output_dir}\n")

    # --- 查找成对的实验目录  ---
    result_pairs = find_result_pairs(results_false_base_dir, results_true_base_dir)
    if not result_pairs: return
        
    # --- 全新的数据加载和合并流程 ---
    all_merged_data = []
    print("--- 开始逐一加载、合并每个实验对的数据 ---")
    
    for path_f, path_t in result_pairs:
        sub_dir_name = os.path.basename(path_f)
        print(f"\n-> G' G' {sub_dir_name}")

        # 1. 加载当前实验对的性能数据 (Y)
        perf_data = load_performance_data(path_f, path_t)
        if perf_data.empty:
            print(f"   - 警告: 未能加载性能数据，跳过此实验。")
            continue
        
        # 2. 定位并加载与当前实验对应的特征文件 (X)
        features_csv_path = os.path.join(path_f, "results", "query_features.csv")
        
        if not os.path.exists(features_csv_path):
            print(f"   - ❌ 错误: 在 {features_csv_path} 中找不到对应的特征文件，跳过此实验。")
            continue
            
        features_df = pd.read_csv(features_csv_path)
        
        # 3. 将性能数据和特征数据进行合并
        merged_df = pd.merge(perf_data, features_df, on='QueryID', how='inner')
        
        if merged_df.empty:
            print(f"   - 警告: 性能数据与特征数据合并后为空，请检查 QueryID 是否匹配。跳过此实验。")
            continue

        print(f"   - ✅ 成功合并 {len(merged_df)} 条记录。")
        all_merged_data.append(merged_df)
            
    # --- 循环结束，连接所有数据 ---
    if not all_merged_data:
        print("\n❌ 错误: 未能成功加载并合并任何数据，程序终止。")
        return
        
    final_dataframe = pd.concat(all_merged_data, ignore_index=True)
    print(f"\n✅ 所有数据加载和合并完成，共获得 {len(final_dataframe)} 条记录用于建模。")
    
    # --- 步骤 4: 执行分析流程 ---
    run_two_stage_analysis(final_dataframe, "idea_selector", model_output_dir)

if __name__ == "__main__":
    main()