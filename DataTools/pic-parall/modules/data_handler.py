import os
import pandas as pd
import numpy as np

def get_query_count(config, dataset_name, query_dir_name, cache):
   """
   Counts the number of queries in a given query file, using a cache to avoid re-reading.
   """
   # 1.  创建一个复合键，确保缓存不会在不同数据集间串用
   cache_key = (dataset_name, query_dir_name)

   # 2.  检查复合键
   if cache_key in cache:
      return cache[cache_key]
   
   try:
      base_data_dir = config['global_settings']['base_data_dir']
      label_file_path = os.path.join(base_data_dir, dataset_name, query_dir_name, f"{dataset_name}_query_labels.txt")
      
      # 改进了打印，能同时显示数据集和查询集
      # print(f"  -> 首次发现查询集 '{query_dir_name}' (数据集: {dataset_name}), 正在从文件计数: {label_file_path}") # <--- [日志] 已注释
      
      with open(label_file_path, 'r') as f:
         count = sum(1 for _ in f)
      if count == 0:
         print(f"     -> 警告: 查询文件为空或不存在，找到 0 个查询。")
      # else:
         # print(f"     -> 成功识别到 {count} 个查询。") # <--- [日志] 已注释
      
      # 3.  使用复合键存入缓存
      cache[cache_key] = count
      return count
   
   except FileNotFoundError:
      print(f"  -> ❌ 错误: 无法找到查询标签文件来确定查询数量: {label_file_path}")
      # 4.  使用复合键存入缓存
      cache[cache_key] = 0
      return 0
   except Exception as e:
      print(f"  -> ❌ 错误: 读取查询标签文件时发生异常: {e}")
      # 5.  使用复合键存入缓存
      cache[cache_key] = 0
      return 0

def load_and_prepare_data(file_path, alg_name, num_queries, dataset_name, config):
   """
   Loads C++ summary data, calculates QPS, and applies pruning logic conditionally.
   (包含了对非数字数据的容错处理)
   """
   # 将 FileNotFoundError 的处理移至此函数的调用方 (run_pipeline.py)
   try:
      df = pd.read_csv(file_path)
   except pd.errors.EmptyDataError:
      return pd.DataFrame()
      
   if df.empty:
      return pd.DataFrame()

   # --- 数据清洗：强制类型转换 ---
   # 1. 强制将 Average_Time_ms 转换为数字，无法转换的变成 NaN
   df['Average_Time_ms'] = pd.to_numeric(df['Average_Time_ms'], errors='coerce')
   
   # 2. 同样处理 Average_Recall (以防万一)
   if 'Average_Recall' in df.columns:
       df['Average_Recall'] = pd.to_numeric(df['Average_Recall'], errors='coerce')

   # 3. 删除 Average_Time_ms 为 NaN 的行 (即脏数据)
   df.dropna(subset=['Average_Time_ms'], inplace=True)
   
   if df.empty:
       return pd.DataFrame()
   # -------------------------------------

   df.reset_index(drop=True, inplace=True)
   
   # 现在可以安全计算了
   df['QPS'] = num_queries / (df['Average_Time_ms'] / 1000) if num_queries > 0 else 0
   
   # =================================================================
   # 获取当前数据集的剪枝配置 
   dataset_conf = config.get('dataset_configurations', {}).get(dataset_name, {})
   pruning_settings = dataset_conf.get('pruning_settings', {})
   should_use_pruning_1 = pruning_settings.get("use_pruning_1", True)
   should_use_pruning_2 = pruning_settings.get("use_pruning_2", True)
   # =================================================================


   # Pruning Logic 1
   if should_use_pruning_1 and alg_name in ["ACORN-1", "ACORN-γ", "UNG", "SmartRoute"]:
      if not df.empty and df['Average_Recall'].notna().any():
            max_recall = df['Average_Recall'].max()
            peak_recall_df = df[df['Average_Recall'] >= (max_recall - 0.002)]

            if not peak_recall_df.empty:
               lsearch_at_max_qps = peak_recall_df.loc[peak_recall_df['QPS'].idxmax()]['Lsearch']
               lsearch_at_first_peak = peak_recall_df.iloc[0]['Lsearch']
               optimal_lsearch = min(lsearch_at_max_qps, lsearch_at_first_peak)
               original_rows = len(df)
               df = df[df['Lsearch'] <= optimal_lsearch].copy()
               
               if (pruned_rows := original_rows - len(df)) > 0:
                  # print(f"  -> (Pruning 1) 为 '{alg_name}' 在Lsearch>{optimal_lsearch}处剪枝了 {pruned_rows} 个无效数据点") # <--- [日志] 已注释
                  pass

   # Pruning Logic 2
   if should_use_pruning_2 and len(df) >= 2:
      original_rows_before_gentle_prune = len(df)
      indices_to_drop = []
      
      # 重置索引以确保循环安全
      df = df.sort_values(by='Average_Recall').reset_index(drop=True)
      
      for i in range(len(df) - 1):
         p1 = df.iloc[i]
         p2 = df.iloc[i+1]
         
         if (p2['Average_Recall'] > p1['Average_Recall'] and p2['QPS'] >= p1['QPS']) or \
            (p2['Average_Recall'] >= p1['Average_Recall'] and p2['QPS'] > p1['QPS']):
            indices_to_drop.append(df.index[i])

      if indices_to_drop:
         df.drop(indices_to_drop, inplace=True)

      pruned_count = len(indices_to_drop)
      if pruned_count > 0:
         # print(f"  -> (Pruning 2) 为 '{alg_name}' 剪枝了 {pruned_count} 个被后续点支配的数据点") # <--- [日志] 已注释
         pass

   return df.sort_values(by='Average_Recall')

def get_op_point_at_target_recall(df_processed, target_recall=0.9):
    """
    (v4_Robust) 获取分子的操作点 (R, QPS)
    逻辑：
    1. 优先找 Recall >= target_recall 的所有点，取其中 QPS 最高的一个 (Best Case)。
    2. 如果找不到，则找 Recall 最高的点。如果最高 Recall 对应多个点，取其中 QPS 最高的一个。
    """
    if df_processed.empty or df_processed['Average_Recall'].isna().all():
        return np.nan, np.nan

    # 逻辑 1: 找到所有 Recall >= 0.9 的点
    df_above_target = df_processed[df_processed['Average_Recall'] >= target_recall]

    if not df_above_target.empty:
        # [修改点] 不再简单的取 iloc[0]，而是取符合条件的点中 QPS 最大的
        # 这可以应对波动，选取该 Recall 段性能最好的表现
        best_idx = df_above_target['QPS'].idxmax()
        op_point = df_above_target.loc[best_idx]
        return op_point['Average_Recall'], op_point['QPS']

    # 逻辑 2 (Fallback): 找不到 >= 0.9 的点
    else:
        # [修改点] 找到 Recall 的最大值
        max_recall_val = df_processed['Average_Recall'].max()
        # 筛选出所有达到这个最大 Recall 的点 (防止有多个点 Recall 相同)
        df_max_recall = df_processed[df_processed['Average_Recall'] == max_recall_val]
        
        # 在最高 Recall 的点中，取 QPS 最高的 (Best Case)
        best_idx = df_max_recall['QPS'].idxmax()
        op_point = df_max_recall.loc[best_idx]
        return op_point['Average_Recall'], op_point['QPS']

def get_op_point_at_or_above_recall(df_processed, recall_level):
    """
    获取分母的操作点。
    逻辑：
    1. 找到所有 Recall >= recall_level 的点。
    2. 找出这些点中 Recall 最小的那个值（即紧挨着 recall_level 的点）。
    3. 在所有等于该最小 Recall 的点中，取 QPS 最低的（最保守）。
    """
    if df_processed.empty or np.isnan(recall_level):
        return np.nan, np.nan

    # 1. 筛选所有满足条件的点
    df_above = df_processed[df_processed['Average_Recall'] >= recall_level]

    if not df_above.empty:
        # 2. 找到“紧挨着”的 Recall 值
        # 因为 df_above 包含 Recall=0.99 的点，不能直接取 min QPS，否则会取到 Recall=0.99 的点。
        # 必须先锁定“离目标最近的 Recall”。
        closest_valid_recall = df_above['Average_Recall'].min()
        
        # 3. 锁定这个 Recall 值对应的所有点 (使用小 epsilon 避免浮点误差)
        # 这确保只在 Recall=0.9001 (举例) 的这些点里找，而不去碰 Recall=0.95 的点
        df_closest_neighbors = df_above[
            df_above['Average_Recall'] <= (closest_valid_recall + 1e-6)
        ]
        
        # 4. 在这些紧邻点中，取 QPS 最低的 (最保守/波动下限)
        worst_idx = df_closest_neighbors['QPS'].idxmin()
        op_point = df_closest_neighbors.loc[worst_idx]
        
        return op_point['Average_Recall'], op_point['QPS']
    else:
        return np.nan, np.nan

def get_op_point_at_max_recall(df_processed):
    """
    获取分母的兜底操作点。
    逻辑：
    1. 找到全局最高的 Recall 值。
    2. 在所有达到该 Recall 的点中，取 QPS 最低的（最保守）。
    """
    if df_processed.empty or df_processed['Average_Recall'].isna().all():
        return np.nan, np.nan

    # 1. 找到最大 Recall 值
    max_recall_val = df_processed['Average_Recall'].max()
    
    # 2. 筛选出所有达到这个最大 Recall 的点 (防止有多个点 Recall 相同)
    df_max_recall = df_processed[
        (df_processed['Average_Recall'] >= max_recall_val - 1e-6)
    ]
    
    # 3. 取 QPS 最低的
    worst_idx = df_max_recall['QPS'].idxmin()
    op_point = df_max_recall.loc[worst_idx]
    
    return op_point['Average_Recall'], op_point['QPS']

def get_max_qps_at_min_recall(df_processed, min_recall=0.9):
   """
   获取 Recall >= min_recall 时的最大 QPS，以及对应的 Recall 和 Time。
   如果无法达到 min_recall，返回 (None, None, None)。
   """
   if df_processed.empty:
      return None, None, None

   # 筛选出 Recall 达标的点
   df_qualified = df_processed[df_processed['Average_Recall'] >= min_recall]
   
   if df_qualified.empty:
      # Baseline 没达到要求
      return None, None, None
   
   # 找到 QPS 最大的那个点的索引 (Best Case)
   best_idx = df_qualified['QPS'].idxmax()
   row = df_qualified.loc[best_idx]
   
   # 返回 QPS, Recall, Time
   return row['QPS'], row['Average_Recall'], row['Average_Time_ms']

def parse_acorn_meta(filepath):
   """Parses ACORN meta file for build time and index size."""
   try:
      with open(filepath, 'r') as f:
         lines = f.readlines()
      time_s = float(lines[0].split(':')[1])
      size_bytes = int(lines[1].split(':')[1])
      return time_s, size_bytes
   except (FileNotFoundError, IndexError, ValueError):
      # print(f"  -> 警告: 无法解析ACORN meta文件或文件不存在: {filepath}") # <--- [日志] 已注释
      return None, None

def parse_ung_meta(filepath):
   """Parses UNG meta file."""
   data = {}
   try:
      with open(filepath, 'r') as f:
         for line in f:
               if len(parts := line.strip().split('=', 1)) == 2:
                  try:
                     data[parts[0]] = float(parts[1])
                  except ValueError:
                     pass
      return data
   except FileNotFoundError:
      # print(f"  -> 警告: UNG meta文件不存在: {filepath}") # <--- [日志] 已注释
      return {}

def extract_build_info_for_all_datasets(config):
   """
   迭代数据集以提取构建性能。
   """
   all_datasets_info = {}
   base_results_dir = config['global_settings']['base_results_dir']
   
   for dataset_name, dataset_conf in config['dataset_configurations'].items():
      # print(f"\n🔍 正在为数据集 '{dataset_name}' 提取索引构建信息...")
      build_params = dataset_conf.get('build_params', {})
      ung_index_handle = dataset_conf['structure_templates']['ung_index_handle'].format(**build_params)

      current_ds_info = {} # 存储此数据集的所有可用信息
      has_any_data = False # 标记是否找到了任何数据

      # --- 1. 串行数据 (Serial Data) ---
      serial_base_path = os.path.join(base_results_dir, dataset_name, "Index", ung_index_handle)
      
      # ACORN-γ (来自 acorn.index.meta)
      acorn_g_time_s, acorn_g_size_b = parse_acorn_meta(os.path.join(serial_base_path, "acorn_output", "acorn.index.meta"))
      if acorn_g_time_s is not None:
         current_ds_info['serial_acorn_gamma_time_s'] = acorn_g_time_s
         current_ds_info['serial_acorn_gamma_size_mb'] = acorn_g_size_b / (1024**2)
         has_any_data = True

      # ACORN-1 (来自 acorn1.index.meta)
      acorn_1_time_s, acorn_1_size_b = parse_acorn_meta(os.path.join(serial_base_path, "acorn_output", "acorn1.index.meta"))
      if acorn_1_time_s is not None:
         current_ds_info['serial_acorn_1_time_s'] = acorn_1_time_s
         current_ds_info['serial_acorn_1_size_mb'] = acorn_1_size_b / (1024**2)
         has_any_data = True

      # UNG (Serial) - 使用 'index_time(ms)' 和 'index_size(MB)'
      ung_s_data = parse_ung_meta(os.path.join(serial_base_path, "index_files", "meta"))
      ung_s_time_ms, ung_s_size_mb = ung_s_data.get('index_time(ms)'), ung_s_data.get('index_size(MB)')
      if ung_s_time_ms is not None and ung_s_size_mb is not None:
         current_ds_info['serial_ung_time_s'] = ung_s_time_ms / 1000.0
         current_ds_info['serial_ung_size_mb'] = ung_s_size_mb
         has_any_data = True


      # --- 2. 并行数据 (Parallel Data - Optional) ---
      parallel_base_path = os.path.join(base_results_dir, dataset_name, "Index_parallel", ung_index_handle)
      
      # ACORN-γ (Parallel)
      acorn_p_g_time_s, acorn_p_g_size_b = parse_acorn_meta(os.path.join(parallel_base_path, "acorn_output", "acorn.index.meta"))
      if acorn_p_g_time_s is not None:
         current_ds_info['parallel_acorn_gamma_time_s'] = acorn_p_g_time_s
         current_ds_info['parallel_acorn_gamma_size_mb'] = acorn_p_g_size_b / (1024**2)
         has_any_data = True

      # (我们假设 ACORN-1 目前没有并行构建)

      # UNG (Parallel) - 使用 'index_time_add_rb(ms)' 和 '_index_size_add_rb(MB)'
      ung_p_data = parse_ung_meta(os.path.join(parallel_base_path, "index_files", "meta"))
      ung_p_time_ms, ung_p_size_mb = ung_p_data.get('index_time_add_rb(ms)'), ung_p_data.get('_index_size_add_rb(MB)')
      if ung_p_time_ms is not None and ung_p_size_mb is not None:
         current_ds_info['parallel_ung_time_s'] = ung_p_time_ms / 1000.0
         current_ds_info['parallel_ung_size_mb'] = ung_p_size_mb
         has_any_data = True
         
      # --- 3. 最终处理 ---
      if has_any_data:
         # 仅在至少有一个并行数据存在时才计算
         p_acorn_time = current_ds_info.get('parallel_acorn_gamma_time_s', 0)
         p_ung_time = current_ds_info.get('parallel_ung_time_s', 0)
         if p_acorn_time > 0 or p_ung_time > 0:
               current_ds_info['parallel_max_time_s'] = max(p_acorn_time, p_ung_time)
         
         # 仅在至少有一个并行数据存在时才计算
         p_acorn_size = current_ds_info.get('parallel_acorn_gamma_size_mb', 0)
         p_ung_size = current_ds_info.get('parallel_ung_size_mb', 0)
         if p_acorn_size > 0 or p_ung_size > 0:
               current_ds_info['parallel_sum_size_mb'] = p_acorn_size + p_ung_size
               
         all_datasets_info[dataset_name] = current_ds_info
         # print(f"  -> 成功提取 '{dataset_name}' 的数据。")
      else:
         # print(f"  -> ❌ 错误: '{dataset_name}' 未找到任何构建数据，已跳过。")
         pass
         
   return all_datasets_info