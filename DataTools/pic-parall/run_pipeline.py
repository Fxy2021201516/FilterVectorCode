import json
import os
from modules import data_handler, plotter, utils
import numpy as np
import pandas as pd 

def print_and_save_plot_data(combined_plot_data, title, dataset_name, task_name):
   """打印 QPS-Recall 坐标的前几行作为示例，并将完整数据保存到 CSV 文件。"""
   # 创建一个用于保存坐标数据的目录
   output_dir = "plot_coordinates"
   os.makedirs(output_dir, exist_ok=True)
   
   print(f"\n--- 坐标数据概览 (任务: {task_name}, 子图: {title}) ---")
   
   for alg_name, df in combined_plot_data.items():
      # 检查DataFrame是否为空
      if df.empty:
         print(f"算法: {alg_name} - 数据为空，已跳过。")
         continue
         
      print(f"\n算法: {alg_name}")
      # 打印 Lsearch、Average_Recall (X轴) 和 QPS (Y轴) 的前几行作为示例
      print(df[['Lsearch', 'Average_Recall', 'QPS']].head())
      
      # 构造安全的 CSV 文件名
      safe_alg_name = alg_name.replace('-', '_').replace('/', '_').replace('+', '_plus').replace('γ', 'gamma')
      safe_task_name = task_name.replace('-', '_').replace(' ', '_')
      safe_title = title.replace('\n', '_').replace(' ', '_').replace('[', '').replace(']', '').replace('=', '_').replace('$', '_math_')

      output_filename = f"{safe_task_name}_{dataset_name}_{safe_title}_{safe_alg_name}.csv"
      output_path = os.path.join(output_dir, output_filename)
      
      # 保存完整数据到 CSV，方便后续分析
      # 包含 Lsearch 有助于理解数据点对应的搜索参数
      df[['Lsearch', 'Average_Recall', 'QPS']].to_csv(output_path, index=False)
      print(f"完整数据已保存到: {os.path.abspath(output_path)}")
   print("-----------------------------------------------------------------")


def fetch_data_for_subplot(config, dataset_name, base_params, algorithms, cache, allow_acorn_ori=False):
   """
   Orchestrates the data fetching process for a single subplot.
   Added allow_acorn_ori flag: If True, tries to load search_time_summary_ori.csv for ACORN algos.
   """
   combined_plot_data = {}
   num_queries = data_handler.get_query_count(config, dataset_name, base_params['query_dir_name'], cache) #
   
   #  更新了算法名称映射...
   alg_name_map_inverted = {
      "ACORN-1": "ACORN-1",
      "ACORN-γ": "ACORN-gamma",
      "ACORN-γ-improved": "ACORN-gamma-improved",
      "UNG": "UNG-nTfalse",
      "IntelLANNS": "method2",
      "SmartRoute": "method3"
   }
   
   if num_queries == 0:
      return None
      
   # 遍历 'algorithms_to_compare' 列表
   for alg_config in algorithms:
   
      display_name = ""
      current_params = base_params.copy() # 默认使用 common_parameters

      if isinstance(alg_config, str):
         # --- 1. 旧格式：只是一个字符串 ---
         display_name = alg_config
         # current_params 已经是 base_params 的副本，无需操作
         
      elif isinstance(alg_config, dict):
         # --- 2. 新格式：是一个对象 ---
         display_name = alg_config.get("name")
         if not display_name:
            print(f"  -> 警告: 发现一个没有 'name' 键的算法对象，已跳过。")
            continue
            
         # 获取特定于算法的参数并覆盖
         specific_params = alg_config.get("params", {})
         current_params.update(specific_params) # [!! 关键：覆盖参数 !!]
         
      else:
         print(f"  -> 警告: 'algorithms_to_compare' 列表中的项目格式无法识别，已跳过。")
         continue


      # 使用映射来获取 C++ 实验中实际使用的文件夹名称
      internal_alg_name = alg_name_map_inverted.get(display_name, display_name) #
      
      current_params['algorithm_name'] = internal_alg_name
      
      try:
         # build_paths_for_exp 现在会使用被覆盖后的 'current_params'
         result_dir = utils.build_paths_for_exp(config, dataset_name, current_params) #
         
         # --- [修改开始] 文件名选择逻辑 ---
         target_filename = "search_time_summary.csv" # 默认文件名
         
         # 仅当：1.允许使用ori(即Ratio图) AND 2.算法名包含ACORN 时，才尝试查找 _ori
         if allow_acorn_ori and ("ACORN" in display_name):
             ori_path = os.path.join(result_dir, "results", "search_time_summary_ori.csv")
             if os.path.exists(ori_path):
                 target_filename = "search_time_summary_ori.csv"
                 # print(f"     -> [Info] {display_name} 正在使用 _ori 数据: {target_filename}") # 可选日志
         
         summary_file_path = os.path.join(result_dir, "results", target_filename) #
         # --- [修改结束] ---
         
         # 使用 data handler 加载和处理数据
         df_plot = data_handler.load_and_prepare_data(summary_file_path, display_name, num_queries, dataset_name, config) #
         
         if not df_plot.empty:
               # 存储时使用 "显示名称" (display_name) 作为 key
               combined_plot_data[display_name] = df_plot
               
      except ValueError as e:
         print(f"  -> 警告: 为算法 '{internal_alg_name}' (显示为 '{display_name}') 构建路径时出错: {e}。已跳过。")
         continue
      except FileNotFoundError:
         print(f"  -> [数据缺失] 未找到文件: {summary_file_path}。已跳过 '{display_name}'。") #
         continue
         
   return combined_plot_data


def handle_qps_recall_tasks(config, font_sizes):

   query_count_cache = {}
   tasks = config.get("global_comparison_tasks", {})
   
   # global_subplot_counter = 1

   # --- 特殊任务键列表 ---
   special_task_keys = ['k_comparison_plots', 'length_comparison_plots', 'selectivity_comparison_plots', 
                        'p_pass_comparison_plots', 'thread_comparison_plots',
                        'curated_k_plots', 'curated_p_pass_plots', 'curated_thread_plots', 
                        'speedup_ratio_tasks']
   
   tasks_by_dataset = {k: v for k, v in tasks.items() if k not in special_task_keys}

   # ==================================================================
   # --- [!! 编号任务开始 !!] ---
   # ==================================================================

   # --- 1. Length 对比图 ('length.png') ---
   print("#"*80 + "\n📈 1. 开始生成 Length 对比图 (编号 1...)...\n" + "#"*80)
   task_list_len = tasks.get("length_comparison_plots", [])
   for task in task_list_len:
      if not task.get("enabled", False): continue
      all_plot_items_for_grid = []
      for dataset_info in task.get('datasets_to_compare', []):
            dataset_name = dataset_info['dataset_name']
            common_params = dataset_info.get('common_parameters', {})
            algorithms = dataset_info.get('algorithms_to_compare', [])
            
            # (这是 'queries_to_compare' 的特定逻辑)
            param_key = "query_dir_name" 
            values_key = "queries_to_compare"
            
            if isinstance(dataset_info.get(values_key), list) and all(isinstance(i, dict) for i in dataset_info.get(values_key, [])):
               for item_config in dataset_info.get(values_key, []):
                  current_iter_params = common_params.copy()
                  specific_params = item_config.get('specific_params', {})
                  current_iter_params.update(specific_params)
                  current_iter_params[param_key] = item_config['query_dir_name']
                  combined_plot_data = fetch_data_for_subplot(config, dataset_name, current_iter_params, algorithms, query_count_cache)
                  if combined_plot_data:
                     all_plot_items_for_grid.append({
                           'data': combined_plot_data, 
                           'title': item_config['subplot_title'],
                           'xlabel': f"Recall@{current_iter_params.get('K', 'N/A')}"
                     })
            # (无 else, 因为 length_comparison_plots 总是使用对象列表)

      if all_plot_items_for_grid:
            # [!!! 已修改 (Mod 3) !!!]
            print(f"  -> 准备为 '{task.get('output_filename')}' 绘制 {len(all_plot_items_for_grid)} 个图表，起始编号: 1")
            plotter.generate_qps_recall_grid(
               all_plot_items_for_grid, 
               task.get('main_title'), 
               task.get('output_filename'), 
               font_sizes, 
               task.get('plot_settings', {}),
               numbering_offset_start=1 # <--- 传入 1
            )


   # --- 2. Curated Selectivity (p_pass) 图 ('p_pass.png') ---
   curated_sel_tasks = tasks.get('curated_p_pass_plots', [])
   if curated_sel_tasks:
      print("#"*80 + f"\n📈 2. 开始生成 [Curated] Selectivity对比图 (编号 1...)...\n" + "#"*80)
      for task in curated_sel_tasks:
         if not task.get("enabled", False): continue
         all_plot_items_for_grid = []
         plot_settings = task.get('plot_settings', {})
         
         queries_to_compare = list(reversed(task.get('queries_to_compare', [])))
         
         datasets_to_compare = task.get('datasets_to_compare', [])

         for query_config in queries_to_compare:
               query_suffix_name = query_config.get('suffix_name')
               subplot_title_suffix = query_config['subplot_title_suffix']
               if not query_suffix_name: continue
               for dataset_info in datasets_to_compare:
                  dataset_name = dataset_info['dataset_name']
                  common_params = dataset_info.get('common_parameters', {}).copy()
                  algorithms = dataset_info.get('algorithms_to_compare', [])
                  query_base_dir = dataset_info.get('query_dir_base')
                  if not query_base_dir:
                     all_plot_items_for_grid.append({'data': {}, 'title': f"{dataset_name}\n(Config Error)", 'xlabel': ""})
                     continue
                  base_name_part = os.path.basename(query_base_dir).replace("query_select_imp_", "")
                  final_query_dir = f"{query_base_dir}/{base_name_part}_{query_suffix_name}"
                  common_params['query_dir_name'] = final_query_dir
                  combined_plot_data = fetch_data_for_subplot(config, dataset_name, common_params, algorithms, query_count_cache)
                  if combined_plot_data:
                     all_plot_items_for_grid.append({'data': combined_plot_data, 'title': f"{dataset_name}\n{subplot_title_suffix}", 'xlabel': f"Recall@{common_params.get('K', 'N/A')}"})
                  else:
                     all_plot_items_for_grid.append({'data': {}, 'title': f"{dataset_name}\n{subplot_title_suffix}\n(No Data)", 'xlabel': f"Recall@{common_params.get('K', 'N/A')}"})

         if all_plot_items_for_grid:
               print(f"  -> 准备为 '{task.get('output_filename')}' 绘制 {len(all_plot_items_for_grid)} 个图表，起始编号: 1")
               plotter.generate_qps_recall_grid(
                  all_plot_items_for_grid, 
                  task.get('main_title'), 
                  task.get('output_filename'), 
                  font_sizes, 
                  plot_settings,
                  numbering_offset_start=1 
               )

   # --- 3. Curated K-Plots ('K.png') ---
   curated_k_tasks = tasks.get('curated_k_plots', [])
   if curated_k_tasks:
      print("#"*80 + f"\n📈 3. 开始生成 [Curated] K值对比图 (编号 1...)...\n" + "#"*80)
      for task in curated_k_tasks:
         if not task.get("enabled", False): continue
         all_plot_items_for_grid = []
         plot_settings = task.get('plot_settings', {})
         for dataset_info in task.get('datasets_to_compare', []):
               dataset_name = dataset_info['dataset_name']
               common_params = dataset_info.get('common_parameters', {})
               algorithms = dataset_info.get('algorithms_to_compare', [])
               if 'K' not in common_params: continue
               combined_plot_data = fetch_data_for_subplot(config, dataset_name, common_params, algorithms, query_count_cache)
               if combined_plot_data:
                  all_plot_items_for_grid.append({'data': combined_plot_data, 'title': f"{dataset_name}\n$\\mathrm{{K}}={common_params['K']}$", 'xlabel': f"Recall@{common_params['K']}"})
               else:
                  all_plot_items_for_grid.append({'data': {}, 'title': f"{dataset_name}\n(No Data)", 'xlabel': f"Recall@{common_params['K']}"})

         if all_plot_items_for_grid:
               print(f"  -> 准备为 '{task.get('output_filename')}' 绘制 {len(all_plot_items_for_grid)} 个图表，起始编号: 1")
               plotter.generate_qps_recall_grid(
                  all_plot_items_for_grid, 
                  task.get('main_title'), 
                  task.get('output_filename'), 
                  font_sizes, 
                  plot_settings,
                  numbering_offset_start=1
               )

   # --- 4. Curated Thread-Plots ('th.png') ---
   curated_thread_tasks = tasks.get('curated_thread_plots', [])
   if curated_thread_tasks:
      print("#"*80 + f"\n📈 4. 开始生成 [Curated] Thread对比图 (编号 1...)...\n" + "#"*80)
      for task in curated_thread_tasks:
         if not task.get("enabled", False): continue
         all_plot_items_for_grid = []
         plot_settings = task.get('plot_settings', {})
         for dataset_info in task.get('datasets_to_compare', []):
               dataset_name = dataset_info['dataset_name']
               common_params = dataset_info.get('common_parameters', {})
               algorithms = dataset_info.get('algorithms_to_compare', [])
               if 'threads' not in common_params: continue
               thread_val = common_params['threads']
               combined_plot_data = fetch_data_for_subplot(config, dataset_name, common_params, algorithms, query_count_cache)
               if combined_plot_data:
                  all_plot_items_for_grid.append({'data': combined_plot_data, 'title': f"{dataset_name}\n$\\mathrm{{th}}={thread_val}$", 'xlabel': f"Recall@{common_params.get('K', 'N/A')}"})
               else:
                  all_plot_items_for_grid.append({'data': {}, 'title': f"{dataset_name}\n(No Data)", 'xlabel': f"Recall@{common_params.get('K', 'N/A')}"})

         if all_plot_items_for_grid:
               print(f"  -> 准备为 '{task.get('output_filename')}' 绘制 {len(all_plot_items_for_grid)} 个图表，起始编号: 1")
               plotter.generate_qps_recall_grid(
                  all_plot_items_for_grid, 
                  task.get('main_title'), 
                  task.get('output_filename'), 
                  font_sizes, 
                  plot_settings,
                  numbering_offset_start=1
               )

   # ==================================================================
   # --- [!! 无编号任务开始 !!] ---
   # ==================================================================
   print("#"*80 + "\n📈 5. 开始生成 [无编号] 其他对比图...\n" + "#"*80)

   # --- 5. 其他对比图 (K, p_pass, thread 的非 Curated 版本) ---
   other_comparison_task_map = {
      "多K值": ("k_comparison_plots", "K", "k_values", "subplot_titles"),
      "多选择性(p_pass)": ("p_pass_comparison_plots", "query_dir_name", "queries_to_compare", None),
      "多线程": ("thread_comparison_plots", "threads", "thread_values", "subplot_titles")
   }

   for idx, (task_type_name, (config_key, param_key, values_key, titles_key)) in enumerate(other_comparison_task_map.items()):
      task_list = tasks.get(config_key, [])
      if not task_list: continue
      
      for task in task_list:
         if not task.get("enabled", False): continue
         all_plot_items_for_grid = []
         print(f"\n[无编号] 处理任务: {task.get('task_name')}")
         for dataset_info in task.get('datasets_to_compare', []):
               dataset_name = dataset_info['dataset_name']
               common_params = dataset_info.get('common_parameters', {})
               algorithms = dataset_info.get('algorithms_to_compare', [])

               if isinstance(dataset_info.get(values_key), list) and all(isinstance(i, dict) for i in dataset_info.get(values_key, [])):
                  for item_config in dataset_info.get(values_key, []):
                     current_iter_params = common_params.copy()
                     specific_params = item_config.get('specific_params', {})
                     current_iter_params.update(specific_params)
                     current_iter_params[param_key] = item_config['query_dir_name']
                     combined_plot_data = fetch_data_for_subplot(config, dataset_name, current_iter_params, algorithms, query_count_cache)
                     if combined_plot_data:
                        all_plot_items_for_grid.append({'data': combined_plot_data, 'title': item_config['subplot_title'], 'xlabel': f"Recall@{current_iter_params.get('K', 'N/A')}"})
               else: 
                  values_to_iterate = dataset_info.get(values_key, [])
                  default_titles_provided = dataset_info.get(titles_key)
                  if default_titles_provided: subplot_titles = default_titles_provided
                  else:
                     if config_key == "thread_comparison_plots": subplot_titles = [f"{dataset_name}\n$\\mathrm{{th}}={v}$" for v in values_to_iterate]
                     elif config_key == "k_comparison_plots": subplot_titles = [f"{dataset_name}\n$\\mathrm{{{param_key.upper()}}}={v}$" for v in values_to_iterate]
                     else: subplot_titles = [f"{dataset_name}\n{param_key}={v}" for v in values_to_iterate]
                  for value, subplot_title in zip(values_to_iterate, subplot_titles):
                     current_iter_params = common_params.copy()
                     current_iter_params[param_key] = value
                     combined_plot_data = fetch_data_for_subplot(config, dataset_name, current_iter_params, algorithms, query_count_cache)
                     if combined_plot_data:
                        all_plot_items_for_grid.append({'data': combined_plot_data, 'title': subplot_title, 'xlabel': f"Recall@{current_iter_params.get('K', common_params.get('K', 'N/A'))}"})

         if all_plot_items_for_grid:
               #  这里不传递 numbering_offset_start
               plotter.generate_qps_recall_grid(
                   all_plot_items_for_grid, 
                   task.get('main_title'), 
                   task.get('output_filename'), 
                   font_sizes, 
                   task.get('plot_settings', {})
               )

   # --- 6. Normal QPS-Recall Plots ---
   if tasks_by_dataset:
      print("#"*80 + "\n📈 6. 开始生成 [无编号] 普通 QPS-Recall 网格对比图...\n" + "#"*80)
      all_plot_items_normal = []
      first_task_settings = {}
      if tasks_by_dataset:
          first_task_settings = next(iter(tasks_by_dataset.values()))[0].get('plot_settings', {})
      for dataset_name, tasks_list in tasks_by_dataset.items():
         for task in tasks_list:
               if not task.get("enabled", False): continue
               common_params = task.get('common_parameters', {})
               if 'query_dir_name' not in common_params: continue
               algorithms = task.get('algorithms_to_compare', [])
               combined_plot_data = fetch_data_for_subplot(config, dataset_name, common_params, algorithms, query_count_cache)
               if combined_plot_data:
                  all_plot_items_normal.append({'data': combined_plot_data, 'title': task.get('title', dataset_name), 'xlabel': f"Recall@{common_params.get('K', 'N/A')}"})
      
      if all_plot_items_normal:
         plotter.generate_qps_recall_grid(all_plot_items_normal, "Overall Performance Comparison", task.get('output_filename'),font_sizes, first_task_settings)


def handle_build_plots(config, font_sizes):
   print("#"*80 + "\n📊 5. 开始生成 索引构建性能对比图...\n" + "#"*80)
   
   # 1. 提取数据
   build_info = data_handler.extract_build_info_for_all_datasets(config)
   
   # 2. 将提取到的数据保存到 CSV 文件
   output_dir = "debug_csvs"
   os.makedirs(output_dir, exist_ok=True)
   csv_path = os.path.join(output_dir, "build_performance_summary.csv")
   
   if build_info:
      try:
         # 将 { 'dataset1': { 'metric1': v1, 'metric2': v2 }, ... } 转换为 DataFrame
         df = pd.DataFrame.from_dict(build_info, orient='index')
         df.index.name = 'dataset'
         
         # 定义一个理想的列顺序
         cols_order = [
               'serial_acorn_1_time_s', 'serial_acorn_1_size_mb',
               'serial_acorn_gamma_time_s', 'serial_acorn_gamma_size_mb',
               'serial_ung_time_s', 'serial_ung_size_mb',
               'parallel_acorn_gamma_time_s', 'parallel_acorn_gamma_size_mb',
               'parallel_ung_time_s', 'parallel_ung_size_mb',
               'parallel_max_time_s', 'parallel_sum_size_mb'
         ]
         
         # 过滤，只保留 DataFrame 中实际存在的列
         existing_cols = [col for col in cols_order if col in df.columns]
         df_ordered = df[existing_cols]
         
         df_ordered.to_csv(csv_path, float_format='%.3f')
         print(f"✅ [DEBUG] 详细构建性能数据已保存到: {os.path.abspath(csv_path)}")
         
      except Exception as e:
         print(f"  -> ❌ 错误: 保存构建性能 CSV 时出错: {e}")
   else:
      # 如果 build_info 为空，也创建一个空文件或带标题的文件，以明确告知用户
      pd.DataFrame(columns=['dataset']).to_csv(csv_path, index=False)
      print(f"  -> [DEBUG] 未提取到数据，已创建空的构建性能文件: {os.path.abspath(csv_path)}")


   # 3. 绘图
   if build_info:
      build_font_sizes = font_sizes
      plotter.generate_build_time_plot(build_info, "build_time_comparison.png", build_font_sizes)
      plotter.generate_index_size_plot(build_info, "index_size_comparison.png", build_font_sizes)
   else:
      print(" -> 警告: 未能提取到任何完整的构建信息，无法生成构建性能对比图。") # 这个警告现在是预期的


def handle_speedup_ratio_tasks(config, font_sizes):
    """
    处理加速比绘图任务
    逻辑更新 (v3 - Numerator Driven):
    1. 优先检查分子 (Intel/Num) 是否达到 target_recall (0.9)。
       - 如果分子达到 0.9:
         - 检查分母是否达到 0.9。
           - 是 -> 计算 Ratio。
           - 否 -> Ratio = Inf (因为分子做到了分母做不到的高精度)。
    2. 如果分子未达到 0.9:
       - 降级检查分子是否达到 0.8。
       - 如果分子达到 0.8:
         - 检查分母是否达到 0.8。
           - 是 -> 计算 Ratio。
           - 否 -> Ratio = Inf。
       - 如果分子连 0.8 都不到 -> Num Failed。
    """
    print("#"*80 + "\n🚀 9. 开始生成 加速比 (Speedup Ratio) 对比图...\n" + "#"*80)
    
    tasks = config.get("speedup_ratio_tasks", [])
    if not tasks:
        print(" -> 未找到 'speedup_ratio_tasks' 任务，已跳过。")
        return
        
    query_count_cache = {}

    for task in tasks:
        if not task.get("enabled", False): continue
        
        all_debug_data = []
        
        # 首选目标 (通常是 0.9)
        primary_target = task.get('target_recall', 0.9)
        secondary_target = 0.8
        
        datasets_to_process = task.get('datasets_to_process', [])
        alg_list_global = task.get('algorithms_to_compare', []) 
        
        ratio_pairs = task.get('ratio_pairs', [])
        ratio_labels = task.get('ratio_labels', [])
        categories = task.get('categories', [])

        # --- 步骤 1: 数据收集与计算 ---
        for dataset_name in datasets_to_process:
            print(f"\n  -> 正在处理数据集: '{dataset_name}'")
            for category in categories:
                category_name = category['name']
                params = category['parameters_to_use'].get(dataset_name)
                
                if not params:
                    continue
                
                # 准备算法列表
                alg_list_to_use = alg_list_global 
                params_to_use = params.copy() 
                if "algorithms_to_compare" in params_to_use:
                    alg_list_to_use = params_to_use["algorithms_to_compare"]
                    print(f"     -> [Info] 为 '{category_name}' 类别使用特定算法列表。")
                    del params_to_use["algorithms_to_compare"]
                    
                try:
                    # 获取数据
                    alg_data = fetch_data_for_subplot(
                        config, dataset_name, params_to_use, alg_list_to_use, 
                        query_count_cache, allow_acorn_ori=False
                    )
                    if not alg_data: continue
                except Exception as e:
                    print(f"     -> ❌ 错误: fetch_data_for_subplot 出错: {e}")
                    continue
                
                # 计算每一对的比率
                for i, (num_alg, den_alg) in enumerate(ratio_pairs):
                    ratio_name = ratio_labels[i] 
                    df_num = alg_data.get(num_alg)
                    df_den = alg_data.get(den_alg)
                    
                    # 初始化结果变量
                    final_ratio = np.nan
                    status = "Init"
                    qps_num = recall_num = time_num = None
                    qps_den = recall_den = time_den = None
                    
                    # 检查数据缺失
                    if df_num is None or df_den is None:
                        status = "Missing Data"
                    else:
                        # =================================================
                        # [核心逻辑修改 v3] 以分子 (Numerator) 的能力为主导
                        # =================================================
                        
                        # 1. 优先尝试 Primary Target (0.9)
                        qps_num_09, rec_num_09, time_num_09 = data_handler.get_max_qps_at_min_recall(df_num, primary_target)
                        
                        if qps_num_09 is not None:
                            # --- 分子达到了 0.9 ---
                            # 无论分母如何，我们都锁定了 0.9 这个标准
                            qps_num, recall_num, time_num = qps_num_09, rec_num_09, time_num_09
                            
                            # 检查分母是否也能达到 0.9
                            qps_den_09, rec_den_09, time_den_09 = data_handler.get_max_qps_at_min_recall(df_den, primary_target)
                            
                            if qps_den_09 is not None:
                                # 分母也能达到 -> 正常计算
                                qps_den, recall_den, time_den = qps_den_09, rec_den_09, time_den_09
                                if qps_den > 0:
                                    final_ratio = qps_num / qps_den
                                    status = f"OK (Recall>={primary_target})"
                                else:
                                    final_ratio = np.inf
                                    status = f"Infinity (Den=0)"
                            else:
                                # 分子到了，分母没到 -> 无穷大
                                final_ratio = np.inf
                                status = f"Infinity (Num>={primary_target}, Den Failed)"
                                # 即使分母Failed，变量保持None即可
                        
                        else:
                            # --- 分子未达到 0.9 -> 降级尝试 Secondary Target (0.8) ---
                            qps_num_08, rec_num_08, time_num_08 = data_handler.get_max_qps_at_min_recall(df_num, secondary_target)
                            
                            if qps_num_08 is not None:
                                # 分子达到了 0.8
                                qps_num, recall_num, time_num = qps_num_08, rec_num_08, time_num_08
                                
                                # 检查分母是否也能达到 0.8
                                qps_den_08, rec_den_08, time_den_08 = data_handler.get_max_qps_at_min_recall(df_den, secondary_target)
                                
                                if qps_den_08 is not None:
                                    qps_den, recall_den, time_den = qps_den_08, rec_den_08, time_den_08
                                    if qps_den > 0:
                                        final_ratio = qps_num / qps_den
                                        status = f"OK (Recall>={secondary_target})"
                                    else:
                                        final_ratio = np.inf
                                        status = f"Infinity (Den=0)"
                                else:
                                    # 分子到了0.8，分母连0.8都没到 -> 无穷大
                                    final_ratio = np.inf
                                    status = f"Infinity (Num>={secondary_target}, Den Failed)"
                            else:
                                # 分子连 0.8 都没到
                                status = f"Num Failed (<{secondary_target})"

                    # 保存结果
                    all_debug_data.append({
                        'dataset': dataset_name, 
                        'category': category_name, 
                        'ratio_name': ratio_name,
                        'ratio_value': final_ratio, 
                        'status': status,
                        'qps_num': qps_num,
                        'recall_num': recall_num,
                        'time_num_ms': time_num,
                        'qps_den': qps_den,
                        'recall_den': recall_den,
                        'time_den_ms': time_den
                    })
        
        # --- 步骤 2: 保存与绘图 ---
        if not all_debug_data:
            print(" -> 警告: 无数据，无法绘图。")
            continue
            
        debug_df = pd.DataFrame(all_debug_data)

        # 保存 CSV
        output_dir = "debug_csvs"
        os.makedirs(output_dir, exist_ok=True)
        csv_filename_base = os.path.splitext(task.get('output_filename', 'debug.png'))[0]
        csv_path = os.path.join(output_dir, f"{csv_filename_base}_calculation_details_priority_num.csv")
        
        cols_to_save = [
            'dataset', 'category', 'ratio_name', 'ratio_value', 'status', 
            'qps_num', 'recall_num', 'time_num_ms', 
            'qps_den', 'recall_den', 'time_den_ms'
        ]
        
        debug_df[[c for c in cols_to_save if c in debug_df.columns]].to_csv(csv_path, index=False, float_format='%.4f')
        print(f"\n✅ [DEBUG] 详细计算过程 (分子优先策略) 已保存到: {os.path.abspath(csv_path)}")

        # 绘图
        all_ratios_list_final = debug_df.to_dict('records')
        if all_ratios_list_final:
            plotter.generate_speedup_ratio_plot(
                all_ratios_list_final, task, font_sizes, task.get('output_filename', 'speedup_ratio.png')
            )


def main():
   """主函数，负责执行整个流程。"""
   config_file = "/home/fengxiaoyao/1023FilterVector/FilterVectorCode/DataTools/pic-parall/config_overall_qps.json"
   try:
      with open(config_file, 'r', encoding='utf-8') as f:
         config = json.load(f)
   except (FileNotFoundError, json.JSONDecodeError) as e:
      print(f"❌ 错误: 无法加载或解析配置文件 {config_file}。\n{e}")
      return

   #  更新字体加载逻辑
   global_cfg = config.get('global_settings', {})
   
   # 1. 加载 QPS 网格图的字体
   font_sizes_grid = global_cfg.get('font_sizes_qps_grid', {})
   if not font_sizes_grid:
      print(" -> 警告: 未找到 'font_sizes_qps_grid' 配置，将使用默认值。")
      font_sizes_grid = {'main_title': 28, 'legend': 16, 'subplot_title': 18, 'axis_label': 14, 'tick_label': 12}
      
   # 2. 加载 Build 图的字体 (如果未定义，则回退到使用 grid 的配置)
   font_sizes_build = global_cfg.get('font_sizes_build_plots', font_sizes_grid)
   
   # 3. 加载 Speedup 图的字体 (如果未定义，则回退到使用 grid 的配置)
   font_sizes_speedup = global_cfg.get('font_sizes_speedup_plot', font_sizes_grid)

   #  Execute tasks, 传入各自的字体配置
   handle_qps_recall_tasks(config, font_sizes_grid)
   handle_build_plots(config, font_sizes_build)
   handle_speedup_ratio_tasks(config, font_sizes_speedup)

   print("\n✅ 所有任务处理完毕。")

if __name__ == "__main__":
   main()