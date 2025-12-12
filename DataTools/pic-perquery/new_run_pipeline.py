import json
import os
import pandas as pd
from functools import reduce
from modules import process, plot
from modules import new_select as select

# ==============================================================================
# 1. 辅助函数
# ==============================================================================   

def find_experiment_config(config, dataset_name, experiment_name_to_find):
    try:
        dataset_conf = config['dataset_configurations'][dataset_name]
        for exp_conf in dataset_conf['experiments']:
            if exp_conf['name'] == experiment_name_to_find:
                return exp_conf
        return None
    except KeyError:
        return None

def build_paths_for_exp(config, dataset_name, exp_config):
    """
    (已更新) 为特定的数据集和实验构建所有必需的文件路径。
    (已更新) 现在也会从 config 中提取 N 并作为 'total_vectors' 返回。
    """
    global_settings = config['global_settings']
    dataset_conf = config['dataset_configurations'][dataset_name]
    base_dir = global_settings['base_results_dir']
    templates = dataset_conf['structure_templates']
    
    # 结合 build_params (含 N) 和 experiment params
    build_params = dataset_conf.get('build_params', {})
    params = exp_config['parameters']
    format_params = {**build_params, **params, 'dataset': dataset_name, **global_settings}
    
    exp_name = exp_config['name']
    paths = {}
    merge_dir = os.path.join(base_dir, dataset_name, "Results", "per-query_results")
    os.makedirs(merge_dir, exist_ok=True)
    paths['output_dir'] = merge_dir
    
    ung_index_handle = templates['ung_index_handle'].format(**format_params)
    ung_gt_handle = templates['ung_gt_handle'].format(**format_params)
    search_params_handle = templates['search_params_handle'].format(**format_params)
    
    format_params.update({
        'ung_index_handle': ung_index_handle,
        'ung_gt_handle': ung_gt_handle,
        'search_params_handle': search_params_handle
    })
    
    result_base_dir = templates['result_dir_template'].format(**format_params)
    
    cpp_details_file = os.path.join(result_base_dir,"results", "query_details_repeat1.csv")
    paths['cpp_details_file'] = cpp_details_file
    
    paths['summary_output_path'] = os.path.join(merge_dir, f"{exp_name}_summary.csv")
    
    # --- 关键: 提取 N (总向量数) ---
    total_vectors = 0
    if 'N' in format_params:
        try:
            total_vectors = int(format_params['N'])
        except ValueError:
            total_vectors = 0
            
    paths['total_vectors'] = total_vectors #
    # ---
    
    paths['attribute_coverage_file'] = templates['attribute_coverage_file'].format(**format_params)
    
    return paths

# ==============================================================================
# 2. 任务处理器
# ==============================================================================

def handle_processing_and_merge_tasks(config):
    # (此函数未修改，保持原样)
    tasks = config.get("global_comparison_tasks", {}).get("qps_recall_plots", [])
    if not tasks: return

    print("\n" + "#"*80)
    print("⚙️ 开始第一步: 生成并合并摘要文件 (Process & Merge)")
    print("#"*80)

    for task in tasks:
        if not task.get("enabled", False):
            continue

        print(f"\n🚀 处理任务: {task['task_name']}")

        if not task["items_to_compare"]:
            print(f"  -> 警告: 任务 '{task['task_name']}' 中没有要比较的项目，跳过。")
            continue
        
        dataset_name = task["items_to_compare"][0]['dataset']
        dataframes_to_merge = []
        
        unified_analysis_params = {}
        first_item = task["items_to_compare"][0]
        first_exp_config = find_experiment_config(config, first_item['dataset'], first_item['experiment_name'])
            
        if first_exp_config and "analysis_params" in first_exp_config:
            unified_analysis_params = first_exp_config["analysis_params"]
            print(f"  -> ℹ️ 找到了统一的 analysis_params (来自 {first_item['experiment_name']})。")
        else:
            print(f"  -> 警告: 未能在第一个实验 '{first_item['experiment_name']}' 中找到 'analysis_params'。")
        
        for item in task["items_to_compare"]:
            exp_config = find_experiment_config(config, item['dataset'], item['experiment_name'])
            if not exp_config:
                print(f"  -> 警告: 找不到实验 '{item['experiment_name']}' 的配置，跳过。")
                continue
            
            paths = build_paths_for_exp(config, item['dataset'], exp_config)
            algorithm_name = exp_config['parameters']['algorithm_name']
            
            summary_df = process.run_processing(paths, algorithm_name, unified_analysis_params)
            if summary_df is not None and not summary_df.empty:
                dataframes_to_merge.append(summary_df)

        if len(dataframes_to_merge) < 1:
            print("  -> 没有生成任何有效的摘要数据，无法合并。")
            continue

        print("\n  -> 开始在内存中合并所有算法的摘要...")
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=['QueryID', 'QuerySize'], how='outer'), dataframes_to_merge)
        
        print("  -> 正在缩短列名以便阅读 (例如: SearchTime_ms_method3 -> ST_M3)...")
        metric_map = {'SearchTime_ms': 'ST', 'Time_ms': 'T', 'Recall': 'R', 'Optimal_Lsearch': 'L'}
        algo_map = {
            'method3': 'M3', 'method2': 'M2', 'method1': 'M1',        
            'ACORN-gamma': 'AG', 'ACORN-1': 'A1',    
            'ACORN-gamma-improved': 'AGI',   
            'UNG-nTfalse': 'UNG', 'UNG-nTtrue': 'UNGT'
        }
        new_columns = []
        for col in df_merged.columns:
            if col in ['QueryID', 'QuerySize']: new_columns.append(col); continue
            parts = col.split('_');
            if len(parts) < 2: new_columns.append(col); continue
            algo_name = parts[-1]; metric_name = '_'.join(parts[:-1])
            metric_short = metric_map.get(metric_name, metric_name); algo_short = algo_map.get(algo_name, algo_name)
            new_columns.append(f"{metric_short}_{algo_short}")
        df_merged.columns = new_columns
        
        output_dir = os.path.join(config['global_settings']['base_results_dir'], dataset_name, "Results", "per-query_results")
        output_path = os.path.join(output_dir, f"{task['task_name']}_merged_summary.csv")
        df_merged.to_csv(output_path, index=False, float_format='%.4f')
        print(f"  -> ✅ 总摘要文件已成功保存到: {os.path.abspath(output_path)}")
        
        
        print("\n" + "#"*80)
        print("🔍 开始第二步: 挑选高价值查询 (Select)")
        print("#"*80)
        
        analysis_params = first_exp_config.get("analysis_params", {}) if first_exp_config else {}
        analysis_params['current_dataset_name'] = dataset_name

        first_item_paths = build_paths_for_exp(config, first_item['dataset'], first_exp_config)
        attribute_coverage_path = first_item_paths.get('attribute_coverage_file')
        
        if not attribute_coverage_path or not os.path.exists(attribute_coverage_path):
            print(f"  -> 警告: 无法找到 'attribute_coverage_file' at {attribute_coverage_path}。")
            attribute_coverage_path = None
        else:
            print(f"  -> ℹ️ 找到了 'attribute_coverage_file'，将传递给筛选脚本: {os.path.basename(attribute_coverage_path)}")

        selection_output_path = os.path.join(output_dir, f"{task['task_name']}_selected_queries.csv")

        select.run_selection(
            merged_summary_path=output_path,
            output_path=selection_output_path,
            params=analysis_params,
            attribute_coverage_path=attribute_coverage_path
        )


def handle_qps_recall_tasks(config):
    tasks = config.get("global_comparison_tasks", {}).get("qps_recall_plots", [])
    if not tasks: return

    print("\n" + "#"*80)
    print("📈 开始第三步: 生成QPS-Recall对比图 (Plot)")
    print("#"*80)

    for task in tasks:
        if not task.get("enabled", False):
            continue

        print(f"\n🚀 执行绘图任务: {task['title']}")

        if not task["items_to_compare"]:
            print(f"  -> 警告: 任务 '{task['task_name']}' 中没有要比较的项目，跳过。")
            continue

        dataset_name = task["items_to_compare"][0]['dataset']
        output_dir_base = os.path.join(config['global_settings']['base_results_dir'], dataset_name, "Results", "per-query_results")
        selection_file_path = os.path.join(output_dir_base, f"{task['task_name']}_selected_queries.csv")
        
        selected_df = None
        try:
            print(f"  -> 尝试加载筛选后的查询文件: {os.path.basename(selection_file_path)}")
            selected_df = pd.read_csv(selection_file_path)
            print(f"      成功加载 {len(selected_df)} 个带权重的查询 (来自新策略)。")
            if 'QuerySize' not in selected_df.columns or 'p_pass' not in selected_df.columns:
                 print(f"      -> 警告: 筛选出的文件缺少 'QuerySize' 或 'p_pass' 列。无法按新策略拆分绘图。")
                 continue
        except FileNotFoundError:
            print(f"      错误: 未找到查询筛选文件 '{selection_file_path}'。这是后续步骤必需的。")
            continue

        # 1. 加载所有算法的详细数据到内存
        all_detailed_dfs = {}
        main_exp_config = None
        total_vectors = 0 # N

        for item in task["items_to_compare"]:
            exp_config = find_experiment_config(config, item['dataset'], item['experiment_name'])
            if not exp_config: continue
            
            if not main_exp_config: main_exp_config = exp_config 

            paths = build_paths_for_exp(config, item['dataset'], exp_config)
            
            if total_vectors == 0:
                total_vectors = paths.get('total_vectors', 0)
            
            algorithm_name = exp_config['parameters']['algorithm_name']
            df_detailed = plot._load_and_merge_data(paths['cpp_details_file'])
            if df_detailed.empty: continue
            
            all_detailed_dfs[algorithm_name] = df_detailed
            print(f"      已加载 '{algorithm_name}' 的 {len(df_detailed)} 行原始数据。")

        if total_vectors == 0:
            print(f"      -> 警告: 未能从 config.json 的 'build_params' 中找到 'N' (total_vectors)。")
            print(f"      -> 'p_pass' 绘图标题将显示原始计数值 (coverage_count)，而不是比例。")
        else:
            print(f"      -> ℹ️ 成功获取 N = {total_vectors}。 'p_pass' 绘图标题将显示为比例。")
        
        # 2. 定义5个绘图子集
        print(f"  -> 正在根据 {len(selected_df)} 个选定查询创建 5 个绘图组...")
        
        N_SPLIT = 500
        
        df_all = selected_df
        
        df_sorted_len = selected_df.sort_values('QuerySize').reset_index(drop=True)
        df_len_small = df_sorted_len.head(N_SPLIT)
        df_len_large = df_sorted_len.tail(N_SPLIT)
        len_small_range = f"[{df_len_small['QuerySize'].min()}-{df_len_small['QuerySize'].max()}]"
        len_large_range = f"[{df_len_large['QuerySize'].min()}-{df_len_large['QuerySize'].max()}]"
        print(f"     -> Length Split: Small (N={len(df_len_small)}, L={len_small_range}), Large (N={len(df_len_large)}, L={len_large_range})")

        df_sorted_ppass = selected_df.sort_values('p_pass').reset_index(drop=True)
        df_ppass_small = df_sorted_ppass.head(N_SPLIT)
        df_ppass_large = df_sorted_ppass.tail(N_SPLIT)

        if total_vectors > 0:
            small_min_ratio = df_ppass_small['p_pass'].min() / total_vectors
            small_max_ratio = df_ppass_small['p_pass'].max() / total_vectors
            large_min_ratio = df_ppass_large['p_pass'].min() / total_vectors
            large_max_ratio = df_ppass_large['p_pass'].max() / total_vectors
            ppass_small_range = f"[{small_min_ratio:.3f}-{small_max_ratio:.3f}]" 
            ppass_large_range = f"[{large_min_ratio:.3f}-{large_max_ratio:.3f}]"
        else:
            ppass_small_range = f"[{df_ppass_small['p_pass'].min()}-{df_ppass_small['p_pass'].max()}]"
            ppass_large_range = f"[{df_ppass_large['p_pass'].min()}-{df_ppass_large['p_pass'].max()}]"

        print(f"     -> p_pass Split: Small (N={len(df_ppass_small)}, P={ppass_small_range}), Large (N={len(df_ppass_large)}, P={ppass_large_range})")

        split_groups = {
            f'All Selected (N={len(df_all)})': df_all,
            f'Small Length (N={len(df_len_small)}, L={len_small_range})': df_len_small,
            f'Large Length (N={len(df_len_large)}, L={len_large_range})': df_len_large,
            f'Small p_pass (N={len(df_ppass_small)}, P={ppass_small_range})': df_ppass_small,
            f'Large p_pass (N={len(df_ppass_large)}, P={ppass_large_range})': df_ppass_large,
        }
        
        # 3. 遍历5个拆分好的查询组，为每个组创建绘图数据
        plot_items_for_grid = []
        
        # ========================================================
        # --- 修改点 START: 增加调试标志位 ---
        # ========================================================
        is_first_subplot = True # 标志：是否为第一个子图 (All Selected)
        # ========================================================
        # --- 修改点 END ---
        # ========================================================

        for group_name, df_group_queries in split_groups.items():
            
            print(f"  -> 正在处理绘图组: '{group_name}'")
            combined_plot_data_for_range = {}
            selected_query_list_for_group = df_group_queries[['QueryID']]
            
            # ========================================================
            # --- 修改点 START: 增加调试标志位 ---
            # ========================================================
            is_first_alg = True # 标志：是否为本子图的第一个算法
            # ========================================================
            # --- 修改点 END ---
            # ========================================================
            
            for alg_name, df_detailed in all_detailed_dfs.items():
                df_weighted_detailed = pd.merge(selected_query_list_for_group, df_detailed, on='QueryID', how='left')
                
                # ========================================================
                # --- 修改点 START: 检查并传递调试标志位 ---
                # ========================================================
                
                # 仅当是第一个子图 且 是第一个算法时，才触发调试
                do_debug = is_first_subplot and is_first_alg
                if do_debug:
                    print(f"\n  -> 📈 [坐标计算流程 DEBUG] - 追踪: '{alg_name}' (子图: '{group_name}')")
                
                plot_data_single_alg = plot._prepare_plot_data(
                    df_weighted_detailed, 
                    alg_name, 
                    query_length_range=None,
                    debug_print=do_debug # <--- 传递调试标志
                ) 
                
                is_first_alg = False # 关闭算法标志
                # ========================================================
                # --- 修改点 END ---
                # ========================================================

                if alg_name in plot_data_single_alg:
                    subplot_title_safe = group_name.replace('[', '').replace(']', '').replace(' ', '_').replace('=', '_').replace(',', '')
                    df_plot = plot_data_single_alg[alg_name]
                    
                    # (已修改 plot.py，此函数不再打印到终端)
                    plot._save_plot_coordinates(df_plot, alg_name, subplot_title_safe, task['task_name'])
                
                combined_plot_data_for_range.update(plot_data_single_alg)
            
            subplot_title = f"{task['title']}\n({group_name})"

            if combined_plot_data_for_range:
                plot_item = {
                    'data': combined_plot_data_for_range,
                    'title': subplot_title,
                    'xlabel': f"Recall@{main_exp_config['parameters']['K']}"
                }
                plot_items_for_grid.append(plot_item)
            
            # ========================================================
            # --- 修改点 START: 更新调试标志位 ---
            # ========================================================
            is_first_subplot = False # 关闭子图标志
            # ========================================================
            # --- 修改点 END ---
            # ========================================================

        # 4. 使用准备好的所有绘图项来生成网格图
        if plot_items_for_grid:
            dataset_name_for_path = task["items_to_compare"][0]['dataset']
            output_dir = os.path.join(config['global_settings']['base_results_dir'], dataset_name, "Results", "per-query_results", "pic", dataset_name_for_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, task["output_filename"])
            
            plot.generate_qps_recall_grid(plot_items_for_grid, output_path, task["title"])
        else:
            print(f"  -> 警告: 没有足够的数据来生成图表 '{task['title']}'")

def handle_build_tasks(config):
    tasks = config.get("global_comparison_tasks", {}).get("build_plots", [])
    if not tasks: return

    print("\n" + "#"*80)
    print("📊 开始第四步: 生成构建性能对比图 (Build Plots)")
    print("#"*80)

    for task in tasks:
        if not task.get("enabled", False):
            print(f"⏭️ 跳过已禁用的任务: {task.get('task_name', 'N/A')}")
            continue

        print(f"\n🚀 执行任务: {task['title']}")
        all_build_data = []

        for item in task["items_to_compare"]:
            build_item_data = prepare_build_data(config, item)
            if build_item_data:
                all_build_data.extend(build_item_data)
        
        if all_build_data:
            output_dir = os.path.join(config['global_settings']['base_results_dir'], "merge_results", "experiments", "pic")
            os.makedirs(output_dir, exist_ok=True)
            
            plot.generate_build_summary_plot(
                all_build_data, 
                output_dir, 
                task["output_filename_prefix"],
                task["title"]
            )

def prepare_build_data(config, item_config):
    dataset_name = item_config['dataset']
    exp_type = item_config['type']
    build_params_from_task = item_config['build_params']
    group_title = item_config['group_title']
    
    dataset_conf = config['dataset_configurations'][dataset_name]
    templates = dataset_conf['structure_templates']
    
    base_build_params = dataset_conf.get('build_params', {})
    params = {**base_build_params, **build_params_from_task, 'dataset': dataset_name}
    
    results = []

    try:
        if exp_type == 'Standalone_Unified':
            ung_index_handle = templates['ung_index_handle'].format(**params)
            params['ung_index_handle'] = ung_index_handle
            unified_dir = templates['standalone_build_dir_template'].format(**params)
            print(f"  -> 正在从统一构建目录解析性能: {unified_dir}")

            acorn_data = plot._parse_acorn_meta(os.path.join(unified_dir, "acorn_output", "acorn.index.meta"))
            if acorn_data:
                results.append({'group_title': group_title, 'Algorithm': 'ACORN', 'Index Time (ms)': acorn_data['time'], 'Index Size (MB)': acorn_data['size']})

            acorn1_data = plot._parse_acorn_meta(os.path.join(unified_dir, "acorn_output", "acorn1.index.meta"))
            if acorn1_data:
                results.append({'group_title': group_title, 'Algorithm': 'ACORN-1', 'Index Time (ms)': acorn1_data['time'], 'Index Size (MB)': acorn1_data['size']})

            ung_data = plot._parse_ung_meta(os.path.join(unified_dir, "index_files", "meta")) 
            if ung_data:
                results.append({'group_title': group_title, 'Algorithm': 'UNG', 'Index Time (ms)': ung_data['time'], 'Index Size (MB)': ung_data['size']})
        
        return results

    except Exception as e:
        print(f"❌ 解析 Build 数据时出错 ('{group_title}'): {e}")
        return []


# ==============================================================================
# 3. 主函数
# ==============================================================================

def main():
    config_file = "/home/fengxiaoyao/1023FilterVector/FilterVectorCode/DataTools/pic-perquery/config.json"
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ 错误: 无法加载或解析配置文件 {config_file}。\n{e}")
        return
    
    handle_processing_and_merge_tasks(config)
    handle_qps_recall_tasks(config)
    handle_build_tasks(config)

    print("\n✅ 所有任务处理完毕。")

if __name__ == "__main__":
    main()