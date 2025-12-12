import json
import os
import pandas as pd
from functools import reduce
from modules import process, select, plot

# ==============================================================================
# 1. 辅助函数
# ==============================================================================   

def find_experiment_config(config, dataset_name, experiment_name_to_find):
    """在复杂的 config 结构中，根据数据集和实验名称找到完整的实验配置。"""
    try:
        dataset_conf = config['dataset_configurations'][dataset_name]
        for exp_conf in dataset_conf['experiments']:
            if exp_conf['name'] == experiment_name_to_find:
                return exp_conf
        return None
    except KeyError:
        return None

def build_paths_for_exp(config, dataset_name, exp_config):
    """为特定的数据集和实验构建所有必需的文件路径。"""
    global_settings = config['global_settings']
    dataset_conf = config['dataset_configurations'][dataset_name]
    base_dir = global_settings['base_results_dir']
    templates = dataset_conf['structure_templates']
    
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
    cpp_summary_file = os.path.join(result_base_dir,"results", "search_time_summary.csv")
    
    paths['cpp_details_file'] = cpp_details_file
    paths['ung_summary_file'] = cpp_summary_file
    
    paths['summary_output_path'] = os.path.join(merge_dir, f"{exp_name}_summary.csv")
    
    total_vectors = 0
    if 'N' in format_params:
        total_vectors = int(format_params['N'])
    paths['total_vectors'] = total_vectors
    
    paths['attribute_coverage_file'] = templates['attribute_coverage_file'].format(**format_params)
    
    return paths

# ==============================================================================
# 2. 任务处理器
# ==============================================================================

def handle_processing_and_merge_tasks(config):
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
        
        # --- 在循环外，从第一个实验中获取统一的分析参数 ---
        unified_analysis_params = {}
        if task["items_to_compare"]:
            first_item = task["items_to_compare"][0]
            first_exp_config = find_experiment_config(config, first_item['dataset'], first_item['experiment_name'])
            
            if first_exp_config and "analysis_params" in first_exp_config:
                unified_analysis_params = first_exp_config["analysis_params"]
                print(f"  -> ℹ️ 找到了统一的 analysis_params (来自 {first_item['experiment_name']})。")
                print(f"  -> ℹ️ 将为本任务所有算法应用策略: '{unified_analysis_params.get('optimal_row_strategy', 'default')}'")
            else:
                print(f"  -> 警告: 未能在第一个实验 '{first_item['experiment_name']}' 中找到 'analysis_params'。")
                print(f"  -> 警告: 所有处理将使用 'default' 策略。")
        # --------------
        
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
            'UNG-nTfalse': 'UNG', 'UNG-nTtrue': 'UNGT'
        }
        
        new_columns = []
        for col in df_merged.columns:
            if col in ['QueryID', 'QuerySize']:
                new_columns.append(col)
                continue
            
            parts = col.split('_')
            if len(parts) < 2: 
                new_columns.append(col)
                continue

            algo_name = parts[-1]
            metric_name = '_'.join(parts[:-1])
            
            metric_short = metric_map.get(metric_name, metric_name)
            algo_short = algo_map.get(algo_name, algo_name)
            
            new_columns.append(f"{metric_short}_{algo_short}")
            
        df_merged.columns = new_columns
        
        output_dir = os.path.join(config['global_settings']['base_results_dir'], dataset_name, "Results", "per-query_results")
        output_path = os.path.join(output_dir, f"{task['task_name']}_merged_summary.csv")
        df_merged.to_csv(output_path, index=False, float_format='%.4f')
        print(f"  -> ✅ 总摘要文件已成功保存到: {os.path.abspath(output_path)}")
        
        
        print("\n" + "#"*80)
        print("🔍 开始第二步: 挑选高价值查询 (Select)")
        print("#"*80)
        
        first_item = task["items_to_compare"][0]
        exp_config = find_experiment_config(config, first_item['dataset'], first_item['experiment_name'])
        analysis_params = exp_config.get("analysis_params", {}) if exp_config else {}

        if not analysis_params:
            print("  -> 警告: 未在config中找到 'analysis_params'，将使用默认值。")
         
        analysis_params['current_dataset_name'] = dataset_name

        selection_output_path = os.path.join(output_dir, f"{task['task_name']}_selected_queries.csv")

        select.run_selection(
            merged_summary_path=output_path,
            output_path=selection_output_path,
            params=analysis_params
        )

def handle_qps_recall_tasks(config):
    tasks = config.get("global_comparison_tasks", {}).get("qps_recall_plots", [])
    if not tasks: return

    print("\n" + "#"*80)
    print("📈 开始第三步: 生成QPS-Recall对比图 (Plot)")
    print("#"*80)

    for task in tasks:
        if not task.get("enabled", False):
            print(f"⏭️ 跳过已禁用的任务: {task.get('task_name', 'N/A')}")
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
            print(f"      成功加载 {len(selected_df)} 个带权重的查询ID (共 {len(selected_df['QueryID'].unique())} 个唯一ID)。")
            print(f"      绘图将根据这 {len(selected_df)} 个查询的分布进行加权平均。")
        except FileNotFoundError:
            print(f"      警告: 未找到查询筛选文件 '{selection_file_path}'。将使用所有数据进行绘图。")
            # (selected_df 保持为 None)

        # 1. 加载所有算法的详细数据到内存
        all_detailed_dfs = {}
        main_exp_config = None
        for item in task["items_to_compare"]:
            exp_config = find_experiment_config(config, item['dataset'], item['experiment_name'])
            if not exp_config: continue
            
            if not main_exp_config: main_exp_config = exp_config 

            paths = build_paths_for_exp(config, item['dataset'], exp_config)
            algorithm_name = exp_config['parameters']['algorithm_name']

            df_detailed = plot._load_and_merge_data(paths['cpp_details_file'])
            if df_detailed.empty: continue
            
            if selected_df is not None:
                initial_rows = len(df_detailed)
                selected_query_list = selected_df[['QueryID']]

                df_weighted_detailed = pd.merge(selected_query_list, df_detailed, on='QueryID', how='left')
                
                final_rows = len(df_weighted_detailed)
                print(f"      '{algorithm_name}' 的数据已根据 {len(selected_df)} 个带权重查询进行合并: {initial_rows} (原始) -> {final_rows} (加权后)。")
                all_detailed_dfs[algorithm_name] = df_weighted_detailed
            else:
                print(f"      '{algorithm_name}' 将使用所有 {len(df_detailed)} 行原始数据。")
                all_detailed_dfs[algorithm_name] = df_detailed

        
        # 2. 从配置中获取查询长度范围
        analysis_params = main_exp_config.get("analysis_params", {}) if main_exp_config else {}
        query_length_ranges = analysis_params.get("query_length_ranges")

        if not query_length_ranges:
            print("  -> 警告: 未在config中找到 'query_length_ranges'，将为整个数据集生成一张图。")
            query_length_ranges = [None] 

        # 3. 遍历长度范围，为每个范围创建绘图数据
        plot_items_for_grid = []
        for q_range in query_length_ranges:
            combined_plot_data_for_range = {}
            for alg_name, df_weighted_detailed in all_detailed_dfs.items():
                
                # (plot._prepare_plot_data 现在会接收加权后的数据)
                # (它内部的 .mean() 会自动计算出加权平均的 QPS 和 Recall)
                plot_data_single_alg = plot._prepare_plot_data(df_weighted_detailed, alg_name, q_range)

                #  新的保存函数
                # =========================================================
                if alg_name in plot_data_single_alg:
                    if q_range:
                        subplot_title = f"{task['title']} (Length {q_range[0]}-{q_range[1]})"
                    else:
                        subplot_title = task['title']
                    df_plot = plot_data_single_alg[alg_name]
                    plot._save_plot_coordinates(df_plot, alg_name, subplot_title, task['task_name'])
                # =========================================================
                combined_plot_data_for_range.update(plot_data_single_alg)
            
            if q_range:
                subplot_title = f"{task['title']} (Length {q_range[0]}-{q_range[1]})"
            else:
                subplot_title = task['title']

            if combined_plot_data_for_range:
                plot_item = {
                    'data': combined_plot_data_for_range,
                    'title': subplot_title,
                    'xlabel': f"Recall@{main_exp_config['parameters']['K']}"
                }
                plot_items_for_grid.append(plot_item)

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
        print(f"❌ 错误: 无法加载或解析配置文件 {config_file}。请检查文件是否存在且格式正确。\n{e}")
        return
    
    handle_processing_and_merge_tasks(config)
    handle_qps_recall_tasks(config)
    handle_build_tasks(config)

    print("\n✅ 所有任务处理完毕。")

if __name__ == "__main__":
    main()