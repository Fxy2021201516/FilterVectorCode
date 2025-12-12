import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker
import seaborn as sns
import re
plt.rcParams['axes.unicode_minus'] = False 

ENABLE_RECALL_PRUNING = False 

# ==============================================================================
# 1. 辅助函数
# ==============================================================================

def _calculate_compatible_y_limit(max_val, num_ticks=5):
    if max_val <= 0: return 10.0, 2.0  
    rough_step = max_val / num_ticks
    magnitude = 10**np.floor(np.log10(rough_step))
    residual = rough_step / magnitude
    if residual > 5: nice_residual = 10
    elif residual > 2.5: nice_residual = 5
    elif residual > 2: nice_residual = 2.5
    elif residual > 1: nice_residual = 2
    else: nice_residual = 1
    nice_step = nice_residual * magnitude
    limit = np.ceil(max_val / nice_step) * nice_step
    return limit, nice_step

def _get_adaptive_xaxis_locator(data_structure):
    global_min_recall, global_max_recall, data_found = 1.0, 0.0, False
    items_to_check = []
    if isinstance(data_structure, dict): items_to_check.append({'data': data_structure})
    elif isinstance(data_structure, list): items_to_check.extend(data_structure)
    for config in items_to_check:
        if config.get('data'):
            for alg, data in config['data'].items():
                if not data.empty and 'Recall' in data.columns:
                    global_min_recall = min(global_min_recall, data['Recall'].min())
                    global_max_recall = max(global_max_recall, data['Recall'].max())
                    data_found = True
    if data_found and global_max_recall > global_min_recall:
        return mticker.MultipleLocator(0.05) if (global_max_recall - global_min_recall) <= 0.2 else mticker.MultipleLocator(0.1)
    return mticker.MultipleLocator(0.1)


def _load_and_merge_data(details_file_path):
    print("  -> Loading C++ processed data file...")
    try:
        merged_df = pd.read_csv(details_file_path)
        print(f"       Successfully loaded {len(merged_df)} rows from {os.path.basename(details_file_path)}")
        return merged_df
    except FileNotFoundError:
        print(f"       ERROR: Data file not found at {details_file_path}")
        return pd.DataFrame()

def _prepare_plot_data(df, algorithm_name, query_length_range=None, debug_print=False):
    if df.empty:
        return {}
    
    if query_length_range and len(query_length_range) == 2:
        df_filtered = df[df['QuerySize'].between(query_length_range[0], query_length_range[1])]
        # print(f"       -> 算法 '{algorithm_name}' 的数据已按长度范围 [{query_length_range[0]}-{query_length_range[1]}] 过滤，剩余 {len(df_filtered)} 行。")
    else:
        df_filtered = df
    
    if df_filtered.empty:
        return {}

    # --- [DEBUG] START ---
    if debug_print:
        print(f"       1. 接收到已加权(合并)的数据 {len(df_filtered)} 行。")
        print(f"          (例如，如果选择了 1000 个查询，Lsearch有40个点，这里应有 1000 * 40 = 40000 行)")
        print(f"       2. 按 'Lsearch' 分组 (Grouping by 'Lsearch')...")
    # --- [DEBUG] END ---

    grouped = df_filtered.groupby('Lsearch')
    time_col = 'Time_ms' # 关键：QPS是基于总时间 Time_ms 计算的
    # time_col = 'search_time_ms'
    
    avg_time = grouped[time_col].mean()
    avg_recall = grouped['Recall'].mean()
    
    if debug_print:
        print(f"       3. 计算每个 'Lsearch' 组的平均时间和平均召回率 (已加权):")
        print("          --- (DEBUG) avg_time (Lsearch: Time_ms) ---")
        print(avg_time.head().to_string())
        print("\n          --- (DEBUG) avg_recall (Lsearch: Recall) ---")
        print(avg_recall.head().to_string())

    qps = 1 / (avg_time / 1000.0)*60
    
    if debug_print:
        print("\n       4. 计算 QPS (QPS = 1 / (avg_time / 1000.0)):")
        print("          --- (DEBUG) QPS (Lsearch: QPS) ---")
        print(qps.head().to_string())
    
    df_plot = pd.DataFrame({'Recall': avg_recall, 'QPS': qps}).sort_values(by='Recall').reset_index()
    
    if debug_print:
        print("\n       5. 组合、排序并重置索引后的最终坐标数据:")
        print("          --- (DEBUG) df_plot (Lsearch, Recall, QPS) ---")
        print(df_plot.head().to_string())
        print("       -> [DEBUG] 坐标计算流程打印完毕。\n")

    if ENABLE_RECALL_PRUNING:
        print(f"       -> 算法 '{algorithm_name}' 已启用Recall剪枝。")
        first_reach_one_idx = df_plot.index[df_plot['Recall'] >= 0.999].tolist()
        end_idx = first_reach_one_idx[0] if first_reach_one_idx else df_plot['Recall'].idxmax()
        final_plot_data = df_plot.iloc[:end_idx + 1]
    else:
        # print(f"       -> 算法 '{algorithm_name}' 已禁用Recall剪枝，将绘制所有数据点。")
        final_plot_data = df_plot
        
    return {algorithm_name: final_plot_data}

# ==============================================================================
# --- 修改点 1: 注释掉所有 print() 语句 ---
# ==============================================================================
def _save_plot_coordinates(df_plot, alg_name, subplot_title, task_name):
    """保存 QPS-Recall 坐标到 CSV 文件，(已禁止打印摘要)。"""
    if df_plot.empty:
        return
        
    output_dir = "plot_coordinates_perquery"
    os.makedirs(output_dir, exist_ok=True)
    
    safe_alg_name = alg_name.replace('-', '_').replace('/', '_')
    safe_task_name = task_name.replace('-', '_').replace(' ', '_')
    safe_title = subplot_title.replace('\n', '_').replace(' ', '_').replace('[', '').replace(']', '').replace('=', '_')

    output_filename = f"{safe_task_name}_{safe_title}_{safe_alg_name}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    df_plot.to_csv(output_path, index=False, float_format='%.6f')
    
    # --- (修改 1: 注释掉以下所有打印) ---
    # print(f"\n  -> [坐标输出] '{alg_name}' (子图: {subplot_title}) 坐标数据摘要:")
    # print(df_plot[['Lsearch', 'Recall', 'QPS']].head().to_string(index=False))
    # print(f"  -> 完整坐标 (共 {len(df_plot)} 个点) 已保存到: {os.path.abspath(output_path)}")
    # --- (修改 1 结束) ---

def _print_performance_summary(plot_data, title):
    """在绘图前，打印关键Recall点上的QPS性能，并增加“Max Recall”列便于调试和分析。"""
    print(f"\n     -> Performance Summary for Plot: '{title}'")
    print("         " + "-"*105)
    print(f"         {'Algorithm':<20} | {'QPS @ R≈0.90':<20} | {'QPS @ R≈0.95':<20} | {'QPS @ R≈0.99':<20} | {'QPS @ Max Recall':<25}")
    print("         " + "-"*105)

    for alg, data in plot_data.items():
        if data.empty:
            print(f"         {alg:<20} | {'No Data':<20} | {'No Data':<20} | {'No Data':<20} | {'No Data':<25}")
            continue

        qps_at_90, qps_at_95, qps_at_99 = "N/A", "N/A", "N/A"

        high_recall_90 = data[data['Recall'] >= 0.90]
        if not high_recall_90.empty:
            qps_at_90 = f"{high_recall_90.iloc[0]['QPS']:.2f}"

        high_recall_95 = data[data['Recall'] >= 0.95]
        if not high_recall_95.empty:
            qps_at_95 = f"{high_recall_95.iloc[0]['QPS']:.2f}"
            
        high_recall_99 = data[data['Recall'] >= 0.99]
        if not high_recall_99.empty:
            qps_at_99 = f"{high_recall_99.iloc[0]['QPS']:.2f}"

        max_recall_row = data.iloc[-1]
        max_recall = max_recall_row['Recall']
        qps_at_max = max_recall_row['QPS']
        qps_at_max_recall_str = f"{qps_at_max:.2f} (R={max_recall:.4f})"

        print(f"         {alg:<20} | {qps_at_90:<20} | {qps_at_95:<20} | {qps_at_99:<20} | {qps_at_max_recall_str:<25}")
        
    print("         " + "-"*105)

def _parse_acorn_meta(file_path):
    try:
        with open(file_path, 'r') as f: content = f.read()
        build_time_s = float(re.search(r'build_time_s:([\d.]+)', content).group(1))
        index_size_bytes = int(re.search(r'index_only_size_bytes:(\d+)', content).group(1))
        return {'time': build_time_s * 1000, 'size': index_size_bytes / (1024 * 1024)}
    except (FileNotFoundError, AttributeError, ValueError) as e:
        return None

def _parse_ung_meta(file_path):
    try:
        with open(file_path, 'r') as f: content = f.read()
        def find_value(pattern, text):
            match = re.search(pattern, text)
            return float(match.group(1)) if match else None
        index_time_ms = find_value(r'index_time\(ms\)=([\d.]+)', content)
        index_size_mb = find_value(r'index_size\(MB\)=([\d.]+)', content)
        index_size_add_rb_mb = find_value(r'_index_size_add_rb\(MB\)=([\d.]+)', content)
        if index_time_ms is None or index_size_mb is None or index_size_add_rb_mb is None:
                raise ValueError("One or more required metrics not found in log.")
        return {'time': index_time_ms, 'size': index_size_mb, 'size_add_rb': index_size_add_rb_mb}
    except (FileNotFoundError, AttributeError, ValueError) as e:
        return None

def _plot_qps_recall_on_ax(ax, plot_data, title, xlabel):
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    ax.set_title(title, fontsize=34)
    if plot_data:
        for j, (alg, data) in enumerate(plot_data.items()):
            if not data.empty:
                ax.plot(data['Recall'], data['QPS'], marker=markers[j], linestyle='-', label=alg)
    ax.set_xlabel(xlabel, fontsize=32)

# ==============================================================================
# 3. 网格生成
# ==============================================================================
def generate_qps_recall_grid(all_plot_items, output_path, main_title):
    if not all_plot_items:
        print("错误：没有可供绘制的 QPS-Recall 数据。")
        return

    x_locator = _get_adaptive_xaxis_locator(all_plot_items)
    global_max_qps = 0
    for item in all_plot_items:
        if item.get('data'):
            for alg, data in item['data'].items():
                if not data.empty:
                    global_max_qps = max(global_max_qps, data['QPS'].max())
    y_upper_limit, y_step = _calculate_compatible_y_limit(global_max_qps)
    
    num_plots = len(all_plot_items)
    n_cols = 2
    n_rows = (num_plots + n_cols - 1) // n_cols
    # 适配5个子图 (3行2列)
    if num_plots == 5:
        n_cols = 2
        n_rows = 3
        
    figsize = (10 * n_cols, 8 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    fig.suptitle(main_title, fontsize=36, y=1.02)

    for i, item in enumerate(all_plot_items):
        ax = axes.flat[i]
        
        if item.get('data'):
            _print_performance_summary(item.get('data'), item.get('title', 'N/A'))
        
        _plot_qps_recall_on_ax(ax, item.get('data'), item.get('title', 'N/A'), item.get('xlabel', 'Recall'))

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for i, label in enumerate(l):
            if label not in labels:
                labels.append(label)
                handles.append(h[i])
    
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(handles), fontsize=34)

    num_filled_plots = len(all_plot_items)
    for i, ax in enumerate(axes.flat):
        if i < num_filled_plots:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='x', labelbottom=True, labelsize=32)
            ax.tick_params(axis='y', labelsize=32)
            ax.xaxis.set_major_locator(x_locator)
            ax.set_ylim(0, y_upper_limit)
            if y_step > 0:
                ax.set_yticks(np.arange(0, y_upper_limit + y_step, y_step))
            if i % n_cols == 0: 
                ax.set_ylabel('QPS', fontsize=32)
        else:
            ax.set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ QPS-Recall 组合图已成功保存到: {os.path.abspath(output_path)}")


def generate_build_summary_plot(all_build_data, output_dir, file_prefix, main_title):
    # (此函数未修改)
    if not all_build_data:
        print("错误：没有可供绘制的构建性能数据。"); return
    df = pd.DataFrame(all_build_data)
    df['Algorithm'] = df['Algorithm'].replace({'Hybrid': 'Our Method (Hybrid)', 'ACORN-1': 'ACORN-1'})
    algo_order = ['UNG', 'ACORN', 'ACORN-1', 'Our Method (Hybrid)']
    palette = sns.color_palette("viridis", n_colors=len(algo_order))
    df['Algorithm'] = pd.Categorical(df['Algorithm'], categories=algo_order, ordered=True)
    all_group_categories = df['group_title'].unique().tolist()
    df['group_title'] = pd.Categorical(df['group_title'], categories=all_group_categories, ordered=True)
    df['Index Time (s)'] = df['Index Time (ms)'] / 1000
    
    max_time = df['Index Time (s)'].max()
    time_y_limit, time_y_step = _calculate_compatible_y_limit(max_time)
    fig_time, ax_time = plt.subplots(figsize=(18, 6), dpi=150)
    sns.barplot(data=df, x='group_title', y='Index Time (s)', hue='Algorithm', ax=ax_time, palette=palette, dodge=True)
    ax_time.set_title('Index Build Time', fontsize=20, pad=80)
    ax_time.set_xlabel(''); ax_time.set_ylabel('Index Construction Time (s)', fontsize=20)
    ax_time.set_ylim(0, time_y_limit)
    if time_y_step > 0: ax_time.set_yticks(np.arange(0, time_y_limit + time_y_step, time_y_step))
    ax_time.tick_params(axis='x', rotation=0, labelsize=18); ax_time.tick_params(axis='y', labelsize=18)
    ax_time.grid(axis='y', linestyle='--', alpha=0.7)
    for container in ax_time.containers: ax_time.bar_label(container, fmt='%.0f', fontsize=14, padding=3)
    handles, labels = ax_time.get_legend_handles_labels()
    if ax_time.get_legend() is not None: ax_time.get_legend().remove()
    fig_time.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.83), ncol=len(algo_order), fontsize=16)
    fig_time.tight_layout(rect=[0, 0, 1, 0.95])
    time_output_path = os.path.join(output_dir, f"{file_prefix}_time.png")
    fig_time.savefig(time_output_path, bbox_inches='tight'); plt.close(fig_time)
    print(f"✅ 构建时间图表已成功保存到: {os.path.abspath(time_output_path)}")

    max_size = df['Index Size (MB)'].max()
    size_y_limit, size_y_step = _calculate_compatible_y_limit(max_size)
    fig_size, ax_size = plt.subplots(figsize=(18, 6), dpi=150)
    sns.barplot(data=df, x='group_title', y='Index Size (MB)', hue='Algorithm', ax=ax_size, palette=palette, dodge=True)
    ax_size.set_title('Index Size', fontsize=20, pad=80)
    ax_size.set_xlabel(''); ax_size.set_ylabel('Index Size (MB)', fontsize=20)
    ax_size.set_ylim(0, size_y_limit)
    if size_y_step > 0: ax_size.set_yticks(np.arange(0, size_y_limit + size_y_step, size_y_step))
    ax_size.tick_params(axis='x', rotation=0, labelsize=18); ax_size.tick_params(axis='y', labelsize=18)
    ax_size.grid(axis='y', linestyle='--', alpha=0.7)
    for container in ax_size.containers: ax_size.bar_label(container, fmt='%.0f', fontsize=14, padding=3)
    handles, labels = ax_size.get_legend_handles_labels()
    if ax_size.get_legend() is not None: ax_size.get_legend().remove()
    fig_size.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.83), ncol=len(algo_order), fontsize=16)
    fig_size.tight_layout(rect=[0, 0, 1, 0.95])
    size_output_path = os.path.join(output_dir, f"{file_prefix}_size.png")
    fig_size.savefig(size_output_path, bbox_inches='tight'); plt.close(fig_size)
    print(f"✅ 索引大小图表已成功保存到: {os.path.abspath(size_output_path)}")