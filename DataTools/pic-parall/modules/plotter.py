import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 将数学字体集设置为 'Computer Modern'，这是 TeX/LaTeX 的默认字体
plt.rcParams['mathtext.fontset'] = 'cm'

import pandas as pd                  
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.interpolate import PchipInterpolator
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle        
from matplotlib.legend_handler import HandlerBase


def calculate_compatible_y_limit(max_val, num_ticks=5):
    """Calculates a 'nice' upper limit for a y-axis."""
    if max_val <= 0: return 10.0, 2.0
    # A more robust calculation for nice steps
    if max_val == 0: return 1.0, 0.2
    power = np.floor(np.log10(max_val))
    base_step = 10**power
    
    # Try steps of 1, 2, 2.5, 5 in the magnitude of the data
    steps = np.array([1, 2, 2.5, 5, 10]) * base_step
    
    # Find a step that creates a reasonable number of ticks
    num_ticks_generated = max_val / steps
    best_step_idx = np.argmin(np.abs(num_ticks_generated - num_ticks))
    nice_step = steps[best_step_idx]
    
    # Calculate the upper limit based on the nice step
    y_upper_limit = np.ceil(max_val / nice_step) * nice_step
    return y_upper_limit, nice_step

def generate_qps_recall_grid(all_plot_items, main_title, output_filename, font_sizes, plot_settings={}, numbering_offset_start=None):

    if not all_plot_items:
        print("错误: 没有可供绘制的数据。")
        return

    # --- 网格布局代码 ---
    grid_layout = plot_settings.get('grid_layout')
    if grid_layout and isinstance(grid_layout, list) and len(grid_layout) == 2:
        n_rows, n_cols = grid_layout
        print(f"  -> 使用自定义网格布局: {n_rows} 行 x {n_cols} 列。")
    else:
        total_items = len(all_plot_items)
        if total_items <= 6: n_cols = total_items if total_items > 0 else 1
        elif total_items <= 12: n_cols = 6
        elif total_items <= 18: n_cols = 6
        else: n_cols = 6
        n_rows = int(np.ceil(total_items / n_cols))

    # --- 固定子图大小 ---
    fixed_subplot_width = 6.0
    fixed_subplot_height = 4.3
    rect_left = 0.0
    rect_right = 1.0
    rect_bottom = 0.03
    rect_top = plot_settings.get('layout_rect_top', 0.88)
    width_fraction = rect_right - rect_left
    height_fraction = rect_top - rect_bottom
    total_fig_width = (fixed_subplot_width * n_cols) / width_fraction
    total_fig_height = (fixed_subplot_height * n_rows) / height_fraction
    fig = plt.figure(figsize=(total_fig_width, total_fig_height))
 
    main_gs = GridSpec(n_rows, n_cols, figure=fig)
    
    # --- 样式策略 ---
    markers_list = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    # 1. 定义颜色映射 (全局固定)
    alg_color_map = {
        "UNG": "tab:blue",
        "ACORN-1": "tab:purple",
        "ACORN-γ": "tab:orange",
        "SmartRoute": "tab:red",
        "ACORN-γ-improved": "tab:green"
    }
    fallback_colors = ['cyan', 'magenta', 'yellow', 'black', 'brown']

    def get_color_for_alg(name):
        if name in alg_color_map:
            return alg_color_map[name]
        return fallback_colors[sum(ord(c) for c in name) % len(fallback_colors)]

    # 2. 定义图例顺序 (Legend Order)
    # 这决定了图例中标签的排列顺序
    default_legend_order = ["UNG", "ACORN-1", "ACORN-γ", "ACORN-γ-improved", "SmartRoute"]
    alg_order = plot_settings.get('custom_alg_order', default_legend_order)

    # 3. 定义绘图层级顺序 (Drawing Order / Z-Order)
    # 列表越靠前的算法，越先被绘制 (即位于图层最底部/Under)
    # 需求：ACORN-γ 在 ACORN-1 之下 -> ACORN-γ 必须排在 ACORN-1 前面
    default_drawing_order = ["UNG", "ACORN-γ", "ACORN-γ-improved", "ACORN-1", "SmartRoute"]
    drawing_order = plot_settings.get('custom_z_order', default_drawing_order)

    def get_marker_for_alg(name):
        # 始终使用 default_legend_order 来索引标记形状，确保同一算法在所有图中形状一致
        if name in default_legend_order:
             return markers_list[default_legend_order.index(name) % len(markers_list)]
        else:
            return markers_list[sum(ord(c) for c in name) % len(markers_list)]

    active_axes = [] 

    # 循环绘制子图
    for i in range(n_rows * n_cols):
        if i >= len(all_plot_items):
            try:
                ax = fig.add_subplot(main_gs[i])
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.tick_params(axis='both', which='major', labelsize=font_sizes.get('tick_label', 18))
            except (IndexError, ValueError):
                pass 
            continue

        item = all_plot_items[i]
        plot_data = item.get('data', {})
        
        max_qps = max((df['QPS'].max() for df in plot_data.values() if not df.empty and 'QPS' in df and df['QPS'].notna().any()), default=0)
        y_upper_limit, y_step = calculate_compatible_y_limit(max_qps)
        subplot_title = item.get('title', '')
        
        ax = fig.add_subplot(main_gs[i])
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        
        active_axes.append(ax)

        # ================= [修改：使用 drawing_order 排序绘图] =================
        def get_draw_priority(name):
            # 如果在绘图顺序列表中，返回其索引（越小越先画）
            if name in drawing_order:
                return drawing_order.index(name)
            # 如果不在，放在最后，并参考图例顺序
            if name in alg_order:
                return len(drawing_order) + alg_order.index(name)
            return 999

        sorted_plot_data = sorted(plot_data.items(), key=lambda x: get_draw_priority(x[0]))
        # ====================================================================

        # --- 绘图逻辑 (已移除插值) ---
        for j, (alg_name, df) in enumerate(sorted_plot_data):
            current_marker = get_marker_for_alg(alg_name)
            current_color = get_color_for_alg(alg_name)
            
            if not df.empty:
                # 确保按 Recall 排序，避免线条乱连
                df_sorted = df.sort_values(by='Average_Recall')
                
                # 直接绘制折线 (不进行平滑)
                ax.plot(df_sorted['Average_Recall'], df_sorted['QPS'], 
                        marker=current_marker, 
                        linestyle='-', 
                        label=alg_name,
                        lw=5,             # 保持原有线宽
                        markersize=12,    # 保持原有标记大小
                        color=current_color)

        if numbering_offset_start is not None:
            current_plot_number = numbering_offset_start + i
            number_font_size = font_sizes.get('num_label', 20)
            ax.set_title(f"({current_plot_number})", 
                         loc='left', 
                         fontsize=number_font_size,   
                         color='black',                 
                         weight='normal')               
            
            ax.set_title(subplot_title, 
                         loc='center', 
                         fontsize=font_sizes.get('subplot_title', 26), 
                         pad=12)
        else:
            ax.set_title(subplot_title, 
                         loc='center', 
                         fontsize=font_sizes.get('subplot_title', 26), 
                         pad=12)

        ax.set_xlabel(item.get('xlabel', 'Recall'), fontsize=font_sizes.get('axis_label', 22))
        ax.set_ylabel('QPS', fontsize=font_sizes.get('axis_label', 22))
        
        # ... (坐标轴通用设置) ...
        use_log_y = plot_settings.get("use_log_scale_y", False)
        
        if use_log_y:
            ax.set_yscale('log')
            all_qps_values = [v for df in plot_data.values() if not df.empty for v in df['QPS'] if v > 0]
            if all_qps_values:
                max_val = max(all_qps_values)
                min_val = min(all_qps_values)
                y_upper_limit_log = 10**np.ceil(np.log10(max_val + 1e-9))
                y_lower_limit_log = 10**np.floor(np.log10(min_val * 0.9))
                ax.set_ylim(bottom=y_lower_limit_log, top=y_upper_limit_log)
                ticks_to_show = [10**i for i in np.arange(np.round(np.log10(y_lower_limit_log)), np.round(np.log10(y_upper_limit_log)) + 1)]
                ax.set_yticks(ticks_to_show)
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            else:
                ax.set_ylim(bottom=1, top=1000)
        else:
            ax.set_ylim(0, y_upper_limit)
            if y_step > 0:
                ax.set_yticks(np.arange(0, y_upper_limit + y_step, y_step))
        
        all_recalls = [v for df in plot_data.values() if not df.empty for v in df['Average_Recall'] if pd.notna(v)]
        if all_recalls:
            min_r = min(all_recalls) if all_recalls else 0.0
            max_r = max(all_recalls) if all_recalls else 1.0
            plot_min_x_raw = np.floor(min_r * 10) / 10.0
            plot_max_x_raw = min(1.0, np.ceil(max_r * 10) / 10.0)
            if plot_max_x_raw - plot_min_x_raw < 0.1:
                plot_max_x_raw = min(1.0, plot_min_x_raw + 0.1)
            data_range_raw = plot_max_x_raw - plot_min_x_raw
            if data_range_raw <= 0.2: step = 0.05
            elif data_range_raw <= 0.5: step = 0.1
            else: step = 0.2
            plot_min_x_adj = np.floor(np.round(plot_min_x_raw / step, 5)) * step
            plot_max_x_adj = np.ceil(np.round(plot_max_x_raw / step, 5)) * step
            ax.set_xlim(plot_min_x_adj, plot_max_x_adj)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
        else:
            ax.set_xlim(0.0, 1.0)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=font_sizes.get('tick_label', 18))
        ax.tick_params(axis='x', which='major', pad=7)

    # --- 共享图例逻辑 (Legend) ---
    # 1. 收集所有出现过的标签
    unique_labels_set = set()
    for ax in active_axes:
        h, l = ax.get_legend_handles_labels()
        for lab in l:
            unique_labels_set.add(lab)
    
    # 2. 按照 alg_order (图例顺序) 排序标签
    # 注意：这里我们不使用 drawing_order，因为我们需要图例保持逻辑上的顺序，而不是绘图的层级顺序
    def sort_key(name):
        if name in alg_order:
            return alg_order.index(name)
        return len(alg_order) + sum(ord(c) for c in name)
        
    sorted_unique_labels = sorted(list(unique_labels_set), key=sort_key)
    
    # 3. 手动创建图例句柄
    legend_handles = []
    for lab in sorted_unique_labels:
        col = get_color_for_alg(lab) 
        mar = get_marker_for_alg(lab) 
        
        proxy = Line2D([], [], color=col, label=lab, 
                       marker=mar, markersize=12, 
                       linewidth=5, linestyle='-')
        legend_handles.append(proxy)

    legend_cols = plot_settings.get('legend_cols', len(legend_handles))
    legend_y = plot_settings.get('legend_anchor_y', 0.92)
    
    if legend_handles and not plot_settings.get('hide_legend', False):
        fig.legend(legend_handles, sorted_unique_labels, 
                   loc='upper center', bbox_to_anchor=(0.5, legend_y), 
                   ncol=legend_cols, fontsize=font_sizes.get('legend', 24),
                   frameon=False)

    # --- 布局和保存 ---
    h_pad_val = plot_settings.get('h_pad', 1.08)
    w_pad_val = plot_settings.get('w_pad', 1.08) 
    main_gs.tight_layout(fig, rect=[rect_left, rect_bottom, rect_right, rect_top], h_pad=h_pad_val, w_pad=w_pad_val)
    
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ (Z-Order Fixed, No Interpolation) 网格图表已成功保存到: {os.path.abspath(output_path)}")


# --- 自定义图例处理器：实现“点在框中”的效果 ---
# --- 修改后的图例处理器：只显示点，没有框 ---
class PointInBoxLegendHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        color = orig_handle.color
        marker = orig_handle.marker
        
        # 计算中心位置
        center_x = xdescent + width / 2
        center_y = ydescent + height / 2
        
        # 【修改点 2 & 3】 去掉了 Rectangle (框)，并增大了 markersize
        # 画点 (前景) - 居中
        line = Line2D(
            [center_x], 
            [center_y], 
            marker=marker, 
            color=color, 
            linestyle='None',
            markersize=24,  # <--- 【增大】图例中的点大小 (原为14)
            markeredgecolor='white',
            markeredgewidth=2.0,
            transform=trans
        )
        
        # 只返回点，不返回框
        return [line]
    

class LegendProxy:
    def __init__(self, color, marker):
        self.color = color
        self.marker = marker

def generate_speedup_ratio_plot(all_ratios_data, task_config, font_sizes, output_filename):

    if not all_ratios_data:
        print("错误: 没有可供绘制的数据。")
        return
            
    df = pd.DataFrame(all_ratios_data)
    
    # --- 1. 计算 Y 轴范围和无穷大逻辑 ---
    # 找出有限值的最大值
    finite_vals = df.loc[df['ratio_value'] != np.inf, 'ratio_value']
    finite_max = finite_vals.max() if not finite_vals.empty else 10.0
    
    # 计算有限值的对数最大值
    base_log_finite_max = np.log10(finite_max)
    
    # 1. 向上取整找到包含数据的最近整数幂 (例如 max=85 -> 10^2)
    next_int_pow = np.ceil(base_log_finite_max)
    # 2. 强制将无穷大线设为再高一个整数幂 (例如 10^3)
    # 这样 10^2 到 10^3 (无穷大) 的距离就是严格的 1 个对数单位，与下方刻度间距一致
    log_top_limit = next_int_pow + 1.0
    y_top_limit = 10**log_top_limit
    
    #  优化无穷大点的显示 (减小波动，整体下移)
    mask_inf = (df['ratio_value'] == np.inf)
    n_inf = mask_inf.sum()
    
    df['ratio_plot'] = df['ratio_value'].copy()
    
    if n_inf > 0:
        # 定义无穷大点的分布范围 (在对数尺度上)
        # log_top_limit 是无穷大线的位置, 让点分布在 [线-0.5, 线-0.2] 之间
        jitter_high = log_top_limit - 0.15
        jitter_low = log_top_limit - 0.30
        
        # 生成随机噪点
        jitter_logs = np.random.uniform(jitter_low, jitter_high, size=n_inf)
        df.loc[mask_inf, 'ratio_plot'] = 10**jitter_logs
    
    # 移除 NaN
    df = df.dropna(subset=['ratio_plot'])
    if df.empty:
        print("警告: 所有数据均为 NaN，无法绘图。")
        return

    # --- 2. 准备绘图参数 ---
    x_labels_display = task_config.get('x_axis_categories', [])
    data_category_names = [cat.get('name', '') for cat in task_config.get('categories', [])]
    
    min_len = min(len(x_labels_display), len(data_category_names))
    x_labels_display = x_labels_display[:min_len]
    data_category_names = data_category_names[:min_len]

    hue_order = task_config.get('ratio_labels', [])
    
    colors_list = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628']
    markers_list = ['o', 's', 'v', '^', 'D', 'P']
    
    palette_map = dict(zip(hue_order, colors_list))
    markers_map = dict(zip(hue_order, markers_list))
    
    # --- 3. 初始化画布 ---
    fig, ax = plt.subplots(figsize=(25, 8)) 
    
    n_bars = len(hue_order)
    total_width = 0.85
    bar_width = total_width / n_bars
    offsets = np.arange(n_bars) - (n_bars - 1) / 2.0
    
    # 重新计算有效的 x_tick 位置和标签 (为了去除 K=5)
    valid_x_ticks = []
    valid_x_labels = []
    
    current_x_idx = 0

    # --- 4. 核心绘图循环 ---
    for i in range(len(x_labels_display)):
        cat_data_name = data_category_names[i]
        display_label = x_labels_display[i]
        
        # 在代码层面过滤掉 K=5
        if "K=5" in display_label or "K=5" in cat_data_name:
            continue
            
        valid_x_ticks.append(current_x_idx)
        valid_x_labels.append(display_label)
        
        x_base = current_x_idx
        
        for j, ratio_name in enumerate(hue_order):
            x_center = x_base + offsets[j] * bar_width
            
            sub_df = df[(df['category'] == cat_data_name) & (df['ratio_name'] == ratio_name)]
            if sub_df.empty: continue
            
            col = palette_map.get(ratio_name, 'black')
            mar = markers_map.get(ratio_name, 'o')
            
            y_vals = sub_df['ratio_plot'].values
            
            # 画散点 (Scatter)
            # s=600 保持大尺寸
            ax.scatter([x_center]*len(y_vals), y_vals, 
                       color=col, marker=mar, 
                       s=600,  
                       edgecolor='white', linewidth=2.0, zorder=10, alpha=0.9)
        
        current_x_idx += 1 

    # --- 5. 坐标轴设置 ---
    ax.set_yscale('log')
    ax.set_ylim(0.1, y_top_limit) # Y轴上限
    
    # 【修改点 C】 设置严格对齐的 Y 轴刻度
    # 从 -1 (0.1) 到 next_int_pow (有限值最大整数幂)
    y_ticks = [10**k for k in range(-1, int(next_int_pow) + 1)]
    # 添加最顶部的无穷大刻度 (它是 next_int_pow + 1)
    y_ticks.append(y_top_limit) 
    
    ax.set_yticks(y_ticks)
    
    y_ticklabels = []
    for y in y_ticks[:-1]:
        exp_val = int(np.log10(y))
        #y_ticklabels.append(f"$10^{{{exp_val}}}$")
        y_ticklabels.append(f"$\\mathsf{{10}}^{{\\mathsf{{{exp_val}}}}}$")
    
    # 最顶部的标签设为无穷大符号
    y_ticklabels.append(r"$\infty$")
    
    ax.set_yticklabels(y_ticklabels, fontsize=font_sizes.get('tick_label_y', 40))
    
    # 使用过滤后的 X 轴 标签
    ax.set_xticks(valid_x_ticks)
    ax.set_xticklabels(valid_x_labels, fontsize=font_sizes.get('tick_label_x', 36))
    
    ax.set_ylabel('QPS Ratio', fontsize=font_sizes.get('axis_label_y', 40))
    ax.set_xlabel('', fontsize=1)
    
    # 画网格线
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    # 顶部画一条实线表示边界
    ax.axhline(y=y_top_limit, color='black', linewidth=1.5, linestyle='-')

    # --- 6. 自定义图例 ---
    legend_proxies = []
    legend_labels = []
    
    reorder_indices = [0, 3, 1, 4, 2, 5] if len(hue_order) == 6 else range(len(hue_order))
    
    for idx in reorder_indices:
        r_name = hue_order[idx]
        col = palette_map[r_name]
        mar = markers_map[r_name]
        
        l_math = task_config.get('ratio_labels_map', {}).get(r_name, r_name)
        label_str = f"{l_math}" 
        
        proxy = LegendProxy(color=col, marker=mar)
        
        legend_proxies.append(proxy)
        legend_labels.append(label_str)

    legend_cols = task_config.get('legend_cols', 4) 

    ax.legend(legend_proxies, legend_labels, 
              handler_map={LegendProxy: PointInBoxLegendHandler()},
              loc='upper center', 
              bbox_to_anchor=(0.5, 1.30), 
              ncol=legend_cols,
              fontsize=font_sizes.get('legend', 48) + 4, 
              frameon=False,
              handletextpad=0.4,
              columnspacing=1.2
             )

    # --- 7. 保存 ---
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ 加速比图表 (优化版: 均匀刻度, 无穷大微调) 已保存到: {os.path.abspath(output_path)}")


def generate_build_time_plot(datasets_info, output_filename, font_sizes):

    labels = list(datasets_info.keys())
    num_labels = len(labels)
    # 至少分配8个槽位，如果数据集更多，则自动扩展
    total_slots = max(8, num_labels) 
    x_tick_labels = labels + [''] * (total_slots - num_labels)
    
    # 使用 .get(key, 0) 来安全地获取数据，如果键不存在则返回0
    serial_acorn_1 = [info.get('serial_acorn_1_time_s', 0) for info in datasets_info.values()]
    serial_acorn_gamma = [info.get('serial_acorn_gamma_time_s', 0) for info in datasets_info.values()]
    serial_ung = [info.get('serial_ung_time_s', 0) for info in datasets_info.values()]
    parallel_max = [info.get('parallel_max_time_s', 0) for info in datasets_info.values()] # This is optional

    # 提取所有非零值用于Y轴范围计算
    all_values = [v for v in (serial_acorn_1 + serial_acorn_gamma + serial_ung + parallel_max) if v > 0]
    if not all_values:
        print(" -> 警告: 没有任何构建时间数据可供绘制。")
        return
        
    # Y轴限制使用对数逻辑
    max_val = max(all_values)
    min_val = min(all_values)
    y_upper_limit = 10**np.ceil(np.log10(max_val + 1e-9))
    y_lower_limit = 10**np.floor(np.log10(min_val * 0.9))
    if y_lower_limit < 0.1: y_lower_limit = 0.1 # 最小 0.1s

    x_bars = np.arange(len(labels))
    
    #  动态计算要绘制的条形图
    bar_groups = []
    bar_labels = {}
    bar_data = {}

    # 1. UNG (顺序调整到最前)
    if any(v > 0 for v in serial_ung): 
       bar_groups.append('ung')
       bar_labels['ung'] = 'UNG'
       bar_data['ung'] = serial_ung
       
    # 2. ACORN-1 (顺序调整到第二)
    if any(v > 0 for v in serial_acorn_1): 
       bar_groups.append('acorn_1')
       bar_labels['acorn_1'] = 'ACORN-1'
       bar_data['acorn_1'] = serial_acorn_1

    # 3. ACORN-γ (顺序调整到第三)
    if any(v > 0 for v in serial_acorn_gamma): 
       bar_groups.append('acorn_gamma')
       bar_labels['acorn_gamma'] = 'ACORN-γ (-improved)'
       bar_data['acorn_gamma'] = serial_acorn_gamma

    # 4. Parallel (顺序调整到最后，并重命名)
    if any(v > 0 for v in parallel_max): 
       bar_groups.append('parallel')
       bar_labels['parallel'] = 'SmartRoute' # <--- 修改点 1
       bar_data['parallel'] = parallel_max
    
    num_bars = len(bar_groups)
    width = 0.8 / num_bars # 动态计算条形图宽度
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 动态计算每个条的中心偏移
    offsets = np.arange(num_bars) - (num_bars - 1) / 2.0
    
    # 循环绘制所有可用的条
    for i, group_key in enumerate(bar_groups):
       # 绘制时替换0值，防止对数坐标出错
       data_to_plot = [v if v > 0 else y_lower_limit * 0.5 for v in bar_data[group_key]]
       ax.bar(x_bars + offsets[i] * width, 
              data_to_plot, 
              width, 
              label=bar_labels[group_key])

    ax.set_ylabel('Build Time (seconds)', fontsize=font_sizes.get('axis_label', 18))
    ax.set_xticks(np.arange(total_slots))
    ax.set_xticklabels(x_tick_labels, fontsize=font_sizes.get('tick_label', 16))
    ax.tick_params(axis='y', labelsize=font_sizes.get('tick_label', 16))
    ax.set_xlim(-0.5, total_slots - 0.5)

    # 设置对数坐标轴和限制
    ax.set_yscale('log')
    ax.set_ylim(bottom=y_lower_limit, top=y_upper_limit)


    # 手动计算显示的刻度（从 y_lower_log 到 y_upper_log 的所有10的幂）
    ticks_to_show = [10**i for i in np.arange(np.log10(y_lower_limit), np.log10(y_upper_limit) + 1)]
    ax.set_yticks(ticks_to_show)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    
    
    #fig.suptitle('Index Build Time Comparison', fontsize=font_sizes.get('main_title', 22))
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=num_bars, fontsize=font_sizes.get('legend', 16), frameon=False)
    
    # ax.grid(axis='y', which='major', linestyle='--', alpha=0.7)
    # ax.grid(axis='y', which='minor', linestyle=':', alpha=0.5)
    
    # 调整图例间距
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_filename), dpi=150)
    plt.close(fig)
    print(f"✅ 构建时间对比图 (Log Scale) 已成功保存到: {os.path.abspath(os.path.join(output_dir, output_filename))}")


def generate_index_size_plot(datasets_info, output_filename, font_sizes):

    labels = list(datasets_info.keys())
    num_labels = len(labels)
    # 至少分配8个槽位，如果数据集更多，则自动扩展
    total_slots = max(8, num_labels)
    x_tick_labels = labels + [''] * (total_slots - num_labels)
    
    serial_acorn_1 = [info.get('serial_acorn_1_size_mb', 0) for info in datasets_info.values()]
    serial_acorn_gamma = [info.get('serial_acorn_gamma_size_mb', 0) for info in datasets_info.values()]
    serial_ung = [info.get('serial_ung_size_mb', 0) for info in datasets_info.values()]
    parallel_sum = [info.get('parallel_sum_size_mb', 0) for info in datasets_info.values()] # Optional

    all_values = [v for v in (serial_acorn_1 + serial_acorn_gamma + serial_ung + parallel_sum) if v > 0]
    if not all_values:
       print(" -> 警告: 没有任何索引大小数据可供绘制。")
       return

    max_val = max(all_values)
    min_val = min(all_values)
    y_upper_limit = 10**np.ceil(np.log10(max_val + 1e-9))
    y_lower_limit = 10**np.floor(np.log10(min_val * 0.9))
    if y_lower_limit < 1: y_lower_limit = 1 # 最小 1MB

    x_bars = np.arange(len(labels))

    #动态计算要绘制的条形图
    bar_groups = []
    bar_labels = {}
    bar_data = {}

    # 1. UNG (顺序调整到最前)
    if any(v > 0 for v in serial_ung): 
       bar_groups.append('ung')
       bar_labels['ung'] = 'UNG'
       bar_data['ung'] = serial_ung

    # 2. ACORN-1 (顺序调整到第二)
    if any(v > 0 for v in serial_acorn_1): 
       bar_groups.append('acorn_1')
       bar_labels['acorn_1'] = 'ACORN-1'
       bar_data['acorn_1'] = serial_acorn_1
       
    # 3. ACORN-γ (顺序调整到第三)
    if any(v > 0 for v in serial_acorn_gamma): 
       bar_groups.append('acorn_gamma')
       bar_labels['acorn_gamma'] = 'ACORN-γ (-improved)'
       bar_data['acorn_gamma'] = serial_acorn_gamma

    # 4. Parallel (顺序调整到最后，并重命名)
    if any(v > 0 for v in parallel_sum): 
       bar_groups.append('parallel')
       bar_labels['parallel'] = 'SmartRoute' # <--- 修改点 2
       bar_data['parallel'] = parallel_sum

    num_bars = len(bar_groups)
    width = 0.8 / num_bars # 动态计算条形图宽度
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 动态计算每个条的中心偏移
    offsets = np.arange(num_bars) - (num_bars - 1) / 2.0
    
    # 循环绘制所有可用的条
    for i, group_key in enumerate(bar_groups):
       # 绘制时替换0值，防止对数坐标出错
       data_to_plot = [v if v > 0 else y_lower_limit * 0.5 for v in bar_data[group_key]]
       ax.bar(x_bars + offsets[i] * width, 
              data_to_plot, 
              width, 
              label=bar_labels[group_key])

    ax.set_ylabel('Index Size (MB)', fontsize=font_sizes.get('axis_label', 18))
    ax.set_xticks(np.arange(total_slots))
    ax.set_xticklabels(x_tick_labels, fontsize=font_sizes.get('tick_label', 16))
    ax.tick_params(axis='y', labelsize=font_sizes.get('tick_label', 16))
    ax.set_xlim(-0.5, total_slots - 0.5)

    #设置对数坐标轴和限制
    ax.set_yscale('log')
    ax.set_ylim(bottom=y_lower_limit, top=y_upper_limit)

    # 手动计算我们要显示的刻度（从 y_lower_log 到 y_upper_log 的所有10的幂）
    ticks_to_show = [10**i for i in np.arange(np.log10(y_lower_limit), np.log10(y_upper_limit) + 1)]
    ax.set_yticks(ticks_to_show)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    #fig.suptitle('Index Size Comparison', fontsize=font_sizes.get('main_title', 22))
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=num_bars, fontsize=font_sizes.get('legend', 16), frameon=False)
    
    #移除Y轴网格线
    # ax.grid(axis='y', which='major', linestyle='--', alpha=0.7)
    # ax.grid(axis='y', which='minor', linestyle=':', alpha=0.5)
    
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_filename), dpi=150)
    plt.close(fig)
    print(f"✅ 索引大小对比图 (Log Scale) 已成功保存到: {os.path.abspath(os.path.join(output_dir, output_filename))}")