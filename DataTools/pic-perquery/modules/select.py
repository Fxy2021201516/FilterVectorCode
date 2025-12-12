import pandas as pd
import os
import numpy as np

# ==============================================================================
# --- 常量定义 ---
# ==============================================================================

METRIC_MAP = {'Time_ms': 'T'}
ALGO_MAP = {
    'method3': 'M3', 'method2': 'M2', 'method1': 'M1',
    'ACORN-gamma': 'AG', 'ACORN-1': 'A1',
    'UNG-nTfalse': 'UNG', 'UNG-nTtrue': 'UNGT'
}
MIN_SPEEDUP_THRESHOLD = 1.5

# ==============================================================================
# --- 辅助函数 ---
# ==============================================================================

def _calculate_and_filter_keys(df, target_alg, baseline_algs, threshold, dataset_name):
    """
    步骤 A: 计算所有排序键，并应用初始阈值过滤。
    """
    print("     -> 步骤 A: 计算所有加速比并应用初始过滤...")
    metric_short = METRIC_MAP['Time_ms']
    
    try:
        target_col = f"{metric_short}_{ALGO_MAP[target_alg]}"
    except KeyError:
        print(f"  -> 错误: 目标算法 '{target_alg}' 未在 ALGO_MAP 中定义。")
        return None
    
    if target_col not in df.columns:
        print(f"  -> 错误: 在摘要文件中找不到目标算法列 '{target_col}'。")
        return None

    df_ratios = df.copy()
    ratio_cols = []
    
    # --- 1. 计算基础加速比 (Speedup_vs_...) ---
    for baseline in baseline_algs:
        try:
            baseline_key = ALGO_MAP[baseline]
        except KeyError:
            print(f"  -> 警告: 基准算法 '{baseline}' 未在 ALGO_MAP 中定义，跳过。")
            continue
            
        baseline_col = f"{metric_short}_{baseline_key}"
        if baseline_col not in df.columns:
            print(f"  -> 警告: 找不到基准算法列 '{baseline_col}'，将跳过此加速比的计算。")
            continue
        
        ratio_col_name = f'Speedup_vs_{baseline}'
        df_ratios[ratio_col_name] = np.divide(df_ratios[baseline_col], df_ratios[target_col])
        ratio_cols.append(ratio_col_name)

    if not ratio_cols:
        print("  -> 错误: 未能成功计算任何加速比。")
        return None
        
    # --- 2a. 计算 "min_speedup" (排序键 1) ---
    df_ratios['min_speedup'] = df_ratios[ratio_cols].min(axis=1)
    df_ratios.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_ratios.dropna(subset=['min_speedup'], inplace=True)
    
    # --- 2b. 应用 *主要* 过滤 (min_speedup) ---
    df_qualified = df_ratios[df_ratios['min_speedup'] > threshold].copy()
    print(f"     -> 共有 {len(df_qualified)} 个查询的最小加速比 > {threshold}。")
    
    if df_qualified.empty:
        print(f"     警告: 没有查询满足最小加速比 > {threshold} 的条件。")
        return None
    
    # --- 2c. 应用 *额外* 的、特定于数据集的过滤 ---
    print(f"     -> 正在为数据集 '{dataset_name}' 应用额外的筛选策略...")
    initial_count = len(df_qualified) 

    if dataset_name == 'Genome':
        print(f"     -> 应用 'Genome' 额外策略: 筛选 UNG-nTtrue / UNG-nTfalse < 1.0。")
        st_ungt_col = f"{metric_short}_{ALGO_MAP['UNG-nTtrue']}"
        st_ungf_col = f"{metric_short}_{ALGO_MAP['UNG-nTfalse']}"
        
        if st_ungt_col in df_qualified.columns and st_ungf_col in df_qualified.columns:
            df_qualified['UNG_T_vs_F_Ratio'] = np.divide(df_qualified[st_ungt_col], df_qualified[st_ungf_col])
            RATIO_THRESHOLD = 1.0 
            df_qualified = df_qualified[df_qualified['UNG_T_vs_F_Ratio'] < RATIO_THRESHOLD].copy()
        else:
            print(f"     -> 警告: 无法计算 'UNG_T_vs_F_Ratio'，跳过此额外过滤。")
    else:
        pass # 默认策略

    if len(df_qualified) < initial_count:
        print(f"     -> 额外筛选后: 查询数从 {initial_count} -> {len(df_qualified)}。")
    else:
        print(f"     -> 'default' 策略: 无额外筛选。")

    if df_qualified.empty:
        print("     警告: 额外筛选后，没有查询满足所有条件。")
        return None
        
    # --- 3. 计算额外的排序键 (用于策略定义) ---
    print("     -> 正在为剩余查询计算额外的排序标准 (max_speedup, max_speedup_vs_ACORN)...")
    
    required_baselines = ['UNG-nTfalse', 'ACORN-1', 'ACORN-gamma']
    if not all(b in baseline_algs for b in required_baselines):
        print(f"  -> 错误: config.json 的 'speedup_baseline_algorithms' 必须包含: {required_baselines}")
        return None

    col_ung = 'Speedup_vs_UNG-nTfalse'
    col_a1 = 'Speedup_vs_ACORN-1'
    col_ag = 'Speedup_vs_ACORN-gamma'
    
    required_cols = [col_ung, col_a1, col_ag]
    if not all(c in df_qualified.columns for c in required_cols):
        print(f"  -> 错误: 在 df_qualified 中找不到一个或多个所需的加速比列: {required_cols}")
        return None

    df_qualified['max_speedup'] = df_qualified[required_cols].max(axis=1)
    df_qualified['max_speedup_vs_ACORN'] = df_qualified[[col_a1, col_ag]].max(axis=1)
    
    return df_qualified


# ==============================================================================
# --- 步骤 B: 筛选与分桶 ---
# ==============================================================================

def _get_bucketing_strategy(dataset_name, length_ranges, verbose=True):
    """
    步骤 B-1: 根据数据集名称，定义分桶排序策略。
    """
    if verbose:
        print(f"\n     -> 步骤 B-1: 检测到数据集 '{dataset_name}'，配置分桶排序策略...")
    
    sort_key_small = None
    sort_key_middle = None
    sort_key_large = None

    if dataset_name == 'bigann':
        if verbose: print("     -> 应用 'bigann' 策略: [Speedup_vs_UNG-nTfalse, Speedup_vs_UNG-nTfalse, min_speedup]")
        sort_key_small = 'Speedup_vs_UNG-nTfalse'
        sort_key_middle = 'Speedup_vs_UNG-nTfalse'
        sort_key_large = 'min_speedup' 
    elif dataset_name == 'celeba':
        if verbose: print("     -> 应用 'celeba' 策略: [Speedup_vs_UNG-nTfalse, max_speedup, max_speedup_vs_ACORN]")
        sort_key_small = 'Speedup_vs_UNG-nTfalse'
        sort_key_middle = 'max_speedup'
        sort_key_large = 'max_speedup_vs_ACORN'
    elif dataset_name == 'words':
        if verbose: print("     -> 应用 'words' 策略: [Speedup_vs_UNG-nTtrue, Speedup_vs_UNG-nTtrue, Speedup_vs_UNG-nTtrue]")
        sort_key_small = 'Speedup_vs_UNG-nTtrue'
        sort_key_middle = 'Speedup_vs_UNG-nTtrue'
        sort_key_large = 'Speedup_vs_UNG-nTtrue'
    elif dataset_name == 'Genome':
        if verbose: print("     -> 应用 'Genome' 策略: [min_speedup, max_speedup, min_speedup]")
        sort_key_small = 'min_speedup'
        sort_key_middle = 'max_speedup'
        sort_key_large = 'min_speedup'
    elif dataset_name == 'MTG':
        if verbose: print("     -> 应用 'MTG' 策略:  [Speedup_vs_UNG-nTfalse, max_speedup, max_speedup_vs_ACORN]")
        sort_key_small = 'Speedup_vs_UNG-nTfalse'
        sort_key_middle = 'max_speedup'
        sort_key_large = 'max_speedup_vs_ACORN'
    elif dataset_name == 'openpmc':
        if verbose: print("     -> 应用 'MTG' 策略:  [Speedup_vs_UNG-nTfalse, Speedup_vs_UNG-nTfalse, Speedup_vs_UNG-nTfalse]")
        sort_key_small = 'Speedup_vs_UNG-nTfalse'
        sort_key_middle = 'Speedup_vs_UNG-nTfalse'
        sort_key_large = 'Speedup_vs_UNG-nTfalse'
    else:
        if verbose: print("     -> 应用 'default' 策略: [min_speedup, max_speedup, min_speedup]")
        sort_key_small = 'min_speedup'
        sort_key_middle = 'max_speedup'
        sort_key_large = 'min_speedup'

    range_configs = [
        {'name': 'Small',  'range': length_ranges[0], 'sort_by': sort_key_small},
        {'name': 'Middle', 'range': length_ranges[1], 'sort_by': sort_key_middle},
        {'name': 'Large',  'range': length_ranges[2], 'sort_by': sort_key_large}
    ]
    
    if not all(cfg['sort_by'] for cfg in range_configs):
        print(f"  -> 错误: 策略未能为所有分桶设置 sort_by 键。")
        return None
        
    if verbose:
        print("     -> 分桶配置:")
        for cfg in range_configs:
              print(f"        - {cfg['name']} (Length {cfg['range'][0]}-{cfg['range'][1]}): 按 {cfg['sort_by']} 排序")
          
    return range_configs


def _perform_bucketing(df_qualified, length_ranges, num_to_select, dataset_name):
    """
    步骤 B-2, B-3, B-4: 执行分桶、按比例挑选和补足差额。
    MODIFIED: 为 'Reviews' 实现了 *条件触发* 的混合挑选策略。
    """
    print(f"\n     -> 步骤 B-2: 执行查询挑选策略...")

    if not length_ranges or len(length_ranges) != 3:
        print(f"  -> 错误: config.json 必须定义3个 'query_length_ranges'。 找到: {length_ranges}")
        return None
        
    # B-1: 获取 *原始* 排序策略
    original_range_configs = _get_bucketing_strategy(dataset_name, length_ranges, verbose=True)
    if original_range_configs is None:
        return None

    # ======================================================================
    # --- 'Reviews' 混合策略 (条件触发) ---
    # ======================================================================
    
    manual_selection = pd.DataFrame()
    df_for_bucketing = df_qualified.copy()
    num_for_bucketing = num_to_select
    configs_for_bucketing = original_range_configs
    
    is_Reviews_special = False # 用于 B-5 和 B-6 的标志
    BUCKET_START_LEN = 3           # L=1/L=2 后的起始长度

    if dataset_name == 'Reviews':
        small_bucket_config = original_range_configs[0]
        small_range = small_bucket_config['range'] # e.g., [1, 3]

        # [核心修改] 检查 'Small' 桶的范围是否 *同时* 包含 1 和 2
        bool_1_in_range = (small_range[0] <= 1 <= small_range[1])
        bool_2_in_range = (small_range[0] <= 2 <= small_range[1])

        if bool_1_in_range and bool_2_in_range:
            is_Reviews_special = True # 激活特殊规则
            print(f"     -> [AppReviews 策略] 1. 'Small' 桶 ({small_range}) 包含 L=1 & L=2。")
            print(f"     -> [AppReviews 策略] 2. 强制挑选最好的 L=1 和 L=2。")
            
            try:
                sort_key_small = small_bucket_config['sort_by']
                if not sort_key_small: 
                    raise ValueError("未能在 range_configs[0] 中找到 'sort_by' 键。")
                
                df_qualified_sorted = df_qualified.sort_values(by=sort_key_small, ascending=False)
                
                best_q_len_1 = df_qualified_sorted[df_qualified_sorted['QuerySize'] == 1].head(1)
                best_q_len_2 = df_qualified_sorted[df_qualified_sorted['QuerySize'] == 2].head(1)
                manual_selection = pd.concat([best_q_len_1, best_q_len_2], ignore_index=True)
                print(f"     -> 已强制选出 {len(manual_selection)} 个查询 (L=1, L=2)。")
                
                # 3. 准备比例分桶 (L=3+)
                num_for_bucketing = num_to_select - len(manual_selection)
                df_for_bucketing = df_qualified[df_qualified['QuerySize'] >= BUCKET_START_LEN].copy()
                
                # 3c. 更新分桶范围: e.g., [[1, 3], [4, 6], [7, 9]] -> [[3, 3], [4, 6], [7, 9]]
                length_ranges_for_bucketing = [r.copy() for r in length_ranges] # 深度复制
                
                if BUCKET_START_LEN <= small_range[1]:
                    # e.g., range=[1, 3]. 3 <= 3. New range = [3, 3]
                    length_ranges_for_bucketing[0] = [BUCKET_START_LEN, small_range[1]]
                else:
                    # e.g., range=[1, 2]. 3 > 2. New range = [3, 2]
                    # 'Small' 桶在 L=3+ 池中将为空，这是正确的。
                    length_ranges_for_bucketing[0] = [BUCKET_START_LEN, small_range[1]]

                # 3d. 重新获取分桶策略 (现在基于 L=3+)
                configs_for_bucketing = _get_bucketing_strategy(dataset_name, length_ranges_for_bucketing, verbose=False)
                
                print(f"     -> [AppReviews 策略] 3. 将 L={BUCKET_START_LEN}+ (共 {len(df_for_bucketing)} 个) 纳入比例分桶 (目标 {num_for_bucketing} 个名额)。")
                
                length_ranges = length_ranges_for_bucketing # 覆盖 B-2 使用的变量

            except (IndexError, KeyError, ValueError) as e:
                print(f"  -> 错误: 无法为 'Reviews' 获取 'Small' 桶的排序键: {e}")
                print("     -> 警告: 将回退到标准分桶逻辑。")
                is_Reviews_special = False # 出错时重置标志
        
        else:
             print(f"     -> [AppReviews 策略] 'Small' 桶 ({small_range}) 未同时包含 L=1 和 L=2。")
             print(f"     -> 将对 'Reviews' 使用标准分桶逻辑。")
             # (is_Reviews_special 保持 False)
    
    # ======================================================================
    # --- B-2: 将查询分入桶中 ---
    # ======================================================================
    # (此步骤现在对 Reviews (如果触发) 使用 L=3+ 的池 和 [[3,3], [4,6], [7,9]] 范围)
    df_buckets = {}
    all_lengths_needed = set()
    for r in length_ranges: # <-- 已为 Reviews (如果触发) 更新
        all_lengths_needed.update(range(r[0], r[1] + 1))
    
    df_qualified_in_ranges = df_for_bucketing[df_for_bucketing['QuerySize'].isin(all_lengths_needed)].copy()
    
    for cfg in configs_for_bucketing: # <-- 已为 Reviews (如果触发) 更新
        is_bigann_large_bucket = (
            dataset_name == 'bigann' and 
            cfg['name'] == 'Large' and 
            cfg['range'][0] == 5 and cfg['range'][1] == 6 
        )
        if is_bigann_large_bucket:
            print(f"     -> [bigann 规则] 'Large' 桶将只包含 Length = 5 的查询。")
            mask = (df_qualified_in_ranges['QuerySize'] == 5)
        else:
            # 确保 range[0] <= range[1]
            if cfg['range'][0] <= cfg['range'][1]:
                mask = df_qualified_in_ranges['QuerySize'].between(cfg['range'][0], cfg['range'][1])
            else:
                mask = pd.Series(False, index=df_qualified_in_ranges.index) # 空范围
        
        df_buckets[cfg['name']] = df_qualified_in_ranges[mask].copy()

    total_qualified_count = len(df_qualified_in_ranges) 
    if total_qualified_count == 0 and not manual_selection.empty:
         print("     警告: L=3+ 合格查询池为空，但已手动选出 L=1/L=2。")
    elif total_qualified_count == 0:
        print("     警告: 合格查询池为空 (在所有指定长度范围内)，无法挑选任何查询。")
        return None

    if not all_lengths_needed:
        print("     -> 合格查询池 (L=N/A) 分布情况:")
    else:
        print(f"     -> 合格查询池 (L={min(all_lengths_needed)}-{max(all_lengths_needed)}) 分布情况:")
        
    for cfg in configs_for_bucketing:
        df_b = df_buckets[cfg['name']]
        range_str = f"L={cfg['range'][0]}-{cfg['range'][1]}"
        print(f"        - {cfg['name']} ({range_str}): {len(df_b)} 个查询")
    print(f"        - 总计: {total_qualified_count} 个查询")

    # ======================================================================
    # --- B-3: 按比例计算并挑选 ---
    # ======================================================================
    num_selected_so_far = 0
    selected_dfs = []
    
    for i, cfg in enumerate(configs_for_bucketing):
        name = cfg['name']
        df_bucket = df_buckets[name]
        
        if i == len(configs_for_bucketing) - 1:
            num_to_select_for_bucket = num_for_bucketing - num_selected_so_far
        else:
            proportion = len(df_bucket) / total_qualified_count if total_qualified_count > 0 else 0
            num_to_select_for_bucket = int(proportion * num_for_bucketing)
        
        num_to_select_for_bucket = max(0, num_to_select_for_bucket)
        num_to_select_for_bucket = min(num_to_select_for_bucket, len(df_bucket))
        
        print(f"     -> 正在为 '{name}' 桶挑选 {num_to_select_for_bucket} / {len(df_bucket)} 个查询...")

        df_bucket_sorted = df_bucket.sort_values(by=cfg['sort_by'], ascending=False)
        selected_df = df_bucket_sorted.head(num_to_select_for_bucket)
        
        selected_dfs.append(selected_df)
        
        # 必须在循环内部更新已选中的数量
        num_selected_so_far += len(selected_df)
    
    # ======================================================================
    # --- B-4: 合并主要挑选结果 ---
    # ======================================================================
    df_bucket_selection = pd.concat(selected_dfs) 
    df_final_selection = pd.concat([manual_selection, df_bucket_selection], ignore_index=True)
    
    num_selected_so_far = len(df_final_selection)
    remaining_to_select = num_to_select - num_selected_so_far
    
    # ======================================================================
    # --- B-5: 补足差额 (Fill-up) ---
    # ======================================================================
    if remaining_to_select > 0:
        print(f"     -> {remaining_to_select} 个名额因比例分配未填满，开始用剩余查询补足...")
        
        all_selected_ids_so_far = set(df_final_selection['QueryID'])
        
        original_lengths_needed = set()
        for r in original_range_configs: 
             original_lengths_needed.update(range(r['range'][0], r['range'][1] + 1))
        
        df_qualified_in_original_ranges = df_qualified[df_qualified['QuerySize'].isin(original_lengths_needed)]
        
        df_remaining_candidates = df_qualified_in_original_ranges[
            ~df_qualified_in_original_ranges['QueryID'].isin(all_selected_ids_so_far)
        ]
        
        # --- [BUG FIX 1] ---
        if is_Reviews_special: # (仅当混合策略被触发时)
            print(f"     -> [AppReviews 规则] 补足差额时，仅从 L={BUCKET_START_LEN}+ 范围的剩余查询中挑选。")
            df_remaining_candidates = df_remaining_candidates[df_remaining_candidates['QuerySize'] >= BUCKET_START_LEN]
        # --- [END FIX] ---

        if not df_remaining_candidates.empty:
            df_remaining_sorted = df_remaining_candidates.sort_values(by='min_speedup', ascending=False)
            df_fillers_unique = df_remaining_sorted.head(remaining_to_select)
            
            print(f"     -> 额外选出 {len(df_fillers_unique)} 个 *唯一* 查询 (按 'min_speedup' 排序) 以填满总数。")
            
            df_final_selection = pd.concat([df_final_selection, df_fillers_unique], ignore_index=True)
        else:
            print(f"     -> 警告: 没有任何剩余的唯一查询可用于补足。")

    # ======================================================================
    # --- B-6: 重复填充 (Duplication) ---
    # ======================================================================
    num_selected_so_far = len(df_final_selection)
    remaining_to_select = num_to_select - num_selected_so_far
    
    if remaining_to_select > 0:
        df_to_duplicate_from = df_final_selection
        if df_to_duplicate_from.empty:
            print(f"     -> 错误: 没有任何查询被选中，无法执行重复填充。")
            return df_final_selection 

        print(f"     -> 警告: 唯一合格查询 ({num_selected_so_far}) 少于目标 ({num_to_select})。")
        print(f"     -> 正在重复使用已选查询以填满剩余的 {remaining_to_select} 个名额...")
        
        # --- [BUG FIX 2] ---
        if is_Reviews_special: # (仅当混合策略被触发时)
            print(f"     -> [AppReviews 规则] 重复填充时，仅从已选的 L={BUCKET_START_LEN}+ 查询中复制。")
            df_to_duplicate_from = df_final_selection[df_final_selection['QuerySize'] >= BUCKET_START_LEN]
            
            if df_to_duplicate_from.empty:
                print(f"     -> 警告: 没有 L={BUCKET_START_LEN}+ 查询可供复制。将从 L=1/L=2 复制以填满。(回退)")
                df_to_duplicate_from = df_final_selection # Fallback
        # --- [END FIX] ---

        num_unique_queries_to_copy = len(df_to_duplicate_from)
        if num_unique_queries_to_copy == 0:
             print(f"     -> 错误: 复制源为空，无法执行重复填充。")
             return df_final_selection
             
        num_repeats = int(np.ceil(remaining_to_select / num_unique_queries_to_copy))
        df_duplicates = pd.concat([df_to_duplicate_from] * num_repeats, ignore_index=True)
        df_fillers_duplicate = df_duplicates.head(remaining_to_select)
        
        df_final_selection = pd.concat([df_final_selection, df_fillers_duplicate], ignore_index=True)
        
        print(f"     -> 成功重复填充。总查询数: {len(df_final_selection)}。")

    return df_final_selection

# ==============================================================================
# --- 步骤 C: 保存结果 ---
# ==============================================================================

def _save_results(original_df, final_stats_df, output_path):
    """
    步骤 C: 保存最终结果并打印统计信息。
    """
    print("\n     -> 步骤 C: 保存最终结果并打印统计信息...")
    
    try:
        original_columns = original_df.columns.tolist()
        final_selected_df_to_save = final_stats_df[original_columns]
    except KeyError as e:
        print(f"  -> 错误: 最终数据中缺少原始列: {e}。将保存所有列。")
        final_selected_df_to_save = final_stats_df

    if not final_stats_df.empty:
        min_val, max_val, mean_val, median_val = final_stats_df['min_speedup'].min(), final_stats_df['min_speedup'].max(), final_stats_df['min_speedup'].mean(), final_stats_df['min_speedup'].median()
        print("\n     -> 最终选出查询的【最小加速比】统计信息:")
        print(f"        - 最小值 (Min Speedup) : {min_val:.2f}x")
        print(f"        - 最大值 (Max Speedup) : {max_val:.2f}x")
        print(f"        - 平均值 (Mean Speedup): {mean_val:.2f}x")
        print(f"        - 中位数 (Median Speedup): {median_val:.2f}x")
    
    if not final_selected_df_to_save.empty:
        print(f"\n     -> 最终选出 {len(final_selected_df_to_save)} 个查询的【查询长度】分布 (含重复):")
        length_distribution = final_selected_df_to_save['QuerySize'].value_counts().sort_index()
        print(length_distribution.to_string())
    
    final_selected_df_to_save.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n  -> ✅ 总共挑选了 {len(final_selected_df_to_save)} 个查询ID (含重复)，结果已保存到: {os.path.abspath(output_path)}")


# ==============================================================================
# --- 主函数 ---
# ==============================================================================

def run_selection(merged_summary_path, output_path, params):

    print(f"\n[Selection] 开始根据策略挑选查询...")

    try:
        print(f"  -> Gagging 加载合并后的摘要文件: {merged_summary_path}")
        df = pd.read_csv(merged_summary_path)
    except FileNotFoundError:
        print(f"  -> 错误: 输入文件未找到 -> {merged_summary_path}")
        raise

    SELECTION_MODE = params.get('selection_mode', 'all')
    print(f"  -> 数据加载成功。当前选择模式: '{SELECTION_MODE}'")

    if SELECTION_MODE == 'all':
        print("  -> 'all' 模式被激活，将选中所有查询。")
        df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"  -> ✅ 全量查询已保存到: {os.path.abspath(output_path)}")
        return
        
    elif SELECTION_MODE == 'min_speedup_selection':
        print(f"     按 'min_speedup_selection' 模式开始筛选...")
        
        target_alg = params.get('speedup_target_algorithm')
        baseline_algs = params.get('speedup_baseline_algorithms', [])
        num_to_select = params.get('num_queries_to_select', 1000)
        length_ranges = params.get('query_length_ranges', [])
        dataset_name = params.get('current_dataset_name', 'unknown') 

        if not all([target_alg, baseline_algs, length_ranges]):
            print("  -> 错误: 'min_speedup_selection' 模式缺少必要的参数。")
            return

        # --- 步骤 A: 计算所有排序键并应用初始过滤 ---
        try:
            df_qualified = _calculate_and_filter_keys(
                df, target_alg, baseline_algs, MIN_SPEEDUP_THRESHOLD, dataset_name
            )
            if df_qualified is None or df_qualified.empty:
                return
        except Exception as e:
            print(f"  -> 错误: 计算排序键时出错: {e}")
            return

        # --- 步骤 B: 按比例分桶筛选法 ---
        final_stats_df = _perform_bucketing(
            df_qualified, length_ranges, num_to_select, dataset_name
        )
        if final_stats_df is None or final_stats_df.empty:
            print("     警告: 未能通过分桶筛选出任何查询。")
            return

        # --- 步骤 C: 保存最终结果 ---
        _save_results(df, final_stats_df, output_path)
        
    else:
        print(f"  -> 错误: 无效或未实现的 SELECTION_MODE: '{SELECTION_MODE}'")