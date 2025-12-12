import pandas as pd
import numpy as np
import os
import sys

# ==============================================================================
# --- 辅助函数：定义算法和指标的简称 ---
# ==============================================================================

METRIC_MAP_SELECT = {'SearchTime_ms': 'ST'}
ALGO_MAP_SELECT = {
    'method3': 'M3', 'method2': 'M2', 'method1': 'M1',
    'ACORN-gamma': 'AG', 'ACORN-1': 'A1',
    'ACORN-gamma-improved': 'AGI',
    'UNG-nTfalse': 'UNG', 'UNG-nTtrue': 'UNGT'
}

# ==============================================================================
# --- 主函数：run_selection ---
# ==============================================================================

def run_selection(merged_summary_path, output_path, params, attribute_coverage_path):
    """
    执行新的筛选逻辑：
    (已更新) ...
    (已更新) 3. 按 QueryID 将合格查询分为 A, B, C 三类。
    (已更新) 4. (新增) 可选地，根据 'c_length_range' 过滤 C 类查询。
    (已更新) 5. 按 config.json 中定义的比例从 A, B, C 中随机抽样。
    """
    
    print("\n[Selection] 开始执行新的'比例抽样'筛选策略...")
    
    # --- 1. 加载数据 ---
    try:
        print(f"  -> 正在加载合并后的摘要文件: {os.path.basename(merged_summary_path)}")
        df_merged = pd.read_csv(merged_summary_path)
    except FileNotFoundError:
        print(f"  -> 错误: 输入文件未找到 -> {merged_summary_path}")
        return

    try:
        print(f"  -> 正在加载覆盖率文件: {os.path.basename(attribute_coverage_path)}")
        df_coverage = pd.read_csv(attribute_coverage_path)
        
        if 'coverage_count' not in df_coverage.columns:
            print(f"  -> 错误: 在 {attribute_coverage_path} 中未找到 'coverage_count' 列。")
            return
            
        df_coverage.rename(columns={'coverage_count': 'p_pass'}, inplace=True)
        df_coverage['QueryID'] = df_coverage.index 
        print(f"     -> 已加载 {len(df_coverage)} 行，已从 0-based 索引生成 'QueryID'。")
        
        df_full = pd.merge(df_merged, df_coverage[['QueryID', 'p_pass']], on='QueryID', how='left')
        
        if df_full['p_pass'].isnull().any():
            nan_count = df_full['p_pass'].isnull().sum()
            print(f"  -> 警告: {nan_count} 个查询在合并 'p_pass' 后为空。将丢弃这些行。")
            df_full.dropna(subset=['p_pass'], inplace=True)
        
        print(f"  -> 成功合并 p_pass，剩余 {len(df_full)} 行有效数据。")
        
    except (FileNotFoundError, TypeError) as e:
        print(f"  -> 错误: 覆盖率文件加载失败: {e}")
        return
    except Exception as e:
        print(f"  -> 错误: 处理覆盖率文件时出错: {e}")
        return
    

    # ==========================================================================
    # --- 全局硬性过滤 (可配置): UNG 或 ACORN-gamma 至少一个达标 ---
    # ==========================================================================
    
    # 1. 读取配置
    global_filter_conf = params.get("global_recall_filter", {})
    is_global_filter_enabled = global_filter_conf.get("enabled", False)
    
    if is_global_filter_enabled:
        min_recall_threshold = global_filter_conf.get("min_recall", 0.8)
        
        # 定义目标列名 (基于 pipeline 的映射规则)
        col_ung = 'R_UNG'  # 对应 UNG-nTfalse
        col_ag = 'R_AG'    # 对应 ACORN-gamma
        
        print(f"\n  -> [Global Filter] 已启用全局过滤: {col_ung} >= {min_recall_threshold} OR {col_ag} >= {min_recall_threshold}")
        
        # 2. 检查列是否存在
        available_check_cols = [c for c in [col_ung, col_ag] if c in df_full.columns]
        
        if not available_check_cols:
            print(f"     -> 警告: 找不到 '{col_ung}' 或 '{col_ag}' 列。无法执行全局过滤，跳过。")
        else:
            count_before = len(df_full)
            
            # 3. 构建过滤掩码 (Mask)
            # 逻辑: 只要有一个算法的 Recall 达标，就保留该查询 (OR 逻辑)
            valid_mask = pd.Series(False, index=df_full.index)
            
            if col_ung in df_full.columns:
                valid_mask |= (df_full[col_ung] >= min_recall_threshold)
            else:
                print(f"     -> 提示: '{col_ung}' 不存在，仅检查 '{col_ag}'。")

            if col_ag in df_full.columns:
                valid_mask |= (df_full[col_ag] >= min_recall_threshold)
            else:
                print(f"     -> 提示: '{col_ag}' 不存在，仅检查 '{col_ung}'。")
            
            # 4. 应用过滤
            df_full = df_full[valid_mask].copy()
            count_after = len(df_full)
            dropped = count_before - count_after
            
            print(f"     -> 过滤完成: {count_before} -> {count_after} 个查询保留。")
            print(f"     -> 剔除了 {dropped} 个查询 (在现有算法上 Recall 均 < {min_recall_threshold})。")
            
            if df_full.empty:
                print("     -> 错误: 所有查询都被全局过滤器剔除了！请检查阈值或数据质量。")
                return
    else:
        print("\n  -> [Global Filter] 全局过滤未启用 (enabled=False)。")

    # ==========================================================================
    # ==========================================================================

    
    # --- 步骤 A: (可选) 加速比筛选 ---
    ENABLE_SPEEDUP_FILTER = params.get("enable_speedup_filter", True)
    df_qualified = None 

    if ENABLE_SPEEDUP_FILTER:
        print("\n  -> 'enable_speedup_filter' 为 True。执行步骤 A：加速比筛选...")
        target_alg = params.get('speedup_target_algorithm')
        baseline_algs = params.get('speedup_baseline_algorithms', [])

        if not target_alg or not baseline_algs:
            print("  -> 错误: config.json 中未定义 'speedup_target_algorithm' 或 'speedup_baseline_algorithms'。")
            return

        try:
            metric_short = METRIC_MAP_SELECT['SearchTime_ms']
            target_col = f"{metric_short}_{ALGO_MAP_SELECT[target_alg]}"
            
            if target_col not in df_full.columns:
                print(f"  -> 错误: 目标列 '{target_col}' 不在摘要文件中。")
                return

            df_ratios = df_full.copy()
            ratio_cols = []

            for baseline in baseline_algs:
                baseline_col = f"{metric_short}_{ALGO_MAP_SELECT[baseline]}"
                if baseline_col in df_ratios.columns:
                    ratio_col_name = f'Speedup_vs_{baseline}'
                    df_ratios[ratio_col_name] = np.divide(df_ratios[baseline_col], df_ratios[target_col])
                    ratio_cols.append(ratio_col_name)
                else:
                    print(f"  -> 警告: 找不到基准列 '{baseline_col}'，跳过。")
            
            if not ratio_cols:
                print("  -> 错误: 未能计算任何加速比。")
                return

            df_ratios['min_speedup'] = df_ratios[ratio_cols].min(axis=1)
            df_ratios.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_ratios.dropna(subset=['min_speedup'], inplace=True)
            
            df_qualified = df_ratios[df_ratios['min_speedup'] > 1.0].copy()
            print(f"     -> 共有 {len(df_qualified)} / {len(df_ratios)} 个查询满足 '最小加速比 > 1.0'。")

            if df_qualified.empty:
                print("     -> 警告: 没有查询满足条件，无法继续筛选。")
                return
                
        except KeyError as e:
            print(f"  -> 错误: 算法名称映射失败: {e}。请检查 ALGO_MAP_SELECT。")
            return
        except Exception as e:
            print(f"  -> 错误: 计算加速比时出错: {e}")
            return
    
    else:
        print("\n  -> 'enable_speedup_filter' 为 False。跳过步骤 A (加速比筛选)。")
        print(f"     -> 将使用所有 {len(df_full)} 个查询进行比例抽样。")
        df_qualified = df_full.copy()

    # --- 步骤 B: 按 QueryID 分类 ---
    print("\n  -> 步骤 B: 按 QueryID 范围将合格查询分为 A, B, C 三类...")
    

    df_A = df_qualified[df_qualified['QueryID'] < 100].copy()
    df_B = df_qualified[(df_qualified['QueryID'] >= 101) & (df_qualified['QueryID'] < 200)].copy()
    df_C = df_qualified[df_qualified['QueryID'] >= 202].copy()
    
    # ========================================================
    # --- 检查并应用 A 类 Recall 过滤器 ---
    # ========================================================
    a_filter_config = params.get("a_recall_filter", {})
    
    if (a_filter_config.get("enabled", False) and 
        "algorithm_name" in a_filter_config and 
        "min_recall" in a_filter_config):
        
        alg_key = a_filter_config["algorithm_name"]
        min_recall = a_filter_config["min_recall"]
        
        # 检查算法是否在映射表 ALGO_MAP_SELECT 中
        if alg_key not in ALGO_MAP_SELECT:
            print(f"\n  -> 警告: 'a_recall_filter' 中的算法 '{alg_key}' 不在 ALGO_MAP_SELECT 中，跳过 A 类过滤器。")
        else:
            # 动态构建列名，例如: 'R_AG' (R 来自 pipeline, AG 来自 select)
            alg_short_name = ALGO_MAP_SELECT[alg_key]
            recall_col = f"R_{alg_short_name}" 
            
            if recall_col not in df_A.columns:
                print(f"\n  -> 警告: 找不到列 '{recall_col}'，无法应用 A 类 Recall 过滤器。")
            else:
                print(f"\n  -> 正在对 A 类查询应用过滤器: {recall_col} >= {min_recall}")
                original_a_count = len(df_A)
                df_A = df_A[df_A[recall_col] >= min_recall]
                print(f"     -> A 类合格查询数从 {original_a_count} 减少到 {len(df_A)}")

    
    # ========================================================
    # --- 检查并应用 C 类 p_pass (覆盖率) [min, max] 过滤 ---
    # ========================================================
    dataset_name = params.get("current_dataset_name", None)
    c_ppass_filters = params.get("c_ppass_filters", {}) # 读取 config.json
    
    if dataset_name in c_ppass_filters:
        ppass_range = c_ppass_filters[dataset_name]
        
        # 1. 验证格式是否为 [min, max] 列表
        if (isinstance(ppass_range, list) and 
            len(ppass_range) == 2 and 
            (isinstance(ppass_range[0], (int, float, type(None))) or 
             isinstance(ppass_range[1], (int, float, type(None))))):
            
            min_ppass, max_ppass = ppass_range[0], ppass_range[1]
            
            print(f"\n  -> 正在对 C 类查询应用 'p_pass' 范围过滤器 (来自 config.json):")
            print(f"     -> 数据集 '{dataset_name}' 要求 p_pass 范围: [{min_ppass or 'N/A'}, {max_ppass or 'N/A'}]")
            
            original_c_count_ppass = len(df_C)
            
            if 'p_pass' not in df_C.columns:
                print("     -> 警告: 找不到 'p_pass' 列，无法应用 C 类 p_pass 过滤器。")
            else:
                # 2. 按需应用过滤器
                if min_ppass is not None:
                    df_C = df_C[df_C['p_pass'] >= min_ppass]
                
                if max_ppass is not None:
                    df_C = df_C[df_C['p_pass'] <= max_ppass]
                
                print(f"     -> C 类合格查询数 (p_pass 过滤后) 从 {original_c_count_ppass} 减少到 {len(df_C)}")

        else:
            print(f"     -> 警告: '{dataset_name}' 的 'c_ppass_filters' 格式不正确。应为 [min, max] 列表。跳过此过滤器。")

    
    # ========================================================
    # --- (新增) 检查并应用 C 类长度过滤器 ---
    # ========================================================
    c_length_range = params.get("c_length_range", None)
    
    # 检查 c_length_range 是否是一个有效的 [min, max] 列表
    if (isinstance(c_length_range, list) and 
        len(c_length_range) == 2 and 
        all(isinstance(x, (int, float)) for x in c_length_range)):
        
        min_len, max_len = c_length_range[0], c_length_range[1]
        print(f"\n  -> 正在对 C 类查询应用长度过滤器: QuerySize 范围 [{min_len}, {max_len}]")
        
        original_c_count = len(df_C)
        
        # 确保 'QuerySize' 列存在 (它在合并时就已加入)
        if 'QuerySize' in df_C.columns:
            df_C = df_C[df_C['QuerySize'].between(min_len, max_len)]
            print(f"     -> C 类合格查询数从 {original_c_count} 减少到 {len(df_C)}")
        else:
            print("     -> 警告: 找不到 'QuerySize' 列，无法应用 C 类长度过滤器。")
    
    # ======================================================
    # --- (新增结束) ---
    # ======================================================

    print(f"\n     -> A类 (QueryID 0-999): {len(df_A)} 个合格查询")
    print(f"     -> B类 (QueryID 1000-1999): {len(df_B)} 个合格查询")
    print(f"     -> C类 (QueryID 2000+): {len(df_C)} 个合格查询 (已按长度过滤)")
    
    if df_qualified.empty:
        print("     -> 警告: 合格查询总数为 0，无法进行抽样。")
        return

    # --- 步骤 C: 按比例随机抽样 ---
    TOTAL_SELECT = 1000
    
    ratios_list = params.get('selection_ratios', [1, 1, 2])
    
    if not isinstance(ratios_list, list) or len(ratios_list) != 3 or not all(isinstance(x, (int, float)) for x in ratios_list):
        print(f"  -> 警告: 'selection_ratios' 格式不正确。将使用默认比例 [1, 1, 2]。")
        ratios_list = [1, 1, 2]
    
    total_ratio_sum = sum(ratios_list)
    if total_ratio_sum <= 0:
        print(f"  -> 错误: 'selection_ratios' 总和为0。将使用默认比例 [1, 1, 2]。")
        ratios_list = [1, 1, 2]
        total_ratio_sum = sum(ratios_list)

    ratio = {'A': ratios_list[0], 'B': ratios_list[1], 'C': ratios_list[2]}
    print(f"\n  -> 步骤 C: 按 {ratio['A']}:{ratio['B']}:{ratio['C']} 比例从 A, B, C 中随机抽样 (目标 {TOTAL_SELECT} 个)...")

    num_to_select = {
        'A': int(TOTAL_SELECT * (ratio['A'] / total_ratio_sum)),
        'B': int(TOTAL_SELECT * (ratio['B'] / total_ratio_sum)),
        'C': int(TOTAL_SELECT * (ratio['C'] / total_ratio_sum))
    }
    
    current_sum = sum(num_to_select.values())
    if current_sum < TOTAL_SELECT:
        max_ratio_key = max(ratio, key=ratio.get)
        num_to_select[max_ratio_key] += (TOTAL_SELECT - current_sum)

    # 确保抽样数不超过 C 类的*新*计数值
    num_A = min(num_to_select['A'], len(df_A))
    num_B = min(num_to_select['B'], len(df_B))
    num_C = min(num_to_select['C'], len(df_C)) # <--- 这里会自动使用过滤后的 df_C

    print(f"     -> 正在从 A 中随机抽取 {num_A} / {len(df_A)} 个...")
    df_selected_A = df_A.sample(n=num_A, random_state=42)
    
    print(f"     -> 正在从 B 中随机抽取 {num_B} / {len(df_B)} 个...")
    df_selected_B = df_B.sample(n=num_B, random_state=42)
    
    print(f"     -> 正在从 C 中随机抽取 {num_C} / {len(df_C)} 个...")
    df_selected_C = df_C.sample(n=num_C, random_state=42)
    
    df_final_selection = pd.concat([df_selected_A, df_selected_B, df_selected_C], ignore_index=True)
    
    # --- 步骤 D: 填满至 1000 ---
    num_selected_so_far = len(df_final_selection)
    remaining_to_select = TOTAL_SELECT - num_selected_so_far
    
    print(f"  -> 步骤 D: 检查总数... (已选 {num_selected_so_far} / {TOTAL_SELECT})")

    if remaining_to_select > 0:
        if df_final_selection.empty:
            print("  -> 错误: 没有任何查询被选中，无法执行重复填充。")
            return

        print(f"     -> 唯一查询不足 {TOTAL_SELECT}。正在重复使用已选查询以填满剩余的 {remaining_to_select} 个名额...")
        
        df_to_duplicate_from = df_final_selection
        num_unique_queries_to_copy = len(df_to_duplicate_from)
        num_repeats = int(np.ceil(remaining_to_select / num_unique_queries_to_copy))
        df_duplicates = pd.concat([df_to_duplicate_from] * num_repeats, ignore_index=True)
        df_fillers_duplicate = df_duplicates.head(remaining_to_select)
        df_final_selection = pd.concat([df_final_selection, df_fillers_duplicate], ignore_index=True)
        print(f"     -> 成功重复填充。总查询数: {len(df_final_selection)}。")

    # --- 步骤 E: 保存结果 ---
    print("  -> 步骤 E: 保存最终的 1000 个查询...")
    
    cols_to_keep = list(df_merged.columns) + ['p_pass', 'min_speedup']
    cols_to_save = [col for col in cols_to_keep if col in df_final_selection.columns]
    
    df_final_selection[cols_to_save].to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n  -> ✅ 新策略筛选完毕: 共 {len(df_final_selection)} 个查询已保存到: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    print("这是一个筛选模块，请通过 new_run_pipeline.py 运行。")