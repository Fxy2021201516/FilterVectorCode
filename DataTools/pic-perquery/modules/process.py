import pandas as pd
from tqdm import tqdm
import os 

tqdm.pandas(desc="Summarizing Queries")

def find_optimal_row(df_group, strategy="default"):
    """
    在一个查询的分组数据中(包含所有Lsearch的行)，根据策略找到最佳性能的那一行。
    
    策略:
    1. "default" (原方法): 
       - 优先选择第一个达到 Recall > 0.9 的行中，Lsearch 值最小的那一行。
       - 如果没有达到要求的 Recall，则选择 Recall 最高的那一行中，Time_ms (总时间) 最小的那一行。
    2. "max_recall_first" (规则2):
       - 直接选择 Recall 最高的那一行中，Time_ms (总时间) 最小的那一行。
    """
    
    # --- 规则 2: 直接找最高recall对应的最短时间 ---
    if strategy == "max_recall_first":
        max_recall = df_group['Recall'].max()
        if pd.isna(max_recall):
            return None 
        
        # 找到所有达到最高 Recall 的行
        max_recall_df = df_group[df_group['Recall'] == max_recall]
        
        # 在这些行中，返回 Time_ms 最小的那一行
        return max_recall_df.loc[max_recall_df['Time_ms'].idxmin()]

    # --- 规则 1: "default" (原方法) ---
    else:
        # 按照 Lsearch 排序，以便找到 "第一个" 满足条件的点
        df_sorted = df_group.sort_values('Lsearch')
        
        # 1. 寻找第一个 Recall > 0.9 的行
        high_recall_df = df_sorted[df_sorted['Recall'] > 0.9]
        if not high_recall_df.empty:
            # .iloc[0] 会返回排序后的第一行，即 Lsearch 最小的那一行
            return high_recall_df.iloc[0]

        # 2. 如果没有行的 Recall > 0.9，则执行备选方案
        max_recall = df_group['Recall'].max()
        if pd.isna(max_recall):
            return None 
        
        # 找到所有达到最高 Recall 的行
        max_recall_df = df_group[df_group['Recall'] == max_recall]
        
        # 在这些行中，返回 Time_ms 最小的那一行
        return max_recall_df.loc[max_recall_df['Time_ms'].idxmin()]


def run_processing(paths, algorithm_name, analysis_params={}):
    """
    输入: 单一算法的详细数据文件路径 (来自C++) 和 analysis_params
    输出: 单一算法的浓缩摘要文件 (每个查询一行)
    MODIFIED: 增加了缓存功能
    MODIFIED: 增加了 analysis_params 以控制筛选策略
    """
    print(f"\n[Processing] 正在为算法 '{algorithm_name}' 生成性能摘要...")
    
    # ==================== 缓存逻辑 START ====================
    summary_output_path = paths.get('summary_output_path')
    if not summary_output_path:
        print("  -> 错误: 未能在 'paths' 配置中找到 'summary_output_path'。")
        return None

    if os.path.exists(summary_output_path):
        print(f"  -> 缓存命中！直接从文件加载摘要: {os.path.basename(summary_output_path)}")
        try:
            df_final = pd.read_csv(summary_output_path)
            print(f"  -> ✅ 已为算法 '{algorithm_name}' 从缓存加载摘要。")
            return df_final
        except Exception as e:
            print(f"  -> 警告: 读取缓存文件失败: {e}。将重新进行处理。")
    # ==================== 缓存逻辑 END ======================

    input_csv_path = paths['cpp_details_file']
    try:
        df_detailed = pd.read_csv(input_csv_path)
        print(f"  -> 已加载详细数据: {input_csv_path}")
    except FileNotFoundError:
        print(f"  -> 错误: 未找到输入文件 {input_csv_path}，跳过处理。")
        return

    print("  -> 缓存未命中，开始进行计算密集型处理...")
    
    # --- 修改点: 从 analysis_params 获取策略 ---
    strategy = analysis_params.get("optimal_row_strategy", "default")
    print(f"  -> 正在使用筛选策略: '{strategy}'")
    # --- 结束修改 ---
    
    # --- 修改点: 将 strategy 传递给 find_optimal_row ---
    df_summary = df_detailed.groupby('QueryID').progress_apply(
        lambda df_group: find_optimal_row(df_group, strategy=strategy)
    )
    # --- 结束修改 ---

    df_summary.dropna(how='all', inplace=True)
    df_summary.reset_index(drop=True, inplace=True)

    if df_summary.empty:
        print(f"  -> 警告: 未能为 '{algorithm_name}' 生成任何摘要数据。")
        return

    core_columns = {
        'QueryID': 'QueryID',
        'Time_ms': f'Time_ms_{algorithm_name}',
        'search_time_ms': f'SearchTime_ms_{algorithm_name}',
        'Recall': f'Recall_{algorithm_name}',
        'Lsearch': f'Optimal_Lsearch_{algorithm_name}',
        'QuerySize': 'QuerySize'
    }
    
    cols_to_rename = {k: v for k, v in core_columns.items() if k in df_summary.columns}
    df_final = df_summary[list(cols_to_rename.keys())].copy()
    df_final.rename(columns=cols_to_rename, inplace=True)

    # ==================== 保存结果逻辑 START ====================
    try:
        df_final.to_csv(summary_output_path, index=False)
        print(f"  -> ✅ 已为算法 '{algorithm_name}' 生成摘要并保存到缓存: {os.path.basename(summary_output_path)}")
    except Exception as e:
        print(f"  -> 错误: 保存摘要文件失败: {e}")
    # ==================== 保存结果逻辑 END ======================
    
    return df_final