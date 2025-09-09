import pandas as pd
import numpy as np
import os
import re
import glob

# ==============================================================================
# 配置区
BASE_RESULTS_DIR = '/data/fxy/FilterVector/FilterVectorResults'
TARGET_RECALL = 0.95  # 目标召回率阈值
dataset_name = "app_reviews"
OUTPUT_DIR = "/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_A/" +dataset_name
# ==============================================================================

def find_optimal_performance_per_query(df, recall_col, time_col, dist_calcs_col):
    """
    核心处理函数：对每个QueryID，根据筛选逻辑找到最优记录。
    """
    if df.empty or not all(c in df.columns for c in [recall_col, time_col, dist_calcs_col, 'QueryID']):
        return pd.DataFrame()
        
    # 确保关键列为数值类型，无效值转为NaN
    cols_to_convert = [recall_col, time_col, dist_calcs_col, 'QueryID', 'EntryGroupT_ms', 'Lsearch']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=[recall_col, time_col, 'QueryID'], inplace=True)
    df['QueryID'] = df['QueryID'].astype(int)

    optimal_rows = []
    
    for query_id, group in df.groupby('QueryID'):
        high_recall_group = group[group[recall_col] >= TARGET_RECALL]
        
        if not high_recall_group.empty:
            optimal_row = high_recall_group.sort_values(by=time_col, ascending=True).iloc[0]
        else:
            max_recall = group[recall_col].max()
            if pd.isna(max_recall): continue
            top_recall_group = group[group[recall_col] == max_recall]
            if top_recall_group.empty: continue
            optimal_row = top_recall_group.sort_values(by=time_col, ascending=True).iloc[0]
            
        optimal_rows.append(optimal_row)
        
    if not optimal_rows:
        return pd.DataFrame()
        
    return pd.concat(optimal_rows, axis=1).T


def process_matched_pair(acorn_csv_path, ung_csv_path):
    """
    处理一对匹配好的ACORN和UNG CSV文件，并返回合并后的DataFrame。
    """
    try:
        # --- 1. 处理 UNG 文件 ---
        ung_df = pd.read_csv(ung_csv_path)
        ung_df.columns = ung_df.columns.str.strip()
        
        for col in ['EntryGroupT_ms', 'Lsearch']:
            if col not in ung_df.columns:
                ung_df[col] = np.nan

        ung_optimal = find_optimal_performance_per_query(
            ung_df, 'Recall', 'SearchT_ms', 'DistCalcs'
        )
        
        ung_cols_to_keep = ['QueryID', 'Recall', 'SearchT_ms', 'DistCalcs', 'EntryGroupT_ms', 'Lsearch']
        ung_rename_map = {
            'Recall': 'Recall_U', 
            'SearchT_ms': 'Time_U(ms)', 
            'DistCalcs': 'DistCalcs_U',
            'EntryGroupT_ms': 'EntryGroupT_U(ms)',
            'Lsearch': 'Lsearch_U'
        }
        ung_final = ung_optimal[ung_cols_to_keep].rename(columns=ung_rename_map) if not ung_optimal.empty else pd.DataFrame()

        # --- 2. 处理 ACORN 文件 ---
        acorn_df = pd.read_csv(acorn_csv_path)
        acorn_df.columns = acorn_df.columns.str.strip()
        
        acorn_base_optimal = find_optimal_performance_per_query(
            acorn_df.copy(), 'acorn_Recall', 'acorn_Time', 'acorn_n3'
        )
        if not acorn_base_optimal.empty:
            acorn_base_final_temp = acorn_base_optimal[['QueryID', 'acorn_Recall', 'acorn_Time', 'acorn_n3']].copy()
            acorn_base_final_temp['acorn_Time'] = acorn_base_final_temp['acorn_Time'] * 1000
            acorn_base_final = acorn_base_final_temp.rename(columns={
                'acorn_Recall': 'Recall_A', 'acorn_Time': 'Time_A(ms)', 'acorn_n3': 'DistCalcs_A'
            })
        else:
            acorn_base_final = pd.DataFrame()

        acorn_1_optimal = find_optimal_performance_per_query(
            acorn_df.copy(), 'ACORN_1_Recall', 'ACORN_1_Time', 'ACORN_1_n3'
        )
        if not acorn_1_optimal.empty:
            acorn_1_final_temp = acorn_1_optimal[['QueryID', 'ACORN_1_Recall', 'ACORN_1_Time', 'ACORN_1_n3']].copy()
            acorn_1_final_temp['ACORN_1_Time'] = acorn_1_final_temp['ACORN_1_Time'] * 1000
            acorn_1_final = acorn_1_final_temp.rename(columns={
                'ACORN_1_Recall': 'Recall_A1', 'ACORN_1_Time': 'Time_A1(ms)', 'ACORN_1_n3': 'DistCalcs_A1'
            })
        else:
            acorn_1_final = pd.DataFrame()

        # --- 3. 合并所有结果 ---
        final_dfs = [df for df in [ung_final, acorn_base_final, acorn_1_final] if not df.empty]
        if not final_dfs: return None

        merged_df = final_dfs[0]
        for df_to_merge in final_dfs[1:]:
            merged_df = pd.merge(merged_df, df_to_merge, on='QueryID', how='outer')
        
        # --- 4. 计算新增的列 ---
        
        # 4.1 【新增】计算 Time_U(ms) - EntryGroupT_U(ms)
        if all(col in merged_df.columns for col in ['Time_U(ms)', 'EntryGroupT_U(ms)']):
            merged_df['Time_U_Adjusted(ms)'] = merged_df['Time_U(ms)'] - merged_df['EntryGroupT_U(ms)']
        else:
            merged_df['Time_U_Adjusted(ms)'] = np.nan

        # 4.2 计算比率 (Time_U-Entry_U)/Time_A
        if all(col in merged_df.columns for col in ['Time_U(ms)', 'EntryGroupT_U(ms)', 'Time_A(ms)']):
            denominator = merged_df['Time_A(ms)'].replace(0, np.nan)
            merged_df['(Time_U-Entry_U)/Time_A'] = (merged_df['Time_U(ms)'] - merged_df['EntryGroupT_U(ms)']) / denominator
        else:
            merged_df['(Time_U-Entry_U)/Time_A'] = np.nan

        if 'QueryID' in merged_df.columns:
            merged_df.dropna(subset=['QueryID'], inplace=True)
            merged_df['QueryID'] = merged_df['QueryID'].astype(int)
            merged_df.sort_values(by='QueryID', inplace=True)
            
        # 重新排列列的顺序，将新增的列放在一起，方便查看
        cols_order = [
            'QueryID', 'Recall_U', 'Time_U(ms)', 'EntryGroupT_U(ms)', 'Lsearch_U', 'DistCalcs_U',
            'Recall_A', 'Time_A(ms)', 'DistCalcs_A', 
            'Recall_A1', 'Time_A1(ms)', 'DistCalcs_A1',
            'Time_U_Adjusted(ms)', '(Time_U-Entry_U)/Time_A'
        ]
        # 过滤掉不存在的列，以防万一
        existing_cols_order = [col for col in cols_order if col in merged_df.columns]
        merged_df = merged_df[existing_cols_order]

        return merged_df

    except Exception as e:
        print(f"  [ERROR] 处理文件对时出错: {e}")
        return None


def main():
    """
    主函数，驱动整个匹配、处理和合并流程。
    """
    if not dataset_name:
        print("[ERROR] 未在配置区指定数据集名称，程序退出。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[INFO] 结果CSV文件将保存在: {os.path.abspath(OUTPUT_DIR)}")
    
    print(f"\n{'='*25}\n[INFO] 开始处理数据集: {dataset_name}\n[INFO] 匹配规则: dataset_name + query + th/threads\n{'='*25}")

    ung_base_dir = os.path.join(BASE_RESULTS_DIR, 'UNG', dataset_name)
    acorn_base_dir = os.path.join(BASE_RESULTS_DIR, 'ACORN', dataset_name)

    if not os.path.isdir(ung_base_dir):
        print(f"[ERROR] 找不到UNG数据集目录: {ung_base_dir}")
        return

    ung_pattern = re.compile(r".*_query(\d+)_(nT(?:true|false))_th(\d+)_.*")
    
    ung_exp_dirs = glob.glob(os.path.join(ung_base_dir, '*'))
    print(f"[INFO] 在UNG下找到 {len(ung_exp_dirs)} 个实验目录，开始以其为基准进行匹配...")

    success_count = 0
    
    for ung_dir in ung_exp_dirs:
        ung_dir_name = os.path.basename(ung_dir)
        match = ung_pattern.match(ung_dir_name)

        if not match:
            print(f"\n [SKIP] 无法从UNG目录名解析参数: {ung_dir_name}")
            continue

        query_val, nT_val,th_val = match.groups()
        print(f"\n[UNG] 找到基准实验: {ung_dir_name} (query={query_val}, th={th_val})")
        
        acorn_search_pattern = os.path.join(acorn_base_dir, f"{dataset_name}_query{query_val}_*_threads{th_val}_*")
        
        print(f"  [SEARCH] 正在查找匹配的ACORN目录...")
        matched_acorn_dirs = glob.glob(acorn_search_pattern)

        if not matched_acorn_dirs:
            print("  [FAIL] 未找到匹配的ACORN实验目录。")
            continue
        
        acorn_dir = matched_acorn_dirs[0]
        print(f"  [SUCCESS] 成功匹配ACORN目录: {os.path.basename(acorn_dir)}")

        all_acorn_csvs = glob.glob(os.path.join(acorn_dir, 'results', '*.csv'))
        acorn_detail_csvs = [f for f in all_acorn_csvs if not f.endswith('_avg.csv')]
        
        ung_csv_path = next(iter(glob.glob(os.path.join(ung_dir, 'results', 'query_details_repeat*.csv'))), None)
        acorn_csv_path = acorn_detail_csvs[0] if acorn_detail_csvs else None
        
        if not acorn_csv_path or not ung_csv_path:
            if not acorn_csv_path:
                print(f"  [FAIL] 未能找到ACORN的详细结果文件(已忽略_avg.csv)。")
            if not ung_csv_path:
                print(f"  [FAIL] 未能找到UNG的结果CSV文件。")
            continue

        print(f"  [INFO] 正在处理UNG文件: {os.path.basename(ung_csv_path)}")
        print(f"  [INFO] 正在处理ACORN文件: {os.path.basename(acorn_csv_path)}")
        
        final_merged_df = process_matched_pair(acorn_csv_path, ung_csv_path)

        if final_merged_df is not None and not final_merged_df.empty:
            output_filename = f"U_A_{dataset_name}_q{query_val}_th{th_val}_{nT_val}.csv"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            final_merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  [SAVE] 结果已保存到: {output_path}")

            # 为当前文件计算并打印平均比率
            if '(Time_U-Entry_U)/Time_A' in final_merged_df.columns:
                # dropna()会移除所有无法计算比率的行
                valid_ratios = final_merged_df['(Time_U-Entry_U)/Time_A'].dropna()
                if not valid_ratios.empty:
                    avg_ratio = valid_ratios.mean()
                    print(f"  [AVERAGE] 该文件内 (Time_U-Entry_U)/Time_A 的平均值为: {avg_ratio:.4f}")
                else:
                    print("  [AVERAGE] 该文件内无可用的比率值来计算平均值。")

            success_count += 1
        else:
            print("  [WARN] 处理结果为空，未生成文件。")
    
    print(f"\n[FINISH] 数据集 '{dataset_name}' 处理完成！")
    print(f"共成功处理并生成了 {success_count} 个CSV结果文件在目录 '{os.path.abspath(OUTPUT_DIR)}' 中。")


if __name__ == '__main__':
    main()