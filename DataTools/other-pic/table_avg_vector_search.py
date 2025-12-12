import pandas as pd
import numpy as np
import os

def load_and_calculate_metrics(csv_path, metric_column='search_time_ms', target_recall=0.9):
    """
    计算逻辑：
    1. 预处理：剔除 Max Recall = 0 的 Query。
    2. 选行：
       - 达标 (Recall >= 0.9): 选 Recall 刚刚达标 (Min Recall >= 0.9) 的所有行。
       - 未达标 (Recall < 0.9): 选 Recall 最大 (Max Recall) 的所有行。
    3. 返回：
       - avg_time: 平均时间
       - total_rows: 此次计算共采纳了多少行数据 (sum of rows selected)
       - avg_recall: 选中行的平均 Recall (mean of selected recalls)
    """
    try:
        cols_to_use = ['QueryID', 'Recall', metric_column]
        try:
            df = pd.read_csv(csv_path, usecols=cols_to_use)
        except ValueError:
             df = pd.read_csv(csv_path) # Fallback
             if not all(col in df.columns for col in cols_to_use):
                 return np.nan, 0, np.nan
        
        if df.empty: return np.nan, 0, np.nan
        
        df.dropna(subset=['Recall', metric_column], inplace=True)

        # 1. 剔除无效 Query
        max_recalls = df.groupby('QueryID')['Recall'].max()
        valid_query_ids = max_recalls[max_recalls > 0].index
        df = df[df['QueryID'].isin(valid_query_ids)]
        
        if df.empty: return np.nan, 0, np.nan
            
        times_collected = []
        counts_collected = []
        recalls_collected = []
        
        for query_id, group in df.groupby('QueryID'):
            if group.empty: continue
            
            # --- 核心选行逻辑 ---
            qualifying = group[group['Recall'] >= target_recall]
            
            if not qualifying.empty:
                # 达标：取刚过线的 Recall
                target_r = qualifying['Recall'].min()
                target_rows = qualifying[qualifying['Recall'] == target_r]
            else:
                # 未达标：取最大 Recall
                target_r = group['Recall'].max()
                target_rows = group[group['Recall'] == target_r]
            
            if not target_rows.empty:
                # 1. 时间
                times_collected.append(target_rows[metric_column].mean())
                # 2. 行数 (该 Query 贡献了多少行)
                counts_collected.append(len(target_rows))
                # 3. Recall (该 Query 实际运行在哪个 Recall)
                recalls_collected.append(target_r)

        if not times_collected:
            return np.nan, 0, np.nan
            
        final_time = np.mean(times_collected)
        total_rows = np.sum(counts_collected) # 总共用了多少行
        final_recall = np.mean(recalls_collected) # 平均运行 Recall
        
        return final_time, total_rows, final_recall

    except Exception as e:
        print(f"    - ❌ 错误: {os.path.basename(csv_path)}: {e}")
        return np.nan, 0, np.nan

# ---------------------------------------------------------------------------
# --- 配置区域 (保持不变) ---
# ---------------------------------------------------------------------------

datasets_to_load = {
    "Genome":{
        "UNG": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Genome/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls10-Le40000-Lp1000_efsS20-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Genome/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls10-Le40000-Lp1000_efsS100-efss1000-efsf1000-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-1": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Genome/Results/ACORN-1/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls10-Le40000-Lp1000_efsS100-efss1000-efsf1000-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma-improved": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Genome/Results/ACORN-gamma-improved/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls10-Le40000-Lp1000_efsS100-efss2000-efsf2000-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "SmartRoute": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Genome/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls10-Le40000-Lp1000_efsS20-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv"
    },
    "Reviews": {
        "UNG": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Reviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Reviews/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS100-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-1": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Reviews/Results/ACORN-1/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS100-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma-improved": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Reviews/Results/ACORN-gamma-improved/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS100-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "SmartRoute": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Reviews/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "Amazon": {
        "UNG": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Amazon/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Amazon/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-1": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Amazon/Results/ACORN-1/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma-improved": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Amazon/Results/ACORN-gamma-improved/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "SmartRoute": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Amazon/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "VariousImg": {
        "UNG": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/VariousImg/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/VariousImg/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-1": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/VariousImg/Results/ACORN-1/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma-improved": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/VariousImg/Results/ACORN-gamma-improved/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS10-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "SmartRoute": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/VariousImg/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "Music": {
        "UNG": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Music/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Music/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss1000-efsf1000-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-1": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Music/Results/ACORN-1/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss1000-efsf1000-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma-improved": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Music/Results/ACORN-gamma-improved/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS20000-efss1000-efsf1000-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "SmartRoute": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Music/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "BookReviews": {
        "UNG": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/BookReviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_imp_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls10-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/BookReviews/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_imp_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls10-Le20000-Lp500_efsS200-efss400-efsf400-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-1": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/BookReviews/Results/ACORN-1/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_imp_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls10-Le20000-Lp500_efsS200-efss400-efsf400-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma-improved": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/BookReviews/Results/ACORN-gamma-improved/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_imp_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls10-Le20000-Lp500_efsS200-efss400-efsf400-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "SmartRoute": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/BookReviews/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_imp_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls10-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "Tiktok": {
        "UNG": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Tiktok/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Tiktok/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-1": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Tiktok/Results/ACORN-1/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma-improved": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Tiktok/Results/ACORN-gamma-improved/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "SmartRoute": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Tiktok/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "Laion": {
        "UNG": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/UNG-nTfalse/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/ACORN-gamma/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss100-efsf100-lt500000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-1": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/ACORN-1/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss100-efsf100-lt500000_K10_th100]/results/query_details_repeat1.csv",
        "ACORN-gamma-improved": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/ACORN-gamma-improved/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss100-efsf100-lt500000_K10_th100]/results/query_details_repeat1.csv",
        "SmartRoute": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/method3/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv",
    },
}

# ---------------------------------------------------------------------------
# --- 3. 主处理循环  ---
# ---------------------------------------------------------------------------

list_time = []
list_count = []
list_recall = []

output_filename = "table_avg_time_with_stats.csv"
algorithms = ["UNG", "ACORN-1", "ACORN-gamma", "ACORN-gamma-improved", "SmartRoute"]

print(f"开始计算 (输出包含行数和AvgRecall)...")

for dataset_name, paths_dict in datasets_to_load.items():
    print(f"\n--- 正在处理: {dataset_name} ---")
    
    row_time = {"Dataset": dataset_name}
    row_count = {"Dataset": dataset_name}
    row_recall = {"Dataset": dataset_name}
    
    for algo_name in algorithms:
        try:
            csv_path = paths_dict[algo_name].replace('\u00a0', '').strip()
            
            t, c, r = load_and_calculate_metrics(
                csv_path, 
                metric_column='Time_ms',
                target_recall=0.9
            )
            
            row_time[algo_name] = t
            row_count[algo_name] = c
            row_recall[algo_name] = r
            
            if not np.isnan(t):
                print(f"  - {algo_name}: {t:.2f} ms (Based on {c} rows, AvgRecall={r:.4f})")
            
        except KeyError:
            row_time[algo_name] = np.nan
            row_count[algo_name] = np.nan
            row_recall[algo_name] = np.nan
        except Exception as e:
            print(f"  - ❌ {algo_name} 错误: {e}")
            row_time[algo_name] = np.nan
            row_count[algo_name] = np.nan
            row_recall[algo_name] = np.nan
            
    list_time.append(row_time)
    list_count.append(row_count)
    list_recall.append(row_recall)

# ---------------------------------------------------------------------------
# --- 4. 表格呈现并保存到 CSV  ---
# ---------------------------------------------------------------------------

def prepare_df(data_list):
    if not data_list: return pd.DataFrame()
    df = pd.DataFrame(data_list)
    if "Dataset" in df.columns:
        df = df.set_index("Dataset")
    return df.T # 转置

print("\n--- 最终结果 ---")

df_time = prepare_df(list_time)
df_count = prepare_df(list_count)
df_recall = prepare_df(list_recall)

# 打印预览
print("Time Table:")
print(df_time.to_string(float_format='%.2f'))

# 保存到同一个 CSV，中间用空行隔开
with open(output_filename, 'w') as f:
    f.write("--- Average Search Time (ms) ---\n")
    df_time.to_csv(f, float_format='%.2f')
    
    f.write("\n\n--- Total Rows Used for Calculation ---\n")
    df_count.to_csv(f, float_format='%.0f')
    
    f.write("\n\n--- Average Operating Recall ---\n")
    df_recall.to_csv(f, float_format='%.4f')

print(f"\n✅ 详细统计结果已保存到 {output_filename}")