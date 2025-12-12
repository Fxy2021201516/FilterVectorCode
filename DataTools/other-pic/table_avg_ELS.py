import pandas as pd
import numpy as np
import os

# --- 1. 定义要加载的数据集 ---
datasets_to_load = {
    "Genome":{
        "baseline": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Genome/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls10-Le40000-Lp1000_efsS20-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "method3": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Genome/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls10-Le40000-Lp1000_efsS20-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv"
    },
    "Reviews": {
        "baseline": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Reviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "method3": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Reviews/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "Amazon": {
        "baseline": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Amazon/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "method3": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Amazon/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "VariousImg": {
        "baseline": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/VariousImg/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "method3": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/VariousImg/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "Music": {
        "baseline": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Music/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "method3": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Music/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "BookReviews": {
        "baseline": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/BookReviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_imp_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls10-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "method3": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/BookReviews/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_imp_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls10-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "Tiktok": {
        "baseline": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Tiktok/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
        "method3": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Tiktok/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
    },
    "Laion": {
        "baseline": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/UNG-nTfalse/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv",
        "method3": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/method3/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv",
    }
}

# --- 2. 辅助函数：按Recall对齐筛选并聚合 ---
def process_dataset_per_query(csv_path, metric_col='MinSupersetT_ms', target_recall=0.9):
    """
    逻辑更新（与其他脚本对齐）：
    1. 剔除 MaxRecall=0 的 Query。
    2. 对每个 Query：
       - 若达标 (Recall >= 0.9): 选 Min Recall >= 0.9 的行。
       - 若未达标 (Recall < 0.9): 选 Max Recall 的行。
    3. 计算该 Query 选中行的 metric 平均值。
    返回: pd.Series (Index=QueryID, Value=MetricAvg)
    """
    cols_to_use = ['QueryID', 'Recall', metric_col]
    try:
        df = pd.read_csv(csv_path, usecols=cols_to_use)
    except ValueError:
        df = pd.read_csv(csv_path) # Fallback

    # 数据清洗
    if df.empty or metric_col not in df.columns:
        return pd.Series(dtype=float)
    
    df.dropna(subset=['Recall', metric_col], inplace=True)
    
    # 1. 过滤无效 Query
    max_recalls = df.groupby('QueryID')['Recall'].max()
    valid_ids = max_recalls[max_recalls > 0].index
    df = df[df['QueryID'].isin(valid_ids)]
    
    selected_values = {}
    
    # 2. 遍历 Query 进行筛选
    for qid, group in df.groupby('QueryID'):
        qualifying = group[group['Recall'] >= target_recall]
        
        if not qualifying.empty:
            # 达标：取刚过线的配置
            target_r = qualifying['Recall'].min()
            rows = qualifying[qualifying['Recall'] == target_r]
        else:
            # 未达标：取尽力而为的配置
            target_r = group['Recall'].max()
            rows = group[group['Recall'] == target_r]
            
        if not rows.empty:
            selected_values[qid] = rows[metric_col].mean()
            
    return pd.Series(selected_values)

# --- 3. 主处理循环 ---
results_list = []
output_filename = "table_avg_ELS_table_aligned.csv"

print(f"开始计算 '平均值的比值' (Recall对齐版)...")
print(f"筛选逻辑: 达标取MinRecall / 未达标取MaxRecall")

for dataset_name, paths in datasets_to_load.items():
    print(f"\n--- 正在处理: {dataset_name} ---")
    
    try:
        path_baseline = paths['baseline'].replace('\u00a0', '').strip()
        path_method3 = paths['method3'].replace('\u00a0', '').strip()
        
        # --- 1. 分别获取处理后的 Series ---
        # 索引都是 QueryID，值为该 Query 在对齐 Recall 下的开销
        s_base = process_dataset_per_query(path_baseline)
        s_m3 = process_dataset_per_query(path_method3)
        
        if s_base.empty or s_m3.empty:
            print(f"    警告: {dataset_name} 某一方数据为空，跳过。")
            continue
            
        # --- 2. 合并 (Inner Join) 确保 Query 对齐 ---
        # 只有两个算法都算出结果的 Query 才参与比较
        df_merged = pd.concat([s_base, s_m3], axis=1, join='inner')
        df_merged.columns = ['base_val', 'm3_val']
        
        count = len(df_merged)
        print(f"    共找到 {count} 个共同有效的 Query。")
        
        if count == 0:
            continue

        # --- 3. 计算总平均值 ---
        avg_base_overall = df_merged['base_val'].mean()
        avg_m3_overall = df_merged['m3_val'].mean()
        
        # --- 4. 计算加速比 ---
        if avg_m3_overall == 0:
            ratio = np.inf
        else:
            ratio = avg_base_overall / avg_m3_overall

        print(f"    UNG-nTfalse (bottom-up): {avg_base_overall:.2f} ms")
        print(f"    method3 (IntelELS):      {avg_m3_overall:.2f} ms")
        print(f"    Speedup Ratio:           {ratio:.2f}") 
        
        results_list.append({
            "Dataset": dataset_name,
            "Avg_bottom-up (ms)": avg_base_overall,
            "Avg_IntelELS (ms)": avg_m3_overall,
            "Ratio_of_Averages": ratio
        })

    except Exception as e:
        print(f"    !!! 错误: {dataset_name}: {e}")

# --- 4. 表格呈现并保存 ---
print("\n--- 最终计算结果表格 (已转置) ---")

if results_list:
    df_results = pd.DataFrame(results_list)
    df_indexed = df_results.set_index("Dataset")
    df_transposed = df_indexed.T
    print(df_transposed.to_string(float_format='%.2f'))
    df_transposed.to_csv(output_filename, float_format='%.2f')
    print(f"\n✅ 结果已保存到 {output_filename}")
else:
    print("没有结果。")