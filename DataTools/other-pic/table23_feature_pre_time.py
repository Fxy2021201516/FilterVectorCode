import pandas as pd
import os
from pathlib import Path
import numpy as np

def process_dataset(file_path):
    """
    处理单个CSV文件，按照“对齐Recall标准”的逻辑计算特征提取和预测时间。
    
    核心逻辑更新：
    1. 预处理：剔除 MaxRecall=0 的 Query。
    2. 选行策略：
       - 若能达标 (Recall >= 0.9): 取 (Recall >= 0.9) 中 Recall 最小 的那些行 -> 平均。
         (含义：达到 0.9 门槛所需的最低开销配置)
       - 若未达标 (Recall < 0.9): 取 Recall 最大 的那些行 -> 平均。
         (含义：尽力而为的最佳表现对应的开销)
    3. 聚合：计算上述筛选出的行在四个指标上的平均值，最后对所有 Query 取总平均。
    """
    try:
        # 1. 读取数据
        # 必须包含 Recall 列用于筛选
        required_cols = ['QueryID', 'Recall', 'idea1SelT_ms', 'idea2SelT_ms', 
                         'idea1_flag_ms', 'idea2_flag_ms', 'MinSupersetT_ms']
        
        # 尝试只读取需要的列以节省内存
        try:
            df = pd.read_csv(file_path, usecols=lambda c: c in required_cols + ['QueryID', 'Recall'])
            # 补全读取所有需要的列 (防止 usecols 漏掉没报错但后续缺列)
            df = pd.read_csv(file_path) 
        except ValueError:
             # 如果列名不匹配，可能是空文件或格式不对
             df = pd.read_csv(file_path)

        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"错误：文件 {os.path.basename(file_path)} 缺少必要列: {missing}")
            return None, None, None, None

        # 2. 计算每行的派生时间指标
        # 预测时间
        df['pred1_ms'] = df['idea1SelT_ms']
        df['pred2_ms'] = df['idea2SelT_ms']
        # 特征提取时间
        df['feat_ext1_ms'] = df['idea1_flag_ms'] - df['idea1SelT_ms']
        # Idea2 特征提取 = (flag - sel) + MinSuperset
        df['feat_ext2_ms'] = (df['idea2_flag_ms'] - df['idea2SelT_ms']) + df['MinSupersetT_ms']

        # 3. 全局过滤：剔除 Max Recall = 0 的 Query (完全失效的查询)
        if df.empty: return None, None, None, None
        
        max_recalls = df.groupby('QueryID')['Recall'].max()
        valid_query_ids = max_recalls[max_recalls > 0].index
        
        if len(valid_query_ids) < len(max_recalls):
            # print(f"  - 剔除 {len(max_recalls) - len(valid_query_ids)} 个 MaxRecall=0 的 Query")
            pass
            
        df = df[df['QueryID'].isin(valid_query_ids)]
        
        if df.empty:
            print(f"警告：过滤后文件 {os.path.basename(file_path)} 为空。")
            return None, None, None, None

        # 4. 按 QueryID 分组，应用新的选行逻辑
        selected_metrics = []
        target_recall = 0.9
        
        # 针对每个 Query 进行处理
        for qid, group in df.groupby('QueryID'):
            if group.empty: continue
            
            # --- 核心选行逻辑 ---
            qualifying_rows = group[group['Recall'] >= target_recall]
            
            if not qualifying_rows.empty:
                # Case A: 达标。选取“刚刚达标”的那些行。
                # 逻辑：在所有 >= 0.9 的行中，找到最小的 Recall 值。
                # 例如：有 0.91, 0.91, 0.95。最小是 0.91。取这两个 0.91 的平均。
                min_valid_recall = qualifying_rows['Recall'].min()
                target_rows = qualifying_rows[qualifying_rows['Recall'] == min_valid_recall]
            else:
                # Case B: 未达标。选取“效果最好”的那些行。
                max_recall = group['Recall'].max()
                target_rows = group[group['Recall'] == max_recall]
            
            # 计算该 Query 在选中行上的平均指标
            if not target_rows.empty:
                # 这里取平均 (mean)，因为可能有多行具有相同的 Recall
                q_avg = target_rows[['feat_ext1_ms', 'pred1_ms', 'feat_ext2_ms', 'pred2_ms']].mean()
                selected_metrics.append(q_avg)

        # 5. 计算所有 Query 的总平均值
        if not selected_metrics:
            return None, None, None, None
            
        final_df = pd.DataFrame(selected_metrics)
        
        final_avg_feat1 = final_df['feat_ext1_ms'].mean()
        final_avg_pred1 = final_df['pred1_ms'].mean()
        final_avg_feat2 = final_df['feat_ext2_ms'].mean()
        final_avg_pred2 = final_df['pred2_ms'].mean()

        return final_avg_feat1, final_avg_pred1, final_avg_feat2, final_avg_pred2

    except Exception as e:
        print(f"处理文件 {os.path.basename(file_path)} 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# --- 脚本主程序 ---

dataset_paths = [
    '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Genome/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls10-Le40000-Lp1000_efsS20-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv',
    '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Reviews/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv',
    '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Amazon/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv',
    '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/VariousImg/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv',
    '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Music/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv',
    '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/BookReviews/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_imp_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls10-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv',
    '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Tiktok/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv',
    '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/method3/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv'
]

results_idea1 = {}
results_idea2 = {}

for path in dataset_paths:
    # 路径解析逻辑 (保持原有逻辑)
    try:
        p = Path(path)
        parts = p.parts
        idx = -1
        for i, part in enumerate(parts):
            if part.lower() == 'results':
                idx = i
                break
        if idx > 0 and idx < len(parts): 
            dataset_name = parts[idx - 1] 
        else:
            dataset_name = os.path.basename(path).replace('.csv', '')
    except Exception as e:
        dataset_name = os.path.basename(path).replace('.csv', '')

    print(f"正在处理: {dataset_name} ...")
    
    # 接收函数返回的四个值
    feat1, pred1, feat2, pred2 = process_dataset(path)
    
    if feat1 is not None:
        if dataset_name in results_idea1:
            dataset_name = f"{dataset_name}_{os.path.basename(path).replace('.csv', '')}"

        results_idea1[dataset_name] = {
            'feature extraction time': feat1,
            'prediction time': pred1
        }
        results_idea2[dataset_name] = {
            'feature extraction time': feat2,
            'prediction time': pred2
        }
    else:
        print(f"  - 跳过: {dataset_name} (无有效数据)")

# --- 打印和保存部分 ---

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

if results_idea1:
    df_idea1 = pd.DataFrame(results_idea1)
    df_idea1 = df_idea1.reindex(['feature extraction time', 'prediction time'])
    print("\n--- Idea1 统计结果 (Statistics Result for Idea1) [Recall Aligned] ---")
    print(df_idea1)
    df_idea1.to_csv('summary_idea1_statistics_aligned.csv')

if results_idea2:
    df_idea2 = pd.DataFrame(results_idea2)
    df_idea2 = df_idea2.reindex(['feature extraction time', 'prediction time'])
    print("\n--- Idea2 统计结果 (Statistics Result for Idea2) [Recall Aligned] ---")
    print(df_idea2)
    df_idea2.to_csv('summary_idea2_statistics_aligned.csv')