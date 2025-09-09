# 验证UNG的csv文件中，找entry group的时间的/UNG时间的比值
import pandas as pd
import numpy as np

def analyze_query_data(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。请确保文件名正确。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 获取所有唯一的QueryID
    query_ids = df['QueryID'].unique()

    # 检查必需的列是否存在
    required_columns = ['QueryID', 'Time_ms', 'Recall', 'EntryGroupT_ms', 'SearchT_ms']
    if not all(col in df.columns for col in required_columns):
        print(f"错误：CSV文件中缺少必需的列。请确保文件包含: {required_columns}")
        return

    # 存储每个QueryID的详细结果
    results_gte_95 = []
    results_lt_95 = []
    # 存储所有QueryID的代表性比率，用于计算总平均值
    all_representative_ratios = []

    # 遍历每个QueryID
    for qid in query_ids:
        group = df[df['QueryID'] == qid]

        # 筛选出Recall >= 0.95的记录
        high_recall_group = group[group['Recall'] >= 0.95]

        if not high_recall_group.empty:
            # 在所有Recall >= 0.95的记录中，找到Time_ms最小的那一条
            target_row = high_recall_group.loc[high_recall_group['Time_ms'].idxmin()]
            
            entry_group_t = target_row['EntryGroupT_ms']
            search_t = target_row['SearchT_ms']

            # 计算比值，处理SearchT_ms为0的情况
            ratio = entry_group_t / search_t if search_t != 0 else np.inf

            results_gte_95.append({
                'QueryID': qid,
                'Time_ms_at_target': target_row['Time_ms'],
                'Recall_at_target': target_row['Recall'],
                'EntryGroupT_ms': entry_group_t,
                'SearchT_ms': search_t,
                'Ratio': ratio
            })
            all_representative_ratios.append(ratio)
        else:
            # 如果没有记录的Recall >= 0.95，找到最大的Recall值
            max_recall = group['Recall'].max()
            max_recall_rows = group[group['Recall'] == max_recall]
            # 在达到最大Recall的记录中，找到Time_ms最小的那一条
            target_row = max_recall_rows.loc[max_recall_rows['Time_ms'].idxmin()]

            entry_group_t = target_row['EntryGroupT_ms']
            search_t = target_row['SearchT_ms']
            
            # 同样为这些QueryID计算比率，用于计算总平均值
            ratio = entry_group_t / search_t if search_t != 0 else np.inf

            results_lt_95.append({
                'QueryID': qid,
                'Max_Recall': max_recall,
                'Min_Time_ms_for_Max_Recall': target_row['Time_ms'],
                'EntryGroupT_ms_at_max_recall': entry_group_t,
                'SearchT_ms_at_max_recall': search_t,
                'Ratio_at_max_recall': ratio
            })
            all_representative_ratios.append(ratio)

    # --- 结果汇总和输出 ---
    
    # 1. 将结果列表转换为DataFrame
    df_gte_95 = pd.DataFrame(results_gte_95)
    df_lt_95 = pd.DataFrame(results_lt_95)

    # 2. 合并两个DataFrame以便输出到一个文件
    # 为了保持所有信息的完整性，我们在合并时将不匹配的列填充为NaN
    output_df = pd.concat([
        df_gte_95.set_index('QueryID'), 
        df_lt_95.set_index('QueryID')
    ], axis=1)
    output_df.index.name = 'QueryID'
    output_df.reset_index(inplace=True)


    # 3. 计算平均值
    # 计算Recall >= 0.95的QueryID的比率平均值
    avg_ratio_gte_95 = df_gte_95['Ratio'].mean() if not df_gte_95.empty else 0
    # 计算所有QueryID的代表性比率的平均值
    avg_ratio_all = np.mean(all_representative_ratios) if all_representative_ratios else 0

    # 4. 保存到CSV文件
    output_filename = 'analysis_results_detailed.csv'
    output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    # 5. 打印总结信息
    print("="*50)
    print("数据分析结果摘要")
    print("="*50)
    print(f"总共分析了 {len(query_ids)} 个独立的QueryID。")
    print("-" * 50)
    
    print(f"有 {len(df_gte_95)} 个QueryID的Recall达到了0.95或更高:")
    print(f"  - 对于这些QueryID，其比率(EntryGroupT_ms / SearchT_ms)的平均值为: {avg_ratio_gte_95:.4f}")
    print("-" * 50)
    
    print(f"有 {len(df_lt_95)} 个QueryID的Recall从未达到0.95。")
    print("-" * 50)
    
    print(f"对于所有 {len(query_ids)} 个QueryID，其代表性比率的总平均值为: {avg_ratio_all:.4f}")
    print("="*50)
    print(f"\n详细分析结果已保存至文件: '{output_filename}'")
    print("文件中包含了每个QueryID的具体数值，包括时间、比率、最大Recall等信息。")


analyze_query_data(file_path='/data/fxy/FilterVector/FilterVectorResults/UNG/VariousTaggedImages/VariousTaggedImages_query7_nTfalse_th32_M32_LB100_alpha1.2_C6_EP16_Ls20000_Le60000_Lp10000_REPEATs3/results/query_details_repeat3.csv')