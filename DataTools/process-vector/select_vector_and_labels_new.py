# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import csv
from collections import defaultdict

# --------------------------------------------------------------------------------
# 文件读写函数
# --------------------------------------------------------------------------------

def read_fvecs(filename, c_contiguous=True):
    """读取 .fvecs 文件，返回一个 numpy 数组。"""
    try:
        with open(filename, 'rb') as f:
            d_bytes = f.read(4)
            if not d_bytes: return np.array([])
            d = np.frombuffer(d_bytes, dtype='int32')[0]
            f.seek(0)
            record_size = 4 + d * 4
            file_content = f.read()
        
        num_vectors = len(file_content) // record_size
        if num_vectors == 0: return np.array([])
            
        data = np.frombuffer(file_content, dtype='float32').reshape(num_vectors, d + 1)
        vectors = data[:, 1:].copy()
        
        return vectors.copy(order='C') if c_contiguous else vectors
    except FileNotFoundError:
        print(f"❌ 错误: .fvecs 文件未找到 -> {filename}")
        return None
    except Exception as e:
        print(f"❌ 错误: 读取 .fvecs 文件时发生错误: {e}")
        return None

def write_fvecs(filename, vecs):
    """将一个 numpy 数组写入 .fvecs 文件。"""
    if vecs.ndim != 2: raise ValueError("输入必须是一个二维数组")
    if vecs.shape[0] == 0: return
    num_vectors, dim = vecs.shape
    with open(filename, 'wb') as f:
        for i in range(num_vectors):
            f.write(np.array([dim], dtype='int32').tobytes())
            f.write(vecs[i, :].astype('float32').tobytes())

def parse_label_line(line, delimiter=','):
    """辅助函数：将标签行字符串解析为整数集合"""
    parts = line.strip().split(delimiter)
    return set(int(p) for p in parts if p)

def write_output_files(output_dir, data_list, dataset, file_suffix, prefix=""):
    """
    辅助函数：将给定的数据列表写入 fvecs, txt, 和 csv 文件。
    (此函数会生成 'profiled_*.csv'，这就是 new_run_pipeline.py 所需的 p_pass 文件)
    """
    if not data_list:
        print(f"   -> 目录 {os.path.basename(output_dir)} 无数据，已跳过。")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    vectors_to_write = np.array([item['vector'] for item in data_list])
    labels_to_write = [item['label'] for item in data_list]
    
    # 'coverage' 字段就是 'p_pass' (覆盖的向量数)
    profiled_to_write = [{'coverage_count': item['coverage'], 'labels': item['sorted_label_str']} for item in data_list]

    fvecs_name = os.path.join(output_dir, f"{dataset}_query.fvecs")
    labels_name = os.path.join(output_dir, f"{dataset}_query_labels.txt")
    csv_path = os.path.join(output_dir, f"profiled_{file_suffix}.csv")

    write_fvecs(fvecs_name, vectors_to_write)
    with open(labels_name, 'w', encoding='utf-8') as f:
        f.writelines(labels_to_write)
    
    # 这个 CSV 就是 'attribute_coverage_file'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['coverage_count', 'labels'])
        writer.writeheader()
        writer.writerows(profiled_to_write)
        
    print(f"   -> 成功为 '{os.path.basename(output_dir)}' 写入 {len(data_list)} 条记录。")


# --------------------------------------------------------------------------------
# 核心分析函数 (与原版一致)
# --------------------------------------------------------------------------------

def analyze_all_queries(query_df, dataset, merged_data_dir, base_labels_path):
    """
    对给定的查询DataFrame进行全面的分析，返回一个包含所有信息的分析结果DataFrame。
    (此函数从 'query_df' 中获取 QueryID，然后从源文件加载 'vector'，
     并重新计算 'coverage' (p_pass) 和 'length' (QuerySize))
    """
    query_indices = query_df['QueryID'].astype(int).tolist()
    temp_ids = query_df['temp_unique_id'].tolist()
    
    print("--- 步骤 1/4: 加载源文件 ---")
    fvecs_path = os.path.join(merged_data_dir, f"{dataset}_query.fvecs")
    txt_path = os.path.join(merged_data_dir, f"{dataset}_query_labels.txt")
    
    all_vectors = read_fvecs(fvecs_path)
    with open(txt_path, 'r', encoding='utf-8') as f: all_labels = f.readlines()
    
    with open(base_labels_path, 'r', encoding='utf-8') as f:
        base_label_sets = [parse_label_line(line, delimiter=',') for line in f if line.strip()]
    total_base_items = len(base_label_sets)
    print(f"✅ 源文件加载完成 ({len(query_indices)} 个查询ID, {total_base_items} 条基础数据)。")

    print("\n--- 步骤 2/4: 创建倒排索引并分析所有查询 ---")
    inverted_index = defaultdict(set)
    for i, base_set in enumerate(base_label_sets):
        for label in base_set: inverted_index[label].add(i)
    
    all_selected_data = []
    for i, index in enumerate(query_indices):
        if index >= len(all_labels) or index >= len(all_vectors): continue
        
        temp_id = temp_ids[i]
        label_line = all_labels[index]
        query_set = parse_label_line(label_line, delimiter=',')
        coverage = 0
        if query_set:
            try:
                posting_lists = [inverted_index[label] for label in query_set]
                coverage = len(set.intersection(*posting_lists))
            except KeyError: coverage = 0
            
        all_selected_data.append({
            'QueryID': index,
            'temp_unique_id': temp_id,
            'vector': all_vectors[index], 
            'label': label_line,
            'coverage': coverage, # <--- 这就是 p_pass (计数值)
            'selectivity': coverage / total_base_items if total_base_items > 0 else 0,
            'length': len(query_set), # <--- 这就是 length (QuerySize)
            'sorted_label_str': " ".join(map(str, sorted(list(query_set))))
        })
    print(f"✅ 所有 {len(all_selected_data)} 条查询分析完成。")

    if not all_selected_data: return pd.DataFrame()

    analysis_df = pd.DataFrame(all_selected_data)
    
    # 合并来自 _selected_queries.csv 的性能数据 (T_AG, T_M3 等)
    query_df_performance = query_df.drop(columns=['QueryID'])
    merged_df = pd.merge(analysis_df, query_df_performance, on='temp_unique_id')
    
    # 我们现在有一个包含 'vector', 'coverage', 'length' 和所有性能列的完整 DataFrame
    return merged_df.drop(columns=['temp_unique_id'])


# =========================================================================
# --- 新的主流程 (MAIN) ---
# =========================================================================

if __name__ == "__main__":
    # ===================== 用户配置区 =====================

    DATASET = "BookReviews" 
    
    # 1. 这是 new_select.py 生成的 *输入* 文件
    SELECTED_CSV_PATH = "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/BookReviews/Results/per-query_results/BookReviews_weighted_sub_base_123456789_K10_Comparison_selected_queries.csv"
    
    # 2. 这是包含 *所有* 原始查询向量和标签的目录
    MERGED_DATA_DIR = "/home/fengxiaoyao/1023FilterVector/FilterVectorData/BookReviews/query_B_C_D-weighted-sub-base-123456789"
    
    # 3. 这是包含 *所有* 基础数据标签的 .txt 文件
    BASE_LABELS_PATH = "/home/fengxiaoyao/1023FilterVector/FilterVectorData/BookReviews/BookReviews_base_labels.txt"
    
    # 4. 这是你希望所有新查询子集写入的 *根* 目录
    BASE_OUTPUT_DIR = "/home/fengxiaoyao/1023FilterVector/FilterVectorData/BookReviews"
    
    # 5. 输出查询目录的前缀 (例如 "query_select_imp")
    OUTPUT_PREFIX = "query_select_imp"
    # =========================================================================
    
    print("="*80)
    print("🚀 开始执行【新】查询分组与分析任务 (Length/p_pass Splits)")
    print("="*80)
    
    try:
        master_df_original = pd.read_csv(SELECTED_CSV_PATH)
        
        # 1. 准备全局 1000 条查询 (或 N 条)
        #    我们使用 reset_index() 来创建 'temp_unique_id'
        #    这在 analyze_all_queries 中用于 1-to-1 合并
        combined_df = master_df_original.reset_index().rename(columns={'index': 'temp_unique_id'})
        
        print(f"已加载 {len(combined_df)} 条查询 (来自 {os.path.basename(SELECTED_CSV_PATH)})。")

        # 2. 对所有已选查询进行全局分析 (获取 vectors, labels, coverage, length)
        #    profiling_df 是一个包含 1000 行的完整 DataFrame
        profiling_df = analyze_all_queries(combined_df, DATASET, MERGED_DATA_DIR, BASE_LABELS_PATH)

        if profiling_df.empty:
            raise ValueError("分析后未生成任何有效数据。")
        
        # 3. 构造主输出目录
        #    (从 MERGED_DATA_DIR 提取 'weighted_weighted_sub_base_123456789')
        dir_name = os.path.basename(MERGED_DATA_DIR.rstrip('/'))
        file_suffix = dir_name.replace("query_", "") # e.g., weighted_weighted_sub_base_123456789
        
        main_folder_name = f"{OUTPUT_PREFIX}_{file_suffix}"
        main_output_dir = os.path.join(BASE_OUTPUT_DIR, main_folder_name)
        
        print(f"\n--- 步骤 3/4: 拆分查询 (Length & p_pass) ---")
        
        N_TOTAL = len(profiling_df)
        N_SPLIT = N_TOTAL // 2
        # 处理 N_TOTAL 为奇数的情况 (e.g., 1001 -> 500 / 501)
        N_REMAINING = N_TOTAL - N_SPLIT 
        
        print(f" - 总查询数: {N_TOTAL}, 每组大小: {N_SPLIT} (Small) / {N_REMAINING} (Large)")
        
        # --- 拆分 1: 按 Length (QuerySize) 拆分 ---
        # 'length' 是 analyze_all_queries 计算出的 QuerySize
        df_sorted_len = profiling_df.sort_values('length').reset_index(drop=True)
        df_len_small = df_sorted_len.head(N_SPLIT)
        df_len_large = df_sorted_len.tail(N_REMAINING)
        print(" - 已按 Length (QuerySize) 拆分。")
        
        # --- 拆分 2: 按 p_pass (coverage) 拆分 ---
        # 'coverage' 是 analyze_all_queries 计算出的 p_pass (计数值)
        df_sorted_ppass = profiling_df.sort_values('coverage').reset_index(drop=True)
        df_ppass_small = df_sorted_ppass.head(N_SPLIT)
        df_ppass_large = df_sorted_ppass.tail(N_REMAINING)
        print(" - 已按 p_pass (coverage) 拆分。")
        
        print(f"\n--- 步骤 4/4: 创建所有输出文件与子目录 ---")
        print(f"主输出目录: {main_output_dir}")

        # 1. 写入完整的 1000 条查询 (根目录)
        print(f" - 正在写入完整的 {N_TOTAL} 条查询 (根目录)...")
        write_output_files(
            output_dir=main_output_dir, 
            data_list=profiling_df.to_dict('records'), 
            dataset=DATASET, 
            file_suffix=file_suffix,
            prefix=f"{OUTPUT_PREFIX}_{file_suffix}"
        )

        # 2. 写入 Length 子目录
        print(" - 正在写入按【Length】分组的 *子目录*...")
        len_small_suffix = f"{file_suffix}_len_small"
        write_output_files(os.path.join(main_output_dir, len_small_suffix), df_len_small.to_dict('records'), DATASET, len_small_suffix, prefix=f"{OUTPUT_PREFIX}_{len_small_suffix}")
        
        len_large_suffix = f"{file_suffix}_len_large"
        write_output_files(os.path.join(main_output_dir, len_large_suffix), df_len_large.to_dict('records'), DATASET, len_large_suffix, prefix=f"{OUTPUT_PREFIX}_{len_large_suffix}")

        # 3. 写入 p_pass (coverage) 子目录
        print(" - 正在写入按【p_pass】分组的 *子目录*...")
        ppass_small_suffix = f"{file_suffix}_ppass_small"
        write_output_files(os.path.join(main_output_dir, ppass_small_suffix), df_ppass_small.to_dict('records'), DATASET, ppass_small_suffix, prefix=f"{OUTPUT_PREFIX}_{ppass_small_suffix}")

        ppass_large_suffix = f"{file_suffix}_ppass_large"
        write_output_files(os.path.join(main_output_dir, ppass_large_suffix), df_ppass_large.to_dict('records'), DATASET, ppass_large_suffix, prefix=f"{OUTPUT_PREFIX}_{ppass_large_suffix}")
        
        print("\n" + "="*80)
        print("🎉 全部任务完成！")
        print("="*80)

    except (FileNotFoundError, ValueError) as e:
        print(f"❌ 致命错误: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")