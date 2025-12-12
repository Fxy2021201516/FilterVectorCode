# -*- coding: utf-8 -*-

import os
import csv

def merge_text_files(input_files, output_file):
    """
    合并多个文本文件到一个文件中。
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for filename in input_files:
                if not os.path.exists(filename):
                    print(f"⚠️ 警告：文件 '{filename}' 不存在，已跳过。")
                    continue
                with open(filename, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    if not content.endswith('\n'):
                        outfile.write('\n')
        print(f"✅ 成功！文本文件合并完成 -> {output_file}")
    except IOError as e:
        print(f"❌ 错误: 合并文本文件时发生错误: {e}")

def merge_binary_files(input_files, output_file):
    """
    合并多个二进制文件（如 .fvecs）到一个文件中。
    """
    try:
        with open(output_file, 'wb') as outfile:
            for filename in input_files:
                if not os.path.exists(filename):
                    print(f"⚠️ 警告：文件 '{filename}' 不存在，已跳过。")
                    continue
                with open(filename, 'rb') as infile:
                    outfile.write(infile.read())
        print(f"✅ 成功！二进制文件合并完成 -> {output_file}")
    except IOError as e:
        print(f"❌ 错误: 合并二进制文件时发生错误: {e}")

def merge_csv_files(input_files, output_file):
    """
    合并多个 CSV 文件到一个文件中，并只保留一个表头。
    """
    try:
        header_written = False
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            for filename in input_files:
                if not os.path.exists(filename):
                    print(f"⚠️ 警告：文件 '{filename}' 不存在，已跳过。")
                    continue
                with open(filename, 'r', encoding='utf-8') as infile:
                    try:
                        reader = csv.reader(infile)
                        header = next(reader) 

                        if not header_written:
                            writer.writerow(header)
                            header_written = True
                        
                        for row in reader:
                            writer.writerow(row)
                    except StopIteration:
                        # 文件为空，跳过
                        print(f"ℹ️ 信息：文件 '{filename}' 为空，已跳过。")
                        continue
        print(f"✅ 成功！CSV 文件合并完成 -> {output_file}")
    except IOError as e:
        print(f"❌ 错误: 合并 CSV 文件时发生错误: {e}")


# --- [修改] ---
# 重写了 generate_and_merge_files 函数，使其更加灵活
def generate_and_merge_files(dataset, folders_to_merge, output_folder_name):
    """
    根据显式提供的文件夹列表执行合并操作。

    :param dataset: 数据集名称 (e.g., 'Russian')
    :param folders_to_merge: 需要合并的查询文件夹名称列表
    :param output_folder_name: 合并后存放的新文件夹名称
    """
    print("="*80)
    print(f"🚀 开始自动合并任务: dataset='{dataset}'")
    print(f"   输出文件夹: '{output_folder_name}'")
    print(f"   合并列表: {folders_to_merge}")
    print("="*80)

    base_path = f"/home/fengxiaoyao/1023FilterVector/FilterVectorData/{dataset}"
    
    # --- 1. 自动生成输出文件路径 ---
    print("\n🔎 正在生成输出文件路径...")
    output_dir = os.path.join(base_path, output_folder_name)

    # 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 输出目录已准备好: {output_dir}")
    except OSError as e:
        print(f"❌ 错误: 无法创建输出目录 {output_dir}. 原因: {e}")
        return

    merged_txt_file = os.path.join(output_dir, f"{dataset}_query_labels.txt")
    merged_fvecs_file = os.path.join(output_dir, f"{dataset}_query.fvecs")
    merged_csv_file = os.path.join(output_dir, "attribute_coverage_merged.csv")
    print("✅ 输出文件路径生成完毕。")


    # --- 2. 自动生成输入文件列表 ---
    txt_files_to_merge = []
    fvecs_files_to_merge = []
    csv_files_to_merge = []

    print("\n🔎 正在根据文件夹列表生成输入文件路径...")
    for folder_name in folders_to_merge:
        query_dir = os.path.join(base_path, folder_name)
        
        # 1. 标签文件 (名称固定)
        # 路径: .../Russian/query_A_12/Russian_query_labels.txt
        txt_files_to_merge.append(os.path.join(query_dir, f"{dataset}_query_labels.txt"))
        
        # 2. 向量文件 (名称固定)
        # 路径: .../Russian/query_A_12/Russian_query.fvecs
        fvecs_files_to_merge.append(os.path.join(query_dir, f"{dataset}_query.fvecs"))
        
        # 3. CSV 文件 (名称不固定，需要推导)
        # 示例:
        # 'query_A_12' -> 'profiled_A_12.csv'
        # 'query_B_5'  -> 'profiled_B_5.csv'
        # 'query_weighted_sub_base_l=1' -> 'profiled_weighted_sub_base_l=1.csv'
        
        if not folder_name.startswith("query_"):
            print(f"⚠️ 警告: 文件夹 '{folder_name}' 命名不规范，跳过 CSV 合并。")
            continue
            
        # 移除 "query_" 前缀
        profile_suffix = folder_name.replace("query_", "", 1) # 得到 'A_12' 或 'B_5' 或 'weighted_sub_base_l=1'
        csv_name = f"profiled_{profile_suffix}.csv"
        csv_files_to_merge.append(os.path.join(query_dir, csv_name))

    print("✅ 输入文件列表生成完毕。")
    # print("DEBUG (txt):", txt_files_to_merge)
    # print("DEBUG (fvecs):", fvecs_files_to_merge)
    # print("DEBUG (csv):", csv_files_to_merge)

    # --- 3. 执行合并操作 ---
    print("\n- 开始合并文本文件 (Labels)...")
    merge_text_files(txt_files_to_merge, merged_txt_file)

    print("\n- 开始合并 .fvecs 文件 (Vectors)...")
    merge_binary_files(fvecs_files_to_merge, merged_fvecs_file)
    
    print("\n- 开始合并 .csv 文件 (Profiles)...")
    merge_csv_files(csv_files_to_merge, merged_csv_file)

    print("\n" + "="*80)
    print("🎉 所有合并任务已完成！")
    print("="*80)


# --- 主程序 ---
if __name__ == "__main__":
    # ===================== 用户配置区 =====================
    DATASET = "BookReviews"
    FOLDERS_TO_MERGE = [
        # "query_A_12",
        "query_D_23",
        "query_B_4",
        "query_B_5",
        # "query_weighted_sub_base_l=1",
        "query_weighted_sub_base_l=2",
        # "query_weighted_sub_base_l=2-1",
        "query_weighted_sub_base_l=3",
        "query_weighted_sub_base_l=4",
        "query_weighted_sub_base_l=5",
        "query_weighted_sub_base_l=6",
        "query_weighted_sub_base_l=7",
        "query_weighted_sub_base_l=8",
        "query_weighted_sub_base_l=9",
        
    ]

    OUTPUT_FOLDER_NAME = "query_B_C_D-weighted-sub-base-123456789"

    # ====================================================

    generate_and_merge_files(
        dataset=DATASET,
        folders_to_merge=FOLDERS_TO_MERGE,
        output_folder_name=OUTPUT_FOLDER_NAME
    )