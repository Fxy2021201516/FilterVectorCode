#!/usr/bin/env python3
import os
import sys
import glob
import pandas as pd

def parse_L_from_filename(filename):
    """
    从文件名提取 L 数值，如 L10 -> 10
    """
    base = os.path.basename(filename)
    L_str = base.split("_")[2]  # 取出 L10 这部分
    return int(L_str[1:])       # 去掉 L 字符并转为整数


def load_query_details_files(results_path):
    """
    加载指定 results 文件夹中的所有 query_details_xx_repeatX.csv 文件
    返回合并后的 DataFrame 和原始文件列表
    """
    csv_files = glob.glob(os.path.join(results_path, "query_details_*.csv"))
    dfs = []
    original_files = []

    for file in csv_files:
        df = pd.read_csv(file)
        L = parse_L_from_filename(file)
        df["L"] = L
        dfs.append(df)
        original_files.append(file)

    if not dfs:
        return None, []

    return pd.concat(dfs, ignore_index=True), original_files


def load_result_avg_files(results_path):
    """
    加载指定 results 文件夹中的所有 result_avg_repeatX.csv 文件
    返回合并后的 DataFrame 和原始文件列表
    """
    csv_files = glob.glob(os.path.join(results_path, "result_avg_repeat*.csv"))
    dfs = []
    original_files = []

    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
        original_files.append(file)

    if not dfs:
        return None, []

    # 合并所有重复实验的数据
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 按 L 值分组并计算平均值
    grouped_df = combined_df.groupby("L").agg({
        "Cmps": "mean",
        "QPS": "mean",
        "Recall": "mean",
        "Time(ms)": "mean",
        "EntryPoints": "mean",
        "LNGDescendants": "mean",
        "entry_group_total_coverage": "mean"
    }).reset_index()
        
    return grouped_df, original_files


def group_by_query_and_L(df):
    """
    按 QueryID 和 L 分组，对其他列取平均
    """
    grouped_df = df.groupby(["QueryID", "L"]).agg({
        "Time(ms)": "mean",
        "DistanceCalcs": "mean",
        "EntryPoints": "mean",
        "LNGDescendants": "mean",
        "entry_group_total_coverage": "mean",
        "QPS": "mean",
        "Recall": "mean",
        "is_global_search": "first"
    }).reset_index()
    return grouped_df


def group_by_L_and_query(df):
    """
    按 L 和 QueryID 分组，对其他列取平均
    """
    grouped_df = df.groupby(["L", "QueryID"]).agg({
        "Time(ms)": "mean",
        "DistanceCalcs": "mean",
        "EntryPoints": "mean",
        "LNGDescendants": "mean",
        "entry_group_total_coverage": "mean",
        "QPS": "mean",
        "Recall": "mean",
        "is_global_search": "first"
    }).reset_index()
    return grouped_df


def save_per_query(grouped_df, output_dir):
    """
    将每个 QueryID 的数据保存为独立的 CSV 文件
    输出路径: <output_dir>/per_query_merged_by_L/query_{QueryID}.csv
    """
    merged_dir = os.path.join(output_dir, "per_query_merged_by_L")
    os.makedirs(merged_dir, exist_ok=True)

    for qid, group in grouped_df.groupby("QueryID"):
        filename = f"query_{qid}.csv"
        filepath = os.path.join(merged_dir, filename)
        group.to_csv(filepath, index=False)


def save_per_L(grouped_df, output_dir):
    """
    将每个 L 值的数据保存为独立的 CSV 文件
    输出路径: <output_dir>/per_L_merged_by_query/L{L}.csv
    """
    merged_dir = os.path.join(output_dir, "per_L_merged_by_query")
    os.makedirs(merged_dir, exist_ok=True)

    for L_val, group in grouped_df.groupby("L"):
        filename = f"L{L_val}.csv"
        filepath = os.path.join(merged_dir, filename)
        group.to_csv(filepath, index=False)


def delete_original_files(file_list):
    """
    删除原始文件列表中的所有文件
    """
    for file in file_list:
        try:
            os.remove(file)
        except OSError as e:
            print(f"删除文件 {file} 时出错: {e}")


def process_experiment_folder(experiment_path):
    """
    处理单个实验文件夹（包含 results 子文件夹）
    """
    results_path = os.path.join(experiment_path, "results")
    if not os.path.exists(results_path):
        print(f"跳过 {experiment_path}: results 目录不存在")
        return

    # 处理 query_details 文件
    combined_df, query_details_files = load_query_details_files(results_path)
    if combined_df is not None and len(combined_df) > 0:
        # 按 QueryID 分组
        query_grouped_df = group_by_query_and_L(combined_df)
        save_per_query(query_grouped_df, results_path)
        
        # 按 L 分组
        L_grouped_df = group_by_L_and_query(combined_df)
        save_per_L(L_grouped_df, results_path)
        
        #delete_original_files(query_details_files)
    else:
        print(f"跳过 {results_path} 的 query_details: 没有找到有效的文件")

    # 处理 result_avg 文件
    result_avg_df, result_avg_files = load_result_avg_files(results_path)
    if result_avg_df is not None and len(result_avg_df) > 0:
        output_file = os.path.join(results_path, "result_avg_merged.csv")
        result_avg_df.to_csv(output_file, index=False)
        print(f"已保存合并的 result_avg 文件: {output_file}")
        #delete_original_files(result_avg_files)
    else:
        print(f"跳过 {results_path} 的 result_avg: 没有找到有效的文件")


def process_dataset(dataset_path):
    """
    处理一个数据集目录下的所有实验文件夹
    """
    for experiment_folder in glob.glob(os.path.join(dataset_path, "*")):
        if not os.path.isdir(experiment_folder):
            continue

        process_experiment_folder(experiment_folder)


def main(results_root):
    """
    主函数：遍历 FilterVectorResults/UNG 下的所有数据集目录
    """
    ung_path = os.path.join(results_root, "UNG")

    if not os.path.exists(ung_path):
        print(f"错误: 路径 {ung_path} 不存在")
        return

    for dataset_folder in glob.glob(os.path.join(ung_path, "*")):
        if not os.path.isdir(dataset_folder):
            continue

        print(f"\n处理数据集目录: {dataset_folder}")
        process_dataset(dataset_folder)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python summarize_csv.py <FilterVectorResults路径>")
        sys.exit(1)

    results_dir = sys.argv[1]
    main(results_dir)