#!/usr/bin/env python3

import os
import sys
import csv
from collections import defaultdict

def parse_efs_from_filename(filename):
    """从文件名中提取 efs 参数"""
    efs_start = filename.find("_efs")
    if efs_start == -1:
        print(f"警告：文件 {filename} 中未找到 efs 参数。跳过该文件。")
        return None
    efs_end = filename.find("_", efs_start + 4)
    efs_str = filename[efs_start+4:efs_end] if efs_end != -1 else filename[efs_start+4:]
    try:
        return int(efs_str)
    except ValueError:
        print(f"警告：无法解析 efs 值 {efs_str}，来自文件 {filename}")
        return None


def process_normal_csv_files(input_dir, stats):
    """处理非 _avg.csv 的原始数据文件"""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv") and not file.startswith("summary") and not file.endswith("_avg.csv"):
                file_path = os.path.join(root, file)

                with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        try:
                            query_id = int(row['QueryID'])
                        except (KeyError, ValueError):
                            continue

                        filename = os.path.basename(file)
                        efs = parse_efs_from_filename(filename)
                        if efs is None:
                            continue

                        key = (query_id, efs)
                        stats[key]['count'] += 1
                        stats[key]['acorn_Time'] += float(row['acorn_Time'])
                        stats[key]['acorn_QPS'] += float(row['acorn_QPS'])
                        stats[key]['acorn_Recall'] += float(row['acorn_Recall'])
                        stats[key]['acorn_n3'] += float(row['acorn_n3'])
                        stats[key]['ACORN_1_Time'] += float(row['ACORN_1_Time'])
                        stats[key]['ACORN_1_QPS'] += float(row['ACORN_1_QPS'])
                        stats[key]['ACORN_1_Recall'] += float(row['ACORN_1_Recall'])
                        stats[key]['ACORN_1_n3'] += float(row['ACORN_1_n3'])
                        stats[key]['FilterMapTime'] += float(row['FilterMapTime'])


def process_avg_csv_files(input_dir, avg_stats):
    """处理 _avg.csv 文件，按 efs 分组统计"""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_avg.csv"):
                file_path = os.path.join(root, file)

                with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        filename = os.path.basename(file)
                        efs = parse_efs_from_filename(filename)
                        if efs is None:
                            continue

                        avg_stats[efs]['count'] += 1
                        avg_stats[efs]['acorn_Time'] += float(row['acorn_Time'])
                        avg_stats[efs]['acorn_QPS'] += float(row['acorn_QPS'])
                        avg_stats[efs]['acorn_Recall'] += float(row['acorn_Recall'])
                        avg_stats[efs]['acorn_n3'] += float(row['acorn_n3'])
                        avg_stats[efs]['ACORN_1_Time'] += float(row['ACORN_1_Time'])
                        avg_stats[efs]['ACORN_1_QPS'] += float(row['ACORN_1_QPS'])
                        avg_stats[efs]['ACORN_1_Recall'] += float(row['ACORN_1_Recall'])
                        avg_stats[efs]['ACORN_1_n3'] += float(row['ACORN_1_n3'])
                        avg_stats[efs]['FilterMapTime'] += float(row['FilterMapTime'])


def write_per_query_files(stats, output_dir, dataset, gamma, M, threads):
    """为每个 QueryID 写入合并了 efs 的结果文件"""
    output_dir_per_query = os.path.join(output_dir, "per_query_merged_by_efs")
    os.makedirs(output_dir_per_query, exist_ok=True)

    fieldnames = [
        'efs',
        'acorn_Time', 'acorn_QPS', 'acorn_Recall', 'acorn_n3',
        'ACORN_1_Time', 'ACORN_1_QPS', 'ACORN_1_Recall', 'ACORN_1_n3',
        'FilterMapTime'
    ]

    queries = defaultdict(list)
    for (qid, efs), data in sorted(stats.items()):
        avg_data = {
            'efs': efs,
            'acorn_Time': data['acorn_Time'] / data['count'],
            'acorn_QPS': data['acorn_QPS'] / data['count'],
            'acorn_Recall': data['acorn_Recall'] / data['count'],
            'acorn_n3': data['acorn_n3'] / data['count'],
            'ACORN_1_Time': data['ACORN_1_Time'] / data['count'],
            'ACORN_1_QPS': data['ACORN_1_QPS'] / data['count'],
            'ACORN_1_Recall': data['ACORN_1_Recall'] / data['count'],
            'ACORN_1_n3': data['ACORN_1_n3'] / data['count'],
            'FilterMapTime': data['FilterMapTime'] / data['count']
        }
        queries[qid].append(avg_data)

    for qid, rows in queries.items():
        filename = f"acorn_{dataset}_query{qid}_gamma{gamma}_M{M}_threads{threads}.csv"
        file_path = os.path.join(output_dir_per_query, filename)

        with open(file_path, mode='w', newline='', encoding='utf-8') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"✅ 已为每个查询生成合并了 efs 的文件，保存在: {output_dir_per_query}")


def write_per_efs_files(stats, output_dir, dataset, gamma, M, threads):
    """为每个 efs 写入合并了 QueryID 的结果文件"""
    output_dir_per_efs = os.path.join(output_dir, "per_efs_merged_by_query")
    os.makedirs(output_dir_per_efs, exist_ok=True)

    fieldnames = [
        'QueryID',
        'acorn_Time', 'acorn_QPS', 'acorn_Recall', 'acorn_n3',
        'ACORN_1_Time', 'ACORN_1_QPS', 'ACORN_1_Recall', 'ACORN_1_n3',
        'FilterMapTime'
    ]

    efs_groups = defaultdict(list)
    for (query_id, efs), data in sorted(stats.items()):
        avg_data = {
            'QueryID': query_id,
            'acorn_Time': data['acorn_Time'] / data['count'],
            'acorn_QPS': data['acorn_QPS'] / data['count'],
            'acorn_Recall': data['acorn_Recall'] / data['count'],
            'acorn_n3': data['acorn_n3'] / data['count'],
            'ACORN_1_Time': data['ACORN_1_Time'] / data['count'],
            'ACORN_1_QPS': data['ACORN_1_QPS'] / data['count'],
            'ACORN_1_Recall': data['ACORN_1_Recall'] / data['count'],
            'ACORN_1_n3': data['ACORN_1_n3'] / data['count'],
            'FilterMapTime': data['FilterMapTime'] / data['count']
        }
        efs_groups[efs].append(avg_data)

    for efs, rows in efs_groups.items():
        filename = f"acorn_{dataset}_efs{efs}_gamma{gamma}_M{M}_threads{threads}.csv"
        file_path = os.path.join(output_dir_per_efs, filename)

        with open(file_path, mode='w', newline='', encoding='utf-8') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"✅ 已为每个 efs 生成合并了 QueryID 的文件，保存在: {output_dir_per_efs}")


def write_avg_by_efs_file(avg_stats, output_dir):
    """将所有 avg 文件按 efs 合并写入一个汇总文件"""
    filename = f"avg_by_efs.csv"
    merged_avg_output = os.path.join(output_dir, filename)
    fieldnames = [
        'efs',
        'acorn_Time', 'acorn_QPS', 'acorn_Recall', 'acorn_n3',
        'ACORN_1_Time', 'ACORN_1_QPS', 'ACORN_1_Recall', 'ACORN_1_n3',
        'FilterMapTime'
    ]

    with open(merged_avg_output, mode='w', newline='', encoding='utf-8') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for efs, data in sorted(avg_stats.items()):
            avg_row = {
                'efs': efs,
                'acorn_Time': data['acorn_Time'] / data['count'],
                'acorn_QPS': data['acorn_QPS'] / data['count'],
                'acorn_Recall': data['acorn_Recall'] / data['count'],
                'acorn_n3': data['acorn_n3'] / data['count'],
                'ACORN_1_Time': data['ACORN_1_Time'] / data['count'],
                'ACORN_1_QPS': data['ACORN_1_QPS'] / data['count'],
                'ACORN_1_Recall': data['ACORN_1_Recall'] / data['count'],
                'ACORN_1_n3': data['ACORN_1_n3'] / data['count'],
                'FilterMapTime': data['FilterMapTime'] / data['count']
            }
            writer.writerow(avg_row)

    print(f"✅ 已生成合并 avg 文件：{merged_avg_output}")


def delete_all_csv_files_in_directory(input_dir):
    """删除 input_dir 下所有 .csv 文件（包括子目录）"""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"❌ 删除文件失败 {file_path}: {e}")

    try:
        os.rmdir(input_dir)
        print(f"🗑️ 已删除空目录: {input_dir}")
    except OSError:
        print(f"⚠️ 目录 {input_dir} 非空或无法删除")


def main(input_dir, output_dir, dataset, gamma, M, threads):
    stats = defaultdict(lambda: {
        'count': 0,
        'acorn_Time': 0.0,
        'acorn_QPS': 0.0,
        'acorn_Recall': 0.0,
        'acorn_n3': 0.0,
        'ACORN_1_Time': 0.0,
        'ACORN_1_QPS': 0.0,
        'ACORN_1_Recall': 0.0,
        'ACORN_1_n3': 0.0,
        'FilterMapTime': 0.0
    })

    avg_stats = defaultdict(lambda: {
        'count': 0,
        'acorn_Time': 0.0,
        'acorn_QPS': 0.0,
        'acorn_Recall': 0.0,
        'acorn_n3': 0.0,
        'ACORN_1_Time': 0.0,
        'ACORN_1_QPS': 0.0,
        'ACORN_1_Recall': 0.0,
        'ACORN_1_n3': 0.0,
        'FilterMapTime': 0.0
    })

    process_normal_csv_files(input_dir, stats)
    process_avg_csv_files(input_dir, avg_stats)

    write_per_query_files(stats, output_dir, dataset, gamma, M, threads)
    write_per_efs_files(stats, output_dir, dataset, gamma, M, threads)
    write_avg_by_efs_file(avg_stats, output_dir)

   #  delete_all_csv_files_in_directory(input_dir)
   #  print("✅ input_dir 及其所有 .csv 文件已清理完成")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("用法: python script.py <input_dir> <output_dir> <dataset> <gamma> <M> <threads>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    dataset = sys.argv[3]
    gamma = sys.argv[4]
    M = sys.argv[5]
    threads = sys.argv[6]

    main(input_dir, output_dir, dataset, gamma, M, threads)
