import os
import pandas as pd
from pathlib import Path
import os
import sys
import csv
from collections import defaultdict

def process_non_avg_files(input_folder, output_file):
    non_avg_files = [f for f in os.listdir(input_folder) if f.endswith('.csv') and '_avg' not in f]
    
    if not non_avg_files:
        print("没有找到非avg文件")
        return

    dfs = []
    for file in non_avg_files:
        df = pd.read_csv(Path(input_folder) / file)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    grouped_df = combined_df.groupby(['efs', 'QueryID']).mean().reset_index()
    grouped_df.to_csv(Path(input_folder) / output_file, index=False)
    print(f"非avg文件处理完成，结果保存到: {output_file}")

def process_avg_files(input_folder, output_file):
    avg_files = [f for f in os.listdir(input_folder) if f.endswith('.csv') and '_avg' in f]
    
    if not avg_files:
        print("没有找到avg文件")
        return
    
    dfs = []
    for file in avg_files:
        df = pd.read_csv(Path(input_folder) / file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    grouped_df = combined_df.groupby(['efs']).mean().reset_index()
    grouped_df.to_csv(Path(input_folder) / output_file, index=False)
    print(f"avg文件处理完成，结果保存到: {output_file}")

def main(input_dir, output_dir, dataset, gamma, M, threads):
   file_name = f"results_{dataset}_gamma{gamma}_M{M}_threads{threads}"
   process_non_avg_files(output_dir, file_name + '.csv')
   process_avg_files(output_dir, file_name + '_avg.csv')


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("用法: python script.py <input_dir> <output_dir> <dataset> <gamma> <M> <threads>")
        print("示例: python script.py results output words 80 32 32")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    dataset = sys.argv[3]
    gamma = sys.argv[4]
    M = sys.argv[5]
    threads = sys.argv[6]

    main(input_dir, output_dir, dataset, gamma, M, threads)