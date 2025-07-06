import os
import pandas as pd
import re
from pathlib import Path

def find_first_above_threshold(group, recall_col, time_col, n3_col, Recall_col, sort_by):
    """ 在 group 中按照 sort_by 排序，找出 Recall == 1 的第一个点；
        若没有，则取 Recall 最大值的第一个出现 """
    sorted_group = group.sort_values(by=sort_by)

    # 查找 Recall == 1 的第一个点
    hit = sorted_group[sorted_group[recall_col] == 1]
    if not hit.empty:
        return hit.iloc[0][[time_col, n3_col, Recall_col]]
    
    # 否则找最大 Recall 的第一个出现
    max_recall = sorted_group[recall_col].max()
    max_recall_rows = sorted_group[sorted_group[recall_col] == max_recall]
    return max_recall_rows.iloc[0][[time_col, n3_col, Recall_col]]


def extract_params_from_path(filepath, keywords=None, alg=""):
    if keywords is None:
        keywords = [r'repeat_\d+',r'query_\d+', r'query\d+', r'gamma\d+', r'LB\d+', r'alpha[\d.]+', r'M\d+', r'C\d+', r'EP\d+', r'ifbfstrue+', r'ifbfsfalse+', r'efs\d+-\d+-\d+']

    dirname = os.path.dirname(filepath)
    pattern = '|'.join(keywords)
    matches = []

    for part in dirname.split(os.sep):
        part_matches = re.findall(pattern, part)
        matches.extend(part_matches)

    unique_matches = []
    for m in matches:
        if m not in unique_matches:
            unique_matches.append(m)

    if unique_matches and alg=="acorn":
        return '_'.join(unique_matches[:10])
    if unique_matches and alg=="ung":
        return '_'.join(unique_matches[4:])    
    return "default"


def process_acorn_data(acorn_file):
    df = pd.read_csv(acorn_file)
    results = {}

    for query_id, group in df.groupby('QueryID'):
        acorn_time_n3_Recall = find_first_above_threshold(
            group, 'acorn_Recall', 'acorn_Time', 'acorn_n3', 'acorn_Recall', sort_by='efs'
        )
        acorn1_time_n3_Recall = find_first_above_threshold(
            group, 'ACORN_1_Recall', 'ACORN_1_Time', 'ACORN_1_n3', 'ACORN_1_Recall', sort_by='efs'
        )

        results[query_id] = {
            'acorn_Time': acorn_time_n3_Recall['acorn_Time'] * 1000,
            'acorn_n3': acorn_time_n3_Recall['acorn_n3'],
            'acorn_Recall': acorn_time_n3_Recall['acorn_Recall'],
            'ACORN_1_Time': acorn1_time_n3_Recall['ACORN_1_Time'] * 1000,
            'ACORN_1_n3': acorn1_time_n3_Recall['ACORN_1_n3'],
            'ACORN_1_Recall': acorn1_time_n3_Recall['ACORN_1_Recall']
        }
    return pd.DataFrame(results).T.reset_index().rename(columns={'index': 'QueryID'})


def process_ung_data(ung_file):
    df = pd.read_csv(ung_file)
    results = {}

    for query_id, group in df.groupby('QueryID'):
        ung_time_n3_Recall = find_first_above_threshold(
            group, 'Recall', 'UNG_time(ms)', 'DistanceCalcs', 'Recall', sort_by='Lsearch'
        )

        entry_points = group['EntryPoints'].iloc[0]
        coverage = group['entry_group_total_coverage'].iloc[0]
        filter_map_time = group['bitmap_time(ms)'].iloc[0]
        flag_time = group['flag_time(ms)'].iloc[0]
        
        # 新增的指标
        descendants_merge_time = group['descendants_merge_time(ms)'].iloc[0]
        coverage_merge_time = group['coverage_merge_time(ms)'].iloc[0]
        lng_descendants = group['LNGDescendants'].iloc[0]


        results[query_id] = {
            'UNG_time': ung_time_n3_Recall['UNG_time(ms)'],
            'DistanceCalcs': ung_time_n3_Recall['DistanceCalcs'],
            'UNG_Recall': ung_time_n3_Recall['Recall'],
            'EntryPoints': entry_points,
            'Coverage': coverage,
            'bitmap_time': filter_map_time,
            'flag_time': flag_time,
            # 添加新的指标
            'descendants_merge_time': descendants_merge_time,
            'coverage_merge_time': coverage_merge_time,
            'LNGDescendants': lng_descendants
        }

    return pd.DataFrame(results).T.reset_index().rename(columns={'index': 'QueryID'})


def merge_datasets(acorn_df, ung_df):
    merged_df = pd.merge(acorn_df, ung_df, on='QueryID', how='outer')

    # ACORN 加上 bitmap_time
    merged_df['acorn_Time'] += merged_df['bitmap_time']
    merged_df['ACORN_1_Time'] += merged_df['bitmap_time']

    # 添加比较列
    merged_df['ACORN_vs_UNG_time_ratio'] = merged_df['acorn_Time'] / merged_df['UNG_time']
    merged_df['ACORN_vs_UNG_n3_ratio'] = merged_df['acorn_n3'] / merged_df['DistanceCalcs']
    
    merged_df['ACORN1_vs_UNG_time_ratio'] = merged_df['ACORN_1_Time'] / merged_df['UNG_time']
    merged_df['ACORN1_vs_UNG_n3_ratio'] = merged_df['ACORN_1_n3'] / merged_df['DistanceCalcs']

    return merged_df


def get_output_filename(acorn_file, ung_file, dataset):
    acorn_params = extract_params_from_path(acorn_file, None, "acorn")
    ung_params = extract_params_from_path(ung_file, None, "ung")

    print("ACORN params:", acorn_params)
    print("UNG params:", ung_params)

    if not acorn_params:
        acorn_params = "acorn"
    if not ung_params:
        ung_params = "ung"

    filename = f"{dataset}_{acorn_params}_{ung_params}_results_summary.csv"
    return filename


def find_matching_file_pairs(base_dir):
    acorn_root = os.path.join(base_dir, "ACORN")
    ung_root = os.path.join(base_dir, "UNG")
    
    file_pairs = []
    
    print("\n开始搜索匹配的ACORN和UNG文件对...")
    print(f"ACORN根目录: {acorn_root}")
    print(f"UNG根目录: {ung_root}\n")
    
    for acorn_dir, _, files in os.walk(acorn_root):
        for file in files:
            if file.endswith('_avg.csv'):
                continue
            if file.endswith('.csv'):
                acorn_path = os.path.join(acorn_dir, file)
                
                rel_path = os.path.relpath(acorn_dir, acorn_root)
                parts = rel_path.split(os.sep)
                dataset = parts[0]
                exp_name = parts[1]
            
                query_num_match = re.search(r'query(\d+)', exp_name)
                if query_num_match:
                    query_num = query_num_match.group(1)
                    ung_search_dir = os.path.join(ung_root, dataset, f"{dataset}_dataset_{dataset}_query{query_num}*")
                    
                    ung_dirs_found = list(Path(ung_search_dir).parent.glob(Path(ung_search_dir).name))
                    
                    if not ung_dirs_found:
                        print(f"⚠️ 警告: 未找到匹配的UNG目录")
                        continue
                    
                    for ung_exp_dir in ung_dirs_found:
                        ung_results_dir = os.path.join(ung_exp_dir, "results")
                        if not os.path.exists(ung_results_dir):
                            print(f"⚠️ 警告: UNG结果目录不存在: {ung_results_dir}")
                            continue
                            
                        for ung_file in os.listdir(ung_results_dir):
                            if ung_file.startswith("query_details") and ung_file.endswith(".csv"):
                                ung_path = os.path.join(ung_results_dir, ung_file)
                                print(f"✅ 找到匹配对:")
                                print(f"ACORN文件: {acorn_path}")
                                print(f"UNG文件:   {ung_path}")
                                print("-" * 80)
                                file_pairs.append((dataset, acorn_path, ung_path))
                else:
                    print(f"⚠️ 警告: 无法从路径中提取查询编号: {exp_name}")
    
    print("\n搜索完成!")
    print(f"共找到 {len(file_pairs)} 组匹配的文件对\n")
    return file_pairs


def main():
    base_dir = "/home/sunyahui/fxy/FilterVectorResults"
    output_dir = os.path.join(base_dir, "merge_results")
    os.makedirs(output_dir, exist_ok=True)
    
    file_pairs = find_matching_file_pairs(base_dir)
    
    if not file_pairs:
        print("No matching file pairs found!")
        return
    
    for dataset, acorn_file, ung_file in file_pairs:
        print(f"\nProcessing dataset: {dataset}")
        print("ACORN file:", acorn_file)
        print("UNG file:", ung_file)

        try:
            acorn_df = process_acorn_data(acorn_file)
            ung_df = process_ung_data(ung_file)
            merged_df = merge_datasets(acorn_df, ung_df)
            
            output_file = get_output_filename(acorn_file, ung_file, dataset)
            output_path = os.path.join(output_dir, output_file)
            merged_df.to_csv(output_path, index=False)
            print(f"Saved summary to {output_path}")
        except Exception as e:
            print(f"Error processing {acorn_file} and {ung_file}: {str(e)}")


if __name__ == "__main__":
    main()