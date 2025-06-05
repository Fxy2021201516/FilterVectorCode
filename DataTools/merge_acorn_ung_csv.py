import os
import pandas as pd
import re

def find_first_above_threshold(group, recall_col, time_col, n3_col, Recall_col, sort_by):
    """ 在 group 中按照 sort_by 排序，找出 recall >= 0.97 的第一个点 """
    sorted_group = group.sort_values(by=sort_by)
    hit = sorted_group[sorted_group[recall_col] >= 0.97]
    if not hit.empty:
        return hit.iloc[0][[time_col, n3_col, Recall_col]]
    else:
        max_recall_row = sorted_group.loc[sorted_group[recall_col].idxmax()]
        return max_recall_row[[time_col, n3_col, Recall_col]]

def extract_params_from_path(filepath, keywords=None, alg=""):
    if keywords is None:
        keywords = [r'query_\d+', r'query\d+', r'gamma\d+', r'LB\d+', r'alpha[\d.]+', r'M\d+', r'C\d+', r'EP\d+']

    dirname = os.path.dirname(filepath)
    pattern = '|'.join(keywords)
    matches = []

    for part in dirname.split(os.sep):
        # 在每个目录名中查找所有匹配的关键字
        part_matches = re.findall(pattern, part)
        matches.extend(part_matches)

    # 去重，保留顺序
    unique_matches = []
    for m in matches:
        if m not in unique_matches:
            unique_matches.append(m)

    if unique_matches and alg=="acorn":
        return '_'.join(unique_matches[:3])
    if unique_matches and alg=="ung":
        return '_'.join(unique_matches[4:])    
    return "default"

def process_acorn_data(acorn_file):
    df = pd.read_csv(acorn_file)
    results = {}

    for query_id, group in df.groupby('QueryID'):
        # 按 efs 升序排序后再找首次达到 0.97 的
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
    return pd.DataFrame(results).T.reset_index(names='QueryID')

def process_ung_data(ung_file):
    df = pd.read_csv(ung_file)
    results = {}

    for query_id, group in df.groupby('QueryID'):
        # 按 Lsearch 升序排序后再找首次达到 0.97 的
        ung_time_n3_Recall = find_first_above_threshold(
            group, 'Recall', 'UNG_time(ms)', 'DistanceCalcs', 'Recall', sort_by='Lsearch'
        )

        entry_points = group['EntryPoints'].iloc[0]
        coverage = group['entry_group_total_coverage'].iloc[0]
        filter_map_time = group['bitmap_time(ms)'].iloc[0]

        results[query_id] = {
            'UNG_time': ung_time_n3_Recall['UNG_time(ms)'],
            'DistanceCalcs': ung_time_n3_Recall['DistanceCalcs'],
            'UNG_Recall': ung_time_n3_Recall['Recall'],
            'EntryPoints': entry_points,
            'Coverage': coverage,
            'bitmap_time(ms)': filter_map_time
        }

    return pd.DataFrame(results).T.reset_index(names='QueryID')

def merge_datasets(acorn_df, ung_df):
    merged_df = pd.merge(acorn_df, ung_df, on='QueryID', how='outer')

    # 加上 bitmap_time 到 ACORN 时间中
    merged_df['acorn_Time'] += merged_df['bitmap_time(ms)']
    merged_df['ACORN_1_Time'] += merged_df['bitmap_time(ms)']

    return merged_df

def get_output_filename(acorn_file, ung_file, dataset):
    acorn_params = extract_params_from_path(acorn_file,None,"acorn")
    ung_params = extract_params_from_path(ung_file,None,"ung")

    print("ACORN params:", acorn_params)
    print("UNG params:", ung_params)

    if not acorn_params:
        acorn_params = "acorn"
    if not ung_params:
        ung_params = "ung"

    filename = f"{dataset}_{acorn_params}_{ung_params}_results_summary.csv"
    return filename

def main():
    base_dir = "/data/fxy/FilterVectorResults"
    datasets = ["MTG"]

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        # 固定路径示例
        acorn_file = "/data/fxy/FilterVectorResults/ACORN/MTG/MTG_query1_M32_gamma80_threads32_repeat10/results/MTG_query_1_M32_gamma80_threads32_repeat10.csv"
        ung_file = "/data/fxy/FilterVectorResults/UNG/MTG/MTG_dataset_MTG_query1_M32_LB100_alpha1.2_C6_EP16_REPEATs5/results/query_details_repeat5.csv"

        print("ACORN file:", acorn_file)
        print("UNG file:", ung_file)

        acorn_df = process_acorn_data(acorn_file)
        ung_df = process_ung_data(ung_file)

        merged_df = merge_datasets(acorn_df, ung_df)

        output_file = get_output_filename(acorn_file, ung_file, dataset)
        output_path = os.path.join(base_dir, "merge_results", output_file)
        merged_df.to_csv(output_path, index=False)
        print(f"Saved summary to {output_path}")

if __name__ == "__main__":
    main()