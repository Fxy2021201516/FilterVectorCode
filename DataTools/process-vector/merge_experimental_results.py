# =======================================================================================
# 脚本名称: merge_experimental_results_multi.py
# 功能说明: 
#   用于批量合并指定列表 (INPUT_SOURCE_DIRS) 中的多个实验结果文件夹。
#   支持多数据集 (DATASETS) 和多算法 (ALGORITHMS) 的自动化处理。
#
# 1. 多源匹配逻辑 (Multi-Source Matching):
#    - 逻辑约定: INPUT_SOURCE_DIRS 列表的【最后一个】文件夹作为 "Base" (基准，通常为 Large)。
#      列表前面的文件夹被视为 "Patch" (补充源，如 Small, Medium)。
#    - 匹配流程: 遍历 Base 目录，解析文件夹前缀 (Index[...] _GT[...]) 和阈值 (_th)，
#      并在列表定义的所有其他源目录中寻找具有相同前缀和阈值的文件夹。
#    - 严格校验: 只有当所有源目录中都找到对应的匹配项时，才会被视为一组有效数据进行合并。
#
# 2. 输出文件夹命名规则 (Naming Convention):
#    - 基底模板: 使用 Base (Large) 文件夹的全名作为基底。
#    - 参数替换: 遍历列表前面的所有文件夹，提取特定参数 (如 `Ls`, `efsS`) 的数值。
#      若提取成功，则使用新数值覆盖 Base 模板名称中的对应部分，以体现合并后的参数范围。
#
# 3. 文件处理逻辑 (File Processing):
#    A. search_time_summary.csv (汇总表):
#       - 动作: 将列表所有源 (Small...Base) 的 DataFrame 进行拼接 (pd.concat)。
#       - 清洗: 按 `Ls` (或 `Average_Efs`) 升序排列。
#       - 去重: 若存在重复的 `Ls`，保留列表顺序中靠前的数据 (keep='first')。
#
#    B. query_details_repeat1.csv (详细日志):
#       - 动作: 将列表所有源的数据拼接。
#       - 排序: 严格按照优先级 `repeat` -> `Ls/Lsearch` -> `efs` -> `QueryID` 升序排列，
#         确保日志在合并后依然有序且连贯。
#
#    C. query_features.csv (特征文件):
#       - 动作: 仅从 Base (列表最后一个源) 文件夹复制，忽略其他源的版本。
# =======================================================================================

import pandas as pd
import os
import re
import shutil

# ================= 配置区域 =================
BASE_ROOT = "/home/fengxiaoyao/1023FilterVector/FilterVectorResults"
DATASETS = ["BookReviews"] 
ALGORITHMS = ["UNG-nTfalse", "method3", "ACORN-gamma-improved","ACORN-gamma","ACORN-1"] #"UNG-nTfalse", "method3", "ACORN-gamma-improved","ACORN-gamma","ACORN-1"

# 规则：
# 1. 列表顺序决定数据拼接顺序 (index 0 在最上面)。
# 2. 列表最后一个元素 (index -1) 被视为 Base，用于驱动遍历循环和提供特征文件。
INPUT_SOURCE_DIRS = ["small-efs","large-efs"] 

FILE_SUMMARY = "search_time_summary.csv"
FILE_DETAILS = "query_details_repeat1.csv"
FILE_FEATURES = "query_features.csv"
# ===========================================

def get_th_value(folder_name):
    """从文件夹名中提取 th 值"""
    match = re.search(r'[_]th(\d+)', folder_name)
    if match:
        return match.group(1)
    return None

def get_param_value(folder_name, param_key):
    """从文件夹名中提取指定参数的数值"""
    match = re.search(rf'{param_key}(\d+)', folder_name)
    if match:
        return match.group(1)
    return None

def generate_new_folder_name(base_name, other_names):
    """
    生成新的文件夹名
    逻辑:
    1. 以 Base 文件夹名 (列表最后一个) 为基底。
    2. 使用其他源的参数覆盖 Base。
    """
    new_name = base_name
    params_from_others = ['Ls', 'efsS']
    
    # 加上 reversed(...) 
    # 这样处于列表 index 0 的 small-efs 会是最后一次循环，从而覆盖掉 middle-efs 的值
    for other_name in reversed(other_names):
        for param in params_from_others:
            val = get_param_value(other_name, param)
            if val:
                pattern = rf'{param}\d+'
                replacement = f'{param}{val}'
                if re.search(pattern, new_name):
                    new_name = re.sub(pattern, replacement, new_name)
    return new_name

def process_search_summary(source_paths, path_out, algo_name):
    """
    处理 search_time_summary.csv
    新增参数: algo_name，用于判断去重逻辑
    """
    dfs = []
    for p in source_paths:
        if os.path.exists(p):
            try:
                dfs.append(pd.read_csv(p))
            except Exception as e:
                print(f"    [警告] 读取失败: {p} - {e}")
    
    if not dfs:
        print(f"    [跳过] Summary (无有效文件)")
        return

    try:
        # 批量合并
        df_merged = pd.concat(dfs, ignore_index=True)
        
        # --- 修改开始: 根据算法名决定去重关键字 (sort_key) ---
        sort_key = None
        
        # 如果是 ACORN 系列算法，优先寻找 Average_Efs 或 efs
        if "ACORN" in algo_name:
            priority_keys = ['Average_Efs', 'efs', 'Lsearch']
        else:
            # 其他算法 (如 UNG, method3)，维持原有优先级: Ls -> Lsearch -> Average_Efs
            priority_keys = ['Ls', 'Lsearch', 'Average_Efs']
            
        for key in priority_keys:
            if key in df_merged.columns:
                sort_key = key
                break
        # --- 修改结束 ---

        if sort_key:
            # 打印一下当前使用的去重键，方便调试
            # print(f"    [提示] 算法: {algo_name}, 使用去重键: {sort_key}")
            
            df_merged = df_merged.drop_duplicates(subset=[sort_key], keep='first')
            df_merged = df_merged.sort_values(by=sort_key, ascending=True)
        
        df_merged.to_csv(path_out, index=False)
        print(f"    [成功] Summary")
    except Exception as e:
        print(f"    [错误] 处理 Summary 失败: {e}")

def process_query_details(source_paths, path_out):
    """处理 query_details_repeat1.csv"""
    dfs = []
    for p in source_paths:
        if os.path.exists(p):
            try:
                dfs.append(pd.read_csv(p))
            except Exception as e:
                print(f"    [警告] 读取失败: {p} - {e}")
    
    if not dfs:
        print(f"    [跳过] Details (无有效文件)")
        return

    try:
        df_merged = pd.concat(dfs, ignore_index=True)
        
        # 多级排序
        possible_cols = ['repeat', 'Ls', 'Lsearch', 'efs', 'QueryID']
        existing_cols = [c for c in possible_cols if c in df_merged.columns]
        
        if existing_cols:
            df_merged = df_merged.sort_values(by=existing_cols, ascending=[True]*len(existing_cols))
        
        df_merged.to_csv(path_out, index=False)
        print(f"    [成功] Details")
    except Exception as e:
        print(f"    [错误] 处理 Details 失败: {e}")

def process_query_features(path_base, path_out):
    """只从 Base (Large) 复制特征文件"""
    if not os.path.exists(path_base):
        print(f"    [跳过] 缺少 features 文件 (Base源不存在)")
        return
    try:
        shutil.copy2(path_base, path_out)
        print(f"    [成功] Features")
    except Exception as e:
        print(f"    [错误] 复制 Features 失败: {e}")

def parse_folder_prefix(folder_name):
    """解析前缀"""
    if "_Search[" in folder_name:
        parts = folder_name.split("_Search[")
        return parts[0], True
    elif folder_name.startswith("Search["):
        return "", True
    return folder_name, False

def find_match_in_dir(base_folder_name, target_dir_path):
    """
    在指定目录中寻找与 base_folder_name 匹配的文件夹
    逻辑：前缀相同 且 th 值相同
    """
    if not os.path.exists(target_dir_path):
        return None

    prefix, is_valid = parse_folder_prefix(base_folder_name)
    if not is_valid:
        return None

    # 1. 筛选前缀匹配的候选者
    if prefix == "": 
        candidates = [f for f in os.listdir(target_dir_path) if f.startswith("Search[")]
    else:
        search_prefix = prefix + "_Search["
        candidates = [f for f in os.listdir(target_dir_path) if f.startswith(search_prefix)]
    
    if not candidates:
        return None

    # 2. 精确匹配 th 值
    base_th = get_th_value(base_folder_name)
    if base_th:
        for cand in candidates:
            if get_th_value(cand) == base_th:
                return cand
        return None 
    else:
        return candidates[0] 

def process_dataset_algo(dataset_name, algo_name):
    print(f"\n====== 正在处理: {dataset_name} | {algo_name} ======")
    
    # 1. 准备路径
    base_dir_key = INPUT_SOURCE_DIRS[-1]
    other_dir_keys = INPUT_SOURCE_DIRS[:-1]
    
    path_base_root = os.path.join(BASE_ROOT, dataset_name, "Results", base_dir_key, algo_name)
    path_output_root = os.path.join(BASE_ROOT, dataset_name, "Results", algo_name)

    if not os.path.exists(path_base_root):
        print(f"Base 路径不存在，跳过: {path_base_root}")
        return

    # 2. 遍历 Base 目录
    for base_folder in os.listdir(path_base_root):
        path_base_full = os.path.join(path_base_root, base_folder)
        
        if not os.path.isdir(path_base_full) or "results" not in os.listdir(path_base_full):
            continue
            
        # 3. 在其他所有源目录中寻找匹配
        current_group_paths = [] 
        other_folder_names = []  
        all_matched = True
        
        for key in other_dir_keys:
            search_path = os.path.join(BASE_ROOT, dataset_name, "Results", key, algo_name)
            match_name = find_match_in_dir(base_folder, search_path)
            
            if match_name:
                full_p = os.path.join(search_path, match_name)
                current_group_paths.append(os.path.join(full_p, "results"))
                other_folder_names.append(match_name)
            else:
                print(f"[跳过] 在 {key} 中未找到对应配置: {base_folder}")
                all_matched = False
                break
        
        if not all_matched:
            continue
            
        # 最后加入 Base
        current_group_paths.append(os.path.join(path_base_full, "results"))
        
        print(f"\n>>> 匹配成功组 (Base: {base_folder}):")
        print(f"  包含源: {other_dir_keys} + {base_dir_key}")

        # 4. 生成新文件夹名
        new_folder_name = generate_new_folder_name(base_folder, other_folder_names)
        
        # 5. 设定输出路径
        output_dir = os.path.join(path_output_root, new_folder_name, "results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 6. 处理文件
        process_search_summary(
            [os.path.join(p, FILE_SUMMARY) for p in current_group_paths],
            os.path.join(output_dir, FILE_SUMMARY),algo_name
        )
        
        process_query_details(
            [os.path.join(p, FILE_DETAILS) for p in current_group_paths],
            os.path.join(output_dir, FILE_DETAILS)
        )
        
        process_query_features(
            os.path.join(path_base_full, "results", FILE_FEATURES),
            os.path.join(output_dir, FILE_FEATURES)
        )

if __name__ == "__main__":
    print(f"当前合并源: {INPUT_SOURCE_DIRS} (最后一个为 Base)")
    print("开始批量处理...")
    for ds in DATASETS:
        for algo in ALGORITHMS:
            process_dataset_algo(ds, algo)
    print("\n所有任务完成。")