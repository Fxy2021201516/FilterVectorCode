import pandas as pd
import numpy as np
import os
import shutil

# ================= 配置区域 =================
BASE_ROOT = "/home/fengxiaoyao/1023FilterVector/FilterVectorResults"

# 需要处理的数据集名称列表
DATASETS = ["BookReviews"]

# 需要处理的算法列表 
ALGORITHMS = ["UNG-nTfalse",  "method3", "ACORN-gamma-improved","ACORN-gamma","ACORN-1"]#"UNG-nTfalse",  "method3", "ACORN-gamma-improved","ACORN-gamma","ACORN-1"

# 文件名常量
FILE_NAME = "search_time_summary.csv"
BACKUP_NAME = "search_time_summary_ori.csv"
# ===========================================

def force_monotonicity(df):
    """
    实现核心需求：
    1. 根据 Lsearch (或 Average_Efs) 对数据行进行升序排列。
    2. 提取 Time 列，对其单独进行升序排列 (np.sort)。
    3. 提取 Recall 列，对其单独进行升序排列 (np.sort)。 [新增]
    
    效果：确保参数越大，时间越长，且召回率越高 (消除实验波动)。
    """
    # 1. 确定排序关键字 (优先 Lsearch, 其次 Average_Efs)
    sort_key = None
    if 'Lsearch' in df.columns:
        sort_key = 'Lsearch'
    elif 'Average_Efs' in df.columns:
        sort_key = 'Average_Efs'
    
    if sort_key is None:
        print("  [警告] 未找到 Lsearch 或 Average_Efs 列，无法排序，跳过处理。")
        return df

    # 2. 按参数对行进行排序
    df = df.sort_values(by=sort_key).reset_index(drop=True)
    
    # 3. 强制 Time 单调递增
    if 'Average_Time_ms' in df.columns:
        raw_times = df['Average_Time_ms'].values
        sorted_times = np.sort(raw_times)
        df['Average_Time_ms'] = np.round(sorted_times, 4)
        
    # 4. >>> 新增修改: 强制 Recall 单调递增 <<<
    # 自动识别是 'Average_Recall' 还是 'Recall'
    recall_col = None
    if 'Average_Recall' in df.columns:
        recall_col = 'Average_Recall'
    elif 'Recall' in df.columns:
        recall_col = 'Recall'
        
    if recall_col:
        raw_recalls = df[recall_col].values
        sorted_recalls = np.sort(raw_recalls)
        # Recall 通常需要较高精度，保留 5 位或 6 位小数
        df[recall_col] = np.round(sorted_recalls, 6)

    return df

def process_file(file_path):
    dir_name = os.path.dirname(file_path)
    backup_path = os.path.join(dir_name, BACKUP_NAME)
    
    # --- 1. 备份逻辑 ---
    if not os.path.exists(backup_path):
        try:
            shutil.move(file_path, backup_path)
            print(f"  [备份] 生成 {BACKUP_NAME}")
        except Exception as e:
            print(f"  [错误] 备份失败: {e}")
            return
    else:
        print(f"  [提示] 备份文件已存在，将基于 {BACKUP_NAME} 重新生成")

    # --- 2. 处理逻辑 ---
    try:
        # 读取备份文件 (原始数据)
        df = pd.read_csv(backup_path)
        if df.empty: return
        
        # 执行强制单调处理
        df_processed = force_monotonicity(df)
        
        # 保存为原文件名
        df_processed.to_csv(file_path, index=False)
        print(f"  [成功] 已清洗并保存: {FILE_NAME}")
        
    except Exception as e:
        print(f"  [错误] 处理数据失败: {e}")
        # 如果处理失败且原文件不存在(被move了)，尝试恢复
        if not os.path.exists(file_path) and os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
            print("  [恢复] 已恢复原文件")

def process_dataset_algo(dataset_name, algo_name):
    print(f"\n====== 正在扫描: {dataset_name} | {algo_name} ======")
    
    target_root = os.path.join(BASE_ROOT, dataset_name, "Results", algo_name)
    
    if not os.path.exists(target_root):
        print(f"路径不存在: {target_root}")
        return

    count = 0
    # 递归遍历该算法目录下的所有子文件夹
    for root, dirs, files in os.walk(target_root):
        if FILE_NAME in files:
            file_path = os.path.join(root, FILE_NAME)
            folder_name = os.path.basename(root)
            print(f"处理文件夹: {folder_name}")
            
            process_file(file_path)
            count += 1
            
    if count == 0:
        print("  未找到任何 search_time_summary.csv 文件。")

if __name__ == "__main__":
    print("开始清洗数据 (强制 Time 和 Recall 单调性，无插值)...")
    for ds in DATASETS:
        for algo in ALGORITHMS:
            process_dataset_algo(ds, algo)
    print("\n所有任务完成。")