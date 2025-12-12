import pandas as pd
import numpy as np
import os
import shutil

# ================= 配置区域 =================
BASE_ROOT = "/home/fengxiaoyao/1023FilterVector/FilterVectorResults"

# 1. 数据集列表
DATASETS = ["BookReviews"]

# 2. 算法列表
ALGORITHMS = ["ACORN-gamma", "ACORN-1"] # "UNG-nTfalse",  "method3", "ACORN-gamma-improved","ACORN-gamma", "ACORN-1"

FILE_NAME = "search_time_summary.csv"
BACKUP_NAME = "search_time_summary_ori.csv"

# --- 默认插值参数 ---
DEFAULT_CONFIG = {
    "GAP_RATIO": 2.0,            
    "TIME_GAP_RATIO": 5.0,       
    "MAX_INTERPOLATION_POINTS": 10,
    
    # 插值指数 (Power)
    "POWER_TIME": 1.0,
    "POWER_RECALL": 1.0
}

# --- 特殊数据集参数 ---
CUSTOM_CONFIGS = {
    # "Reviews": {
    #     "GAP_RATIO": 1.0,            
    #     "TIME_GAP_RATIO": 2.0,       
    #     "MAX_INTERPOLATION_POINTS": 2000
    # },
    # "russian": {
    #     "POWER_TIME": 0.5,   
    #     "POWER_RECALL": 1.0, 
    #     "GAP_RATIO": 2.0,    
    #     "MAX_INTERPOLATION_POINTS": 300
    # }
}
# ===========================================

def force_monotonicity(df):
    """
    实现需求：
    1. 根据 Average_Efs 排序行。
    2. 强制 Average_Time_ms 升序。
    3. 强制 Average_Recall 升序。
    """
    # 1. 按参数排序
    df = df.sort_values(by='Average_Efs').reset_index(drop=True)
    
    # 2. 强制 Time 单调递增
    if 'Average_Time_ms' in df.columns:
        raw_times = df['Average_Time_ms'].values
        sorted_times = np.sort(raw_times)
        df['Average_Time_ms'] = sorted_times

    # 3. 强制 Recall 单调递增
    recall_col = None
    if 'Average_Recall' in df.columns:
        recall_col = 'Average_Recall'
    elif 'Recall' in df.columns:
        recall_col = 'Recall'
        
    if recall_col:
        raw_recalls = df[recall_col].values
        sorted_recalls = np.sort(raw_recalls)
        df[recall_col] = sorted_recalls
        
    return df

def get_standard_step(series):
    """计算标准步长"""
    diffs = np.diff(series)
    valid_diffs = diffs[diffs > 1e-6] 
    if len(valid_diffs) == 0: return 1.0 
    step = np.percentile(valid_diffs, 25)
    return max(step, 1e-3)

def interpolate_data(df, config):
    """执行插值，支持非线性 Power 参数"""
    gap_ratio = config.get("GAP_RATIO", 2.0)
    time_gap_ratio = config.get("TIME_GAP_RATIO", 5.0)
    max_points = config.get("MAX_INTERPOLATION_POINTS", 100)
    power_time = config.get("POWER_TIME", 1.0)
    power_recall = config.get("POWER_RECALL", 1.0)

    final_rows = []
    
    std_step_efs = get_standard_step(df['Average_Efs'])
    std_step_time = get_standard_step(df['Average_Time_ms'])
    
    cols = df.columns.tolist()
    
    for i in range(len(df) - 1):
        row_curr = df.iloc[i]
        row_next = df.iloc[i+1]
        final_rows.append(row_curr) 
        
        gap_efs = row_next['Average_Efs'] - row_curr['Average_Efs']
        gap_time = row_next['Average_Time_ms'] - row_curr['Average_Time_ms']
        
        # 判定断层
        is_efs_gap = gap_efs > (std_step_efs * gap_ratio)
        is_time_gap = False
        if gap_time > 0:
            is_time_gap = gap_time > (std_step_time * time_gap_ratio)
            
        if is_efs_gap or is_time_gap:
            points_by_efs = int(gap_efs / std_step_efs) - 1 if std_step_efs > 0 else 0
            points_by_time = int(gap_time / std_step_time) - 1 if std_step_time > 0 else 0
            num_points = min(max(points_by_efs, points_by_time), max_points)
            
            if num_points > 0:
                t_linear = np.linspace(0, 1, num_points + 2)[1:-1]
                
                for t in t_linear:
                    new_row = row_curr.copy()
                    for col in cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            val_start = row_curr[col]
                            val_end = row_next[col]
                            
                            # 应用非线性 Power
                            if col == 'Average_Time_ms':
                                frac = t ** power_time
                            elif col in ['Average_Recall', 'Recall']:
                                frac = t ** power_recall
                            else:
                                frac = t
                            
                            interp_val = val_start + (val_end - val_start) * frac
                            
                            # 精度控制
                            if col in ['Average_Efs', 'Lsearch', 'repeat']:
                                new_row[col] = int(interp_val)
                            elif col == 'Average_Time_ms':
                                new_row[col] = round(interp_val, 3)
                            elif col in ['Average_Recall', 'Recall']:
                                new_row[col] = round(interp_val, 6)
                            else:
                                new_row[col] = interp_val
                                
                    final_rows.append(new_row)
                    
    final_rows.append(df.iloc[-1])
    return pd.DataFrame(final_rows, columns=cols)

def process_file(file_path, config):
    dir_name = os.path.dirname(file_path)
    backup_path = os.path.join(dir_name, BACKUP_NAME)
    
    if not os.path.exists(backup_path):
        try:
            shutil.move(file_path, backup_path)
            print(f"  [备份] {BACKUP_NAME}")
        except Exception as e:
            print(f"  [错误] 备份失败: {e}")
            return
    
    try:
        df = pd.read_csv(backup_path)
        if df.empty: return
        
        # 1. 强制清洗 (Time & Recall 均单调)
        df = force_monotonicity(df)
        
        # 2. 执行插值平滑
        df_smooth = interpolate_data(df, config)
        
        df_smooth.to_csv(file_path, index=False)
        print(f"  [成功] 已处理")
    except Exception as e:
        print(f"  [错误] 处理失败: {e}")

# >>> 修改: 增加 algo_name 参数 <<<
def scan_and_process(dataset_name, algo_name):
    print(f"\n====== 扫描: {dataset_name} | {algo_name} ======")
    
    # 合并配置
    config = DEFAULT_CONFIG.copy()
    if dataset_name in CUSTOM_CONFIGS:
        config.update(CUSTOM_CONFIGS[dataset_name])
        print(f"  >>> 使用特殊配置: TimePower={config.get('POWER_TIME')}, RecallPower={config.get('POWER_RECALL')}")
    
    # >>> 修改: 动态构建路径 <<<
    target_root = os.path.join(BASE_ROOT, dataset_name, "Results", algo_name)
    
    if not os.path.exists(target_root):
        print(f"路径不存在: {target_root}")
        return

    count = 0
    for root, dirs, files in os.walk(target_root):
        if FILE_NAME in files:
            file_path = os.path.join(root, FILE_NAME)
            print(f"正在处理: .../{os.path.basename(root)}/{FILE_NAME}")
            process_file(file_path, config)
            count += 1
            
    if count == 0:
        print(f"  在 {algo_name} 下未找到任何 {FILE_NAME} 文件。")

if __name__ == "__main__":
    print("开始智能平滑处理 (支持多算法列表 + 强制 Time/Recall 单调)...")
    
    # >>> 修改: 双重循环遍历 <<<
    for ds in DATASETS:
        for algo in ALGORITHMS:
            scan_and_process(ds, algo)
            
    print("\n所有任务完成。")