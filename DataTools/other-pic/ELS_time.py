import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import matplotlib.patches as patches

# --- 1. 全局控制参数 ---
FILTER_RANGE = (0.0, 1.0)
Y_AXIS_RANGE = (0.0, 1.0)
LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 22

# --- 8 种颜色配色方案 (统一深青色) ---
USER_PALETTE = [
    '#2b7488', '#2b7488', '#2b7488', '#2b7488', 
    '#2b7488', '#2b7488', '#2b7488', '#2b7488'
]

# --- 2. 数据加载与处理函数 (单文件模式) ---
def process_single_dataset(dataset_name, path_data_file):
    """
    直接加载一个CSV文件，计算 MinSupersetT_ms / Time_ms 的比例。
    不再进行额外的 QueryID 筛选比对。
    """
    print(f"正在处理数据集: {dataset_name} ...")
    try:
        # 1. 加载数据
        # 假设文件中必须包含计算所需的列
        cols_to_use = ['QueryID', 'MinSupersetT_ms', 'Time_ms']
        
        # 读取数据
        df_raw = pd.read_csv(path_data_file, usecols=cols_to_use)
        
        # 2. 数据聚合 (防止同一个 QueryID 有多行数据，取平均值)
        # 如果数据已经是每个 QueryID 一行，这一步不会改变数据，但保留它更安全
        df_data = df_raw.groupby('QueryID').mean()
        
        print(f"    成功加载数据，共 {len(df_data)} 个查询点。")

        # 3. 计算比例
        # 防止除以 0 导致的错误，先处理 Time_ms 为 0 的情况（如果有）
        df_data = df_data[df_data['Time_ms'] > 0]
        
        df_data['Proportion'] = df_data['MinSupersetT_ms'] / df_data['Time_ms']
        
        # 转换格式为长格式以便绘图
        df_processed = df_data[['Proportion']].copy()
        df_processed['Dataset'] = dataset_name
        
        # 4. 数据清洗
        # 过滤掉 Inf (无穷大) 和 NaN (空值)
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan).dropna(subset=['Proportion'])
        
        # 5. 范围过滤 (例如只保留 0.0 到 1.0 之间的数据)
        min_val, max_val = FILTER_RANGE
        df_to_plot = df_processed[(df_processed['Proportion'] >= min_val) & (df_processed['Proportion'] <= max_val)]
        
        # 打印过滤统计
        removed_count = len(df_processed) - len(df_to_plot)
        print(f"    范围过滤 [{min_val}, {max_val}]: 保留 {len(df_to_plot)} 个点 (剔除 {removed_count} 个异常点)")
        
        return df_to_plot

    except FileNotFoundError:
        print(f"错误：文件未找到 -> {path_data_file}")
        return pd.DataFrame()
    except KeyError as e:
        print(f"错误：文件 {dataset_name} 中缺少必要的列: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"错误：处理 {dataset_name} 时发生未知异常: {e}")
        return pd.DataFrame()
    

# --- 3. 定义数据集路径 (请在此处填入您筛选好的单一 CSV 文件路径) ---
# 格式: "数据集名称": "CSV文件绝对路径"
datasets_to_load = {
    "Genome": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Genome/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls10-Le40000-Lp1000_efsS20-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Reviews": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Reviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Amazon": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Amazon/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "VariousImg": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/VariousImg/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Music": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Music/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "BookReviews": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/BookReviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_imp_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls10-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Tiktok": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Tiktok/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
    "Laion": "/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/UNG-nTfalse/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss10-efsf10-lt500000_K10_th100]/results/query_details_repeat1.csv",
}


# --- 4. 循环加载所有数据 ---
all_data_frames = []

# 为了保证绘图顺序，我们按照 keys 的顺序处理
dataset_order = [
    "Genome", "Reviews", "Amazon", "VariousImg",
    "Music", "BookReviews", "Tiktok", "Laion"
]

for dataset_name in dataset_order:
    if dataset_name in datasets_to_load:
        path = datasets_to_load[dataset_name]
        # 去除路径中可能存在的不可见字符
        clean_path = path.replace('\u00a0', '')
        
        df_processed = process_single_dataset(dataset_name, clean_path)
        
        if not df_processed.empty:
            all_data_frames.append(df_processed)

# --- 5. 开始绘图 ---
if not all_data_frames:
    print("\n错误：没有加载到任何有效数据，无法绘图。请检查路径。")
else:
    final_df = pd.concat(all_data_frames)
    print("\n所有数据加载完成，开始绘图...")
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 3)) 
    
    # 确保 X 轴顺序与数据加载顺序一致 (或者是 final_df 中存在的)
    x_axis_order = [name for name in dataset_order if name in final_df['Dataset'].unique()]
    
    palette_colors = USER_PALETTE 
    # 确保颜色映射正确
    color_map = {name: USER_PALETTE[i % len(USER_PALETTE)] for i, name in enumerate(dataset_order)}

    # --- A. 绘制散点 (Stripplot) ---
    ax = sns.stripplot(
        data=final_df,
        x='Dataset',         
        y='Proportion',      
        order=x_axis_order,  
        hue='Dataset',       
        hue_order=x_axis_order, 
        jitter=0.2,          
        palette=color_map,   
        s=3.5,
        alpha=1.0,
        legend=False,        
        zorder=2 
    )
    
    # --- B. 绘制均值线 (Pointplot) ---
    sns.pointplot(
        data=final_df,
        x='Dataset',
        y='Proportion',
        order=x_axis_order,  
        linestyle='none',   
        markers='_',         
        markersize=15,
        linewidth=2.5,
        color='black',
        errorbar=None,
        ax=ax, 
        legend=False,
        zorder=10 
    )
    
    # --- C. 绘制极值框 (Custom Rectangles) ---
    present_datasets = set(final_df['Dataset'])
    box_width = 0.5 
    
    for i, dataset_name in enumerate(x_axis_order):
        if dataset_name in present_datasets:
            color_for_box = color_map.get(dataset_name, '#2b7488')
            
            dataset_data = final_df[final_df['Dataset'] == dataset_name]
            if dataset_data.empty:
                continue
            
            y_bottom = dataset_data['Proportion'].min()
            y_top = dataset_data['Proportion'].max()
            height = y_top - y_bottom
            
            x_left = i - (box_width / 2)
            
            rect = patches.Rectangle(
                (x_left, y_bottom),  
                box_width,           
                height,              
                linewidth=2.5,
                edgecolor=color_for_box, 
                facecolor='none',
                zorder=1 
            )
            ax.add_patch(rect)
    
    # --- 6. 格式化图表 ---
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0)) 
    ax.set_ylim(Y_AXIS_RANGE[0], Y_AXIS_RANGE[1])
    ax.set_xlabel(None) 
    ax.set_ylabel("Time Proportion", fontsize=LABEL_FONT_SIZE, color='black')
    
    ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.tick_params(axis='both', colors='black', labelsize=TICK_FONT_SIZE)
    
    plt.tight_layout()
    save_name = "ELS time.png"
    plt.savefig(save_name, dpi=150)
    print(f"绘图完成，图像已保存为 {save_name}")