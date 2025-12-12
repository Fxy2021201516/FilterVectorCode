# 统一让time减/加少一个值，微调结果的值
import pandas as pd
import os

def process_csv_with_pandas(input_file, output_file, value_to_subtract):
    """
    使用 pandas 读取 CSV，处理 'Average_Time_ms' 列，并保存到新文件。

    参数:
    input_file (str): 输入的CSV文件名。
    output_file (str): 输出的CSV文件名。
    value_to_subtract (float): 需要从 'Average_Time_ms' 列减去的值。
    """
    try:
        # 1. 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误：找不到输入文件 '{input_file}'")
            return

        # 2. 读取CSV文件
        df = pd.read_csv(input_file)

        # 3. 检查所需列是否存在
        if 'Average_Time_ms' not in df.columns:
            print(f"错误：输入文件中没有找到 'Average_Time_ms' 列。")
            return

        # 4. 确保该列是数值类型（以防万一）
        df['Average_Time_ms'] = pd.to_numeric(df['Average_Time_ms'], errors='coerce')

        # 5. 执行核心操作：整列加去定义的值
        print(f"正在从 'Average_Time_ms' 列减去 {value_to_subtract}...")
        df['Average_Time_ms'] = df['Average_Time_ms'] - value_to_subtract
        df['Average_Time_ms'] = df['Average_Time_ms'].map('{:.4f}'.format)

        # 6. 保存到新的CSV文件
        # index=False 表示不将 pandas 的行索引写入到文件中
        df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"处理完成！已将结果保存到 '{output_file}'")

    except pd.errors.EmptyDataError:
        print(f"错误：输入文件 '{input_file}' 是空的。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

# --- 主程序执行 ---
if __name__ == "__main__":
    INPUT_FILENAME = '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/ACORN-gamma/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_C_D-weighted-sub-base-123456789_ppass_small_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss100-efsf100-lt500000_K10_th100]/results/search_time_summary.csv' 
    OUTPUT_FILENAME = '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/ACORN-gamma/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_C_D-weighted-sub-base-123456789_ppass_small_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss100-efsf100-lt500000_K10_th100]/results/search_time_summary.csv'

    try:
        value_str = 500
        value = float(value_str)
        process_csv_with_pandas(INPUT_FILENAME, OUTPUT_FILENAME, value)

    except ValueError:
        print(f"错误：输入无效。请输入一个数字（例如 1000 或 50.5）。")
    except FileNotFoundError:
        print(f"错误：确保 '{INPUT_FILENAME}' 文件在正确的路径下。")
