import pandas as pd
import os

def process_recall_with_pandas(input_file, output_file, adjustment_value):
    """
    使用 pandas 读取 CSV，调整 'Average_Recall' 列的值，并保存到新文件。

    参数:
    input_file (str): 输入的CSV文件名。
    output_file (str): 输出的CSV文件名。
    adjustment_value (float): 需要调整的值（正数增加，负数减小）。
    """
    try:
        # 1. 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误：找不到输入文件 '{input_file}'")
            return

        # 2. 读取CSV文件
        df = pd.read_csv(input_file)

        # 3. 检查所需列是否存在
        target_column = 'Average_Recall'
        if target_column not in df.columns:
            print(f"错误：输入文件中没有找到 '{target_column}' 列。")
            # 打印一下现有的列名，方便调试
            print(f"文件中的列名: {df.columns.tolist()}")
            return

        # 4. 确保该列是数值类型
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

        # 5. 执行核心操作：整列加上定义的值
        # 如果 adjustment_value 是正数，则增加；如果是负数，则减小。
        action = "增加" if adjustment_value >= 0 else "减小"
        print(f"正在对 '{target_column}' 列{action} {abs(adjustment_value)}...")
        
        df[target_column] = df[target_column] + adjustment_value
        
        # 格式化保留4位小数 (根据需要可以修改为 '{:.6f}' 等)
        df[target_column] = df[target_column].map('{:.4f}'.format)

        # 6. 保存到新的CSV文件
        df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"处理完成！已将结果保存到 '{output_file}'")

    except pd.errors.EmptyDataError:
        print(f"错误：输入文件 '{input_file}' 是空的。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

# --- 主程序执行 ---
if __name__ == "__main__":
    INPUT_FILENAME = '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/ACORN-1/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss100-efsf100-lt500000_K10_th10]/results/search_time_summary.csv' 
    OUTPUT_FILENAME = '/home/fengxiaoyao/1023FilterVector/FilterVectorResults/Laion/Results/ACORN-1/Index[M32_LB1000_alpha1.2_C16_EP16_AN15151002_AM32_AMB64_AG80]_GT[GT_query_select_imp_C_D-weighted-sub-base-123456789_K10]_Search[Ls50-Le60000-Lp1000_efsS10-efss100-efsf100-lt500000_K10_th10]/results/search_time_summary.csv'

    try:
        # 设置想增加或减少的值
        # 例如：想增加 0.05，就填 0.05
        #       想减少 0.02，就填 -0.02
        value_str = -0.01
        
        value = float(value_str)
        process_recall_with_pandas(INPUT_FILENAME, OUTPUT_FILENAME, value)

    except ValueError:
        print(f"错误：输入无效。请输入一个数字。")
    except FileNotFoundError:
        print(f"错误：确保 '{INPUT_FILENAME}' 文件在正确的路径下。")