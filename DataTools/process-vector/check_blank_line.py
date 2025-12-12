def count_empty_lines(file_path):
    """
    统计文件中空行的数量（包括只含空白字符的行）。
    
    Args:
        file_path (str): 要检查的文件路径。
    
    Returns:
        int: 空行的数量。
    """
    empty_line_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():  # 如果去除两端空白后为空，则视为空行
                    empty_line_count += 1
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

    print(f"文件 '{file_path}' 中共有 {empty_line_count} 个空行。")
    return empty_line_count

# 示例用法
if __name__ == '__main__':
    file_path = '/home/fengxiaoyao/1023FilterVector/FilterVectorData/bigann/bigann_base_labels_ori.txt'
    count_empty_lines(file_path)