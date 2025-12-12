import os

def fill_empty_labels(input_path, output_path, fill_value="6000"):
    print(f"正在处理文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"空标签填充值: {fill_value}")
    
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}")
        return

    empty_count = 0
    line_count = 0

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line_count += 1
                # 去除首尾空白字符判断是否为空
                if not line.strip():
                    # 如果是空行，写入填充值
                    f_out.write(f"{fill_value}\n")
                    empty_count += 1
                else:
                    # 如果不是空行，写入原内容
                    f_out.write(line)
                
                if line_count % 1000000 == 0:
                    print(f"已处理 {line_count} 行...")

    except Exception as e:
        print(f"发生错误: {e}")
        return

    print("-" * 30)
    print(f"处理完成！")
    print(f"总行数: {line_count}")
    print(f"修复的空行数: {empty_count}")
    print(f"新文件已保存至: {output_path}")

if __name__ == "__main__":
    # --- 配置区域 ---
    
    # 输入文件的路径
    input_file = "/home/fengxiaoyao/1023FilterVector/FilterVectorData/Laion/Laion_base_labels.txt"
    
    # 输出文件的路径
    output_file = "/home/fengxiaoyao/1023FilterVector/FilterVectorData/Laion/Laion_base_labels_fixed.txt"
    
    # 填充的标签ID
    fill_val = "6000"
    
    # --- 执行 ---
    fill_empty_labels(input_file, output_file, fill_val)