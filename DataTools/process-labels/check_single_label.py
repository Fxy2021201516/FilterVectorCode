import os

def check_single_label_vectors(file_path):
    single_label_count = 0
    total_vectors = 0
    
    print(f"正在检查文件: {file_path} ...")
    print("-" * 40)

    if not os.path.exists(file_path):
        print(f"错误：找不到文件 '{file_path}'，请确认路径是否正确。")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                
                # 跳过空行
                if not line:
                    continue
                
                total_vectors += 1
                
                # 使用逗号分割
                parts = line.split(',')
                
                # 过滤掉可能因为末尾逗号产生的空字符串（例如 "1,"）
                labels = [p for p in parts if p.strip()]
                
                # 判断标签数量
                if len(labels) == 1:
                    print(f"[发现单标签] 行号: {i}, 内容: {line}")
                    single_label_count += 1

        print("-" * 40)
        print(f"检查完成！")
        print(f"总向量数: {total_vectors}")
        print(f"单标签向量数: {single_label_count}")
        
        if single_label_count > 0:
            print("结论: \033[91m存在\033[0m 单标签向量。") # 红色高亮
        else:
            print("结论: \033[92m不存在\033[0m 单标签向量。") # 绿色高亮

    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    # ==========================================
    input_filename = "/home/fengxiaoyao/1023FilterVector/FilterVectorData/Laion/Laion_base_labels.txt" 
    # ==========================================
    
    check_single_label_vectors(input_filename)