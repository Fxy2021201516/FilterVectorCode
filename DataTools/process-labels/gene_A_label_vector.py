import os
import random
import struct
import numpy as np
import sys # 用于在错误时退出

# --- 1. 主配置 (开关) ---
RUN_TASK_MODIFY_BASE = True    # 是否执行 [更新base向量]
RUN_TASK_GENERATE_QUERIES = True # 是否执行 [生成对应 query task]


# --- 2. 配置区域 A (修改 Base 标签) ---

if RUN_TASK_MODIFY_BASE:
    BASE_INPUT_LABELS = '/home/fengxiaoyao/1023FilterVector/FilterVectorData/Laion/Laion_base_labels_reorder_ori.txt'
    BASE_OUTPUT_LABELS = '/home/fengxiaoyao/1023FilterVector/FilterVectorData/Laion/Laion_A_base_labels.txt'
    ATTRIBUTES_TO_ADD_CONFIG = {
        1: 0.8
    }

# --- 3. 配置区域 B (生成查询任务) ---
if RUN_TASK_GENERATE_QUERIES:
    # 输入: Base 向量文件 (用于从中抽样)
    BASE_VECTORS_FILE = '/home/fengxiaoyao/1023FilterVector/FilterVectorData/Laion/Laion_base.fvecs' 
    
    # 输出: 新生成的查询标签文件
    OUTPUT_QUERY_LABELS = '/home/fengxiaoyao/1023FilterVector/FilterVectorData/Laion/query_A_12/Laion_query_labels.txt'
    
    # 输出: 新生成的查询向量文件
    OUTPUT_QUERY_VECTORS = '/home/fengxiaoyao/1023FilterVector/FilterVectorData/Laion/query_A_12/Laion_query.fvecs'
    
    # 查询组合 (标签: 数量)
    QUERY_COMPOSITION = {
        '1': 10
    }


# --- 4. Fvecs 辅助函数 (用于功能 B) ---

def read_fvecs(filename):
    """ 读取 fvecs 文件并返回一个 numpy 数组 """
    vectors = []
    d = 0
    try:
        with open(filename, 'rb') as f:
            while True:
                header = f.read(4)
                if not header:
                    break
                d_new, = struct.unpack('i', header)
                if d == 0:
                    d = d_new
                elif d_new != d:
                    raise IOError(f"Inconsistent vector dimensions: expected {d}, got {d_new}")
                
                data = f.read(d * 4) # 4 bytes per float
                if not data:
                    break
                
                vectors.append(np.frombuffer(data, dtype=np.float32))
    except FileNotFoundError:
        print(f"\n[错误]：Fvecs 文件未找到: {filename}")
        return None
    except Exception as e:
        print(f"\n[错误]：读取 Fvecs 文件时出错: {e}")
        return None
        
    if not vectors:
        print(f"\n[警告]：Fvecs 文件为空或格式错误: {filename}")
        return None

    return np.array(vectors)

def write_fvecs(filename, vectors):
    """ 将 numpy 数组写入 fvecs 文件 """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
        
    n, d = vectors.shape
    
    with open(filename, 'wb') as f:
        for vec in vectors:
            f.write(struct.pack('i', d)) # 写入维度 (int)
            f.write(vec.tobytes())       # 写入向量数据 (float32 * d)

# --- 5. 功能 A: 修改 Base 标签 ---

def modify_base_labels():
    print(f"\n--- [功能 A] 开始: 修改 Base 标签 ---")
    print(f"读取文件: {BASE_INPUT_LABELS}")
    print(f"将按以下配置随机添加属性:")
    for attr, percent in ATTRIBUTES_TO_ADD_CONFIG.items():
        print(f"  - 属性 {attr} 将被添加到约 {percent*100:.0f}% 的行中")

    attr_newly_added_counters = {attr: 0 for attr in ATTRIBUTES_TO_ADD_CONFIG}
    attr_intent_counters = {attr: 0 for attr in ATTRIBUTES_TO_ADD_CONFIG}
    total_lines_processed = 0

    try:
        with open(BASE_INPUT_LABELS, 'r', encoding='utf-8') as infile, \
             open(BASE_OUTPUT_LABELS, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                total_lines_processed += 1
                cleaned_line = line.strip()
                
                if not cleaned_line:
                    outfile.write(line)
                    continue
                
                parts = cleaned_line.split(',')
                
                try:
                    existing_attributes = {int(p) for p in parts if p.strip()}
                except ValueError as e:
                    print(f"\n[警告] A: 在第 {total_lines_processed} 行发现非数字属性，已跳过此行。错误: {e}")
                    outfile.write(line) 
                    continue

                for attr, percentage in ATTRIBUTES_TO_ADD_CONFIG.items():
                    if random.random() < percentage:
                        attr_intent_counters[attr] += 1
                        if attr not in existing_attributes:
                            existing_attributes.add(attr)
                            attr_newly_added_counters[attr] += 1
                
                sorted_attributes = sorted(list(existing_attributes))
                attributes_str_list = [str(attr) for attr in sorted_attributes]
                new_line = ','.join(attributes_str_list) 
                
                outfile.write(new_line + '\n')

        print(f"\n--- [功能 A] 处理完成 ---")
        print(f"总共处理了 {total_lines_processed} 行。")
        print(f"新文件已保存为: {BASE_OUTPUT_LABELS}")
        print("\n实际添加统计:")
        
        for attr, intent_count in attr_intent_counters.items():
            percent_intent = (intent_count / total_lines_processed) * 100 if total_lines_processed > 0 else 0
            newly_added_count = attr_newly_added_counters[attr]
            
            print(f"  - 属性 {attr}:")
            print(f"    - 尝试添加 (命中概率): {intent_count} 行 (占 {percent_intent:.2f}%)")
            print(f"    - 实际新添加 (原先没有): {newly_added_count} 行")
        
        return True

    except FileNotFoundError:
        print(f"\n[错误] A: 找不到输入文件 '{BASE_INPUT_LABELS}'。")
        return False
    except Exception as e:
        print(f"\n[发生意外错误] A: {e}")
        return False

# --- 6. 功能 B: 生成查询任务 ---

def generate_queries():
    print(f"\n--- [功能 B] 开始: 生成查询任务 ---")
    
    # --- 步骤 B1: 加载 Base 向量 ---
    print(f"正在加载 Base 向量: {BASE_VECTORS_FILE} ...")
    base_vectors = read_fvecs(BASE_VECTORS_FILE)
    
    if base_vectors is None:
        print("[错误] B: 无法加载 Base 向量，功能 B 中止。")
        return False
        
    ntotal, dimension = base_vectors.shape
    print(f"加载成功。共 {ntotal} 个向量, 维度 {dimension}。")

    # --- 步骤 B2: 生成标签列表和抽样索引 ---
    print("正在生成查询标签并随机抽样...")
    final_query_labels = []
    sampled_indices = []
    
    total_queries = 0
    for label, count in QUERY_COMPOSITION.items():
        print(f"  - 生成 {count} 个带标签 '{label}' 的查询...")
        for _ in range(count):
            final_query_labels.append(label)
            random_idx = random.randint(0, ntotal - 1)
            sampled_indices.append(random_idx)
        total_queries += count
    
    print(f"共生成 {total_queries} 个查询。")
    if total_queries != len(final_query_labels) or total_queries != len(sampled_indices):
        print("[警告] B: 生成的查询数量与索引数量不匹配!")
        
    # --- 步骤 B3: 提取查询向量 ---
    print("正在从 Base 向量中提取查询向量...")
    final_query_vectors = base_vectors[sampled_indices]

        # --- 步骤 B4: 保存查询标签文件 (.txt) ---
    print(f"正在保存查询标签到: {OUTPUT_QUERY_LABELS} ...")
    try:
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(OUTPUT_QUERY_LABELS), exist_ok=True)
        with open(OUTPUT_QUERY_LABELS, 'w', encoding='utf-8') as f:
            for label in final_query_labels:
                f.write(label + '\n')
        print("  - 标签保存成功。")
    except Exception as e:
        print(f"  - [错误] B: 保存标签失败: {e}")
        return False

    # --- 步骤 B5: 保存查询向量文件 (.fvecs) ---
    print(f"正在保存查询向量到: {OUTPUT_QUERY_VECTORS} ...")
    try:
        # 同样确保目录存在
        os.makedirs(os.path.dirname(OUTPUT_QUERY_VECTORS), exist_ok=True)
        write_fvecs(OUTPUT_QUERY_VECTORS, final_query_vectors)
        print("  - 向量保存成功。")
    except Exception as e:
        print(f"  - [错误] B: 保存向量失败: {e}")
        return False
    
    return True 

# --- 7. 主执行函数 ---

def main():
    print("--- 启动 Python 脚本 ---")
    
    if not RUN_TASK_MODIFY_BASE and not RUN_TASK_GENERATE_QUERIES:
        print("两个功能都已关闭，脚本退出。")
        return

    if RUN_TASK_MODIFY_BASE:
        success_A = modify_base_labels()
        if not success_A:
            print("\n[紧急] 功能 A 失败，请检查错误信息。")
            sys.exit(1) # 如果希望功能 A 失败时停止脚本，取消此行注释
            
    if RUN_TASK_GENERATE_QUERIES:
        success_B = generate_queries()
        if not success_B:
            print("\n[紧急] 功能 B 失败，请检查错误信息。")
            
    print("\n--- 所有任务执行完毕 ---")

if __name__ == "__main__":
    main()