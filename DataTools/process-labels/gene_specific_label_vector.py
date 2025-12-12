import os
import random
import struct
import numpy as np
import sys

# ================= 配置区域 =================

# 1. 输入: Base 向量库文件路径 (.fvecs)
#    脚本需要从这里随机抽取向量
INPUT_BASE_VECTORS = '/home/fengxiaoyao/1023FilterVector/FilterVectorData/BookReviews/BookReviews_base.fvecs'

# 2. 输出: 生成的查询标签文件路径 (.txt)
OUTPUT_QUERY_LABELS = '/home/fengxiaoyao/1023FilterVector/FilterVectorData/BookReviews/query_D_23/BookReviews_query_labels.txt'

# 3. 输出: 生成的查询向量文件路径 (.fvecs)
OUTPUT_QUERY_VECTORS = '/home/fengxiaoyao/1023FilterVector/FilterVectorData/BookReviews/query_D_23/BookReviews_query.fvecs'

# 4. 核心任务配置: { '标签名': 数量 }
#    例如: {'1': 10, '100': 5} 表示生成 10 个标签为 '1' 的查询，和 5 个标签为 '100' 的查询
QUERY_TASKS = {
    '2': 50,
    '3': 50
}

# =======================================================

def read_fvecs(filename):
    """ 读取 fvecs 文件 """
    vectors = []
    d = 0
    try:
        print(f"正在读取底库: {filename} ...")
        with open(filename, 'rb') as f:
            while True:
                header = f.read(4)
                if not header: break
                d_new, = struct.unpack('i', header)
                if d == 0: d = d_new
                elif d_new != d: raise IOError("向量维度不一致")
                data = f.read(d * 4)
                if not data: break
                vectors.append(np.frombuffer(data, dtype=np.float32))
        return np.array(vectors)
    except FileNotFoundError:
        print(f"[错误] 文件未找到: {filename}")
        return None

def write_fvecs(filename, vectors):
    """ 写入 fvecs 文件 """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1: vectors = vectors.reshape(1, -1)
    n, d = vectors.shape
    with open(filename, 'wb') as f:
        for vec in vectors:
            f.write(struct.pack('i', d))
            f.write(vec.tobytes())

def main():
    # 1. 加载底库
    base_vectors = read_fvecs(INPUT_BASE_VECTORS)
    if base_vectors is None:
        return
    
    ntotal, dim = base_vectors.shape
    print(f"底库加载完毕: 共 {ntotal} 个向量, 维度 {dim}")

    # 2. 生成任务列表
    final_labels = []
    sampled_indices = []

    print("正在生成任务列表...")
    for label, count in QUERY_TASKS.items():
        print(f"  - 生成标签 '{label}': {count} 个")
        for _ in range(count):
            final_labels.append(label)
            # 核心逻辑: 随机生成一个索引
            rand_idx = random.randint(0, ntotal - 1)
            sampled_indices.append(rand_idx)

    if not final_labels:
        print("[警告] 没有配置任何查询任务，程序结束。")
        return

    # 3. 提取向量 (利用 numpy 的切片功能一次性提取)
    print("正在提取对应的向量...")
    query_vectors = base_vectors[sampled_indices]

    # 4. 保存结果
    # 确保目录存在
    os.makedirs(os.path.dirname(OUTPUT_QUERY_LABELS), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_QUERY_VECTORS), exist_ok=True)

    # 保存标签
    try:
        with open(OUTPUT_QUERY_LABELS, 'w', encoding='utf-8') as f:
            for lbl in final_labels:
                f.write(f"{lbl}\n")
        print(f"成功保存标签至: {OUTPUT_QUERY_LABELS}")
    except Exception as e:
        print(f"[错误] 保存标签失败: {e}")

    # 保存向量
    try:
        write_fvecs(OUTPUT_QUERY_VECTORS, query_vectors)
        print(f"成功保存向量至: {OUTPUT_QUERY_VECTORS}")
    except Exception as e:
        print(f"[错误] 保存向量失败: {e}")

    print("\n--- 全部完成 ---")

if __name__ == "__main__":
    main()