#!/bin/bash

# =============================================================================
# 脚本: gene_B.sh
# 编译和运行 gene_B_label_vector.cpp
# =============================================================================

# 获取 gene_B.sh 脚本自己的绝对路径
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ACORN_SOURCE_DIR="/home/fengxiaoyao/1023FilterVector/FilterVectorCode/ACORN"

# 检查 jq 是否已安装
if ! command -v jq &> /dev/null; then
    echo "错误: 核心依赖 'jq' 未安装。" 
    exit 1
fi

# C++ 源文件
CPP_SOURCE_FILE="$SCRIPT_DIR/gene_B_label_vector.cpp" 
if [ ! -f "$CPP_SOURCE_FILE" ]; then
    echo "错误: C++ 源文件 '$CPP_SOURCE_FILE' 未找到。"
    exit 1
fi

CONFIG_FILE="$SCRIPT_DIR/gene_B.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 '$CONFIG_FILE' 不存在。"
    exit 1
fi

# 遍历JSON配置文件中的所有实验
cat "$CONFIG_FILE" | jq -c '.experiments_B[]' | while read -r experiment; do
    # --- 提取参数 ---
    dataset=$(echo "$experiment" | jq -r '.dataset')
    
    echo -e "\n============================================"
    echo "开始处理 Gene_B 任务: $dataset"
    echo "============================================"
    
    # --- 1. 编译 FAISS 和 ACORN 依赖 ---
    echo "--- [1/3] 正在编译 Faiss/ACORN 依赖... ---"

    BUILD_DIR="/home/fengxiaoyao/1023FilterVector/build_genB_$dataset"
    rm -rf "$BUILD_DIR"

    cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B "$BUILD_DIR" "$ACORN_SOURCE_DIR"
    if [ $? -ne 0 ]; then echo "CMake 失败"; exit 1; fi

    make -C "$BUILD_DIR" -j faiss
    if [ $? -ne 0 ]; then echo "make faiss 失败"; exit 1; fi

    echo "--- Faiss/ACORN 编译完成 ---"


    # --- 2. 编译 C++ 修改脚本 ---
    echo "--- [2/3] 正在编译 $CPP_SOURCE_FILE ... ---"

    FAISS_LIB_DIR="$BUILD_DIR/faiss"
    FAISS_INCLUDE_DIR="$ACORN_SOURCE_DIR" 
    EXECUTABLE_PATH="$BUILD_DIR/demos/gene_B_label_vector"
    mkdir -p "$BUILD_DIR/demos"

    g++ -std=c++17 -o "$EXECUTABLE_PATH" "$CPP_SOURCE_FILE" \
        -I"$FAISS_INCLUDE_DIR" \
        -L"$FAISS_LIB_DIR" \
        -lfaiss -fopenmp -lpthread -lstdc++fs

    if [ $? -ne 0 ]; then
        echo "错误: '$CPP_SOURCE_FILE' 编译失败。"
        exit 1
    fi
    
    echo "--- $CPP_SOURCE_FILE 编译成功 ---"


    # --- 3. 提取所有参数并执行 ---
    echo "--- [3/3] 正在执行 Gene_B 标签修改和查询生成... ---"
    
    index_path=$(echo "$experiment" | jq -r '.index_path')
    input_base_labels_A_path=$(echo "$experiment" | jq -r '.input_base_labels_A_path')
    output_base_labels_B_path=$(echo "$experiment" | jq -r '.output_base_labels_B_path')
    output_query_labels_B_path=$(echo "$experiment" | jq -r '.output_query_labels_B_path')
    output_query_vectors_B_path=$(echo "$experiment" | jq -r '.output_query_vectors_B_path')
    
    target_label_length=$(echo "$experiment" | jq -r '.target_label_length')
    num_queries_to_generate=$(echo "$experiment" | jq -r '.num_queries_to_generate')
    num_neighbors_to_label=$(echo "$experiment" | jq -r '.num_neighbors_to_label')
    num_neighbors_to_find=$(echo "$experiment" | jq -r '.num_neighbors_to_find')

    # 设置 LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$FAISS_LIB_DIR:$LD_LIBRARY_PATH
    echo "  (临时设置 LD_LIBRARY_PATH 为: $FAISS_LIB_DIR)"

    # --- 构造 C++ 命令 (9 个参数) ---
    "$EXECUTABLE_PATH" \
        "$index_path" \
        "$input_base_labels_A_path" \
        "$output_base_labels_B_path" \
        "$output_query_labels_B_path" \
        "$output_query_vectors_B_path" \
        "$target_label_length" \
        "$num_queries_to_generate" \
        "$num_neighbors_to_label" \
        "$num_neighbors_to_find"

    if [ $? -ne 0 ]; then
        echo "错误: C++ 程序执行失败。"
        exit 1
    fi

    echo -e "\n任务 ($dataset) 处理完成"
    echo "============================================"
done

echo -e "\n所有 Gene_B 实验任务已完成!"