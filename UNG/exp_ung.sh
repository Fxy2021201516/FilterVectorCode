#!/bin/bash

# 检查jq是否安装
if ! command -v jq &> /dev/null; then
    echo "错误: jq 未安装。请先安装jq (https://stedolan.github.io/jq/)"
    exit 1
fi

# 读取JSON配置
CONFIG_FILE="../../FilterVectorCode/UNG/experiments.json"

# 遍历所有实验
cat "$CONFIG_FILE" | jq -c '.experiments[]' | while read -r experiment; do
    dataset=$(echo "$experiment" | jq -r '.dataset')
    
    echo -e "============================================"
    echo "开始处理数据集: $dataset"
    echo "============================================"
    
    ./run.sh \
        --dataset "$dataset" \
        --data_dir "$(echo "$experiment" | jq -r '.data_dir')" \
        --output_dir "$(echo "$experiment" | jq -r '.output_dir')" \
        --num_query_sets "$(echo "$experiment" | jq -r '.num_query_sets')" \
        --max_degree "$(echo "$experiment" | jq -r '.max_degree')" \
        --Lbuild "$(echo "$experiment" | jq -r '.Lbuild')" \
        --alpha "$(echo "$experiment" | jq -r '.alpha')" \
        --num_cross_edges "$(echo "$experiment" | jq -r '.num_cross_edges')" \
        --num_entry_points "$(echo "$experiment" | jq -r '.num_entry_points')" \
        --Lsearch_values "$(echo "$experiment" | jq -r '.Lsearch_values')" \
        --build_dir "build_$dataset" \
        --num_threads "$(echo "$experiment" | jq -r '.num_threads')" \
        --K "$(echo "$experiment" | jq -r '.K')" \
        --num_repeats "$(echo "$experiment" | jq -r '.num_repeats')" \
        --generate_query "$(echo "$experiment" | jq -r '.generate_query')" \
        --generate_query_task "$(echo "$experiment" | jq -r '.generate_query_task')" 
   
    echo "数据集 $dataset 处理完成"
    echo "============================================"
done

# echo "所有数据集处理完成，开始汇总结果..."
# python3 ./data/summarize_csv.py ../../FilterVectorResults
# echo "所有实验处理完成，结果已汇总。"
echo "============================================="