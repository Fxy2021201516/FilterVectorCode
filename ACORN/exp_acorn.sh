#!/bin/bash

# 检查jq是否安装
if ! command -v jq &> /dev/null; then
    echo "错误: jq 未安装。请先安装jq (https://stedolan.github.io/jq/)"
    exit 1
fi

# 读取JSON配置
CONFIG_FILE="experiments.json" 

# 遍历所有实验
cat "$CONFIG_FILE" | jq -c '.experiments[]' | while read -r experiment; do
    dataset=$(echo "$experiment" | jq -r '.dataset')
    
    echo -e "============================================"
    echo "开始处理数据集: $dataset"
    echo "============================================"
    
    # 提取参数
    N=$(echo "$experiment" | jq -r '.N')
    M=$(echo "$experiment" | jq -r '.M')
    M_beta=$(echo "$experiment" | jq -r '.M_beta')
    gamma=$(echo "$experiment" | jq -r '.gamma')
    query_num=$(echo "$experiment" | jq -r '.query_num')
    threads=$(echo "$experiment" | jq -r '.threads')
    efs_list=$(echo "$experiment" | jq -r '.efs_list')
    repeat_num=$(echo "$experiment" | jq -r '.repeat_num')  
    if_bfs_filter=$(echo "$experiment" | jq -r '.if_bfs_filter')  

    # 运行测试脚本
    ./run_more_efs.sh "$dataset" "$N" "$M" "$M_beta" "$gamma" "$query_num" "$threads" "$efs_list" "$repeat_num" "$if_bfs_filter"
    
    echo "数据集 $dataset 处理完成"
    echo "============================================"
done

echo "所有数据集测试已完成!"