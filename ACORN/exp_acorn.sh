#!/bin/bash

# 数据集配置 (名称 N M M_beta gamma query0/1/2 thread [efs_begin efs_step efs_end] run_num)
declare -A DATASET_CONFIGS=(
    ["words"]="8000 32 64 80 1 32 [4 4 128] 5"
    ["MTG"]="40274 32 64 80 1 32 [4 4 128] 5"
   #  ["arxiv"]="157606 32 64 80 1 32 [4 4 128] 10"
)

# 运行所有数据集实验
for dataset in "${!DATASET_CONFIGS[@]}"; do
    echo "========================================"
    echo "开始处理数据集: $dataset"
    echo "参数: ${DATASET_CONFIGS[$dataset]}"
    echo "========================================"
    
    # 提取参数
    params=(${DATASET_CONFIGS[$dataset]})
    N=${params[0]}
    M=${params[1]}
    M_beta=${params[2]}
    gamma=${params[3]}
    query_num=${params[4]}
    threads=${params[5]}
    efs_begin=${params[6]//[\[\]]/} # 去掉方括号
    efs_step=${params[7]//[\[\]]/}  # 去掉方括号
    efs_end=${params[8]//[\[\]]/}    # 去掉方括号
    run_num=${params[9]//[\[\]]/}    # 去掉方括号

    # 运行测试脚本
    ./run_more_efs.sh "$dataset" "$N" "$M" "$M_beta" "$gamma" "$query_num" "$threads" "$efs_begin" "$efs_step" "$efs_end" "$run_num"

    # 执行python汇总csv文件
    parent_dir="../../FilterVectorResults/ACORN/${dataset}_query_${query_num}_M${M}_gamma${gamma}_threads${threads}/per_query_task_dir"
    input_dir=$(ls -dt $parent_dir | head -n1)
    output_dir="../../FilterVectorResults/ACORN/${dataset}_query_${query_num}_M${M}_gamma${gamma}_threads${threads}/results"
    if [ -d "$input_dir" ]; then
       python3 summarize_csv.py "$input_dir" "$output_dir" "$dataset" "$gamma" "$M" "$threads"
    else
       echo "❌ 未找到结果目录: $input_dir"
    fi
    
    echo "========================================"
    echo "完成数据集: $dataset"
    echo "========================================"
    echo ""
done

echo "所有数据集测试已完成!"