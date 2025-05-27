#!/bin/bash

# 参数检查
if [ "$#" -ne 7 ]; then
    echo "用法: $0 <dataset_name> <N> <M> <M_beta> <gamma> <query_num> <threads>"
    exit 1
fi

dataset=$1
N=$2
M=$3
M_beta=$4
gamma=$5
query_num=$6
threads=$7

# 清理旧构建
rm -rf build_$dataset

# 使用CMake构建
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build_$dataset
make -C build_$dataset -j faiss
make -C build_$dataset utils
make -C build_$dataset test_acorn

##########################################
# 测试配置
##########################################
now=$(date +"%Y%m%d_%H%M%S")
parent_dir="../../FilterVectorResults/ACORN/${dataset}_query_${query_num}_M${M}_gamma${gamma}_threads${threads}"
mkdir -p $parent_dir
per_query_task_dir="${parent_dir}/per_query_task_dir"
mkdir -p "$per_query_task_dir"

# 写入实验配置experiment_config.txt
config_file="${parent_dir}/experiment_config.txt"
echo "实验配置参数:" > $config_file
echo "数据集: $dataset" >> $config_file
echo "数据量(N): $N" >> $config_file
echo "M: $M" >> $config_file
echo "M_beta: $M_beta" >> $config_file
echo "gamma: $gamma" >> $config_file
echo "查询数量: $query_num" >> $config_file
echo "线程数: $threads" >> $config_file
echo "实验时间: $now" >> $config_file

##########################################
# 运行测试
##########################################
base_path="../../FilterVectorData/${dataset}" # base vector dir
for i in $(seq 1 $query_num); do
    query_path="../../FilterVectorData/${dataset}/query_${i}"
    base_label_path="../../FilterVectorData/${dataset}/base_${i}"
    
    for efs in $(seq 4 4 8); do
        for run in {1..2}; do
            csv_path="${per_query_task_dir}/${dataset}_query_${query_num}_M${M}_gamma${gamma}_threads${threads}_efs${efs}_run${run}.csv"
            avg_csv_path="${per_query_task_dir}/${dataset}_query_${query_num}_M${M}_gamma${gamma}_threads${threads}_efs${efs}_run${run}_avg.csv"
            dis_output_path="${parent_dir}/dis_output"
            
            # 仅第一次运行时生成JSON
            if [ "$efs" -eq 4 ] && [ "$run" -eq 1 ]; then
               generate_dist_output="1"
               echo "为查询 ${i} 生成JSON (仅第一次运行)"
            else
               generate_dist_output="0"
            fi
            
            echo "运行测试: 数据集=${dataset}, 查询=${i}, efs=${efs}, 运行=${run}, gamma=${gamma}, M=${M}, 线程=${threads}"
            ./build_$dataset/demos/test_acorn $N $gamma $dataset $M $M_beta $efs "$base_path" "$base_label_path" "$query_path" "$csv_path" "$avg_csv_path" "$dis_output_path" "$generate_dist_output" &>> "${parent_dir}/output_log.log"
        done
    done
done

echo "测试完成! 结果保存在: ${parent_dir}"
echo "配置文件: $config_file"