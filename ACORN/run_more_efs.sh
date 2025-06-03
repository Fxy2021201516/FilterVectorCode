#!/bin/bash

# 参数检查
if [ "$#" -ne 9 ]; then
    echo "用法: $0 <dataset_name> <N> <M> <M_beta> <gamma> <query_num> <threads> <efs_list> <repeat_num>"
    exit 1
fi

dataset=$1
N=$2
M=$3
M_beta=$4
gamma=$5
query_num=$6
threads=$7
efs_list=$8
repeat_num=$9

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
parent_dir="../../FilterVectorResults/ACORN/${dataset}_query_${query_num}_M${M}_gamma${gamma}_threads${threads}_repeat${repeat_num}"
mkdir -p $parent_dir
results="${parent_dir}/results"
mkdir -p "$results"

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

    csv_path="${results}/${dataset}_query_${query_num}_M${M}_gamma${gamma}_threads${threads}_repeat${repeat_num}.csv" 
    avg_csv_path="${results}/${dataset}_query_${query_num}_M${M}_gamma${gamma}_threads${threads}_repeat${repeat_num}_avg.csv"
    dis_output_path="${parent_dir}/dis_output"

    echo "运行测试: 数据集=${dataset}, 查询=${i}, gamma=${gamma}, M=${M}, 线程=${threads}, 重复=${repeat_num}"
    ./build_$dataset/demos/test_acorn $N $gamma $dataset $M $M_beta "$base_path" "$base_label_path" "$query_path" "$csv_path" "$avg_csv_path" "$dis_output_path" "$threads" "$repeat_num" "$efs_list"&>> "${parent_dir}/output_log.log"
done

echo "测试完成! 结果保存在: ${parent_dir}"
echo "配置文件: $config_file"