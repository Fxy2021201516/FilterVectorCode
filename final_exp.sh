#!/bin/bash
set -e

echo "=== 开始执行实验... ==="

# echo "$(date): [步骤 1] 运行 experiments-Reviews-1-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Reviews-1-2.json

# echo "$(date): [步骤 2] 运行 experiments-AllNews-1-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-AllNews-1-2.json

# echo "$(date): [步骤 3] 运行 experiments-Amazon-1-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Amazon-1-2.json

# echo "$(date): [步骤 4] 运行 experiments-Genome-1-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Genome-1-2.json

# echo "$(date): [步骤 5] 运行 experiments-Music-1-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Music-1-2.json

# echo "$(date): [步骤 6] 运行 experiments-Tiktok-1-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Tiktok-1-2.json

# echo "$(date): [步骤 7] 运行 experiments-VariousImg-1-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-VariousImg-1-2.json

# echo "$(date): [步骤 8] 运行 experiments-Laion-1..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Laion-1.json



echo "$(date): === 变化efs的任务 ==="

# echo "$(date): [步骤 9] 运行 experiments-AllNews-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-AllNews-2.json

# echo "$(date): [步骤 10] 运行 experiments-Amazon-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Amazon-2.json

# echo "$(date): [步骤 11] 运行 experiments-Genome-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Genome-2.json

# echo "$(date): [步骤 12] 运行 experiments-Music-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Music-2.json

echo "$(date): [步骤 13] 运行 experiments-Reviews-2..."
cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
./exp.sh experiment_json/experiments-Reviews-2.json

# echo "$(date): [步骤 14] 运行 experiments-Tiktok-2..."
# cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Tiktok-2.json

echo "$(date): [步骤 15] 运行 experiments-VariousImg-2..."
cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
./exp.sh experiment_json/experiments-VariousImg-2.json

echo "$(date): [步骤 16] 运行 experiments-Laion-2..."
cd /home/fengxiaoyao/1023FilterVector/FilterVectorCode
./exp.sh experiment_json/experiments-Laion-2.json


echo "$(date): === 所有任务执行完毕。 ==="