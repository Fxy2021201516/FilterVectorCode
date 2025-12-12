#!/bin/bash

set -e # 如果任何命令失败，则立即退出

# --- 定位脚本和配置文件 ---
if [ -z "$1" ]; then
    echo "错误: 请提供一个 JSON 配置文件作为第一个参数。"
    echo "用法: ./exp.sh [config_file.json]"
    exit 1
fi

CONFIG_FILE="$1"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件未找到，请检查路径: $CONFIG_FILE"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "错误: jq 未安装。请先安装 jq (https://stedolan.github.io/jq/)"
    exit 1
fi

echo "成功找到配置文件: $CONFIG_FILE"
echo "开始执行实验..."

cat "$CONFIG_FILE" | jq -c '.experiments[]' | while read -r dataset_config; do
    
    # --- 【步骤A】提取所有共享参数作为默认值 ---
    DATASET=$(echo "$dataset_config" | jq -r '.dataset_name')
    SHARED_CONFIG=$(echo "$dataset_config" | jq '.shared_config')
    
    DATA_DIR=$(echo "$SHARED_CONFIG" | jq -r '.data_dir')
    BASE_OUTPUT_DIR=$(echo "$SHARED_CONFIG" | jq -r '.output_dir')
    BUILD_MODE=$(echo "$SHARED_CONFIG" | jq -r '.build_mode')
    MAX_DEGREE=$(echo "$SHARED_CONFIG" | jq -r '.max_degree')
    LBUILD=$(echo "$SHARED_CONFIG" | jq -r '.Lbuild')
    ALPHA=$(echo "$SHARED_CONFIG" | jq -r '.alpha')
    NUM_CROSS_EDGES=$(echo "$SHARED_CONFIG" | jq -r '.num_cross_edges')
    NUM_ENTRY_POINTS=$(echo "$SHARED_CONFIG" | jq -r '.num_entry_points')
    K=$(echo "$SHARED_CONFIG" | jq -r '.K')
    LSEARCH_START=$(echo "$SHARED_CONFIG" | jq -r '.Lsearch_start')
    LSEARCH_END=$(echo "$SHARED_CONFIG" | jq -r '.Lsearch_end')
    LSEARCH_STEP=$(echo "$SHARED_CONFIG" | jq -r '.Lsearch_step')
    NUM_THREADS=$(echo "$SHARED_CONFIG" | jq -r '.num_threads')
    NUM_REPEATS=$(echo "$SHARED_CONFIG" | jq -r '.num_repeats')
    # 读取共享的ACORN构建参数
    ACORN_N=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.N')
    ACORN_M=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.M')
    ACORN_M_BETA=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.M_beta')
    ACORN_GAMMA=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.gamma')
    LSEARCH_THRESHOLD=$(echo "$SHARED_CONFIG" | jq -r '.acorn_params.lsearch_threshold')

    export UNG_BUILD_DIR="/home/fengxiaoyao/1023FilterVector/build_para_${DATASET}/ung"
    export ACORN_BUILD_DIR="/home/fengxiaoyao/1023FilterVector/build_para_${DATASET}/acorn"
    
    # --- 中层循环: 遍历任务 (查询) ---
    echo "$dataset_config" | jq -c '.tasks[]' | while read -r task; do
        QUERY_DIR_NAME=$(echo "$task" | jq -r '.query_dir_name')

        # --- 【核心修改】加载并覆盖任务专属参数 ---
        ACORN_EFS_START=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_start')
        ACORN_EFS_STEP_SLOW=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_step_slow')
        ACORN_EFS_STEP_FAST=$(echo "$task" | jq -r '.acorn_search_params.acorn_efs_step_fast')

        # 检查是否成功读取，如果为null或空，则给出错误提示（确保配置完整）
        if [[ "$ACORN_EFS_START" == "null" || -z "$ACORN_EFS_START" ]]; then
            echo "错误: 任务 '$QUERY_DIR_NAME' 缺少 'acorn_efs_start' 参数！"
            exit 1
        fi

        # --- 内层循环: 遍历算法名称 ---
        echo "$task" | jq -r '.algorithms[]' | while read -r ALGORITHM_NAME; do
            
            echo -e "\n=========================================================="
            echo "Processing: Dataset=[$DATASET], Query=[$QUERY_DIR_NAME], Algorithm=[$ALGORITHM_NAME]"
            echo "Using ACORN search params: efs_start=${ACORN_EFS_START}, efs_step_slow=${ACORN_EFS_STEP_SLOW}, efs_step_fast=${ACORN_EFS_STEP_FAST}"
            echo "=========================================================="

            # 根据算法名称设置详细参数
            case "$ALGORITHM_NAME" in
                "method1")
                    FORCE_USE_ALG=0; IS_IDEA2_AVAILABLE=false; IS_NEW_TRIE_METHOD=true; IS_REC_MORE_START=true; IS_BFS_FILTER=true;;
                "method2")
                    FORCE_USE_ALG=0; IS_IDEA2_AVAILABLE=true; IS_NEW_TRIE_METHOD=false; IS_REC_MORE_START=false; IS_BFS_FILTER=true;;
                "method3")
                    FORCE_USE_ALG=0; IS_IDEA2_AVAILABLE=true; IS_NEW_TRIE_METHOD=true; IS_REC_MORE_START=true; IS_BFS_FILTER=true;;
                "UNG-nTfalse")
                    FORCE_USE_ALG=1; IS_IDEA2_AVAILABLE=false; IS_NEW_TRIE_METHOD=false; IS_REC_MORE_START=false; IS_BFS_FILTER=true;;
                "UNG-nTtrue")
                    FORCE_USE_ALG=2; IS_IDEA2_AVAILABLE=false; IS_NEW_TRIE_METHOD=false; IS_REC_MORE_START=false; IS_BFS_FILTER=true;;
                "ACORN-gamma")
                    FORCE_USE_ALG=3; IS_IDEA2_AVAILABLE=false; IS_NEW_TRIE_METHOD=false; IS_REC_MORE_START=false; IS_BFS_FILTER=true;;
                "ACORN-1")
                    FORCE_USE_ALG=4; IS_IDEA2_AVAILABLE=false; IS_NEW_TRIE_METHOD=false; IS_REC_MORE_START=false; IS_BFS_FILTER=true;;
                "ACORN-gamma-improved") # FORCE_USE_ALG 依然是 3 (走 ACORN 逻辑)，但 IS_BFS_FILTER 设为 false
                    FORCE_USE_ALG=3; IS_IDEA2_AVAILABLE=false; IS_NEW_TRIE_METHOD=false; IS_REC_MORE_START=false; IS_BFS_FILTER=false;;
                *)
                    echo "错误: 未知的算法名称 '$ALGORITHM_NAME'。请在 exp.sh 的 case 语句中定义它。"
                    exit 1;;
            esac
            
            SHARED_DATASET_DIR="${BASE_OUTPUT_DIR}/${DATASET}"
            ALGO_RESULT_DIR="${SHARED_DATASET_DIR}/Results/${ALGORITHM_NAME}"
            
            # --- 调用 build_hybrid.sh ---
            # build_hybrid.sh 内部会负责编译、数据转换和索引构建
            echo "Preparing build index..."

            ./build_hybrid.sh \
               --build_mode "$BUILD_MODE" \
               --query_dir_name "$QUERY_DIR_NAME" \
               --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" \
               --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" \
               --num_cross_edges "$NUM_CROSS_EDGES" --num_entry_points "$NUM_ENTRY_POINTS" \
               --acorn_n "$ACORN_N" --acorn_m "$ACORN_M" --acorn_m_beta "$ACORN_M_BETA" --acorn_gamma "$ACORN_GAMMA"
            
            # 新增判断：如果 build_mode 不是 '串行'，则构建任务已完成，直接跳过 GT 生成和搜索，进入下一个实验。
            if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "acorn_only" || "$BUILD_MODE" == "ung_only" ]]; then
               echo "[INFO] Skipping GT generation and search steps."
               echo "--- The current experimental configuration processing has been completed (BUILD ONLY) ---"
               continue
            fi
            
            # --- 调用 generate_gt.sh ---
            echo "Preparing Ground Truth (K=$K)..."
            ./generate_gt.sh \
               --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" --build_dir "$UNG_BUILD_DIR" \
               --query_dir_name "$QUERY_DIR_NAME" \
               --K "$K"

            # --- 调用 search.sh ---
            INDEX_DIR_NAME="M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_AN${ACORN_N}_AM${ACORN_M}_AMB${ACORN_M_BETA}_AG${ACORN_GAMMA}"
            echo "Begin search (K=$K)..."
            ./search.sh \
               --dataset "$DATASET" --data_dir "$DATA_DIR" \
               --query_dir_name "$QUERY_DIR_NAME" \
               --shared_output_dir "$SHARED_DATASET_DIR" \
               --algo_result_dir "$ALGO_RESULT_DIR" \
               --build_dir "$UNG_BUILD_DIR" \
               --index_dir_name "$INDEX_DIR_NAME" \
               --build_mode "$BUILD_MODE" \
               --num_entry_points "$NUM_ENTRY_POINTS" \
               --Lsearch_start "$LSEARCH_START" --Lsearch_end "$LSEARCH_END" --Lsearch_step "$LSEARCH_STEP" \
               --num_threads "$NUM_THREADS" --K "$K" --num_repeats "$NUM_REPEATS" \
               --is_new_trie_method "$IS_NEW_TRIE_METHOD" --is_rec_more_start "$IS_REC_MORE_START" \
               --is_idea2_available "$IS_IDEA2_AVAILABLE" --force_use_alg "$FORCE_USE_ALG" \
               --is_bfs_filter "$IS_BFS_FILTER" \
               --efs_start "$ACORN_EFS_START" \
               --efs_step_slow "$ACORN_EFS_STEP_SLOW" --efs_step_fast "$ACORN_EFS_STEP_FAST" --lsearch_threshold "$LSEARCH_THRESHOLD"
                    
            echo "--- Finished: Dataset=[$DATASET], Query=[$QUERY_DIR_NAME], Algorithm=[$ALGORITHM_NAME] ---"
        done
    done
done

echo -e "\n所有实验已完成！"