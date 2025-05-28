#!/bin/bash

# Step1:解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_query_sets)
            NUM_QUERY_SETS="$2"
            shift 2
            ;;
        --max_degree)
            MAX_DEGREE="$2"
            shift 2
            ;;
        --Lbuild)
            LBUILD="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --num_cross_edges)
            NUM_CROSS_EDGES="$2"
            shift 2
            ;;
        --num_entry_points)
            NUM_ENTRY_POINTS="$2"
            shift 2
            ;;
        --Lsearch_values)
            LSEARCH_VALUES="$2"
            shift 2
            ;;
        --build_dir)
            BUILD_DIR="$2"
            shift 2
            ;;
         --num_threads)
            NUM_THREADS="$2"
            shift 2
            ;;
         --K)
            K="$2"
            shift 2
            ;;
         --num_repeats)
            NUM_REPEATS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# Step2:删除旧的构建文件夹
if [ -d "$BUILD_DIR" ]; then
    echo "删除 $BUILD_DIR 文件夹及其内容..."
    rm -rf "$BUILD_DIR"
fi

# Step3:创建构建目录并编译代码
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit
cmake -DCMAKE_BUILD_TYPE=Release ../codes/
make -j
cd .. || exit

OUTPUT_DIR="${OUTPUT_DIR}/${DATASET}_dataset_${DATASET}_query${NUM_QUERY_SETS}_M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_Lsearch${LSEARCH_VALUES}"
mkdir -p "$OUTPUT_DIR"
OTHER_DIR="$OUTPUT_DIR/others"
mkdir -p "$OTHER_DIR"
RESULT_DIR="$OUTPUT_DIR/results"
mkdir -p "$RESULT_DIR"

# Step4:转换基础数据格式
./"$BUILD_DIR"/tools/fvecs_to_bin --data_type float --input_file "$DATA_DIR/${DATASET}_base.fvecs" --output_file "$DATA_DIR/${DATASET}_base.bin"

# Step5:构建index + 生成查询任务文件
./"$BUILD_DIR"/apps/build_UNG_index \
    --data_type float --dist_fn L2 --num_threads "$NUM_THREADS" --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" --num_cross_edges "$NUM_CROSS_EDGES"\
    --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
    --base_label_file "$DATA_DIR/base_${NUM_QUERY_SETS}/${DATASET}_base_labels.txt" \
    --index_path_prefix "$OUTPUT_DIR/index_files/" \
    --result_path_prefix "$RESULT_DIR/" \
    --scenario general \
    --generate_query true --query_file_path "$DATA_DIR/query_${NUM_QUERY_SETS}" \
    --dataset "$DATASET" > "$OTHER_DIR/${DATASET}_build_index_output.txt" 2>&1

# Step6:转换查询数据格式
for ((i=1; i<=NUM_QUERY_SETS; i++))
do
    INPUT_FILE="$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_query.fvecs"
    OUTPUT_FILE="$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_query.bin"
    
    echo "Processing set $i: $INPUT_FILE -> $OUTPUT_FILE"
    ./"$BUILD_DIR"/tools/fvecs_to_bin --data_type float --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE"
done

# Step7:生成gt
for ((i=1; i<=NUM_QUERY_SETS; i++))
do
    ./"$BUILD_DIR"/tools/compute_groundtruth \
        --data_type float --dist_fn L2 --scenario containment --K "$K" --num_threads "$NUM_THREADS" \
        --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
        --base_label_file "$DATA_DIR/base_${NUM_QUERY_SETS}/${DATASET}_base_labels.txt" \
        --query_bin_file "$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_query.bin" \
        --query_label_file "$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_query_labels.txt" \
        --gt_file "$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_gt_labels_containment.bin"
    
    if [ $? -ne 0 ]; then
        echo "Error generating GT for set $i"
        exit 1
    fi
done
echo -e "All ground truth files generated successfully!"

# Step8:执行搜索
for ((i=1; i<=NUM_QUERY_SETS; i++))
do    
    echo -e "\nRunning query$i..."
    QUERY_DIR="$DATA_DIR/query_${NUM_QUERY_SETS}"

    ./"$BUILD_DIR"/apps/search_UNG_index \
        --data_type float --dist_fn L2 --num_threads "$NUM_THREADS" --K "$K" --is_new_method true --is_ori_ung true  --num_repeats "$NUM_REPEATS"\
        --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
        --base_label_file "$DATA_DIR/base_${NUM_QUERY_SETS}/${DATASET}_base_labels.txt" \
        --query_bin_file "$QUERY_DIR/${DATASET}_query.bin" \
        --query_label_file "$QUERY_DIR/${DATASET}_query_labels.txt" \
        --gt_file "$QUERY_DIR/${DATASET}_gt_labels_containment.bin" \
        --index_path_prefix "$OUTPUT_DIR/index_files/" \
        --result_path_prefix "$RESULT_DIR/" \
        --scenario containment \
        --num_entry_points "$NUM_ENTRY_POINTS" \
        --Lsearch $LSEARCH_VALUES > "$OTHER_DIR/${DATASET}_search_output.txt" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "Error in iteration $i"
        exit 1
    fi
done

echo -e "All search iterations completed successfully for dataset $DATASET!"