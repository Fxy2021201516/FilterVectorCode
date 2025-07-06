# 这份代码将扁平的标签处理成深LNG的标签
import collections
import logging
from typing import List, Set, Dict, Any, Optional, Tuple
import multiprocessing
from tqdm import tqdm

# --- 1. 参数设置 ---
MIN_SUPPORT_COUNT = 10
MIN_PATH_DEPTH = 3
COVERAGE_TARGET = 0.99
BATCH_SIZE = 1000 

# --- 日志系统设置 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler('processing_log.txt', mode='w')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# --- 2. 数据加载与预处理  ---
def load_labels(filepath: str) -> List[Set[int]]:
    logger.info(f"开始从文件加载数据: {filepath}")
    all_labels = []
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                all_labels.append(set(map(int, line.split(','))))
                if (i + 1) % 100000 == 0: logger.info(f"已加载 {i + 1} 行...")
        logger.info(f"数据加载完成，共 {len(all_labels)} 行。")
    except FileNotFoundError:
        logger.error(f"错误：输入文件未找到 at {filepath}"); return []
    return all_labels

def build_inverted_index(all_data: List[Set[int]]) -> Dict[int, Set[int]]:
    logger.info("正在构建倒排索引，请耐心等待...")
    inverted_index = collections.defaultdict(set)
    for i, labels in enumerate(all_data):
        for label in labels:
            inverted_index[label].add(i)
    logger.info(f"倒排索引构建完成。索引了 {len(inverted_index)} 个独立标签。")
    return inverted_index

# --- 3. 阶段一：构建“主干树”森林 ---
TreeNode = Dict[str, Any]
def build_tree_recursive(parent_path: List[int], parent_row_indices: Set[int], inverted_index: Dict[int, Set[int]], original_data: List[Set[int]]) -> List[TreeNode]:
    logger.debug(f"递归构建子树，当前路径: {parent_path}, 数据子集大小: {len(parent_row_indices)}")
    if not parent_row_indices: return []
    counter = collections.Counter()
    parent_path_set = set(parent_path)
    for row_idx in parent_row_indices:
        counter.update(original_data[row_idx] - parent_path_set)
    strong_children_labels = [label for label, count in counter.items() if count >= MIN_SUPPORT_COUNT]
    if strong_children_labels:
        nodes = []
        for child_label in strong_children_labels:
            child_row_indices = parent_row_indices.intersection(inverted_index.get(child_label, set()))
            children = build_tree_recursive(parent_path + [child_label], child_row_indices, inverted_index, original_data)
            nodes.append({'label': child_label, 'children': children})
        return nodes
    leaf_nodes = [{'label': label, 'children': []} for label, count in counter.most_common() if count > 0]
    return leaf_nodes

def find_single_theme_tree(data_indices: Set[int], inverted_index: Dict[int, Set[int]], original_data: List[Set[int]]) -> Optional[TreeNode]:
    if not data_indices: return None
    logger.info("在当前数据子集上寻找最佳根节点...")
    sub_counter = collections.Counter(label for idx in data_indices for label in original_data[idx])
    best_root_candidate = None
    for label, count in sub_counter.most_common():
        if count >= MIN_SUPPORT_COUNT:
            best_root_candidate = (label, count); break
    if not best_root_candidate:
        logger.warning(f"数据池中已没有任何标签满足最小支持度 {MIN_SUPPORT_COUNT}。"); return None
    root_label, root_count = best_root_candidate
    logger.info(f"已选定本轮候选根: {root_label} (支持度: {root_count})")
    root_row_indices = data_indices.intersection(inverted_index.get(root_label, set()))
    children = build_tree_recursive([root_label], root_row_indices, inverted_index, original_data)
    tree = {'label': root_label, 'children': children}
    def get_max_depth(node: TreeNode) -> int:
        if not node.get('children'): return 1
        return 1 + max([get_max_depth(child) for child in node['children']], default=0)
    max_depth = get_max_depth(tree)
    logger.info(f"为根 {root_label} 构建的树，最大深度为: {max_depth}")
    if max_depth >= MIN_PATH_DEPTH:
        return tree
    else:
        logger.warning(f"根 {root_label} 的树不满足最小深度要求 {MIN_PATH_DEPTH}。"); return None

def build_theme_forest(all_data: List[Set[int]], inverted_index: Dict[int, Set[int]]) -> List[TreeNode]:
    logger.info("\n==================================================")
    logger.info("--- 阶段一：开始构建主干树森林 ---")
    forest = []
    remaining_indices = set(range(len(all_data)))
    total_data_count = len(all_data)
    iteration_count = 1
    while True:
        logger.info(f"\n--- 森林构建迭代轮次: {iteration_count} ---")
        if not remaining_indices: logger.info("所有数据已处理完毕。"); break
        processed_count = total_data_count - len(remaining_indices)
        coverage_ratio = processed_count / total_data_count if total_data_count > 0 else 0
        if coverage_ratio >= COVERAGE_TARGET:
            logger.info(f"数据覆盖率 {coverage_ratio:.2%} 已达到目标，停止挖掘。"); break
        logger.info(f"当前剩余数据: {len(remaining_indices)} 行。已处理覆盖率: {coverage_ratio:.2%}")
        new_tree = find_single_theme_tree(remaining_indices, inverted_index, all_data)
        if not new_tree: logger.warning("在剩余数据中已无法找到新树，挖掘结束。"); break
        forest.append(new_tree)
        root_label = new_tree['label']
        covered_indices = remaining_indices.intersection(inverted_index.get(root_label, set()))
        remaining_indices -= covered_indices
        iteration_count += 1
    logger.info(f"\n--- 阶段一结束：共发现 {len(forest)} 棵主干树 ---")
    return forest


# --- 4. 阶段二：匹配与重构 ---

def find_deepest_path_in_tree_recursive(node: TreeNode, original_labels: Set[int]) -> Optional[List[int]]:
    if node['label'] not in original_labels: return None
    if not node['children']: return [node['label']]
    best_child_path = []
    for child in node['children']:
        path_from_child = find_deepest_path_in_tree_recursive(child, original_labels)
        if path_from_child and len(path_from_child) > len(best_child_path):
            best_child_path = path_from_child
    return [node['label']] + best_child_path

# 并行工作函数现在会返回结果和统计信息
def process_batch(batch_data: List[Tuple[int, Set[int]]]) -> Tuple[List[Tuple[int, List[int]]], Dict]:
    batch_results = []
    batch_stats = {
        "unmatched_count": 0,
        "path_length_dist": collections.Counter(),
        "tree_assignment_dist": collections.Counter()
    }
    
    for original_index, original_labels in batch_data:
        best_path_overall = []
        best_tree_index = -1
        for tree_idx, tree in enumerate(forest): # forest 是全局变量
            path = find_deepest_path_in_tree_recursive(tree, original_labels)
            if path and len(path) > len(best_path_overall):
                best_path_overall = path
                best_tree_index = tree_idx
        
        if best_path_overall:
            batch_results.append((original_index, best_path_overall))
            batch_stats["path_length_dist"][len(best_path_overall)] += 1
            batch_stats["tree_assignment_dist"][best_tree_index] += 1
        else:
            batch_results.append((original_index, list(original_labels)))
            batch_stats["unmatched_count"] += 1
            batch_stats["path_length_dist"][len(original_labels)] += 1
            
    return batch_results, batch_stats

def reconstruct_and_save_labels_parallel(original_data: List[Set[int]], forest: List[TreeNode], output_filepath: str, num_workers: int):
    logger.info("\n==================================================")
    logger.info(f"--- 阶段二：开始并行匹配与重构（使用 {num_workers} 个核心） ---")

    # 将数据和原始行号打包分批
    indexed_data = list(enumerate(original_data))
    batches = [indexed_data[i:i + BATCH_SIZE] for i in range(0, len(indexed_data), BATCH_SIZE)]
    
    final_results_with_indices = []
    # 初始化总统计信息
    final_stats = {
        "unmatched_count": 0,
        "path_length_dist": collections.Counter(),
        "tree_assignment_dist": collections.Counter()
    }

    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=len(batches), desc="重构标签进度") as pbar:
            # imap_unordered可以在每个批次完成后立即处理结果
            for result_batch, stats_batch in pool.imap_unordered(process_batch, batches):
                final_results_with_indices.extend(result_batch)
                # 聚合每个批次返回的统计信息
                final_stats["unmatched_count"] += stats_batch["unmatched_count"]
                final_stats["path_length_dist"].update(stats_batch["path_length_dist"])
                final_stats["tree_assignment_dist"].update(stats_batch["tree_assignment_dist"])
                pbar.update(1)

    logger.info("所有并行任务处理完成。正在整理和保存结果...")
    
    # 根据原始行号排序，确保输出文件顺序与输入一致
    final_results_with_indices.sort(key=lambda x: x[0])
    final_labels = [labels for index, labels in final_results_with_indices]

    with open(output_filepath, 'w') as f:
        for labels in final_labels:
            f.write(','.join(map(str, labels)) + '\n')
    logger.info(f"文件写入完成: {output_filepath}")
    
    return final_labels, final_stats


# --- 5. 统计报告模块  ---
def analyze_tree_structure(node: TreeNode, level: int, level_dist: collections.Counter) -> Tuple[int, int]:
    level_dist[level] += 1
    if not node.get('children'): return 1, level
    total_nodes, max_depth = 1, level
    for child in node['children']:
        child_nodes, child_depth = analyze_tree_structure(child, level + 1, level_dist)
        total_nodes += child_nodes
        max_depth = max(max_depth, child_depth)
    return total_nodes, max_depth

def log_statistics(forest: List[TreeNode], final_labels: List[List[int]], stats: Dict):
    logger.info("\n==================================================")
    logger.info("--- 最终统计报告 ---")
    
    # 1. 森林统计
    logger.info("\n[森林结构统计]")
    if not forest: logger.info("未发现任何主干树。"); return
    logger.info(f"共发现 {len(forest)} 棵主干树。")
    for i, tree in enumerate(forest):
        level_dist = collections.Counter()
        total_nodes, max_depth = analyze_tree_structure(tree, 1, level_dist)
        logger.info(f"\n--- 树 {i+1} (根: {tree['label']}) ---")
        logger.info(f"  - 最大深度: {max_depth}")
        logger.info(f"  - 节点总数: {total_nodes}")
        logger.info("  - 每层节点数分布:")
        for level in sorted(level_dist.keys()):
            logger.info(f"    - 层 {level}: {level_dist[level]} 个节点")

    # 2. 重构结果统计
    total_count = len(final_labels)
    matched_count = total_count - stats['unmatched_count']
    logger.info("\n[重构结果统计]")
    logger.info(f"总处理向量数: {total_count}")
    logger.info(f"成功匹配到主干树的数量: {matched_count} ({matched_count/total_count:.2%})")
    logger.info(f"未匹配（保留原始标签）的数量: {stats['unmatched_count']} ({stats['unmatched_count']/total_count:.2%})")

    logger.info("\n[各主干树覆盖的向量数]")
    if not stats['tree_assignment_dist']:
        logger.info("无数据匹配到任何树。")
    else:
        for tree_idx in sorted(stats['tree_assignment_dist'].keys()):
            count = stats['tree_assignment_dist'][tree_idx]
            root_label = forest[tree_idx]['label']
            logger.info(f"  - 树 {tree_idx+1} (根: {root_label}): 覆盖 {count} 个向量 ({count/total_count:.2%})")

    logger.info("\n[最终标签长度分布]")
    if not stats['path_length_dist']:
        logger.info("无有效数据。")
    else:
        total_len_sum = sum(k * v for k, v in stats['path_length_dist'].items())
        avg_len = total_len_sum / total_count if total_count > 0 else 0
        logger.info(f"  - 平均标签长度: {avg_len:.2f}")
        for length in sorted(stats['path_length_dist'].keys()):
            count = stats['path_length_dist'][length]
            logger.info(f"  - 长度为 {length} 的向量: {count} 个 ({count/total_count:.2%})")

# --- 主程序 ---
if __name__ == '__main__':
    INPUT_FILE = '/home/sunyahui/fxy/Data/data/app_reviews/app_reviews/app_reviews_base_labels.txt'
    OUTPUT_FILE = 'hierarchical_labels.txt'

    original_data = load_labels(INPUT_FILE)

    if not original_data:
        logger.critical("输入数据为空或文件不存在，程序退出。")
    else:
        inverted_index = build_inverted_index(original_data)
        
        # 为了让并行进程能访问到 forest，将其设为全局变量
        global forest
        forest = build_theme_forest(original_data, inverted_index)
        
        if forest:
            # 计算要使用的CPU核心数，取总数的三分之一，且至少为1
            total_cpus = multiprocessing.cpu_count()
            num_workers = max(1, total_cpus // 3)
            logger.info(f"总CPU核心数: {total_cpus}，本次任务将使用: {num_workers} 个核心。")

            # 执行并行化的阶段二
            reconstructed_labels, stats = reconstruct_and_save_labels_parallel(
                original_data, forest, OUTPUT_FILE, num_workers
            )
            
            # 使用聚合后的统计信息打印报告
            log_statistics(forest, reconstructed_labels, stats)

            logger.info("\n==================================================")
            logger.info("所有任务已成功完成！")
        else:
            logger.warning("未能发现任何主干树。程序退出，不会生成输出文件。")
            logger.info("\n==================================================")
            logger.info("处理结束。")