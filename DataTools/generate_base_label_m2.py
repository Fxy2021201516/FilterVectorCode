from collections import deque, defaultdict
import random
from typing import Optional, List, Set, Dict, Tuple, Union, FrozenSet

class LabelSetNode:
    def __init__(self, label_set):
        self.label_set = frozenset(label_set)
        self.children = []
        self.vectors = []

    def add_child(self, child_node):
        self.children.append(child_node)

class LNGVectorGenerator:
    def __init__(self, max_depth=10, branch_factor=2, max_groups=None, base_attr_start=1):
        """
        构建一个层级化标签集合树（LNG）
        :param max_depth: 最大深度
        :param branch_factor: 每个节点下最多有几个子节点
        :param max_groups: 最多允许生成的 group 数量（None 表示不限制）
        :param base_attr_start: 初始属性编号
        """
        self.max_depth: int = max_depth
        self.branch_factor: int = branch_factor
        self.max_groups: Optional[int] = max_groups
        self.base_attr_start: int = base_attr_start
        self.root = None
        self.nodes_by_depth = defaultdict(list)  # depth -> [nodes]
        self.vector_to_labels = {}  # vector_id -> label_set
        self.label_set_to_vectors = defaultdict(list)  # label_set -> [vector_ids]

    def build(self, total_vectors: int):
         """ 构建整个 LNG 树，并分配指定数量的向量 """
         if self.max_groups is not None and total_vectors < self.max_groups:
            raise ValueError("total_vectors 必须 >= max_groups")

         self.total_vectors = total_vectors
         self.generated_groups = 0  # 初始化计数器

         root_label = [self.base_attr_start]
         self.root = LabelSetNode(root_label)
         self._build_recursive(self.root, depth=1)

         all_nodes = self._get_all_nodes()
         num_groups = len(all_nodes)

         print(f"Generated {num_groups} groups with max_depth={self.max_depth}, branch_factor={self.branch_factor}")

         # 分配向量
         self._assign_vectors_evenly(num_groups)

         # 构建映射关系
         self._build_vector_label_mappings()

    def _build_recursive(self, node, depth):
         """ 递归构建 LNG 树 """
         self.nodes_by_depth[depth].append(node)

         if depth >= self.max_depth:
            return

         current_attrs = list(node.label_set)
         next_attr_base = max(current_attrs) + 1

         # 全局 group 计数器
         if not hasattr(self, "generated_groups"):
            self.generated_groups = 0

         # 控制分支数量
         actual_branch_factor = 0
         for i in range(self.branch_factor):
            if self.max_groups is not None and self.generated_groups >= self.max_groups-1:
                  break  # 达到最大组数，停止生成子节点

            new_attr = next_attr_base + i
            new_label_set = current_attrs + [new_attr]
            child_node = LabelSetNode(new_label_set)
            node.add_child(child_node)
            self.generated_groups += 1
            self._build_recursive(child_node, depth + 1)
            actual_branch_factor += 1

    def _get_all_nodes(self):
        """ 获取所有节点用于遍历 """
        result = []
        q = deque([self.root])
        while q:
            node = q.popleft()
            result.append(node)
            if node is not None:
                q.extend(node.children)
        return result

    def _assign_vectors_evenly(self, num_groups):
         """ 按层级加权分配向量，上层获得更多向量 """

         all_nodes = self._get_all_nodes()
         assert len(all_nodes) <= num_groups, f"实际生成了 {len(all_nodes)} group，超过了预期 {num_groups}"

         # Step 0: 获取每个 group 的层级信息
         group_depth_map = {}
         for depth, nodes in self.nodes_by_depth.items():
            for node in nodes:
                  group_depth_map[node] = depth

         # Step 1: 每个 group 至少一个向量
         guaranteed_ids = list(range(len(all_nodes)))
         remaining_ids = list(range(len(all_nodes), self.total_vectors))

         for idx, node in enumerate(all_nodes):
            node.vectors = [idx]

         # Step 2: 计算每组的权重（越浅权重越高）
         weights = []
         for node in all_nodes:
            depth = group_depth_map[node]
            weight = (self.max_depth - depth + 1)  # 越靠近根节点权重越大
            weights.append(weight)

         # 归一化权重
         total_weight = sum(weights)
         normalized_weights = [w / total_weight for w in weights]

         # Step 3: 按照权重分配剩余向量
         import bisect
         from collections import Counter

         # 构造前缀和数组用于快速选择
         prefix_sums = []
         current_sum = 0
         for w in normalized_weights:
            current_sum += w
            prefix_sums.append(current_sum)

         # 分配剩余向量
         allocator = [0] * len(all_nodes)  # 分配计数器
         random.shuffle(remaining_ids)

         for vid in remaining_ids:
            # 二分查找选择 group
            r = random.random()
            idx = bisect.bisect_left(prefix_sums, r)
            idx = min(idx, len(all_nodes) - 1)  # 防止越界
            chosen_node = all_nodes[idx]
            chosen_node.vectors.append(vid)
            allocator[idx] += 1

         print("📊 向量分配完成")

    def _build_vector_label_mappings(self):
        """ 构建双向映射关系 """
        for node in self._get_all_nodes():
            for vec_id in node.vectors:
                self.vector_to_labels[vec_id] = node.label_set
                self.label_set_to_vectors[node.label_set].append(vec_id)

    def get_layer_labels(self, target_depth):
        """ 查询某一层的所有标签集合 """
        return self.nodes_by_depth.get(target_depth, [])

    def get_layer_vectors(self, target_depth):
        """ 查询某一层的所有向量ID列表 """
        result = []
        for node in self.nodes_by_depth.get(target_depth, []):
            result.extend(node.vectors)
        return result

    def print_tree(self):
        """ 打印整个树结构 """
        q = deque([(self.root, 1)])
        while q:
            node, depth = q.popleft()
            if node is not None:
                print(f"Depth {depth}: {sorted(node.label_set)} | Vectors: {len(node.vectors)}")
                for child in node.children:
                    q.append((child, depth + 1))

    def save_to_file(self, filename="generated_vectors.txt"):
        """ 保存向量和对应的标签集到文件 """
        with open(filename, 'w') as f:
            for vec_id in sorted(self.vector_to_labels.keys()):
                labels = sorted(self.vector_to_labels[vec_id])
                line = ','.join(map(str, labels))
                f.write(line + '\n')
        print(f"Vectors saved to {filename}")

    def get_group_stats(self):
        """ 返回每个 group 的向量数量统计 """
        stats = {}
        for node in self._get_all_nodes():
            stats[frozenset(node.label_set)] = len(node.vectors)
        return stats
    
    def print_stats(self):
         """ 打印每层的统计信息 """
         total_groups = 0
         total_vectors = 0
         max_depth = len(self.nodes_by_depth)
         print(f"Total layers: {max_depth}")
         
         for depth in sorted(self.nodes_by_depth.keys()):
            nodes = self.nodes_by_depth[depth]
            layer_vector_count = sum(len(node.vectors) for node in nodes)
            print(f"Layer {depth}: {len(nodes)} groups, {layer_vector_count} vectors")
            total_groups += len(nodes)
            total_vectors += layer_vector_count
         
         print(f"\nTotal groups: {total_groups}")
         print(f"Total vectors: {total_vectors}")


# 示例运行函数
if __name__ == "__main__":
    generator = LNGVectorGenerator(
        max_depth=400,         # 最大深度
        branch_factor=4,      # 每个节点最多有几个子节点
        max_groups=40000,      # 最多生成个 group
        base_attr_start=1     # 属性从 1 开始
    )
    generator.build(total_vectors=157606)

   #  # 打印树结构
   #  print("🌲 LNG Tree Structure:")
   #  generator.print_tree()

    # 打印统计信息
    print("\n📈 Layer-wise Statistics:")
    generator.print_stats()

    # 查询第3层的标签组合
    print("\n🔍 Layer 1-8 Labels:")
    layer1_labels = generator.get_layer_labels(1)
    for node in layer1_labels:
       print(sorted(node.label_set))
    layer2_labels = generator.get_layer_labels(2)
    for node in layer2_labels:
       print(sorted(node.label_set))
    layer3_labels = generator.get_layer_labels(3)
    for node in layer3_labels:
       print(sorted(node.label_set))
    layer4_labels = generator.get_layer_labels(4)
    for node in layer4_labels:
       print(sorted(node.label_set))
    layer5_labels = generator.get_layer_labels(5)
    for node in layer5_labels:
       print(sorted(node.label_set))
    layer6_labels = generator.get_layer_labels(6)
    for node in layer6_labels:
       print(sorted(node.label_set))
    layer7_labels = generator.get_layer_labels(7)
    for node in layer7_labels:
       print(sorted(node.label_set))
    layer8_labels = generator.get_layer_labels(8)
    for node in layer8_labels:
       print(sorted(node.label_set))

    # 查询第3层的向量ID
    print("\n🔢 Layer 3 Vector IDs:")
    layer3_vecs = generator.get_layer_vectors(3)
    print(layer3_vecs[:10])  # 只打印前10个

    # 保存数据到文件
    generator.save_to_file("arxiv_base_labels.txt")