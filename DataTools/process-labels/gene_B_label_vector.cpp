#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <queue>
#include <unordered_set>
#include <random>       
#include <algorithm>    
#include <stdexcept>    
#include <cstdint>      
#include <climits>      // 用于 INT_MAX
#include <filesystem>   // 用于创建目录
#include <unordered_set>  // 用于统计唯一的 ID

// 包含 Faiss 和 ACORN 的头文件
#include <faiss/index_io.h>
#include <faiss/IndexACORN.h>
#include <faiss/impl/ACORN.h> 
#include <faiss/utils/distances.h>    
#include <faiss/impl/DistanceComputer.h> 
#include <faiss/IndexFlat.h> 

// 使用 ACORN 内部的 MinimaxHeap 和 NodeDistFarther
using MinimaxHeap = faiss::ACORN::MinimaxHeap;
using NodeDistFarther = faiss::ACORN::NodeDistFarther;


// [新] 添加这个辅助函数，用于打印 vector
void print_labels(const std::vector<int>& labels) {
    std::cout << "[";
    for (size_t i = 0; i < labels.size(); ++i) {
        std::cout << labels[i] << (i == labels.size() - 1 ? "" : ",");
    }
    std::cout << "]";
}


// --- Faiss/IO 辅助函数 ---

/**
 * @brief 将向量写入 fvecs 文件
 */
void write_fvecs(const std::string &filename, const std::vector<std::vector<float>> &vectors)
{
   if (vectors.empty())
      return;

   // 自动创建目录
   try {
       std::filesystem::path p(filename);
       std::filesystem::create_directories(p.parent_path());
   } catch (const std::filesystem::filesystem_error& e) {
       std::cerr << "\n警告: 创建目录失败: " << e.what() << std::endl;
   }

   std::ofstream output(filename, std::ios::binary);
   if (!output)
   {
      throw std::runtime_error("Cannot open file for writing: " + filename);
   }
   int dimension = vectors[0].size();
   for (const auto &vec : vectors)
   {
      if (vec.size() != static_cast<size_t>(dimension))
      {
         throw std::runtime_error("Inconsistent vector dimensions when writing to fvecs file.");
      }
      output.write(reinterpret_cast<const char *>(&dimension), sizeof(int));
      output.write(reinterpret_cast<const char *>(vec.data()), dimension * sizeof(float));
   }
}

/**
 * @brief 加载多标签元数据
 */
std::vector<std::vector<int>> load_metadata_multi(const std::string& filename, size_t N) {
    std::vector<std::vector<int>> metadata(N);
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "错误: 无法打开元数据文件: " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    int node_id = 0;
    while (std::getline(infile, line) && node_id < N) {
        
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            node_id++;
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        
        while (std::getline(ss, token, ',')) { // 按 ',' 分割
            try {
                metadata[node_id].push_back(std::stoi(token)); 
            } catch (const std::invalid_argument& e) {
            } catch (const std::out_of_range& e) {
            }
        }
        // 保证加载的标签是排序的，这对于“取前5个”很重要
        std::sort(metadata[node_id].begin(), metadata[node_id].end()); 
        node_id++;
    }
    
    if (node_id < N) {
         std::cerr << "警告: 元数据文件行数 " << node_id 
                   << " 少于索引 ntotal " << N << std::endl;
    }
    return metadata;
}

/**
 * @brief 保存标签到文件
 */
void save_labels_to_file(const std::vector<std::vector<int>>& labels_data, const std::string& filename) {
    
    // 自动创建目录
    try {
        std::filesystem::path p(filename);
        std::filesystem::create_directories(p.parent_path());
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "\n警告: 创建目录失败: " << e.what() << std::endl;
    }

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "错误: 无法创建输出标签文件: " << filename << std::endl;
        exit(1);
    }
    
    for (const auto& labels : labels_data) {
        std::vector<int> sorted_labels = labels;
        std::sort(sorted_labels.begin(), sorted_labels.end());
        
        for (size_t i = 0; i < sorted_labels.size(); ++i) {
            outfile << sorted_labels[i] << (i == sorted_labels.size() - 1 ? "" : ",");
        }
        outfile << "\n";
    }
}


/**
 * @brief 距离引导的贪心搜索 (Greedy Search)
 */
std::vector<int32_t> greedy_search_neighbors(
    const faiss::ACORN& acorn, 
    faiss::DistanceComputer* dc,
    int32_t start_node, 
    int num_to_find) 
{
    std::priority_queue<NodeDistFarther> candidates; 
    std::unordered_set<int32_t> visited;
    std::vector<int32_t> explored_nodes;

    float d_start = (*dc)(start_node); 
    candidates.emplace(d_start, start_node); 
    visited.insert(start_node);
    
    while (!candidates.empty() && explored_nodes.size() < num_to_find) {
        
        const NodeDistFarther& currEv = candidates.top();
        int32_t currNode = currEv.id;
        candidates.pop();

        explored_nodes.push_back(currNode);
        
        size_t begin, end;
        acorn.neighbor_range(currNode, 0, &begin, &end); 

        for (size_t j = begin; j < end; j++) {
            int32_t neighbor_id = acorn.neighbors[j]; 
            if (neighbor_id < 0) break; 

            if (visited.find(neighbor_id) == visited.end()) {
                if (neighbor_id >= acorn.offsets.size() - 1) continue; 
                visited.insert(neighbor_id);
                
                float dis = (*dc)(neighbor_id); 
                candidates.emplace(dis, neighbor_id); 
            }
        }
    }
    
    while (!candidates.empty() && explored_nodes.size() < num_to_find) {
        const NodeDistFarther& currEv = candidates.top();
        candidates.pop();
        explored_nodes.push_back(currEv.id);
    }

    return explored_nodes;
}


/**
 * @brief 【修改】辅助函数：修改元数据，返回 bool
 * @return 如果标签列表发生实际改变，返回 true
 */
bool apply_labels(std::vector<int>& target_labels, const std::vector<int>& labels_to_copy) {
    // 使用 set 来高效合并和检查变化
    std::unordered_set<int> label_set(target_labels.begin(), target_labels.end());
    size_t original_size = label_set.size();
    
    for (int label : labels_to_copy) {
        label_set.insert(label);
    }

    if (label_set.size() > original_size) {
        // 只有在发生变化时才重新分配和排序
        target_labels.assign(label_set.begin(), label_set.end());
        std::sort(target_labels.begin(), target_labels.end());
        return true;
    }
    
    return false;
}


int main(int argc, char* argv[]) {
    // --- 1. 解析参数 (保留不变) ---
    if (argc != 10) {
        std::cerr << "用法错误！需要 9 个参数。" << std::endl;
        std::cerr << "正确用法: " << argv[0] 
                  << " <index_path>"
                  << " <input_base_labels_A_path>"
                  << " <output_base_labels_B_path>"
                  << " <output_query_labels_B_path>"
                  << " <output_query_vectors_B_path>"
                  << " <TARGET_LABEL_LENGTH>"
                  << " <NUM_QUERIES_TO_GENERATE>"
                  << " <NUM_NEIGHBORS_TO_LABEL>"
                  << " <NUM_NEIGHBORS_TO_FIND>"
                  << std::endl;
        return 1;
    }

    std::string index_path = argv[1];
    std::string input_base_labels_A_path = argv[2];
    std::string output_base_labels_B_path = argv[3]; 
    std::string output_query_labels_B_path = argv[4];
    std::string output_query_vectors_B_path = argv[5];
    
    int TARGET_LABEL_LENGTH = std::stoi(argv[6]); 
    int NUM_QUERIES_TO_GENERATE = std::stoi(argv[7]); 
    int NUM_NEIGHBORS_TO_LABEL = std::stoi(argv[8]); 
    int L_FIND_CANDIDATES = std::stoi(argv[9]); 

    std::cout << "--- ACORN 标签修改器 (Gene_B) ---" << std::endl;
    std::cout << "  索引: " << index_path << std::endl;
    std::cout << "  输入 Base 标签 (A): " << input_base_labels_A_path << std::endl;
    std::cout << "  输出 Base 标签 (B): " << output_base_labels_B_path << std::endl;
    std::cout << "  锚点截断长度 (Target_L): " << TARGET_LABEL_LENGTH << " (取前 " << TARGET_LABEL_LENGTH << " 个)" << std::endl;
    std::cout << "  查询/锚点总数 (Q_total): " << NUM_QUERIES_TO_GENERATE << std::endl;
    std::cout << "  邻居修改数 (L_label): " << NUM_NEIGHBORS_TO_LABEL << std::endl;
    std::cout << "  贪心搜索候选 (L_find): " << L_FIND_CANDIDATES << std::endl;
    std::cout << "--------------------------" << std::endl;


    // --- 2. 加载 ACORN 索引 (保留不变) ---
    std::cout << "正在加载索引... " << std::flush;
    faiss::Index* index = faiss::read_index(index_path.c_str());
    faiss::IndexACORNFlat* hybrid_index = dynamic_cast<faiss::IndexACORNFlat*>(index); 
    
    if (!hybrid_index) {
        std::cerr << "错误: 加载的索引不是 IndexACORNFlat 类型。" << std::endl;
        delete index;
        return 1;
    }
    size_t N = hybrid_index->ntotal; 
    int D = hybrid_index->d;         
    std::cout << "完成。索引 Ntotal = " << N << ", 维度 D = " << D << std::endl;

    // --- 3. 【新逻辑】加载 Base 元数据 (累积修改) ---
    std::string path_to_load;
    
    // 检查 B 阶段的输出文件是否已存在
    if (std::filesystem::exists(output_base_labels_B_path)) {
        // 如果存在，加载它，以便在它之上进行修改
        path_to_load = output_base_labels_B_path;
        std::cout << "  [累积模式]：发现已存在的 " << output_base_labels_B_path << "，将加载此文件进行累积修改。" << std::endl;
    } else {
        // 如果不存在 (第一次运行)，加载 A 阶段的文件
        path_to_load = input_base_labels_A_path;
        std::cout << "  [首次运行]：未发现 " << output_base_labels_B_path << "，将加载 " << input_base_labels_A_path << "。" << std::endl;
    }
    
    // [修改] 变量重命名为 current_base_labels
    std::vector<std::vector<int>> current_base_labels = load_metadata_multi(path_to_load, N);
    
    // --- 4. 【新逻辑】查找所有符合条件的锚点 ID ---
    std::cout << "正在查找标签长度 > " << TARGET_LABEL_LENGTH << " 且最短的 Base 向量..." << std::flush;
    
    int min_eligible_length = INT_MAX;
    
    // 第一次遍历: 找到最短的合格长度 (e.g., 6)
    for (int32_t i = 0; i < N; ++i) {
        // [修改] 使用 current_base_labels
        int current_length = current_base_labels[i].size(); 
        if (current_length > TARGET_LABEL_LENGTH) {
            if (current_length < min_eligible_length) {
                min_eligible_length = current_length;
            }
        }
    }

    if (min_eligible_length == INT_MAX) {
        std::cerr << "\n错误: 在 " << path_to_load 
                  << " 中找不到任何标签长度 > " << TARGET_LABEL_LENGTH << " 的向量。" << std::endl;
        delete index;
        return 1;
    }

    std::cout << " 找到。最短合格长度为 " << min_eligible_length << "。" << std::endl;
    std::cout << "正在收集所有长度为 " << min_eligible_length << " 的向量 ID..." << std::flush;

    // 第二次遍历: 收集所有该长度的 ID
    std::vector<int32_t> eligible_anchor_ids;
    for (int32_t i = 0; i < N; ++i) {
        // [修改] 使用 current_base_labels
        if (current_base_labels[i].size() == min_eligible_length) {
            eligible_anchor_ids.push_back(i);
        }
    }
    
    if (eligible_anchor_ids.empty()) {
        std::cerr << "\n错误: 无法收集到合格的锚点 ID (逻辑错误)。" << std::endl;
        delete index;
        return 1;
    }
    std::cout << " 完成。共找到 " << eligible_anchor_ids.size() << " 个符合条件的向量。" << std::endl;


    // ----------------------------------------------------
    // 【5. 新核心逻辑：抽样 + 截断 + 注入 + 生成】
    // ----------------------------------------------------
    std::cout << "正在执行 " << NUM_QUERIES_TO_GENERATE << " 次抽样、修改和查询生成..." << std::endl;
    
    // [修改] 从 current_base_labels 深拷贝
    std::vector<std::vector<int>> modified_base_labels_B = current_base_labels; 
    std::vector<std::vector<int>> final_query_labels;
    std::vector<std::vector<float>> final_query_vectors;
    
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int32_t> anchor_sampler(0, eligible_anchor_ids.size() - 1); 

    int total_modifications = 0; 
    int total_attempts = 0;      

    std::unordered_set<int32_t> modified_base_ids;
    int32_t sample_anchor_id = -1;
    int32_t sample_neighbor_id = -1;
    std::vector<int> sample_labels_copied;
    std::vector<int> sample_neighbor_before;
    
    for (int i = 0; i < NUM_QUERIES_TO_GENERATE; ++i) {
        
        int32_t anchor_base_id = eligible_anchor_ids[anchor_sampler(gen)]; 
        
        // [修改] 使用 current_base_labels
        const std::vector<int>& full_anchor_labels = current_base_labels[anchor_base_id]; 
        
        std::vector<int> labels_to_copy(
            full_anchor_labels.begin(), 
            full_anchor_labels.begin() + TARGET_LABEL_LENGTH
        );
        
        std::vector<float> anchor_node_vec(D);
        hybrid_index->reconstruct(anchor_base_id, anchor_node_vec.data());
        
        final_query_labels.push_back(labels_to_copy);
        final_query_vectors.push_back(anchor_node_vec);
        
        // 【修改】锚点自身标签修改
        bool anchor_changed = apply_labels(modified_base_labels_B[anchor_base_id], labels_to_copy);
        if (anchor_changed) {
            modified_base_ids.insert(anchor_base_id);
        }

        faiss::DistanceComputer* dc = hybrid_index->storage->get_distance_computer();
        dc->set_query(anchor_node_vec.data()); 

        std::vector<int32_t> candidates = greedy_search_neighbors(
            hybrid_index->acorn, 
            dc,
            anchor_base_id, 
            L_FIND_CANDIDATES 
        );
        
        MinimaxHeap closest_neighbors(NUM_NEIGHBORS_TO_LABEL); 
        
        for (int32_t neighbor_id : candidates) {
            if (neighbor_id == anchor_base_id) continue; 
            float dis = (*dc)(neighbor_id); 
            closest_neighbors.push(neighbor_id, dis); 
        }
        delete dc; 

        // 【修改】邻居修改逻辑
        for(int k=0; k < closest_neighbors.k; ++k) {
            int32_t node_to_label_id = closest_neighbors.ids[k];
            if (node_to_label_id != -1 && node_to_label_id < N) { 
                
                total_attempts++; // 统计尝试次数

                std::vector<int> neighbor_labels_before = modified_base_labels_B[node_to_label_id];

                bool neighbor_changed = apply_labels(modified_base_labels_B[node_to_label_id], labels_to_copy);
                
                if (neighbor_changed) {
                    modified_base_ids.insert(node_to_label_id); 
                    total_modifications++; 

                    if (sample_anchor_id == -1 && node_to_label_id != anchor_base_id) {
                        sample_anchor_id = anchor_base_id;
                        sample_neighbor_id = node_to_label_id;
                        sample_labels_copied = labels_to_copy;
                        sample_neighbor_before = neighbor_labels_before; 
                    }
                }
            }
        }
        
        if ((i + 1) % 100 == 0) {
            std::cout << "\rProgress: " << i + 1 << "/" << NUM_QUERIES_TO_GENERATE << " queries generated..." << std::flush;
        }
    }
    
    // 打印新的统计信息
    std::cout << "\n完成。" << std::endl;
    std::cout << "\n--- 统计与抽样 ---" << std::endl;
    std::cout << "  总共尝试附加次数: " << total_attempts << " (仅邻居)" << std::endl;
    std::cout << "  实际成功附加次数: " << total_modifications << " (仅邻居，标签实际变化)" << std::endl;
    std::cout << "  唯一修改 Base 向量数: " << modified_base_ids.size() << " (包括锚点和邻居)" << std::endl;

    if (sample_anchor_id != -1) {
        std::cout << "\n--- 修改样本 (首个真实变化) ---" << std::endl;
        std::cout << "  锚点 ID: " << sample_anchor_id << std::endl;
        std::cout << "  注入标签 (前 " << TARGET_LABEL_LENGTH << " 个): ";
        print_labels(sample_labels_copied);
        std::cout << std::endl;
        
        std::cout << "  邻居 ID: " << sample_neighbor_id << std::endl;
        std::cout << "    - Before: ";
        print_labels(sample_neighbor_before); // 打印修改前
        std::cout << std::endl;
        
        std::cout << "    - After:  ";
        print_labels(modified_base_labels_B[sample_neighbor_id]); // 打印修改后
        std::cout << std::endl;
    } else {
        std::cout << "\n[注] 未抓取到邻居修改样本 (可能 L_label 为 0，或所有邻居都已包含标签)。" << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;


    // --- 6. 保存 B 阶段 Base 元数据 ---
    std::cout << "正在保存 B 阶段 Base 元数据到 " << output_base_labels_B_path << "... " << std::flush;
    save_labels_to_file(modified_base_labels_B, output_base_labels_B_path);
    std::cout << "完成。" << std::endl;
    
    // --- 7. 保存 B 阶段 Query 标签 ---
    std::cout << "正在保存 B 阶段 Query Labels (" << final_query_labels.size() << ") 到 " << output_query_labels_B_path << "... " << std::flush;
    save_labels_to_file(final_query_labels, output_query_labels_B_path);
    std::cout << "完成。" << std::endl;

    // --- 8. 保存 B 阶段 Query 向量 ---
    std::cout << "正在保存 B 阶段 Query Vectors (" << final_query_vectors.size() << ") 到 " << output_query_vectors_B_path << "... " << std::flush;
    try {
        write_fvecs(output_query_vectors_B_path, final_query_vectors); 
        std::cout << "完成。" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: 写入 fvecs 文件失败: " << e.what() << std::endl;
    }

    // --- 9. 清理 ---
    delete index;
    std::cout << "\n--- [功能 B] 所有任务完成 ---" << std::endl;
    return 0;
}