#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <map>
#include <queue>
#include <sstream>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <boost/program_options.hpp>

#pragma once

// 定义基础类型，确保与UNG项目兼容
namespace ANNS {
    using IdxType = uint32_t;
    using LabelType = uint32_t;

    struct TrieNode {
        std::map<LabelType, std::shared_ptr<TrieNode>> children;
        IdxType group_id = 0;
        IdxType group_size = 0;
    };

    class TrieIndex {
    public:
        TrieIndex() : root(std::make_shared<TrieNode>()) {}

        // 插入一个标签集
        IdxType insert(const std::vector<LabelType>& labels, IdxType& new_group_id_counter) {
            auto node = root;
            std::vector<LabelType> sorted_labels = labels;
            std::sort(sorted_labels.begin(), sorted_labels.end());

            for (auto label : sorted_labels) {
                if (node->children.find(label) == node->children.end()) {
                    node->children[label] = std::make_shared<TrieNode>();
                }
                node = node->children[label];
            }

            if (node->group_id == 0) {
                node->group_id = new_group_id_counter++;
                if (!sorted_labels.empty()) {
                    max_label_id = std::max(max_label_id, sorted_labels.back());
                }
            }
            node->group_size++;
            return node->group_id;
        }

        // 精确查找
        std::shared_ptr<TrieNode> find_exact_match(const std::vector<LabelType>& labels) const {
            auto node = root;
            std::vector<LabelType> sorted_labels = labels;
            std::sort(sorted_labels.begin(), sorted_labels.end());
            for (auto label : sorted_labels) {
                if (node->children.find(label) == node->children.end()) {
                    return nullptr;
                }
                node = node->children[label];
            }
            return node;
        }
        
        // 获取所有超集入口
        void get_super_set_entrances(const std::vector<LabelType>& query_labels, std::vector<std::shared_ptr<TrieNode>>& entrances, bool avoid_self, bool need_containment) const {
            auto node = root;
            std::vector<LabelType> sorted_labels = query_labels;
            std::sort(sorted_labels.begin(), sorted_labels.end());

            std::queue<std::pair<std::shared_ptr<TrieNode>, size_t>> q;
            q.push({root, 0});

            while (!q.empty()) {
                auto curr = q.front();
                q.pop();
                auto curr_node = curr.first;
                auto label_idx = curr.second;

                if (label_idx == sorted_labels.size()) {
                    if(!avoid_self || curr_node->group_id == 0 || find_exact_match(sorted_labels) != curr_node) {
                        entrances.push_back(curr_node);
                    }
                    continue;
                }

                for (auto const& [key, val] : curr_node->children) {
                    if (key > sorted_labels[label_idx]) {
                        q.push({val, label_idx});
                    } else if (key == sorted_labels[label_idx]) {
                        q.push({val, label_idx + 1});
                    }
                }
            }
        }

        LabelType get_max_label_id() const { return max_label_id; }
        std::shared_ptr<TrieNode> get_root() const { return root; }

    private:
        std::shared_ptr<TrieNode> root;
        LabelType max_label_id = 0;
    };
}


// 读取fvecs文件
void load_fvecs(const std::string& filename, std::vector<std::vector<float>>& data, uint32_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }
    
    // 读取维度
    input.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    if (file_size == 0) {
        dim = 0;
        data.clear();
        return;
    }
    size_t num_vectors = file_size / ((dim + 1) * sizeof(float));
    input.seekg(0, std::ios::beg);

    data.resize(num_vectors, std::vector<float>(dim));
    uint32_t temp_dim;

    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&temp_dim), sizeof(uint32_t));
        if (temp_dim != dim) {
            throw std::runtime_error("文件中的维度不一致!");
        }
        input.read(reinterpret_cast<char*>(data[i].data()), dim * sizeof(float));
    }
    input.close();
}

// 写入fvecs文件
void write_fvecs(const std::string& filename, const std::vector<std::vector<float>>& data, uint32_t dim) {
    std::ofstream output(filename, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("无法创建文件: " + filename);
    }
    for (const auto& vec : data) {
        output.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
        output.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
    }
    output.close();
}


namespace po = boost::program_options;

// 查询模板结构体
struct QueryTemplate {
    std::vector<ANNS::LabelType> labels;
    ANNS::IdxType coverage_count;
};

// --- 复用并扩展Trie树的功能 ---

// 1. 检查一个标签集是否至少有K个潜在答案 (直接复用)
bool has_k_answers(const ANNS::TrieIndex& trie_index, const std::vector<ANNS::LabelType>& label_set, ANNS::IdxType K) {
    if (label_set.empty()) return true; // 空标签集覆盖所有，一定满足
    
    std::vector<std::shared_ptr<ANNS::TrieNode>> super_set_entrances;
    trie_index.get_super_set_entrances(label_set, super_set_entrances, false, true);

    ANNS::IdxType cnt = 0;
    std::queue<std::shared_ptr<ANNS::TrieNode>> q;
    for (const auto& node : super_set_entrances) {
        q.push(node);
    }
    
    // 使用一个足够大的set来避免重复访问group_id
    std::vector<bool> visited_groups(trie_index.get_max_label_id() * 200, false);
    if (trie_index.get_max_label_id() == 0) visited_groups.resize(10000);


    while (!q.empty()) {
        auto cur = q.front();
        q.pop();

        if(cur->group_id != 0 && cur->group_id < visited_groups.size() && !visited_groups[cur->group_id]) {
           cnt += cur->group_size;
           visited_groups[cur->group_id] = true;
        }

        if (cnt >= K) return true;

        for (const auto& child : cur->children) {
            q.push(child.second);
        }
    }
    return false;
}

// 2. 精确计算一个标签集的覆盖数量
ANNS::IdxType calculate_exact_coverage(const ANNS::TrieIndex& trie_index, const std::vector<ANNS::LabelType>& label_set) {
    if (label_set.empty()) { // 空标签集覆盖所有向量
        ANNS::IdxType total_coverage = 0;
        std::queue<std::shared_ptr<ANNS::TrieNode>> q;
        q.push(trie_index.get_root());
        while(!q.empty()){
            auto curr = q.front();
            q.pop();
            total_coverage += curr->group_size;
            for(auto const& [key, val] : curr->children){
                q.push(val);
            }
        }
        return total_coverage;
    }

    std::vector<std::shared_ptr<ANNS::TrieNode>> super_set_entrances;
    trie_index.get_super_set_entrances(label_set, super_set_entrances, false, true);

    ANNS::IdxType total_coverage = 0;
    std::queue<std::shared_ptr<ANNS::TrieNode>> q;
    std::vector<bool> visited_groups(trie_index.get_max_label_id() * 200, false);
    if (trie_index.get_max_label_id() == 0) visited_groups.resize(10000);

    for (const auto& node : super_set_entrances) {
        q.push(node);
    }

    while (!q.empty()) {
        auto cur = q.front();
        q.pop();
        
        if (cur->group_id != 0 && cur->group_id < visited_groups.size() && !visited_groups[cur->group_id]) {
            total_coverage += cur->group_size;
            visited_groups[cur->group_id] = true;
        }

        for (const auto& child : cur->children) {
            q.push(child.second);
        }
    }
    return total_coverage;
}

// 随机生成一个候选标签集
std::vector<ANNS::LabelType> generate_random_label_set(ANNS::LabelType max_label_id, int max_labels_per_set, std::mt19937& gen) {
    if (max_label_id == 0) return {};
    std::uniform_int_distribution<> num_labels_dist(1, std::min((int)max_label_id, max_labels_per_set));
    std::uniform_int_distribution<ANNS::LabelType> label_dist(1, max_label_id);
    
    int num_labels = num_labels_dist(gen);
    std::vector<ANNS::LabelType> labels;
    std::vector<bool> used(max_label_id + 1, false);

    while (labels.size() < num_labels) {
        ANNS::LabelType new_label = label_dist(gen);
        if (!used[new_label]) {
            labels.push_back(new_label);
            used[new_label] = true;
        }
    }
    std::sort(labels.begin(), labels.end());
    return labels;
}

int main(int argc, char** argv) {
    // --- 1. 参数解析 ---
    std::string base_labels_file, base_vectors_file, output_prefix, coverage_bins_str, mix_ratios_str;
    ANNS::IdxType total_queries, K;
    int max_labels_per_set;
    
    po::options_description desc{"混合覆盖率查询生成器参数"};
    desc.add_options()
        ("help,h", "打印帮助信息")
        ("base_labels", po::value<std::string>(&base_labels_file)->required(), "基础数据集的标签文件路径 (必需)")
        ("base_vectors", po::value<std::string>(&base_vectors_file)->required(), "基础数据集的向量文件路径 (.fvecs) (必需)")
        ("output_prefix", po::value<std::string>(&output_prefix)->required(), "输出文件的前缀 (必需)")
        ("total_queries", po::value<ANNS::IdxType>(&total_queries)->default_value(10000), "要生成的查询总数")
        ("K", po::value<ANNS::IdxType>(&K)->default_value(10), "查询必须拥有的最少潜在答案数")
        ("max_labels_per_set", po::value<int>(&max_labels_per_set)->default_value(5), "随机生成查询模板时，每个模板包含的最大标签数")
        ("coverage_bins", po::value<std::string>(&coverage_bins_str)->default_value("0.001,0.01,0.1"), "覆盖率区间的阈值，逗号分隔")
        ("mix_ratios", po::value<std::string>(&mix_ratios_str)->default_value("0.25,0.25,0.25,0.25"), "各区间查询的混合比例，逗号分隔");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "参数错误: " << e.what() << std::endl << desc << std::endl;
        return 1;
    }

    // --- 2. 初始化 ---
    std::ofstream log_file(output_prefix + "_log.txt");
    auto start_time = std::chrono::high_resolution_clock::now();
    log_file << "--- 查询生成开始 ---" << std::endl;
    log_file << "时间: " << std::fixed << std::setprecision(0) << std::chrono::duration_cast<std::chrono::seconds>(start_time.time_since_epoch()).count() << std::endl;
    log_file << "参数:" << std::endl;
    log_file << "  基础标签文件: " << base_labels_file << std::endl;
    log_file << "  基础向量文件: " << base_vectors_file << std::endl;
    log_file << "  输出前缀: " << output_prefix << std::endl;
    log_file << "  查询总数: " << total_queries << std::endl;
    log_file << "  最小答案数K: " << K << std::endl;
    log_file << "  每个模板最大标签数: " << max_labels_per_set << std::endl;
    log_file << "  覆盖率区间阈值: " << coverage_bins_str << std::endl;
    log_file << "  混合比例: " << mix_ratios_str << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());

    // 解析区间和比例参数
    std::vector<float> coverage_bins;
    std::stringstream ss_bins(coverage_bins_str);
    std::string bin_val;
    while(std::getline(ss_bins, bin_val, ',')) coverage_bins.push_back(std::stof(bin_val));
    std::sort(coverage_bins.begin(), coverage_bins.end());

    std::vector<float> mix_ratios;
    std::stringstream ss_ratios(mix_ratios_str);
    std::string ratio_val;
    while(std::getline(ss_ratios, ratio_val, ',')) mix_ratios.push_back(std::stof(ratio_val));

    if (mix_ratios.size() != coverage_bins.size() + 1) {
        std::cerr << "错误: 混合比例的数量必须比区间阈值的数量多1。" << std::endl;
        log_file << "错误: 混合比例的数量必须比区间阈值的数量多1。" << std::endl;
        return 1;
    }

    // --- 3. 阶段一：分析与模板库构建 ---
    log_file << "\n--- 阶段一：分析与模板库构建 ---" << std::endl;
    
    // 加载基础标签并构建Trie
    ANNS::TrieIndex trie_index;
    ANNS::IdxType total_base_vectors = 0;
    std::ifstream infile(base_labels_file);
    if (!infile.is_open()) {
        std::cerr << "错误: 无法打开基础标签文件 " << base_labels_file << std::endl;
        log_file << "错误: 无法打开基础标签文件 " << base_labels_file << std::endl;
        return 1;
    }
    
    log_file << "正在读取基础标签并构建Trie树..." << std::endl;
    ANNS::IdxType new_label_set_id = 1;
    std::string line, label_str;
    while (std::getline(infile, line)) {
        std::vector<ANNS::LabelType> label_set;
        std::stringstream ss(line);
        while (std::getline(ss, label_str, ',')) {
            if (!label_str.empty()) label_set.emplace_back(std::stoul(label_str));
        }
        trie_index.insert(label_set, new_label_set_id);
        total_base_vectors++;
    }
    infile.close();
    log_file << "Trie树构建完毕。共 " << total_base_vectors << " 个向量, " << new_label_set_id - 1 << " 个唯一标签集。" << std::endl;
    if (total_base_vectors == 0) {
        log_file << "错误：基础数据集中没有向量。" << std::endl;
        return 1;
    }


    // "Generate, Test, Bin" 过程
    std::vector<std::vector<QueryTemplate>> template_bins(coverage_bins.size() + 1);
    std::vector<ANNS::IdxType> target_template_counts(mix_ratios.size());
    for(size_t i = 0; i < mix_ratios.size(); ++i) {
        target_template_counts[i] = static_cast<ANNS::IdxType>(total_queries * mix_ratios[i]);
    }

    log_file << "正在通过“生成-测试-分类”方法构建模板库..." << std::endl;
    log_file << "目标模板数量: ";
    for(auto count : target_template_counts) log_file << count << " ";
    log_file << std::endl;

    long long attempts = 0;
    ANNS::LabelType max_label_id = trie_index.get_max_label_id();
    auto last_log_time = std::chrono::high_resolution_clock::now();

    while (true) {
        bool all_bins_full = true;
        for (size_t i = 0; i < template_bins.size(); ++i) {
            if (template_bins[i].size() < target_template_counts[i]) {
                all_bins_full = false;
                break;
            }
        }
        if (all_bins_full) break;

        attempts++;
        auto current_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_log_time).count() >= 5) {
            std::cout << "\r尝试次数: " << attempts << ", 当前模板库大小: [";
            for(size_t i=0; i < template_bins.size(); ++i) std::cout << template_bins[i].size() << "/" << target_template_counts[i] << (i == template_bins.size()-1 ? "" : ", ");
            std::cout << "]" << std::flush;
            last_log_time = current_time;
        }

        // a. 生成随机候选
        auto candidate_labels = generate_random_label_set(max_label_id, max_labels_per_set, gen);
        
        // b. 验证 (>= K)
        if (!has_k_answers(trie_index, candidate_labels, K)) {
            continue;
        }

        // c. 计算精确覆盖率
        ANNS::IdxType coverage_count = calculate_exact_coverage(trie_index, candidate_labels);
        float coverage_ratio = static_cast<float>(coverage_count) / total_base_vectors;
        
        // d. 分类
        size_t bin_index = std::upper_bound(coverage_bins.begin(), coverage_bins.end(), coverage_ratio) - coverage_bins.begin();

        // e. 存入模板库 (如果对应箱子未满)
        if (template_bins[bin_index].size() < target_template_counts[bin_index]) {
            template_bins[bin_index].push_back({candidate_labels, coverage_count});
        }
    }
    std::cout << std::endl;
    log_file << "模板库构建完成，共尝试 " << attempts << " 次。" << std::endl;
    log_file << "各区间模板库最终大小:" << std::endl;
    for(size_t i=0; i < template_bins.size(); ++i) {
        log_file << "  区间 " << i << ": " << template_bins[i].size() << " 个模板" << std::endl;
    }


    // --- 4. 阶段二：按比例采样与生成 ---
    log_file << "\n--- 阶段二：按比例采样与生成 ---" << std::endl;
    
    // 加载基础向量
    log_file << "正在加载基础向量..." << std::endl;
    std::vector<std::vector<float>> base_vectors;
    uint32_t dim;
    try {
        load_fvecs(base_vectors_file, base_vectors, dim);
    } catch(const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        log_file << "错误: " << e.what() << std::endl;
        return 1;
    }
    log_file << "基础向量加载完毕，共 " << base_vectors.size() << " 个, 维度为 " << dim << std::endl;
    if (base_vectors.empty()){
        log_file << "错误：基础向量文件为空。" << std::endl;
        return 1;
    }
    
    // 准备输出文件
    std::ofstream query_labels_file(output_prefix + "_labels.txt");
    std::vector<std::vector<float>> query_vectors_data;

    // 按比例采样并生成
    log_file << "正在采样并生成最终查询文件..." << std::endl;
    std::uniform_int_distribution<size_t> vec_dist(0, base_vectors.size() - 1);

    for (size_t i = 0; i < template_bins.size(); ++i) {
        size_t num_to_generate = target_template_counts[i];
        log_file << "  从区间 " << i << " 生成 " << num_to_generate << " 个查询..." << std::endl;
        if (template_bins[i].empty()) {
            log_file << "    警告: 区间 " << i << " 的模板库为空，无法生成查询。" << std::endl;
            continue;
        }

        std::uniform_int_distribution<size_t> template_dist(0, template_bins[i].size() - 1);
        for (size_t j = 0; j < num_to_generate; ++j) {
            const auto& chosen_template = template_bins[i][template_dist(gen)];
            
            // 写标签
            for (size_t l = 0; l < chosen_template.labels.size(); ++l) {
                query_labels_file << chosen_template.labels[l] << (l == chosen_template.labels.size() - 1 ? "" : ",");
            }
            query_labels_file << std::endl;

            // 随机选一个向量
            size_t random_vec_idx = vec_dist(gen);
            query_vectors_data.push_back(base_vectors[random_vec_idx]);
        }
    }

    // 写入向量文件
    write_fvecs(output_prefix + ".fvecs", query_vectors_data, dim);
    query_labels_file.close();

    log_file << "查询文件生成完毕。" << std::endl;

    // --- 5. 总结 ---
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    log_file << "\n--- 查询生成结束 ---" << std::endl;
    log_file << "总耗时: " << duration.count() << "毫秒" << std::endl;
    log_file << "输出文件:" << std::endl;
    log_file << "  - 日志: " << output_prefix + "_log.txt" << std::endl;
    log_file << "  - 标签: " << output_prefix + "_labels.txt" << std::endl;
    log_file << "  - 向量: " << output_prefix + ".fvecs" << std::endl;
    log_file.close();
    
    std::cout << "混合覆盖率查询生成完毕，详情请查看日志文件: " << output_prefix + "_log.txt" << std::endl;

    return 0;
}