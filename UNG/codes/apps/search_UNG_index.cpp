#include <chrono>
#include <fstream>
#include <numeric>
#include <iostream>
#include <bitset>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "uni_nav_graph.h"
#include "utils.h"
#include <roaring/roaring.h>
#include <roaring/roaring.hh>
#include <set>
#include <sstream>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

// 辅助函数：计算单个查询的recall
float calculate_single_query_recall(const std::pair<ANNS::IdxType, float> *gt,
                                    const std::pair<ANNS::IdxType, float> *results,
                                    ANNS::IdxType K)
{
    std::unordered_set<ANNS::IdxType> gt_set;
    for (int i = 0; i < K; ++i) {
        if (gt[i].first != -1) {
            gt_set.insert(gt[i].first);
        }
    }
    int correct = 0;
    for (int i = 0; i < K; ++i) {
        if (results[i].first != -1 && gt_set.count(results[i].first)) {
            correct++;
        }
    }
    return static_cast<float>(correct) / gt_set.size();
}

// 辅助函数：用于打印属性集合
void print_label_set(const std::set<ANNS::IdxType>& labels) {
    std::cout << "{ ";
    bool first = true;
    for (const auto& label : labels) {
        if (!first) {
            std::cout << ", ";
        }
        std::cout << label;
        first = false;
    }
    std::cout << " }";
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, scenario;
    std::string base_bin_file, query_bin_file, base_label_file, query_label_file, gt_file, index_path_prefix, result_path_prefix, query_group_id_file;
    ANNS::IdxType K, num_entry_points;
    std::vector<ANNS::IdxType> Lsearch_list;
    uint32_t num_threads;
    bool is_new_method = false;
    bool is_ori_ung = false;
    bool is_new_trie_method = false, is_rec_more_start = false;
    bool is_ung_more_entry = false;
    int num_repeats = 1;

    try {
        po::options_description desc{"Arguments"};
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(), "distance function <L2/IP/cosine>");
        desc.add_options()("base_bin_file", po::value<std::string>(&base_bin_file)->required(), "包含原始基准向量的二进制文件");
        desc.add_options()("base_label_file", po::value<std::string>(&base_label_file)->required(), "原始基准属性文件(txt格式)");
        desc.add_options()("query_bin_file", po::value<std::string>(&query_bin_file)->required(), "包含查询向量的二进制文件");
        desc.add_options()("query_label_file", po::value<std::string>(&query_label_file)->default_value(""), "查询属性文件(txt格式)");
        desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(), "Ground truth二进制文件");
        desc.add_options()("K", po::value<ANNS::IdxType>(&K)->required(), "最近邻数量");
        desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(ANNS::default_paras::NUM_THREADS), "线程数");
        desc.add_options()("result_path_prefix", po::value<std::string>(&result_path_prefix)->required(), "查询结果保存路径");
        desc.add_options()("query_group_id_file", po::value<std::string>(&query_group_id_file)->required(), "查询来源组ID文件");
        desc.add_options()("scenario", po::value<std::string>(&scenario)->default_value("containment"), "场景 <equality/containment/overlap/nofilter>");
        desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(), "索引加载路径前缀");
        desc.add_options()("num_entry_points", po::value<ANNS::IdxType>(&num_entry_points)->default_value(ANNS::default_paras::NUM_ENTRY_POINTS), "每个入口组的入口点数量");
        desc.add_options()("Lsearch", po::value<std::vector<ANNS::IdxType>>(&Lsearch_list)->multitoken()->required(), "搜索候选集大小");
        desc.add_options()("is_new_method", po::value<bool>(&is_new_method)->required(), "is_new_method");
        desc.add_options()("is_ori_ung", po::value<bool>(&is_ori_ung)->required(), "is_ori_ung");
        desc.add_options()("is_new_trie_method", po::value<bool>(&is_new_trie_method)->required(), "is_new_trie_method");
        desc.add_options()("is_rec_more_start", po::value<bool>(&is_rec_more_start)->required(), "is_rec_more_start");
        desc.add_options()("is_ung_more_entry", po::value<bool>(&is_ung_more_entry)->required(), "is_ung_more_entry");
        desc.add_options()("num_repeats", po::value<int>(&num_repeats)->default_value(1), "重复次数");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    if (scenario != "containment" && scenario != "equality" && scenario != "overlap") {
        std::cerr << "Invalid scenario: " << scenario << std::endl;
        return -1;
    }

    // 加载查询数据
    std::shared_ptr<ANNS::IStorage> query_storage = ANNS::create_storage(data_type);
    query_storage->load_from_file(query_bin_file, query_label_file);

    // 加载原始基准数据，用于通过其原始ID查询最近邻的属性
    std::cout << "为调试打印功能加载原始基准数据..." << std::endl;
    std::shared_ptr<ANNS::IStorage> base_storage = ANNS::create_storage(data_type);
    base_storage->load_from_file(base_bin_file, base_label_file);
    std::cout << "原始基准数据加载完毕，包含 " << base_storage->get_num_points() << " 个数据点。" << std::endl;

    // 加载已构建好的索引
    ANNS::UniNavGraph index(query_storage->get_num_points());
    index.load(index_path_prefix, data_type);
    index.load_bipartite_graph(index_path_prefix + "vector_attr_graph");

    // 加载查询来源组ID文件
    std::vector<ANNS::IdxType> true_query_group_ids;
    std::ifstream source_group_file(query_group_id_file);
    if (source_group_file.is_open()) {
        ANNS::IdxType group_id;
        while (source_group_file >> group_id) {
            true_query_group_ids.push_back(group_id);
        }
        source_group_file.close();
        std::cout << "成功加载 " << true_query_group_ids.size() << " 个查询的来源组ID。" << std::endl;
    } else {
        std::cerr << "警告：未找到查询来源组ID文件: " << query_group_id_file << std::endl;
    }

    // 准备工作
    auto num_queries = query_storage->get_num_points();
    std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);
    auto gt = new std::pair<ANNS::IdxType, float>[num_queries * K];
    ANNS::load_gt_file(gt_file, gt, num_queries, K);
    auto results = new std::pair<ANNS::IdxType, float>[num_queries * K];

    // 计算属性位图
    std::vector<std::pair<std::bitset<10000001>, double>> bitmap_and_time(num_queries);
    std::vector<std::bitset<10000001>> bitmap(num_queries);
    #pragma omp parallel for
    for (int id = 0; id < num_queries; id++) {
        bitmap_and_time[id] = index.compute_attribute_bitmap(query_storage->get_label_set(id));
        bitmap[id] = bitmap_and_time[id].first;
    }

    // 初始化查询统计
    std::vector<std::vector<std::vector<ANNS::QueryStats>>> query_stats(num_repeats, std::vector<std::vector<ANNS::QueryStats>>(Lsearch_list.size(), std::vector<ANNS::QueryStats>(num_queries)));

    for (int repeat = 0; repeat < num_repeats; ++repeat) {
        std::cout << "\n=== 重复 " << (repeat + 1) << "/" << num_repeats << " ===" << std::endl;

        std::cout << "开始查询..." << std::endl;
        for (int LsearchId = 0; LsearchId < Lsearch_list.size(); LsearchId++) {
            std::vector<float> num_cmps(num_queries);
            auto start_time = std::chrono::high_resolution_clock::now();

            if (!is_new_method) {
                // index.search(...); // 原始搜索函数调用
            } else {
                index.search_hybrid(query_storage, distance_handler, num_threads, Lsearch_list[LsearchId],
                                    num_entry_points, scenario, K, results, num_cmps, query_stats[repeat][LsearchId], bitmap, is_ori_ung, is_new_trie_method, is_rec_more_start, is_ung_more_entry, true_query_group_ids);
            }
            auto time_cost = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();

            // 在搜索结束后，打印前5个查询的详细信息
            std::cout << "\n--- Lsearch=" << Lsearch_list[LsearchId] << " 的前5个查询调试信息 ---" << std::endl;
            for (int i = 0; i < std::min((size_t)5, (size_t)num_queries); ++i) {
                // 获取并打印查询属性
                auto temp_query_labels_vec = query_storage->get_label_set(i);
                std::set<ANNS::IdxType> query_labels(temp_query_labels_vec.begin(), temp_query_labels_vec.end());

                std::cout << "查询 " << i << " 的属性: ";
                print_label_set(query_labels);
                std::cout << std::endl;

                // 打印找到的K个最近邻的属性
                std::cout << "  找到的 " << K << " 个最近邻:" << std::endl;
                for (int k = 0; k < K; ++k) {
                    // 'results' 中的ID已经是原始ID，可安全地用于原始 base_storage
                    ANNS::IdxType neighbor_id = results[i * K + k].first;
                    float distance = results[i * K + k].second;

                    if (neighbor_id == -1) {
                        std::cout << "    - 邻居 " << k << ": [无效]" << std::endl;
                        continue;
                    }
                    
                    std::cout << "    - 邻居 " << k << " (原始ID: " << neighbor_id << ", 距离: " << distance << "): 属性 ";
                    if (neighbor_id < base_storage->get_num_points()) {
                        auto temp_neighbor_labels_vec = base_storage->get_label_set(neighbor_id);
                        std::set<ANNS::IdxType> neighbor_labels(temp_neighbor_labels_vec.begin(), temp_neighbor_labels_vec.end());
                        print_label_set(neighbor_labels);
                    } else {
                        std::cout << "{ 错误: 原始ID越界 }";
                    }
                    std::cout << std::endl;
                }
                std::cout << "----------------------------------------------------" << std::endl;
            }

            // 为所有查询计算召回率
            #pragma omp parallel for
            for (int i = 0; i < num_queries; ++i)
                query_stats[repeat][LsearchId][i].recall = calculate_single_query_recall(gt + i * K, results + i * K, K);
        }
    }

    // 输出详细的CSV结果文件
    std::ofstream detail_out(result_path_prefix + "query_details_repeat" + std::to_string(num_repeats) + ".csv");
    detail_out << "repeat,Lsearch,QueryID,Time_ms,MinSupersetT_ms,EntryGroupT_ms,DescMergeT_ms,CovMergeT_ms,"
               << "FlagT_ms,BitmapT_ms,SearchT_ms,DistCalcs,NumEntries,NumDescendants,TotalCoverage,QuerySize,"
               << "CandSize,SuccessChecks,HitRatio,RecurCalls,PruneEvents,PruneEff,TrieNodePass,M1TrieReNode,NumNodeVisited,"
               << "QPS,Recall,IsGlobal\n";
    for (int repeat = 0; repeat < num_repeats; repeat++) {
        for (int LsearchId = 0; LsearchId < Lsearch_list.size(); LsearchId++) {
            for (int i = 0; i < num_queries; ++i) {
                detail_out << repeat << ","
                           << Lsearch_list[LsearchId] << ","
                           << i << ","
                           << query_stats[repeat][LsearchId][i].time_ms << ","
                           << query_stats[repeat][LsearchId][i].get_min_super_sets_time_ms << ","
                           << query_stats[repeat][LsearchId][i].get_group_entry_time_ms << ","
                           << query_stats[repeat][LsearchId][i].descendants_merge_time_ms << ","
                           << query_stats[repeat][LsearchId][i].coverage_merge_time_ms << ","
                           << query_stats[repeat][LsearchId][i].flag_time_ms << ","
                           << bitmap_and_time[i].second << ","
                           << query_stats[repeat][LsearchId][i].time_ms - query_stats[repeat][LsearchId][i].flag_time_ms << ","
                           << query_stats[repeat][LsearchId][i].num_distance_calcs << ","
                           << query_stats[repeat][LsearchId][i].num_entry_points << ","
                           << query_stats[repeat][LsearchId][i].num_lng_descendants << ","
                           << query_stats[repeat][LsearchId][i].entry_group_total_coverage << ","
                           << query_stats[repeat][LsearchId][i].query_length << ","
                           << query_stats[repeat][LsearchId][i].candidate_set_size << ","
                           << query_stats[repeat][LsearchId][i].successful_checks << ","
                           << query_stats[repeat][LsearchId][i].shortcut_hit_ratio << ","
                           << query_stats[repeat][LsearchId][i].recursive_calls << ","
                           << query_stats[repeat][LsearchId][i].pruning_events << ","
                           << query_stats[repeat][LsearchId][i].pruning_efficiency << ","
                           << query_stats[repeat][LsearchId][i].trie_nodes_traversed << ","
                           << query_stats[repeat][LsearchId][i].redundant_upward_steps << ","
                           << query_stats[repeat][LsearchId][i].num_nodes_visited << ","
                           << 1000.0 / (query_stats[repeat][LsearchId][i].time_ms) << ","
                           << query_stats[repeat][LsearchId][i].recall << ","
                           << query_stats[repeat][LsearchId][i].is_global_search << "\n";
            }
        }
    }
    detail_out.close();

    // 释放内存
    delete[] gt;
    delete[] results;

    std::cout << "- 全部完成" << std::endl;
    return 0;
}