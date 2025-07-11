#include <omp.h>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <vector>
#include <queue>
#include <stack>
#include <bitset>

#include <random>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <filesystem>
#include <bitset>
#include <boost/dynamic_bitset.hpp>

#include "utils.h"
#include "vamana/vamana.h"
#include "include/uni_nav_graph.h"
#include <roaring/roaring.h>
#include <roaring/roaring.hh>

namespace fs = boost::filesystem;
using BitsetType = boost::dynamic_bitset<>;

// 文件格式常量
const std::string FVEC_EXT = ".fvecs";
const std::string TXT_EXT = ".txt";
const size_t FVEC_HEADER_SIZE = sizeof(uint32_t);
struct QueryTask
{
   ANNS::IdxType vec_id;         // 向量ID
   std::vector<uint32_t> labels; // 查询标签集
};

struct TreeInfo
{
   ANNS::IdxType root_group_id;
   ANNS::LabelType root_label_id; // 存储概念上的根标签ID
   size_t coverage_count;
   // 用于按覆盖率降序排序
   bool operator<(const TreeInfo &other) const
   {
      return coverage_count > other.coverage_count;
   }
};

namespace ANNS
{

   void UniNavGraph::build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                           std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                           IdxType max_degree, IdxType Lbuild, float alpha)
   {
      auto all_start_time = std::chrono::high_resolution_clock::now();
      _base_storage = base_storage;
      _num_points = base_storage->get_num_points();
      _distance_handler = distance_handler;
      std::cout << "- Scenario: " << scenario << std::endl;

      // index parameters
      _index_name = index_name;
      _num_cross_edges = num_cross_edges;
      _max_degree = max_degree;
      _Lbuild = Lbuild;
      _alpha = alpha;
      _num_threads = num_threads;
      _scenario = scenario;

      std::cout << "Dividing groups and building the trie tree index ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      build_trie_and_divide_groups();
      _graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      _global_graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      std::cout << "begin prepare_group_storages_graphs" << std::endl;
      prepare_group_storages_graphs();
      _label_processing_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
      std::cout << "- Finished in " << _label_processing_time << " ms" << std::endl;

      // build graph index for each group
      build_graph_for_all_groups();
      // build_global_vamana_graph();
      build_vector_and_attr_graph(); // fxy_add

      // for label equality scenario, there is no need for label navigating graph and cross-group edges
      if (_scenario == "equality")
      {
         add_offset_for_uni_nav_graph();
      }
      else
      {

         // build the label navigating graph
         build_label_nav_graph();
         get_descendants_info(); // fxy_add

         // calculate the coverage ratio
         cal_f_coverage_ratio(); // fxy_add

         // initialize_lng_descendants_coverage_bitsets();
         initialize_roaring_bitsets();

         // build cross-group edges
         build_cross_group_edges();
      }

      // index time
      _index_time = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - all_start_time)
                        .count();
   }

   void UniNavGraph::build_trie_and_divide_groups()
   {
      // create groups for base label sets
      IdxType new_group_id = 1;
      for (auto vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         auto group_id = _trie_index.insert(label_set, new_group_id);

         // deal with new label setinver
         if (group_id + 1 > _group_id_to_vec_ids.size())
         {
            _group_id_to_vec_ids.resize(group_id + 1);
            _group_id_to_label_set.resize(group_id + 1);
            _group_id_to_label_set[group_id] = label_set;
         }
         _group_id_to_vec_ids[group_id].emplace_back(vec_id);
      }

      // logs
      _num_groups = new_group_id - 1;
      std::cout << "- Number of groups: " << _num_groups << std::endl;
   }

   void UniNavGraph::get_min_super_sets(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                        bool avoid_self, bool need_containment)
   {
      min_super_set_ids.clear();

      // obtain the candidates
      std::vector<std::shared_ptr<TrieNode>> candidates;
      _trie_index.get_super_set_entrances(query_label_set, candidates, avoid_self, need_containment);

      // special cases
      if (candidates.empty())
         return;
      if (candidates.size() == 1)
      {
         min_super_set_ids.emplace_back(candidates[0]->group_id);
         return;
      }

      // obtain the minimum size
      std::sort(candidates.begin(), candidates.end(),
                [](const std::shared_ptr<TrieNode> &a, const std::shared_ptr<TrieNode> &b)
                {
                   return a->label_set_size < b->label_set_size;
                });
      auto min_size = _group_id_to_label_set[candidates[0]->group_id].size();

      // get the minimum super sets
      for (auto candidate : candidates)
      {
         const auto &cur_group_id = candidate->group_id;
         const auto &cur_label_set = _group_id_to_label_set[cur_group_id];
         bool is_min = true;

         // check whether contains existing minimum super sets (label ids are in ascending order)
         if (cur_label_set.size() > min_size)
         {
            for (auto min_group_id : min_super_set_ids)
            {
               const auto &min_label_set = _group_id_to_label_set[min_group_id];
               if (std::includes(cur_label_set.begin(), cur_label_set.end(), min_label_set.begin(), min_label_set.end()))
               {
                  is_min = false;
                  break;
               }
            }
         }

         // add to the minimum super sets
         if (is_min)
            min_super_set_ids.emplace_back(cur_group_id);
      }
   }

   void UniNavGraph::prepare_group_storages_graphs()
   {
      _new_vec_id_to_group_id.resize(_num_points);

      // reorder the vectors
      _group_id_to_range.resize(_num_groups + 1);
      _new_to_old_vec_ids.resize(_num_points);
      IdxType new_vec_id = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         _group_id_to_range[group_id].first = new_vec_id;
         for (auto old_vec_id : _group_id_to_vec_ids[group_id])
         {
            _new_to_old_vec_ids[new_vec_id] = old_vec_id;
            _new_vec_id_to_group_id[new_vec_id] = group_id;
            ++new_vec_id;
         }
         _group_id_to_range[group_id].second = new_vec_id;
      }

      // reorder the underlying storage
      _base_storage->reorder_data(_new_to_old_vec_ids);

      // init storage and graph for each group
      _group_storages.resize(_num_groups + 1);
      _group_graphs.resize(_num_groups + 1);
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         auto start = _group_id_to_range[group_id].first;
         auto end = _group_id_to_range[group_id].second;
         _group_storages[group_id] = create_storage(_base_storage, start, end);
         _group_graphs[group_id] = std::make_shared<Graph>(_graph, start, end);
      }
   }

   void UniNavGraph::build_graph_for_all_groups()
   {
      std::cout << "Building graph for each group ..." << std::endl;
      omp_set_num_threads(_num_threads);
      auto start_time = std::chrono::high_resolution_clock::now();

      // build vamana index
      if (_index_name == "Vamana")
      {
         _vamana_instances.resize(_num_groups + 1);
         _group_entry_points.resize(_num_groups + 1);

#pragma omp parallel for schedule(dynamic, 1)
         for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         {
            // if (group_id % 100 == 0)
            //    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;

            // if there are less than _max_degree points in the group, just build a complete graph
            const auto &range = _group_id_to_range[group_id];
            if (range.second - range.first <= _max_degree)
            {
               build_complete_graph(_group_graphs[group_id], range.second - range.first);
               _vamana_instances[group_id] = std::make_shared<Vamana>(_group_storages[group_id], _distance_handler,
                                                                      _group_graphs[group_id], 0);

               // build the vamana graph
            }
            else
            {
               _vamana_instances[group_id] = std::make_shared<Vamana>(false);
               _vamana_instances[group_id]->build(_group_storages[group_id], _distance_handler,
                                                  _group_graphs[group_id], _max_degree, _Lbuild, _alpha, 1);
            }

            // set entry point
            _group_entry_points[group_id] = _vamana_instances[group_id]->get_entry_point() + range.first;
         }

         // if none of the above
      }
      else
      {
         std::cerr << "Error: invalid index name " << _index_name << std::endl;
         exit(-1);
      }
      _build_graph_time = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count();
      std::cout << "\r- Finished in " << _build_graph_time << " ms" << std::endl;
   }

   // fxy_add：构建全局Vamana图
   void UniNavGraph::build_global_vamana_graph()
   {
      std::cout << "Building global Vamana graph..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _global_vamana = std::make_shared<Vamana>(false);

      _global_vamana->build(_base_storage, _distance_handler, _global_graph, _max_degree, _Lbuild, _alpha, 1);

      auto build_time = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      _global_vamana_entry_point = _global_vamana->get_entry_point();

      std::cout << "- Global Vamana graph built in " << build_time << " ms" << std::endl;
   }
   //=====================================begin 数据预处理：构建向量-属性二分图=========================================
   // fxy_add: 构建向量-属性二分图
   void UniNavGraph::build_vector_and_attr_graph()
   {
      std::cout << "Building vector-attribute bipartite graph..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // 初始化属性到ID的映射和反向映射
      _attr_to_id.clear();
      _id_to_attr.clear();
      _vector_attr_graph.clear();

      // 第一遍：收集所有唯一属性并分配ID
      AtrType attr_id = 0;
      for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         for (const auto &label : label_set)
         {
            if (_attr_to_id.find(label) == _attr_to_id.end())
            {
               _attr_to_id[label] = attr_id;
               _id_to_attr[attr_id] = label;
               attr_id++;
            }
         }
      }

      // 初始化图结构（向量节点 + 属性节点）
      _vector_attr_graph.resize(_num_points + static_cast<size_t>(attr_id));

      // 第二遍：构建图结构
      for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         for (const auto &label : label_set)
         {
            AtrType a_id = _attr_to_id[label];

            // 添加双向边
            // 向量节点ID范围: [0, _num_points-1]
            // 属性节点ID范围: [_num_points, _num_points+attr_id-1]
            _vector_attr_graph[vec_id].push_back(_num_points + static_cast<IdxType>(a_id));
            _vector_attr_graph[_num_points + static_cast<IdxType>(a_id)].push_back(vec_id);
         }
      }

      // 统计信息
      _num_attributes = attr_id;
      _build_vector_attr_graph_time = std::chrono::duration<double, std::milli>(
                                          std::chrono::high_resolution_clock::now() - start_time)
                                          .count();
      std::cout << "- Finish in " << _build_vector_attr_graph_time << " ms" << std::endl;
      std::cout << "- Total edges: " << count_graph_edges() << std::endl;

      // 可选：保存图结构供调试
      // save_bipartite_graph_info();
   }

   // fxy_add: 计算向量-属性二分图的边数
   size_t UniNavGraph::count_graph_edges() const
   {
      size_t total_edges = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         total_edges += neighbors.size();
      }
      return total_edges / 2; // 因为是双向边，实际边数是总数的一半
   }

   // fxy_add: 保存二分图信息到txt文件调试用
   void UniNavGraph::save_bipartite_graph_info() const
   {
      std::ofstream outfile("bipartite_graph_info.txt");
      if (!outfile.is_open())
      {
         std::cerr << "Warning: Could not open file to save bipartite graph info" << std::endl;
         return;
      }

      outfile << "Bipartite Graph Information\n";
      outfile << "==========================\n";
      outfile << "Total vectors: " << _num_points << "\n";
      outfile << "Total attributes: " << _num_attributes << "\n";
      outfile << "Total edges: " << count_graph_edges() << "\n\n";

      // 输出属性映射
      outfile << "Attribute to ID Mapping:\n";
      for (const auto &pair : _attr_to_id)
      {
         outfile << pair.first << " -> " << static_cast<IdxType>(pair.second) << "\n";
      }
      outfile << "\n";

      // 输出部分图结构示例
      outfile << "Sample Graph Connections (first 10 vectors and attributes):\n";
      outfile << "Vector connections:\n";
      for (IdxType i = 0; i < std::min(_num_points, static_cast<IdxType>(10)); ++i)
      {
         outfile << "Vector " << i << " connects to attributes: ";
         for (auto a_node : _vector_attr_graph[i])
         {
            AtrType attr_id = static_cast<AtrType>(a_node - _num_points);
            outfile << _id_to_attr.at(attr_id) << " ";
         }
         outfile << "\n";
      }

      outfile << "\nAttribute connections:\n";
      for (AtrType i = 0; i < std::min(_num_attributes, static_cast<AtrType>(5)); ++i)
      {
         outfile << "Attribute " << _id_to_attr.at(i) << " connects to vectors: ";
         for (auto v_node : _vector_attr_graph[_num_points + static_cast<IdxType>(i)])
         {
            outfile << v_node << " ";
         }
         outfile << "\n";
      }

      outfile.close();
      std::cout << "- Bipartite graph info saved to bipartite_graph_info.txt" << std::endl;
   }

   uint32_t UniNavGraph::compute_checksum() const
   {
      // 简单的校验和计算示例
      uint32_t sum = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         for (IdxType node : neighbors)
         {
            sum ^= (node << (sum % 32));
         }
      }
      return sum;
   }

   // fxy_add: 保存二分图到文件
   void UniNavGraph::save_bipartite_graph(const std::string &filename)
   {
      std::ofstream out(filename, std::ios::binary);
      if (!out)
      {
         throw std::runtime_error("Cannot open file for writing: " + filename);
      }

      // 1. 写入文件头标识和版本
      const char header[8] = {'B', 'I', 'P', 'G', 'R', 'P', 'H', '1'};
      out.write(header, 8);

      // 2. 写入基本元数据
      out.write(reinterpret_cast<const char *>(&_num_points), sizeof(IdxType));
      out.write(reinterpret_cast<const char *>(&_num_attributes), sizeof(AtrType));

      // 3. 写入属性映射表
      // 3.1 先写入条目数量
      uint64_t map_size = _attr_to_id.size();
      out.write(reinterpret_cast<const char *>(&map_size), sizeof(uint64_t));

      // 3.2 写入每个映射条目（LabelType是uint16_t，直接存储）
      for (const auto &[label, id] : _attr_to_id)
      {
         out.write(reinterpret_cast<const char *>(&label), sizeof(LabelType));
         out.write(reinterpret_cast<const char *>(&id), sizeof(AtrType));
      }

      // 4. 写入邻接表数据
      // 4.1 先写入节点总数
      uint64_t total_nodes = _vector_attr_graph.size();
      out.write(reinterpret_cast<const char *>(&total_nodes), sizeof(uint64_t));

      // 4.2 写入每个节点的邻居列表
      for (const auto &neighbors : _vector_attr_graph)
      {
         // 先写入邻居数量
         uint32_t neighbor_count = neighbors.size();
         out.write(reinterpret_cast<const char *>(&neighbor_count), sizeof(uint32_t));

         // 写入邻居ID列表
         if (!neighbors.empty())
         {
            out.write(reinterpret_cast<const char *>(neighbors.data()),
                      neighbors.size() * sizeof(IdxType));
         }
      }

      // 5. 写入文件尾校验和
      uint32_t checksum = compute_checksum();
      out.write(reinterpret_cast<const char *>(&checksum), sizeof(uint32_t));

      std::cout << "Successfully saved bipartite graph to " << filename
                << " (" << out.tellp() << " bytes)" << std::endl;
   }

   // fxy_add: 读取二分图
   void UniNavGraph::load_bipartite_graph(const std::string &filename)
   {
      std::cout << "Loading bipartite graph from " << filename << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      std::ifstream in(filename, std::ios::binary);
      if (!in)
      {
         throw std::runtime_error("Cannot open file for reading: " + filename);
      }

      // 1. 验证文件头
      char header[8];
      in.read(header, 8);
      if (std::string(header, 8) != "BIPGRPH1")
      {
         throw std::runtime_error("Invalid file format");
      }

      // 2. 读取基本元数据
      in.read(reinterpret_cast<char *>(&_num_points), sizeof(IdxType));
      in.read(reinterpret_cast<char *>(&_num_attributes), sizeof(AtrType));

      // 3. 读取属性映射表
      _attr_to_id.clear();
      _id_to_attr.clear();

      // 3.1 读取条目数量
      uint64_t map_size;
      in.read(reinterpret_cast<char *>(&map_size), sizeof(uint64_t));

      // 3.2 读取每个映射条目
      for (uint64_t i = 0; i < map_size; ++i)
      {
         LabelType label;
         AtrType id;

         in.read(reinterpret_cast<char *>(&label), sizeof(LabelType));
         in.read(reinterpret_cast<char *>(&id), sizeof(AtrType));

         _attr_to_id[label] = id;
         _id_to_attr[id] = label;
      }

      // 4. 读取邻接表数据
      _vector_attr_graph.clear();

      // 4.1 读取节点总数
      uint64_t total_nodes;
      in.read(reinterpret_cast<char *>(&total_nodes), sizeof(uint64_t));
      _vector_attr_graph.resize(total_nodes);

      // 4.2 读取每个节点的邻居列表
      for (uint64_t i = 0; i < total_nodes; ++i)
      {
         uint32_t neighbor_count;
         in.read(reinterpret_cast<char *>(&neighbor_count), sizeof(uint32_t));

         _vector_attr_graph[i].resize(neighbor_count);
         if (neighbor_count > 0)
         {
            in.read(reinterpret_cast<char *>(_vector_attr_graph[i].data()),
                    neighbor_count * sizeof(IdxType));
         }
      }

      // 5. 验证校验和
      uint32_t stored_checksum;
      in.read(reinterpret_cast<char *>(&stored_checksum), sizeof(uint32_t));

      uint32_t computed_checksum = compute_checksum();
      if (stored_checksum != computed_checksum)
      {
         throw std::runtime_error("Checksum verification failed");
      }

      std::cout << "- Loaded bipartite graph with " << _num_points << " vectors and "
                << _num_attributes << " attributes in "
                << std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - start_time)
                       .count()
                << " ms" << std::endl;
   }

   // fxy_add: 比较两个向量-属性二分图
   bool UniNavGraph::compare_graphs(const ANNS::UniNavGraph &g1, const ANNS::UniNavGraph &g2)
   {
      // 1. 验证基本属性
      if (g1._num_points != g2._num_points)
      {
         std::cerr << "Mismatch in _num_points: "
                   << g1._num_points << " vs " << g2._num_points << std::endl;
         return false;
      }

      if (g1._num_attributes != g2._num_attributes)
      {
         std::cerr << "Mismatch in _num_attributes: "
                   << g1._num_attributes << " vs " << g2._num_attributes << std::endl;
         return false;
      }

      // 2. 验证属性映射
      if (g1._attr_to_id.size() != g2._attr_to_id.size())
      {
         std::cerr << "Mismatch in _attr_to_id size" << std::endl;
         return false;
      }

      for (const auto &[label, id] : g1._attr_to_id)
      {
         auto it = g2._attr_to_id.find(label);
         if (it == g2._attr_to_id.end())
         {
            std::cerr << "Label " << label << " missing in g2" << std::endl;
            return false;
         }
         if (it->second != id)
         {
            std::cerr << "Mismatch ID for label " << label
                      << ": " << id << " vs " << it->second << std::endl;
            return false;
         }
      }

      // 3. 验证反向属性映射
      for (const auto &[id, label] : g1._id_to_attr)
      {
         auto it = g2._id_to_attr.find(id);
         if (it == g2._id_to_attr.end())
         {
            std::cerr << "ID " << id << " missing in g2" << std::endl;
            return false;
         }
         if (it->second != label)
         {
            std::cerr << "Mismatch label for ID " << id
                      << ": " << label << " vs " << it->second << std::endl;
            return false;
         }
      }

      // 4. 验证邻接表
      if (g1._vector_attr_graph.size() != g2._vector_attr_graph.size())
      {
         std::cerr << "Mismatch in graph size" << std::endl;
         return false;
      }

      for (size_t i = 0; i < g1._vector_attr_graph.size(); ++i)
      {
         const auto &neighbors1 = g1._vector_attr_graph[i];
         const auto &neighbors2 = g2._vector_attr_graph[i];

         if (neighbors1.size() != neighbors2.size())
         {
            std::cerr << "Mismatch in neighbors count for node " << i << std::endl;
            return false;
         }

         for (size_t j = 0; j < neighbors1.size(); ++j)
         {
            if (neighbors1[j] != neighbors2[j])
            {
               std::cerr << "Mismatch in neighbor " << j << " for node " << i
                         << ": " << neighbors1[j] << " vs " << neighbors2[j] << std::endl;
               return false;
            }
         }
      }

      std::cout << "Graphs are identical!" << std::endl;

      return true;
   }
   //=====================================end 数据预处理：构建向量-属性二分图=========================================

   //=====================================begein 查询过程：计算bitmap=========================================

   // fxy_add: 构建bitmap
   std::pair<std::bitset<10000001>, double> UniNavGraph::compute_attribute_bitmap(const std::vector<LabelType> &query_attributes) const
   {
      // 1. 初始化全true的bitmap（表示开始时所有点都满足条件）
      std::bitset<10000001> bitmap;
      bitmap.set(); // 设置所有位为1
      double per_query_bitmap_time = 0.0;

      // 2. 处理每个查询属性
      for (LabelType attr_label : query_attributes)
      {
         //  2.1 查找属性ID
         auto it = _attr_to_id.find(attr_label);
         if (it == _attr_to_id.end())
         {
            // 属性不存在，没有任何点能满足所有条件
            return {std::bitset<10000001>(), 0.0};
         }

         // 2.2 获取属性节点ID
         AtrType attr_id = it->second;
         IdxType attr_node_id = _num_points + static_cast<IdxType>(attr_id);

         // 2.3 创建临时bitmap记录当前属性的满足情况
         std::bitset<10000001> temp_bitmap;
         auto start_time = std::chrono::high_resolution_clock::now();
         for (IdxType vec_id : _vector_attr_graph[attr_node_id])
         {
            temp_bitmap.set(vec_id); // 使用set()方法设置对应的位为1
         }

         // 2.4 与主bitmap进行AND操作
         bitmap &= temp_bitmap; // 使用&=操作符进行位与操作

         per_query_bitmap_time += std::chrono::duration<double, std::milli>(
                                      std::chrono::high_resolution_clock::now() - start_time)
                                      .count();
      }

      return {bitmap, per_query_bitmap_time};
   }

   //====================================end 查询过程：计算bitmap=========================================
   void UniNavGraph::build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points)
   {
      for (auto i = 0; i < num_points; ++i)
         for (auto j = 0; j < num_points; ++j)
            if (i != j)
               graph->neighbors[i].emplace_back(j);
   }

   //=====================================begin LNG中每个f覆盖率计算=========================================
   // fxy_add: 递归打印孩子节点及其覆盖率
   void print_children_recursive(const std::shared_ptr<ANNS::LabelNavGraph> graph, IdxType group_id, std::ofstream &outfile, int indent_level)
   {
      // 根据缩进级别生成前缀空格
      std::string indent(indent_level * 4, ' '); // 每一层缩进4个空格

      // 打印当前节点
      outfile << "\n"
              << indent << "[" << group_id << "] (" << graph->coverage_ratio[group_id] << ")";

      // 如果有子节点，递归打印子节点
      if (!graph->out_neighbors[group_id].empty())
      {
         for (size_t i = 0; i < graph->out_neighbors[group_id].size(); ++i)
         {
            auto child_group_id = graph->out_neighbors[group_id][i];
            print_children_recursive(graph, child_group_id, outfile, indent_level + 1); // 增加缩进级别
         }
      }
   }

   // fxy_add: 输出覆盖率及孩子节点,调用print_children_recursive
   void output_coverage_ratio(const std::shared_ptr<ANNS::LabelNavGraph> _label_nav_graph, IdxType _num_groups, std::ofstream &outfile)
   {
      outfile << "Coverage Ratio for each group\n";
      outfile << "=====================================\n";
      outfile << "Format: [GroupID] -> CoverageRatio \n";
      outfile << "-> [child_GroupID (CoverageRatio)]\n\n";

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // 打印当前组的覆盖率
         outfile << "[" << group_id << "] -> " << _label_nav_graph->coverage_ratio[group_id];

         // 打印孩子节点
         if (!_label_nav_graph->out_neighbors[group_id].empty())
         {
            outfile << " ->";
            for (size_t i = 0; i < _label_nav_graph->out_neighbors[group_id].size(); ++i)
            {
               auto child_group_id = _label_nav_graph->out_neighbors[group_id][i];
               print_children_recursive(_label_nav_graph, child_group_id, outfile, 1); // 第一层子节点缩进为1
            }
         }
         outfile << "\n";
      }
   }

   // fxy_add：计算每个label set的向量覆盖比率
   void UniNavGraph::cal_f_coverage_ratio()
   {
      std::cout << "Calculating coverage ratio..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // Step 0: 初始化covered_sets
      _label_nav_graph->coverage_ratio.clear();
      _label_nav_graph->covered_sets.clear();
      _label_nav_graph->covered_sets.resize(_num_groups + 1);

      // Step 1: 初始化每个 group 的覆盖集合
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &vec_ids = _group_id_to_vec_ids[group_id];
         if (vec_ids.empty())
            continue;

         _label_nav_graph->covered_sets[group_id].insert(vec_ids.begin(), vec_ids.end());
      }

      // Step 2: 找出所有叶子节点（出度为 0）
      std::queue<IdxType> q;
      std::vector<int> out_degree(_num_groups + 1, 0); // 复制一份出度用于拓扑传播

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         out_degree[group_id] = _label_nav_graph->out_neighbors[group_id].size();
         if (out_degree[group_id] == 0)
            q.push(group_id);
      }
      std::cout << "- Number of leaf nodes: " << q.size() << std::endl;

      // Step 3: 自底向上合并集合
      while (!q.empty())
      {
         IdxType current = q.front();
         q.pop();

         for (auto parent : _label_nav_graph->in_neighbors[current])
         {
            // 将当前节点的集合合并到父节点中
            _label_nav_graph->covered_sets[parent].insert(
                _label_nav_graph->covered_sets[current].begin(),
                _label_nav_graph->covered_sets[current].end());

            // 减少父节点剩余未处理的子节点数
            out_degree[parent]--;
            if (out_degree[parent] == 0)
            {
               q.push(parent);
            }
         }
      }

      // Step 4: 计算最终覆盖率
      _label_nav_graph->coverage_ratio.resize(_num_groups + 1, 0.0);
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         size_t covered_count = _label_nav_graph->covered_sets[group_id].size();
         _label_nav_graph->coverage_ratio[group_id] = static_cast<double>(covered_count) / _num_points;
      }

      // Step 5: 输出时间
      _cal_coverage_ratio_time = std::chrono::duration<double, std::milli>(
                                     std::chrono::high_resolution_clock::now() - start_time)
                                     .count();
      std::cout << "- Finish in " << _cal_coverage_ratio_time << " ms" << std::endl;

      // // step4：递归存储所有孩子的覆盖比率
      // std::ofstream outfile0("LNG_coverage_ratio.txt");
      // output_coverage_ratio(_label_nav_graph, _num_groups, outfile0);
      // outfile0.close();

      // // step5：存储每个group的覆盖比率和标签集 Format: GroupID CoverageRatio LabelSet
      // std::ofstream outfile("group_coverage_ratio.txt");
      // for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      // {
      //    outfile << group_id << " " << _label_nav_graph->coverage_ratio[group_id] << " ";
      //    for (const auto &label : _group_id_to_label_set[group_id])
      //       outfile << label << " ";
      //    outfile << "\n";
      // }

      // // step6：存储每个group里面有几个向量 Format: GroupID NumVectors
      // std::ofstream outfile1("group_num_vectors.txt");
      // for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      //    outfile1 << group_id << " " << _group_id_to_vec_ids[group_id].size() << "\n";
   }
   // =====================================end LNG中每个f覆盖率计算=========================================

   // =====================================begin 计算LNG中后代的个数=========================================
   // fxy_add: 计算所有节点的后代数量，并更新_lng_descendants_num和_lng_descendants
   void UniNavGraph::get_descendants_info()
   {
      std::cout << "Calculating descendants info..." << std::endl;
      using PairType = std::pair<IdxType, int>;
      std::vector<PairType> descendants_num(_num_groups);                    // 存储后代个数
      std::vector<std::unordered_set<IdxType>> descendants_set(_num_groups); // 存储后代集合

      auto start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         std::vector<bool> visited(_num_groups + 1, false);
         std::stack<IdxType> stack;
         stack.push(group_id);

         size_t count = 0;
         std::unordered_set<IdxType> temp_set;

         while (!stack.empty())
         {
            IdxType current = stack.top();
            stack.pop();

            if (visited[current])
            {
               continue;
            }
            visited[current] = true;

            for (auto child_id : _label_nav_graph->out_neighbors[current])
            {
               if (child_id == current)
                  continue; // 跳过自环

               if (!visited[child_id])
               {
                  count += 1;
                  temp_set.insert(child_id);
                  stack.push(child_id);
               }
            }
         }

         descendants_num[group_id - 1] = PairType(group_id, static_cast<int>(count));
         descendants_set[group_id - 1] = std::move(temp_set);
      }

      // 单线程写入类成员变量
      _label_nav_graph->_lng_descendants_num = std::move(descendants_num);
      _label_nav_graph->_lng_descendants = std::move(descendants_set);

      // 计算平均后代个数
      double total_descendants = 0;
      for (const auto &pair : _label_nav_graph->_lng_descendants_num)
      {
         total_descendants += pair.second;
      }
      _label_nav_graph->avg_descendants = _num_groups > 0 ? total_descendants / _num_groups : 0;
      _cal_descendants_time = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count();
      std::cout << "- Finish in " << _cal_descendants_time << " ms" << std::endl;
      std::cout << "- Number of groups: " << _num_groups << std::endl;
      std::cout << "- Average number of descendants per group: " << _label_nav_graph->avg_descendants << std::endl;
   }

   // =====================================end 计算LNG中后代的个数=========================================

   void UniNavGraph::build_label_nav_graph()
   {
      std::cout << "Building label navigation graph... " << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
      omp_set_num_threads(_num_threads);

// obtain out-neighbors
#pragma omp parallel for schedule(dynamic, 256)
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // if (group_id % 100 == 0)
         //    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
         std::vector<IdxType> min_super_set_ids;
         get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
         _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;
      }

      // obtain in-neighbors
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         for (auto each : _label_nav_graph->out_neighbors[group_id])
            _label_nav_graph->in_neighbors[each].emplace_back(group_id);

      _build_LNG_time = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      std::cout << "\r- Finished in " << _build_LNG_time << " ms" << std::endl;
   }

   // fxy_add : 打印信息的build_label_nav_graph
   //    void UniNavGraph::build_label_nav_graph()
   //    {
   //       std::cout << "Building label navigation graph... " << std::endl;
   //       auto start_time = std::chrono::high_resolution_clock::now();
   //       _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
   //       omp_set_num_threads(_num_threads);
   //       std::ofstream outfile("lng_structure.txt");
   //       if (!outfile.is_open())
   //       {
   //          std::cerr << "Error: Could not open lng_structure.txt for writing!" << std::endl;
   //          return;
   //       }
   //       outfile << "Label Navigation Graph (LNG) Structure\n";
   //       outfile << "=====================================\n";
   //       outfile << "Format: [GroupID] {LabelSet} -> [OutNeighbor1]{LabelSet}, [OutNeighbor2]{LabelSet}, ...\n\n";
   // // obtain out-neighbors
   // #pragma omp parallel for schedule(dynamic, 256)
   //       for (auto group_id = 1; group_id <= _num_groups; ++group_id)
   //       {
   //          if (group_id % 100 == 0)
   //          {
   // #pragma omp critical
   //             std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
   //          }
   //          std::vector<IdxType> min_super_set_ids;
   //          get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
   //          _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;
   //          _label_nav_graph->out_degree[group_id] = min_super_set_ids.size();
   // #pragma omp critical
   //          {
   //             outfile << "[" << group_id << "] {";
   //             // 打印标签集
   //             for (const auto &label : _group_id_to_label_set[group_id])
   //                outfile << label << ",";
   //             outfile << "} -> ";
   //             // 打印出边（包含目标节点的标签集）
   //             for (size_t i = 0; i < min_super_set_ids.size(); ++i)
   //             {
   //                auto target_id = min_super_set_ids[i];
   //                outfile << "[" << target_id << "] {";
   //                // 打印目标节点的标签集
   //                for (const auto &label : _group_id_to_label_set[target_id])
   //                {
   //                   outfile << label << ",";
   //                }
   //                outfile << "}";
   //                if (i != min_super_set_ids.size() - 1)
   //                   outfile << ", ";
   //             }
   //             outfile << "\n";
   //          }
   //       }
   //       // obtain in-neighbors (不需要打印入边，但保留原有逻辑)
   //       for (auto group_id = 1; group_id <= _num_groups; ++group_id)
   //       {
   //          for (auto each : _label_nav_graph->out_neighbors[group_id])
   //          {
   //             _label_nav_graph->in_neighbors[each].emplace_back(group_id);
   //             _label_nav_graph->in_degree[group_id] += 1;
   //          }
   //       }
   //       // outfile.close();
   //       _build_LNG_time = std::chrono::duration<double, std::milli>(
   //                             std::chrono::high_resolution_clock::now() - start_time)
   //                             .count();
   //       std::cout << "\r- Finished in " << _build_LNG_time << " ms" << std::endl;
   //       std::cout << "- LNG structure saved to lng_structure.txt" << std::endl;
   //    }

   // fxy_add:初始化求flag的几个数据结构
   void UniNavGraph::initialize_lng_descendants_coverage_bitsets()
   {
      std::cout << "Initializing LNG descendants and coverage bitsets..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _lng_descendants_bits.resize(_num_groups + 1);
      _covered_sets_bits.resize(_num_groups + 1);

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // 初始化大小
         _lng_descendants_bits[group_id].resize(_num_groups);
         _covered_sets_bits[group_id].resize(_num_points);

         // 填充后代的group的集合
         const auto &descendants = _label_nav_graph->_lng_descendants[group_id];
         for (auto id : descendants)
         {
            _lng_descendants_bits[group_id].set(id);
         }

         // 填充覆盖的向量的集合
         const auto &coverage = _label_nav_graph->covered_sets[group_id];
         for (auto id : coverage)
         {
            _covered_sets_bits[group_id].set(id);
         }
      }
   }

   // fxy_add:初始化求flag的几个数据结构

   void UniNavGraph::initialize_roaring_bitsets()
   {
      std::cout << "enter initialize_roaring_bitsets" << std::endl;
      _lng_descendants_rb.resize(_num_groups + 1);
      _covered_sets_rb.resize(_num_groups + 1);
      // std::cout << "begin for " << std::endl;

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // std::cout << "group_id: " << group_id << std::endl;
         const auto &descendants = _label_nav_graph->_lng_descendants[group_id];
         const auto &coverage = _label_nav_graph->covered_sets[group_id];

         // 初始化后代 bitset
         for (auto id : descendants)
         {
            _lng_descendants_rb[group_id].add(id);
         }

         // 初始化覆盖 bitset
         for (auto id : coverage)
         {
            _covered_sets_rb[group_id].add(id);
         }
      }

      std::cout << "Roaring bitsets initialized." << std::endl;
   }

   // 将分组内的局部索引转换为全局索引
   void UniNavGraph::add_offset_for_uni_nav_graph()
   {
      omp_set_num_threads(_num_threads);
#pragma omp parallel for schedule(dynamic, 4096)
      for (auto i = 0; i < _num_points; ++i)
         for (auto &neighbor : _graph->neighbors[i])
            neighbor += _group_id_to_range[_new_vec_id_to_group_id[i]].first;
   }

   void UniNavGraph::build_cross_group_edges()
   {
      std::cout << "Building cross-group edges ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // allocate memory for storaging cross-group neighbors
      std::vector<SearchQueue> cross_group_neighbors;
      cross_group_neighbors.resize(_num_points);
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         cross_group_neighbors[point_id].reserve(_num_cross_edges);

      // allocate memory for search caches
      size_t max_group_size = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         max_group_size = std::max(max_group_size, _group_id_to_vec_ids[group_id].size());
      SearchCacheList search_cache_list(_num_threads, max_group_size, _Lbuild);
      omp_set_num_threads(_num_threads);

      // for each group
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (_label_nav_graph->in_neighbors[group_id].size() > 0)
         {
            // if (group_id % 100 == 0)
            //    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
            IdxType offset = _group_id_to_range[group_id].first;

            // query vamana index
            if (_index_name == "Vamana")
            {
               auto index = _vamana_instances[group_id];
               if (_num_cross_edges > _Lbuild)
               {
                  std::cerr << "Error: num_cross_edges should be less than or equal to Lbuild" << std::endl;
                  exit(-1);
               }

               // for each in-neighbor group
               for (auto in_group_id : _label_nav_graph->in_neighbors[group_id])
               {
                  const auto &range = _group_id_to_range[in_group_id];

// take each vector in the group as the query
#pragma omp parallel for schedule(dynamic, 1)
                  for (auto vec_id = range.first; vec_id < range.second; ++vec_id)
                  {
                     const char *query = _base_storage->get_vector(vec_id);
                     auto search_cache = search_cache_list.get_free_cache();
                     index->iterate_to_fixed_point(query, search_cache);

                     // update the cross-group edges for vec_id
                     for (auto k = 0; k < search_cache->search_queue.size(); ++k)
                        cross_group_neighbors[vec_id].insert(search_cache->search_queue[k].id + offset,
                                                             search_cache->search_queue[k].distance);
                     search_cache_list.release_cache(search_cache);
                  }
               }

               // if none of the above
            }
            else
            {
               std::cerr << "Error: invalid index name " << _index_name << std::endl;
               exit(-1);
            }
         }
      }

      // add additional edges
      std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges(_num_groups + 1);
#pragma omp parallel for schedule(dynamic, 256)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &cur_range = _group_id_to_range[group_id];
         std::unordered_set<IdxType> connected_groups;

         // obtain connected groups
         for (IdxType i = cur_range.first; i < cur_range.second; ++i)
            for (IdxType j = 0; j < cross_group_neighbors[i].size(); ++j)
               connected_groups.insert(_new_vec_id_to_group_id[cross_group_neighbors[i][j].id]);

         // add additional cross-group edges for unconnected groups
         for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
            if (connected_groups.find(out_group_id) == connected_groups.end())
            {
               IdxType cnt = 0;
               for (auto vec_id = cur_range.first; vec_id < cur_range.second && cnt < _num_cross_edges; ++vec_id)
               {
                  auto search_cache = search_cache_list.get_free_cache();
                  _vamana_instances[out_group_id]->iterate_to_fixed_point(_base_storage->get_vector(vec_id), search_cache);

                  for (auto k = 0; k < search_cache->search_queue.size() && k < _num_cross_edges / 2; ++k)
                  {
                     additional_edges[group_id].emplace_back(vec_id,
                                                             search_cache->search_queue[k].id + _group_id_to_range[out_group_id].first);
                     cnt += 1;
                  }
                  search_cache_list.release_cache(search_cache);
               }
            }
      }

      // add offset for uni-nav graph
      add_offset_for_uni_nav_graph();

// merge cross-group edges
#pragma omp parallel for schedule(dynamic, 4096)
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         for (auto k = 0; k < cross_group_neighbors[point_id].size(); ++k)
            _graph->neighbors[point_id].emplace_back(cross_group_neighbors[point_id][k].id);

// merge additional cross-group edges
#pragma omp parallel for schedule(dynamic, 256)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         for (const auto &[from_id, to_id] : additional_edges[group_id])
            _graph->neighbors[from_id].emplace_back(to_id);
      }

      _build_cross_edges_time = std::chrono::duration<double, std::milli>(
                                    std::chrono::high_resolution_clock::now() - start_time)
                                    .count();
      std::cout << "\r- Finish in " << _build_cross_edges_time << " ms" << std::endl;
   }

   void UniNavGraph::search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler,
                            uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                            IdxType K, std::pair<IdxType, float> *results, std::vector<float> &num_cmps,
                            std::vector<std::bitset<10000001>> &bitmap)
   {
      auto num_queries = query_storage->get_num_points();
      _query_storage = query_storage;
      _distance_handler = distance_handler;
      _scenario = scenario;

      // preparation
      if (K > Lsearch)
      {
         std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
         exit(-1);
      }
      SearchCacheList search_cache_list(num_threads, _num_points, Lsearch);

      // run queries
      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
      for (auto id = 0; id < num_queries; ++id)
      {
         auto search_cache = search_cache_list.get_free_cache();
         const char *query = _query_storage->get_vector(id);
         SearchQueue cur_result;

         // for overlap or nofilter scenario
         if (scenario == "overlap" || scenario == "nofilter")
         {
            num_cmps[id] = 0;
            search_cache->visited_set.clear();
            cur_result.reserve(K);

            // obtain entry group
            std::vector<IdxType> entry_group_ids;
            if (scenario == "overlap")
               get_min_super_sets(_query_storage->get_label_set(id), entry_group_ids, false, false);
            else
               get_min_super_sets({}, entry_group_ids, true, true);

            // for each entry group
            for (const auto &group_id : entry_group_ids)
            {
               std::vector<IdxType> entry_points;
               get_entry_points_given_group_id(num_entry_points, search_cache->visited_set, group_id, entry_points);

               // graph search and dump to current result
               num_cmps[id] += iterate_to_fixed_point(query, search_cache, id, entry_points, true, false);
               for (auto k = 0; k < search_cache->search_queue.size() && k < K; ++k)
                  cur_result.insert(search_cache->search_queue[k].id, search_cache->search_queue[k].distance);
            }

            // for the other scenarios: containment, equality
         }
         else
         {

            // obtain entry points
            auto entry_points = get_entry_points(_query_storage->get_label_set(id), num_entry_points, search_cache->visited_set);
            if (entry_points.empty())
            {
               num_cmps[id] = 0;
               for (auto k = 0; k < K; ++k)
                  results[id * K + k].first = -1;
               continue;
            }

            // graph search
            num_cmps[id] = iterate_to_fixed_point(query, search_cache, id, entry_points);
            cur_result = search_cache->search_queue;
         }

         // write results
         for (auto k = 0; k < K; ++k)
         {
            if (k < cur_result.size())
            {
               results[id * K + k].first = _new_to_old_vec_ids[cur_result[k].id];
               results[id * K + k].second = cur_result[k].distance;
            }
            else
               results[id * K + k].first = -1;
         }

         // clean
         search_cache_list.release_cache(search_cache);
      }
   }

   // fxy_add
   void UniNavGraph::search_hybrid(std::shared_ptr<IStorage> query_storage,
                                   std::shared_ptr<DistanceHandler> distance_handler,
                                   uint32_t num_threads, IdxType Lsearch,
                                   IdxType num_entry_points, std::string scenario,
                                   IdxType K, std::pair<IdxType, float> *results,
                                   std::vector<float> &num_cmps,
                                   std::vector<QueryStats> &query_stats,
                                   std::vector<std::bitset<10000001>> &bitmaps,
                                   bool is_ori_ung)
   {
      auto num_queries = query_storage->get_num_points();
      _query_storage = query_storage;
      _distance_handler = distance_handler;
      _scenario = scenario;

      // 初始化统计信息
      query_stats.resize(num_queries);

      // 参数检查
      if (K > Lsearch)
      {
         std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
         exit(-1);
      }

      // 搜索参数
      const float COVERAGE_THRESHOLD = 0.8f;
      const int MIN_LNG_DESCENDANTS_THRESHOLD = _num_points / 2.5;
      SearchCacheList search_cache_list(num_threads, _num_points, Lsearch);

      // 并行查询处理
      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
      for (auto id = 0; id < num_queries; ++id)
      {
         auto &stats = query_stats[id];
         auto total_search_start_time = std::chrono::high_resolution_clock::now();

         auto search_cache = search_cache_list.get_free_cache();
         const char *query = _query_storage->get_vector(id);
         SearchQueue cur_result;
         cur_result.reserve(K);

         // 获取查询标签集
         const auto &query_labels = _query_storage->get_label_set(id);
         for (auto i = 0; i < query_labels.size(); ++i)
         {
            std::cout << query_labels[i] << " ";
         }

         // 计算入口组信息
         std::vector<IdxType> entry_group_ids;
         get_min_super_sets(query_labels, entry_group_ids, true, true);
         stats.num_entry_points = entry_group_ids.size();

         // 使用局部作用域限制变量生命周期
         auto flag_start_time = std::chrono::high_resolution_clock::now();

         // 1. 处理 descendants
         stats.num_lng_descendants = [&]()
         {
            roaring::Roaring combined_descendants;
            auto desc_start = std::chrono::high_resolution_clock::now();
            for (auto group_id : entry_group_ids)
            {
               if (group_id > 0 && group_id <= _num_groups)
               {
                  combined_descendants |= _lng_descendants_rb[group_id];
               }
            }
            auto desc_end = std::chrono::high_resolution_clock::now();
            stats.descendants_merge_time_ms = std::chrono::duration<double, std::milli>(desc_end - desc_start).count();
            return combined_descendants.cardinality();
         }();

         // 2. 处理 coverage
         float total_unique_coverage = [&]()
         {
            roaring::Roaring combined_coverage;
            auto cov_start = std::chrono::high_resolution_clock::now();
            for (auto group_id : entry_group_ids)
            {
               if (group_id > 0 && group_id <= _num_groups)
               {
                  combined_coverage |= _covered_sets_rb[group_id];
               }
            }
            auto cov_end = std::chrono::high_resolution_clock::now();
            stats.coverage_merge_time_ms = std::chrono::duration<double, std::milli>(cov_end - cov_start).count();
            return static_cast<float>(combined_coverage.cardinality()) / _num_points;
         }();

         stats.entry_group_total_coverage = total_unique_coverage;
         bool use_global_search = (total_unique_coverage > COVERAGE_THRESHOLD) ||
                                  (stats.num_lng_descendants > MIN_LNG_DESCENDANTS_THRESHOLD);

         stats.flag_time_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - flag_start_time)
                                  .count();

         if (is_ori_ung)
            use_global_search = false;
         stats.is_global_search = use_global_search;

         // 4. 执行搜索
         if (use_global_search)
         {
            // 4.1 全局图搜索模式
            search_cache->visited_set.clear();

            // 获取全局入口点
            std::vector<IdxType> global_entry_points;
            if (_global_vamana_entry_point != -1)
            {
               global_entry_points.push_back(_global_vamana_entry_point);
            }
            else
            {
               // 随机选择入口点
               for (int i = 0; i < num_entry_points; ++i)
               {
                  global_entry_points.push_back(rand() % _num_points);
               }
            }

            // 记录初始距离计算次数
            num_cmps[id] = iterate_to_fixed_point_global(query, search_cache, id, global_entry_points);
            stats.num_distance_calcs = num_cmps[id];

            // 过滤结果
            int valid_count = 0;
            for (size_t k = 0; k < search_cache->search_queue.size() && valid_count < K; k++)
            {
               auto candidate = search_cache->search_queue[k];
               const auto &candidate_labels = _base_storage->get_label_set(candidate.id);

               // 检查候选是否满足查询条件
               bool is_valid = true;
               if (scenario == "equality")
               {
                  is_valid = (candidate_labels == query_labels);
               }

               // 使用bitmaps进行过滤
               if (bitmaps.size() > id && bitmaps[id].size() > candidate.id)
               {
                  if (scenario == "containment")
                  {
                     is_valid = bitmaps[id][candidate.id];
                  }
                  else
                  {
                     is_valid = is_valid && bitmaps[id][candidate.id];
                  }
               }

               if (is_valid)
               {
                  cur_result.insert(candidate.id, candidate.distance);
                  valid_count++;
               }
            }
         }
         else
         {
            // 4.2 传统搜索模式
            search_cache->visited_set.clear();

            if (scenario == "overlap" || scenario == "nofilter")
            {
               // 获取入口点
               std::vector<IdxType> entry_points;
               for (const auto &group_id : entry_group_ids)
               {
                  std::vector<IdxType> group_entry_points;
                  get_entry_points_given_group_id(num_entry_points,
                                                  search_cache->visited_set,
                                                  group_id,
                                                  group_entry_points);
                  entry_points.insert(entry_points.end(),
                                      group_entry_points.begin(),
                                      group_entry_points.end());
               }

               // 执行搜索
               num_cmps[id] = iterate_to_fixed_point(query, search_cache, id,
                                                     entry_points, true, false);
               stats.num_distance_calcs = num_cmps[id];

               // 收集结果
               for (auto k = 0; k < search_cache->search_queue.size() && k < K; ++k)
               {
                  cur_result.insert(search_cache->search_queue[k].id,
                                    search_cache->search_queue[k].distance);
               }
            }
            else
            {
               // containment/equality场景
               auto entry_points = get_entry_points(query_labels, num_entry_points,
                                                    search_cache->visited_set);
               if (entry_points.empty())
               {
                  stats.num_distance_calcs = 0;
                  continue;
               }

               num_cmps[id] = iterate_to_fixed_point(query, search_cache, id, entry_points);
               stats.num_distance_calcs = num_cmps[id];
               cur_result = search_cache->search_queue;
            }
         }

         // 5. 记录结果
         for (auto k = 0; k < K; ++k)
         {
            if (k < cur_result.size())
            {
               results[id * K + k].first = _new_to_old_vec_ids[cur_result[k].id];
               results[id * K + k].second = cur_result[k].distance;
            }
            else
            {
               results[id * K + k].first = -1;
            }
         }

         // 6. 记录统计信息
         stats.time_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - total_search_start_time)
                             .count();

         search_cache_list.release_cache(search_cache);
      }
   }

   std::vector<IdxType> UniNavGraph::get_entry_points(const std::vector<LabelType> &query_label_set,
                                                      IdxType num_entry_points, VisitedSet &visited_set)
   {
      std::vector<IdxType> entry_points;
      entry_points.reserve(num_entry_points);
      visited_set.clear();

      // obtain entry points for label-equality scenario
      if (_scenario == "equality")
      {
         auto node = _trie_index.find_exact_match(query_label_set);
         if (node == nullptr)
            return entry_points;
         get_entry_points_given_group_id(num_entry_points, visited_set, node->group_id, entry_points);

         // obtain entry points for label-containment scenario
      }
      else if (_scenario == "containment")
      {
         std::vector<IdxType> min_super_set_ids;
         get_min_super_sets(query_label_set, min_super_set_ids);
         for (auto group_id : min_super_set_ids)
            get_entry_points_given_group_id(num_entry_points, visited_set, group_id, entry_points);
      }
      else
      {
         std::cerr << "Error: invalid scenario " << _scenario << std::endl;
         exit(-1);
      }

      return entry_points;
   }

   void UniNavGraph::get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet &visited_set,
                                                     IdxType group_id, std::vector<IdxType> &entry_points)
   {
      const auto &group_range = _group_id_to_range[group_id];

      // not enough entry points, use all of them
      if (group_range.second - group_range.first <= num_entry_points)
      {
         for (auto i = 0; i < group_range.second - group_range.first; ++i)
            entry_points.emplace_back(i + group_range.first);
         return;
      }

      // add the entry point of the group
      const auto &group_entry_point = _group_entry_points[group_id];
      visited_set.set(group_entry_point);
      entry_points.emplace_back(group_entry_point);

      // randomly sample the other entry points
      for (auto i = 1; i < num_entry_points; ++i)
      {
         auto entry_point = rand() % (group_range.second - group_range.first) + group_range.first;
         if (visited_set.check(entry_point) == false)
         {
            visited_set.set(entry_point);
            entry_points.emplace_back(i + group_range.first);
         }
      }
   }

   IdxType UniNavGraph::iterate_to_fixed_point(const char *query, std::shared_ptr<SearchCache> search_cache,
                                               IdxType target_id, const std::vector<IdxType> &entry_points,
                                               bool clear_search_queue, bool clear_visited_set)
   {
      auto dim = _base_storage->get_dim();
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      std::vector<IdxType> neighbors;
      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      // entry point
      for (const auto &entry_point : entry_points)
         search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
      IdxType num_cmps = entry_points.size();

      // greedily expand closest nodes
      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();

         // iterate neighbors
         {
            std::lock_guard<std::mutex> lock(_graph->neighbor_locks[cur.id]);
            neighbors = _graph->neighbors[cur.id];
         }
         for (auto i = 0; i < neighbors.size(); ++i)
         {

            // prefetch
            if (i + 1 < neighbors.size() && visited_set.check(neighbors[i + 1]) == false)
               _base_storage->prefetch_vec_by_id(neighbors[i + 1]);

            // skip if visited
            auto &neighbor = neighbors[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);

            // push to search queue
            search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
            num_cmps++;
         }
      }
      return num_cmps;
   }

   // fxy_add
   IdxType UniNavGraph::iterate_to_fixed_point_global(const char *query, std::shared_ptr<SearchCache> search_cache,
                                                      IdxType target_id, const std::vector<IdxType> &entry_points,
                                                      bool clear_search_queue, bool clear_visited_set)
   {
      auto dim = _base_storage->get_dim();
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      std::vector<IdxType> neighbors;
      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      // entry point
      for (const auto &entry_point : entry_points)
         search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
      IdxType num_cmps = entry_points.size();

      // greedily expand closest nodes
      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();

         // iterate neighbors
         {
            std::lock_guard<std::mutex> lock(_global_graph->neighbor_locks[cur.id]);
            neighbors = _global_graph->neighbors[cur.id];
         }
         for (auto i = 0; i < neighbors.size(); ++i)
         {

            // prefetch
            if (i + 1 < neighbors.size() && visited_set.check(neighbors[i + 1]) == false)
               _base_storage->prefetch_vec_by_id(neighbors[i + 1]);

            // skip if visited
            auto &neighbor = neighbors[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);

            // push to search queue
            search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
            num_cmps++;
         }
      }
      return num_cmps;
   }

   void UniNavGraph::save(std::string index_path_prefix, std::string results_path_prefix)
   {
      fs::create_directories(index_path_prefix);
      auto start_time = std::chrono::high_resolution_clock::now();

      // save meta data
      std::map<std::string, std::string> meta_data;
      statistics();
      meta_data["num_points"] = std::to_string(_num_points);
      meta_data["num_groups"] = std::to_string(_num_groups);
      meta_data["index_name"] = _index_name;
      meta_data["max_degree"] = std::to_string(_max_degree);
      meta_data["Lbuild"] = std::to_string(_Lbuild);
      meta_data["alpha"] = std::to_string(_alpha);
      meta_data["build_num_threads"] = std::to_string(_num_threads);
      meta_data["scenario"] = _scenario;
      meta_data["num_cross_edges"] = std::to_string(_num_cross_edges);
      meta_data["graph_num_edges"] = std::to_string(_graph_num_edges);
      meta_data["LNG_num_edges"] = std::to_string(_LNG_num_edges);
      meta_data["index_size(MB)"] = std::to_string(_index_size);
      meta_data["index_time(ms)"] = std::to_string(_index_time);
      meta_data["label_processing_time(ms)"] = std::to_string(_label_processing_time);
      meta_data["build_graph_time(ms)"] = std::to_string(_build_graph_time);
      meta_data["build_vector_attr_graph_time(ms)"] = std::to_string(_build_vector_attr_graph_time);
      meta_data["cal_descendants_time(ms)"] = std::to_string(_cal_descendants_time);
      meta_data["cal_coverage_ratio_time(ms)"] = std::to_string(_cal_coverage_ratio_time);
      meta_data["build_LNG_time(ms)"] = std::to_string(_build_LNG_time);
      meta_data["build_cross_edges_time(ms)"] = std::to_string(_build_cross_edges_time);
      std::string meta_filename = index_path_prefix + "meta";
      write_kv_file(meta_filename, meta_data);

      // save build_time to csv
      std::string build_time_filename = results_path_prefix + "build_time.csv";
      std::ofstream build_time_file(build_time_filename);
      build_time_file << "Index Name,Build Time (ms)\n";
      build_time_file << "index_time" << "," << _index_time << "\n";
      build_time_file << "label_processing_time" << "," << _label_processing_time << "\n";
      build_time_file << "build_graph_time" << "," << _build_graph_time << "\n";
      build_time_file << "build_vector_attr_graph_time" << "," << _build_vector_attr_graph_time << "\n";
      build_time_file << "cal_descendants_time" << "," << _cal_descendants_time << "\n";
      build_time_file << "cal_coverage_ratio_time" << "," << _cal_coverage_ratio_time << "\n";
      build_time_file << "build_LNG_time" << "," << _build_LNG_time << "\n";
      build_time_file << "build_cross_edges_time" << "," << _build_cross_edges_time << "\n";
      build_time_file.close();

      // save vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage->write_to_file(bin_file, label_file);

      // save group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      write_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // save group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      write_2d_vectors(group_id_to_range_filename, _group_id_to_range);

      // save group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      write_1d_vector(group_entry_points_filename, _group_entry_points);

      // save new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      write_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

      // save trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.save(trie_filename);

      // save graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph->save(graph_filename);

      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph->save(global_graph_filename);

      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      write_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      // save LNG coverage ratio
      std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
      write_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);

      // save covered_sets in LNG
      std::string covered_sets_filename = index_path_prefix + "covered_sets";
      write_2d_vectors(covered_sets_filename, _label_nav_graph->covered_sets);
      std::cout << "LNG covered_sets saved." << std::endl;

      // save LNG descendant num
      std::string lng_descendants_num_filename = index_path_prefix + "lng_descendants_num";
      write_1d_pair_vector(lng_descendants_num_filename, _label_nav_graph->_lng_descendants_num);

      // save LNG descendants
      std::string lng_descendants_filename = index_path_prefix + "lng_descendants";
      write_2d_vectors(lng_descendants_filename, _label_nav_graph->_lng_descendants);

      // save vector attr graph data
      std::string vector_attr_graph_filename = index_path_prefix + "vector_attr_graph";
      save_bipartite_graph(vector_attr_graph_filename);

      // // save _lng_descendants_bits and _covered_sets_bits
      // std::string lng_descendants_bits_filename = index_path_prefix + "lng_descendants_bits";
      // write_bitset_vector(lng_descendants_bits_filename, _lng_descendants_bits);
      // std::string covered_sets_bits_filename = index_path_prefix + "covered_sets_bits";
      // write_bitset_vector(covered_sets_bits_filename, _covered_sets_bits);

      // save lng_descendants_rb and _covered_sets_rb
      std::string lng_descendants_rb_filename = index_path_prefix + "lng_descendants_rb.bin";
      save_roaring_vector(lng_descendants_rb_filename, _lng_descendants_rb);
      std::string covered_sets_rb_filename = index_path_prefix + "covered_sets_rb.bin";
      save_roaring_vector(covered_sets_rb_filename, _covered_sets_rb);

      // print
      std::cout << "- Index saved in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   void UniNavGraph::load(std::string index_path_prefix, const std::string &data_type)
   {
      std::cout << "Loading index from " << index_path_prefix << " ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // load meta data
      std::string meta_filename = index_path_prefix + "meta";
      auto meta_data = parse_kv_file(meta_filename);
      _num_points = std::stoi(meta_data["num_points"]);

      // load vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage = create_storage(data_type, false);
      _base_storage->load_from_file(bin_file, label_file);

      // load group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      load_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // load group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      load_2d_vectors(group_id_to_range_filename, _group_id_to_range);

      // load group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      load_1d_vector(group_entry_points_filename, _group_entry_points);

      // load new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      load_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

      // load trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.load(trie_filename);

      // load graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _graph->load(graph_filename);

      // fxy_add:load global graph data
      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _global_graph->load(global_graph_filename);

      // fxy_add: load global vamana entry point
      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      load_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      // fxy_add: load LNG coverage ratio
      std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
      std::cout << "_label_nav_graph->coverage_ratio size: " << _label_nav_graph->coverage_ratio.size() << std::endl;
      load_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);
      std::cout << "LNG coverage ratio loaded." << std::endl;

      // fxy_add: load LNG descendants num
      std::string lng_descendants_num_filename = index_path_prefix + "lng_descendants_num";
      load_1d_pair_vector(lng_descendants_num_filename, _label_nav_graph->_lng_descendants_num);
      std::cout << "LNG descendants num loaded." << std::endl;

      // fxy_add: load LNG descendants
      std::string lng_descendants_filename = index_path_prefix + "lng_descendants";
      load_2d_vectors(lng_descendants_filename, _label_nav_graph->_lng_descendants);

      // fxy_add: load covered_sets in LNG
      std::string covered_sets_filename = index_path_prefix + "covered_sets";
      load_2d_vectors(covered_sets_filename, _label_nav_graph->covered_sets);
      std::cout << "LNG covered_sets loaded." << std::endl;

      // // fxy_add: load  _lng_descendants_bits and _covered_sets_bits
      // std::string lng_descendants_bits_filename = index_path_prefix + "lng_descendants_bits";
      // load_bitset_vector(lng_descendants_bits_filename, _lng_descendants_bits);
      // std::cout << "_lng_descendants_bits loaded." << std::endl;
      // std::string covered_sets_bits_filename = index_path_prefix + "covered_sets_bits";
      // load_bitset_vector(covered_sets_bits_filename, _covered_sets_bits);
      // std::cout << "_covered_sets_bits loaded." << std::endl;

      // fxy_add: load lng_descendants_rb and _covered_sets_rb
      std::string lng_descendants_rb_filename = index_path_prefix + "lng_descendants_rb.bin";
      load_roaring_vector(lng_descendants_rb_filename, _lng_descendants_rb);
      std::cout << "_lng_descendants_rb loaded." << std::endl;
      std::string covered_sets_rb_filename = index_path_prefix + "covered_sets_rb.bin";
      load_roaring_vector(covered_sets_rb_filename, _covered_sets_rb);
      std::cout << "_covered_sets_rb loaded." << std::endl;

      // print
      std::cout << "- Index loaded in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   void UniNavGraph::statistics()
   {

      // number of edges in the unified navigating graph
      _graph_num_edges = 0;
      for (IdxType i = 0; i < _num_points; ++i)
         _graph_num_edges += _graph->neighbors[i].size();

      // number of edges in the label navigating graph
      _LNG_num_edges = 0;
      if (_label_nav_graph != nullptr)
         for (IdxType i = 1; i <= _num_groups; ++i)
            _LNG_num_edges += _label_nav_graph->out_neighbors[i].size();

      // index size
      _index_size = 0;
      for (IdxType i = 1; i <= _num_groups; ++i)
         _index_size += _group_id_to_label_set[i].size() * sizeof(LabelType);
      _index_size += _group_id_to_range.size() * sizeof(IdxType) * 2;
      _index_size += _group_entry_points.size() * sizeof(IdxType);
      _index_size += _new_to_old_vec_ids.size() * sizeof(IdxType);
      _index_size += _trie_index.get_index_size();
      _index_size += _graph->get_index_size();

      // return as MB
      _index_size /= 1024 * 1024;
   }

   // ===================================begin：生成query task========================================
   // fxy_add: 根据LNG中f点，生成查询向量和标签
   void UniNavGraph::query_generate(std::string &output_prefix, int n, float keep_prob, bool stratified_sampling, bool verify)
   {
      std::ofstream fvec_file(output_prefix + FVEC_EXT, std::ios::binary);
      std::ofstream txt_file(output_prefix + "_labels" + TXT_EXT);

      uint32_t dim = _base_storage->get_dim();
      std::cout << "Vector dimension: " << dim << std::endl;
      std::cout << "Number of points: " << _num_points << std::endl;

      // 随机数生成器
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

      std::vector<QueryTask> all_queries;
      size_t total_queries = 0;

      // min(7000个查询,_num_groups)
      total_queries = std::min(300, int(_num_groups));
      for (ANNS::IdxType group_id = 1; group_id <= total_queries; ++group_id)
      {
         // std::cout << "group_id: " << group_id << std::endl;
         auto [start, end] = _group_id_to_range[group_id];
         size_t group_size = end - start;
         if (group_size == 0)
            continue;

         // 计算该组实际采样数量
         int sample_num = n;
         if (stratified_sampling)
         {
            sample_num = std::max(1, static_cast<int>(n * group_size / _num_points)); // 分层采样：按组大小比例调整采样数
         }
         sample_num = std::min(sample_num, static_cast<int>(group_size));

         // 非重复采样
         std::vector<ANNS::IdxType> vec_ids(group_size);
         std::iota(vec_ids.begin(), vec_ids.end(), start);
         std::shuffle(vec_ids.begin(), vec_ids.end(), gen);

         for (int i = 0; i < sample_num; ++i) // 每个组采样的个数
         {
            ANNS::IdxType vec_id = vec_ids[i];
            if (_base_storage->get_label_set(vec_id).empty())
            {
               continue; // 跳过无 base 属性的向量
            }
            QueryTask task;
            task.vec_id = vec_id;

            // 生成查询标签集
            for (auto label : _group_id_to_label_set[group_id])
            {
               if (prob_dist(gen) <= keep_prob)
               {
                  task.labels.push_back(label);
               }
            }

            // 确保至少保留一个标签
            if (task.labels.empty())
            {
               task.labels.push_back(*_group_id_to_label_set[group_id].begin());
            }

            // 验证标签是组的子集
            if (verify)
            {
               auto &group_labels = _group_id_to_label_set[group_id];
               for (auto label : task.labels)
               {
                  assert(std::find(group_labels.begin(), group_labels.end(), label) != group_labels.end());
               }
            }

            all_queries.push_back(task);
         }
      }

      // 写入fvecs文件前验证标签
      std::ofstream verify_file(output_prefix + "_verify.txt"); // 新增验证输出文件
      for (const auto &task : all_queries)
      {
         // 获取基础存储中的原始标签集
         const auto &original_labels = _base_storage->get_label_set(task.vec_id);

         // 写入验证文件（无论是否开启verify都记录）
         verify_file << task.vec_id << " base_labels:";
         for (auto l : original_labels)
            verify_file << " " << l;
         verify_file << " query_labels:";
         for (auto l : task.labels)
            verify_file << " " << l;
         verify_file << "\n";

         // 验证查询标签是原始标签的子集
         if (verify)
         {
            for (auto label : task.labels)
            {
               if (std::find(original_labels.begin(), original_labels.end(), label) == original_labels.end())
               {
                  std::cerr << "Error: Label " << label << " not found in original label set for vector " << task.vec_id << std::endl;
                  std::cerr << "Original labels: ";
                  for (auto l : original_labels)
                     std::cerr << l << " ";
                  std::cerr << "\nQuery labels: ";
                  for (auto l : task.labels)
                     std::cerr << l << " ";
                  std::cerr << std::endl;
                  throw std::runtime_error("Label verification failed");
               }
            }
         }
      }
      verify_file.close(); // 关闭验证文件

      // 写入fvecs文件
      for (const auto &task : all_queries)
      {
         const char *vec_data = _base_storage->get_vector(task.vec_id);
         fvec_file.write((char *)&dim, sizeof(uint32_t)); // 每个向量前写维度
         fvec_file.write(vec_data, dim * sizeof(float));
      }

      // 写入txt文件
      for (const auto &task : all_queries)
      {
         // txt_file << task.vec_id;
         for (auto label : task.labels)
         {
            txt_file << label << ",";
         }
         txt_file << "\n";
      }

      std::cout << "Generated " << all_queries.size() << " queries\n";
      std::cout << "FVECS file: " << output_prefix + FVEC_EXT << "\n";
      std::cout << "TXT file: " << output_prefix + TXT_EXT << "\n";
   }

   // fxy_add:根据LNG中f点，生成多个查询任务
   void UniNavGraph::generate_multiple_queries(
       std::string dataset,
       UniNavGraph &index,
       const std::string &base_output_path,
       int num_sets,
       int n_per_set,
       float keep_prob,
       bool stratified_sampling,
       bool verify)
   {
      std::cout << "enter generate_multiple_queries" << std::endl;
      namespace fs = std::filesystem;

      // 确保基础目录存在
      fs::create_directories(base_output_path);

      for (int i = 1; i <= num_sets; ++i)
      {
         std::cout << "Generating query set " << i << "..." << std::endl;
         std::string folder_name = base_output_path;

         // 创建目录（包括所有必要的父目录）
         fs::create_directories(folder_name);

         std::string output_prefix = folder_name + "/" + dataset + "_query"; // 路径在文件夹内
         index.query_generate(output_prefix, n_per_set, keep_prob, stratified_sampling, verify);

         std::cout << "Generated query set " << i << " at " << folder_name << std::endl;
      }
   }

   // fxy_add:极端方法1数据，生成覆盖率高的查询任务
   void UniNavGraph::generate_queries_method1_high_coverage(std::string &output_prefix, std::string dataset, int query_n, std::string &base_label_file, float coverage_threshold)
   {
      // Step 1: 读取base_label_file并统计每列标签频率
      std::ifstream label_file(base_label_file);
      if (!label_file.is_open())
      {
         std::cerr << "Failed to open label file: " << base_label_file << std::endl;
         return;
      }

      std::vector<std::unordered_map<int, int>> column_label_counts;
      std::vector<std::vector<int>> all_label_rows;
      std::string line;
      int total_lines = 0;
      size_t num_columns = 0;

      while (std::getline(label_file, line))
      {
         std::vector<int> labels;
         std::stringstream ss(line);
         std::string label_str;

         while (std::getline(ss, label_str, ','))
         {
            labels.push_back(std::stoi(label_str));
         }

         if (!labels.empty())
         {
            if (num_columns == 0)
            {
               num_columns = labels.size();
               column_label_counts.resize(num_columns);
            }
            if (labels.size() == num_columns)
            {
               for (size_t col = 0; col < num_columns; ++col)
               {
                  column_label_counts[col][labels[col]]++;
               }
               all_label_rows.push_back(labels);
               total_lines++;
            }
         }
      }
      label_file.close();

      if (total_lines == 0 || num_columns == 0)
      {
         std::cerr << "No valid labels found in base label file" << std::endl;
         return;
      }

      // Step 2: 计算每列标签覆盖率，筛选高覆盖率标签
      struct LabelInfo
      {
         int label;
         float coverage;
         size_t column; // 记录标签所属列
      };
      std::vector<std::vector<LabelInfo>> high_coverage_labels_per_column(num_columns);
      std::vector<LabelInfo> all_high_coverage_labels;

      for (size_t col = 0; col < num_columns; ++col)
      {
         for (const auto &entry : column_label_counts[col])
         {
            float coverage = static_cast<float>(entry.second) / total_lines;
            if (coverage >= coverage_threshold)
            {
               high_coverage_labels_per_column[col].push_back({entry.first, coverage, col});
               all_high_coverage_labels.push_back({entry.first, coverage, col});
            }
         }
         // 按覆盖率降序排序
         std::sort(high_coverage_labels_per_column[col].begin(), high_coverage_labels_per_column[col].end(),
                   [](const auto &a, const auto &b)
                   { return a.coverage > b.coverage; });
      }

      // Step 3: 构建最高覆盖率标签组合并计算幂集覆盖率
      std::vector<int> max_coverage_labels;
      std::vector<size_t> label_to_column;

      // 动态选择每列覆盖率最高的标签
      for (size_t col = 0; col < num_columns; ++col)
      {
         int max_label = 0;
         float max_coverage = 0.0f;
         for (const auto &entry : column_label_counts[col])
         {
            float coverage = static_cast<float>(entry.second) / total_lines;
            if (coverage > max_coverage)
            {
               max_coverage = coverage;
               max_label = entry.first;
            }
         }
         if (max_coverage > 0.0f)
         {
            max_coverage_labels.push_back(max_label);
            label_to_column.push_back(col);
         }
      }

      std::vector<std::pair<std::vector<int>, float>> high_coverage_combinations;

      // 生成幂集（非空子集）
      if (!max_coverage_labels.empty())
      {
         int total_subsets = (1 << max_coverage_labels.size()) - 1;
         for (int mask = 1; mask <= total_subsets; ++mask)
         {
            std::vector<int> subset;
            std::vector<size_t> subset_columns; // 记录子集对应的列
            for (size_t i = 0; i < max_coverage_labels.size(); ++i)
            {
               if (mask & (1 << i))
               {
                  subset.push_back(max_coverage_labels[i]);
                  subset_columns.push_back(label_to_column[i]);
               }
            }

            // 计算子集覆盖率
            int combo_count = 0;
            for (const auto &row : all_label_rows)
            {
               bool match = true;
               for (size_t i = 0; i < subset.size(); ++i)
               {
                  if (row[subset_columns[i]] != subset[i])
                  {
                     match = false;
                     break;
                  }
               }
               if (match)
                  combo_count++;
            }
            float coverage = static_cast<float>(combo_count) / total_lines;
            if (coverage >= coverage_threshold)
            {
               high_coverage_combinations.emplace_back(subset, coverage);
            }
         }
      }

      // 按覆盖率降序排序组合
      std::sort(high_coverage_combinations.begin(), high_coverage_combinations.end(),
                [](const auto &a, const auto &b)
                { return a.second > b.second; });

      // Step 4: 准备候选标签（组合 + 单个标签）
      struct CandidateLabel
      {
         std::vector<int> labels; // 组合或单个标签
         float coverage;
         bool is_combination; // 是否为组合
      };
      std::vector<CandidateLabel> candidate_labels;

      // 添加高覆盖率组合
      for (const auto &combo : high_coverage_combinations)
      {
         candidate_labels.push_back({combo.first, combo.second, true});
      }

      // 添加单个高覆盖率标签
      for (const auto &label_info : all_high_coverage_labels)
      {
         candidate_labels.push_back({{label_info.label}, label_info.coverage, false});
      }

      if (candidate_labels.empty())
      {
         std::cerr << "No labels or combinations with coverage >= " << coverage_threshold << std::endl;
         return;
      }

      // Step 5: 准备输出文件
      std::ofstream txt_file(output_prefix + "/" + dataset + "_query_labels.txt");
      std::ofstream fvec_file(output_prefix + "/" + dataset + "_query.fvecs", std::ios::binary);
      std::ofstream stats_file(output_prefix + "/" + dataset + "_query_stats.txt");

      if (!txt_file.is_open() || !fvec_file.is_open() || !stats_file.is_open())
      {
         std::cerr << "Failed to open output files" << std::endl;
         return;
      }

      // Step 6: 写入统计信息
      stats_file << "Coverage Threshold: " << coverage_threshold << std::endl;
      stats_file << "Number of Columns: " << num_columns << std::endl;
      stats_file << "Total Rows: " << total_lines << std::endl;
      stats_file << "\nPer-Column Max-Coverage Labels:\n";
      for (size_t col = 0; col < num_columns; ++col)
      {
         stats_file << "Column " << col + 1 << ":\n";
         // 找到该列覆盖率最高的标签
         int max_label = 0;
         float max_coverage = 0.0f;
         for (const auto &entry : column_label_counts[col])
         {
            float coverage = static_cast<float>(entry.second) / total_lines;
            if (coverage > max_coverage)
            {
               max_coverage = coverage;
               max_label = entry.first;
            }
         }
         if (max_coverage > 0.0f)
         {
            stats_file << "  Label: " << max_label << ", Coverage: " << max_coverage << "\n";
         }
         else
         {
            stats_file << "  No labels found\n";
         }
      }
      stats_file << "\nHigh-Coverage Combinations (from max coverage labels):\n";
      for (const auto &combo : high_coverage_combinations)
      {
         stats_file << "  Labels: ";
         for (size_t i = 0; i < combo.first.size(); ++i)
         {
            stats_file << combo.first[i];
            if (i != combo.first.size() - 1)
               stats_file << ",";
         }
         stats_file << ", Coverage: " << combo.second << "\n";
      }

      // Step 7: 生成查询任务
      uint32_t dim = _base_storage->get_dim();
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<size_t> label_dis(0, candidate_labels.size() - 1);
      std::uniform_int_distribution<ANNS::IdxType> vec_dis(0, _base_storage->get_num_points() - 1);
      std::unordered_set<ANNS::IdxType> used_vec_ids;

      for (int i = 0; i < query_n; ++i)
      {
         // 选择候选标签（循环使用）
         const auto &candidate = candidate_labels[i % candidate_labels.size()];

         // 写入标签文件
         for (size_t j = 0; j < candidate.labels.size(); ++j)
         {
            txt_file << candidate.labels[j];
            if (j != candidate.labels.size() - 1)
               txt_file << ",";
         }
         txt_file << std::endl;

         // 随机选择不重复的向量
         ANNS::IdxType vec_id;
         do
         {
            vec_id = vec_dis(gen);
         } while (used_vec_ids.count(vec_id) > 0 && used_vec_ids.size() < _base_storage->get_num_points());

         if (used_vec_ids.size() >= _base_storage->get_num_points())
         {
            std::cerr << "Warning: Not enough unique vectors" << std::endl;
            break;
         }
         used_vec_ids.insert(vec_id);

         // 写入向量文件
         const char *vec_data = _base_storage->get_vector(vec_id);
         fvec_file.write((char *)&dim, sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));
      }

      txt_file.close();
      fvec_file.close();
      stats_file.close();

      // Step 8: 输出统计信息到控制台
      std::cout << "Generated " << std::min(query_n, (int)used_vec_ids.size()) << " high-coverage queries" << std::endl;
      std::cout << "Labels written to: " << output_prefix + "/" + dataset + "_query_labels.txt" << std::endl;
      std::cout << "Vectors written to: " << output_prefix + "/" + dataset + "_query.fvecs" << std::endl;
      std::cout << "Statistics written to: " << output_prefix + "/" + dataset + "_query_stats.txt" << std::endl;
   }

   // fxy_add:极端方法1数据，生成覆盖率低的查询任务:选出覆盖率在 (0, coverage_threshold] 区间且出现次数 ≥ K 的组合
   void UniNavGraph::generate_queries_method1_low_coverage(
       std::string &output_prefix,
       std::string dataset,
       int query_n,
       std::string &base_label_file,
       int num_of_per_query_labels,
       float coverage_threshold,
       int K)
   {
      // Step 1: 从标签文件加载并分析标签分布
      std::ifstream label_file(base_label_file);
      if (!label_file.is_open())
      {
         std::cerr << "Failed to open label file: " << base_label_file << std::endl;
         return;
      }

      // 统计标签出现频率
      std::unordered_map<int, int> label_counts;
      std::vector<std::vector<int>> all_label_sets;
      std::string line;

      while (std::getline(label_file, line))
      {
         std::vector<int> labels;
         std::stringstream ss(line);
         std::string label_str;

         while (std::getline(ss, label_str, ','))
         {
            int label = std::stoi(label_str);
            labels.push_back(label);
            label_counts[label]++;
         }

         std::sort(labels.begin(), labels.end());
         all_label_sets.push_back(labels);
      }
      label_file.close();

      // 如果没有标签数据，使用默认标签
      if (all_label_sets.empty())
      {
         std::cerr << "No label data found, using default label 1" << std::endl;
         all_label_sets.push_back({1});
         label_counts[1] = 1;
      }

      // Step 2: 识别低覆盖率标签组合（排除覆盖率为0的组合）
      struct LabelSetInfo
      {
         std::vector<int> labels;
         int count;
         float coverage;
      };

      std::vector<LabelSetInfo> valid_low_coverage_sets;
      int total_vectors = all_label_sets.size();

      // 统计标签组合的覆盖率（最多num_of_per_query_labels个标签的组合）
      std::map<std::vector<int>, int> combination_counts;
      for (const auto &labels : all_label_sets)
      {
         // 生成所有可能的1-num_of_per_query_labels标签组合
         for (int k = 1; k <= num_of_per_query_labels && k <= labels.size(); ++k)
         {
            std::vector<bool> mask(labels.size(), false);
            std::fill(mask.begin(), mask.begin() + k, true);

            do
            {
               std::vector<int> combination;
               for (size_t i = 0; i < mask.size(); ++i)
               {
                  if (mask[i])
                     combination.push_back(labels[i]);
               }
               combination_counts[combination]++;
            } while (std::prev_permutation(mask.begin(), mask.end()));
         }
      }

      // 筛选出覆盖率大于0且低于阈值的组合
      for (const auto &entry : combination_counts)
      {
         // 跳过覆盖率为0的组合
         if (entry.second == 0)
            continue;

         float coverage = static_cast<float>(entry.second) / total_vectors;
         if (coverage <= coverage_threshold && entry.second >= K)
         {
            valid_low_coverage_sets.push_back({entry.first, entry.second, coverage});
         }
      }

      // 按覆盖率升序排序
      std::sort(valid_low_coverage_sets.begin(), valid_low_coverage_sets.end(),
                [](const auto &a, const auto &b)
                {
                   return a.coverage < b.coverage;
                });

      // 检查是否有足够的有效组合
      if (valid_low_coverage_sets.empty())
      {
         std::cerr << "Error: No valid low-coverage label combinations found (all either have coverage=0 or > "
                   << coverage_threshold << ")" << std::endl;
         return;
      }

      // Step 3: 生成查询文件和标签
      std::ofstream txt_file(output_prefix + "/" + dataset + "_query_labels.txt");
      std::ofstream txt__coverage_file(output_prefix + "/" + dataset + "_query_and_coverage_labels.txt");
      std::ofstream fvec_file(output_prefix + "/" + dataset + "_query.fvecs", std::ios::binary);

      if (!txt_file.is_open() || !fvec_file.is_open())
      {
         std::cerr << "Failed to open output files." << std::endl;
         return;
      }

      uint32_t dim = _base_storage->get_dim();
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<ANNS::IdxType> vec_dis(0, _base_storage->get_num_points() - 1);
      std::unordered_set<ANNS::IdxType> used_vec_ids;

      // 选择前query_n个最稀有的有效标签组合
      int queries_generated = 0;
      for (size_t i = 0; i < valid_low_coverage_sets.size() && queries_generated < query_n; ++i)
      {
         const auto &label_set = valid_low_coverage_sets[i].labels;

         // 写入标签文件
         for (size_t j = 0; j < label_set.size(); ++j)
         {
            txt_file << label_set[j];
            if (j != label_set.size() - 1)
               txt_file << ",";

            txt__coverage_file << label_set[j];
            if (j != label_set.size() - 1)
               txt__coverage_file << ",";
         }
         txt__coverage_file << " coverage:" << valid_low_coverage_sets[i].coverage;
         txt_file << std::endl;
         txt__coverage_file << std::endl;

         // 随机选择一个向量（确保不重复）
         ANNS::IdxType vec_id;
         do
         {
            vec_id = vec_dis(gen);
         } while (used_vec_ids.count(vec_id) > 0 &&
                  used_vec_ids.size() < _base_storage->get_num_points());

         used_vec_ids.insert(vec_id);
         const char *vec_data = _base_storage->get_vector(vec_id);

         // 写入向量文件
         fvec_file.write((char *)&dim, sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));

         queries_generated++;
      }

      txt_file.close();
      txt__coverage_file.close();
      fvec_file.close();

      std::cout << "Generated " << queries_generated << " low-coverage queries with 0 < coverage <= "
                << coverage_threshold << std::endl;
      std::cout << "Labels written to: " << output_prefix + "/" + dataset + "_lowcov_query_labels.txt" << std::endl;
      std::cout << "Vectors written to: " << output_prefix + "/" + dataset + "_lowcov_query.fvecs" << std::endl;

      // 输出统计信息
      if (!valid_low_coverage_sets.empty())
      {
         std::cout << "\nStatistics of generated low-coverage queries:" << std::endl;
         std::cout << "Min coverage: " << valid_low_coverage_sets.front().coverage << std::endl;
         std::cout << "Max coverage: " << valid_low_coverage_sets.back().coverage << std::endl;
         std::cout << "Median coverage: "
                   << valid_low_coverage_sets[valid_low_coverage_sets.size() / 2].coverage << std::endl;
      }
   }

   // fxy_add: 极端方法2的查询任务生成
   void UniNavGraph::generate_queries_method2_high_coverage(int N, int K, int top_M_trees, std::string dataset, const std::string &output_prefix, const std::string &base_label_tree_roots)
   {
      std::cout << "\n==================================================" << std::endl;
      std::cout << "--- 开始生成极端查询任务 ---" << std::endl;

      if (_num_groups == 0)
      {
         std::cerr << "错误: 索引未构建或不包含任何分组。" << std::endl;
         return;
      }

      // --- 预处理阶段 ---

      // 1. 读取Python生成的树根标签ID
      std::cout << "步骤1: 读取 'tree_roots.txt' 文件..." << std::endl;
      std::vector<LabelType> conceptual_root_labels;
      std::ifstream roots_file(base_label_tree_roots);
      if (!roots_file.is_open())
      {
         std::cerr << "错误: 无法打开 tree_roots.txt 文件！请确保已运行Python脚本生成此文件。" << std::endl;
         return;
      }
      LabelType root_label_from_file;
      while (roots_file >> root_label_from_file)
      {
         conceptual_root_labels.push_back(root_label_from_file);
      }
      roots_file.close();
      std::cout << "成功读取 " << conceptual_root_labels.size() << " 个概念树根标签。" << std::endl;

      // 2. 识别树结构，并将概念树根(Label ID)与物理树根(Group ID)关联，同时计算覆盖率
      std::cout << "步骤2: 识别树结构并计算覆盖率..." << std::endl;
      std::vector<TreeInfo> sorted_trees;
      std::unordered_map<IdxType, IdxType> group_to_root_group;
      {
         std::vector<int> parent(_num_groups + 1, 0);
         for (ANNS::IdxType i = 1; i <= _num_groups; ++i)
         {
            for (auto child : _label_nav_graph->out_neighbors[i])
            {
               parent[child] = i;
            }
         }
         for (ANNS::IdxType i = 1; i <= _num_groups; ++i)
         {
            ANNS::IdxType current = i;
            while (parent[current] != 0)
               current = parent[current];
            group_to_root_group[i] = current;
         }
      }

      std::map<IdxType, size_t> physical_root_coverage;
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         physical_root_coverage[group_to_root_group[group_id]] += _group_id_to_vec_ids[group_id].size();
      }

      for (LabelType root_label : conceptual_root_labels)
      {
         auto node = _trie_index.find_exact_match({root_label});
         if (node)
         {
            IdxType root_group_id = node->group_id;
            if (physical_root_coverage.count(root_group_id))
            {
               sorted_trees.push_back({root_group_id, root_label, physical_root_coverage.at(root_group_id)});
            }
         }
         else
         {
            std::cout << "警告: 无法在Trie中找到标签 {" << root_label << "} 对应的Group。" << std::endl;
         }
      }
      std::sort(sorted_trees.begin(), sorted_trees.end());

      // 3. 筛选Top-M树并构建深度样本池
      if (top_M_trees > 0 && top_M_trees < sorted_trees.size())
      {
         sorted_trees.resize(top_M_trees);
      }
      std::cout << "将从覆盖率最高的 " << sorted_trees.size() << " 棵树中生成查询。" << std::endl;

      std::map<IdxType, int> tree_max_depth;
      std::map<IdxType, std::vector<IdxType>> per_tree_deep_ids;
      for (const auto &tree_info : sorted_trees)
      {
         IdxType root_group_id = tree_info.root_group_id;
         int current_max_depth = 0;
         for (ANNS::IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            if (group_to_root_group.count(group_id) && group_to_root_group.at(group_id) == root_group_id)
            {
               current_max_depth = std::max(current_max_depth, (int)_group_id_to_label_set[group_id].size());
            }
         }
         tree_max_depth[root_group_id] = current_max_depth;

         int deep_threshold = static_cast<int>(current_max_depth * 0.67);
         for (ANNS::IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            if (group_to_root_group.count(group_id) && group_to_root_group.at(group_id) == root_group_id && _group_id_to_label_set[group_id].size() > deep_threshold)
            {
               per_tree_deep_ids[root_group_id].insert(per_tree_deep_ids[root_group_id].end(), _group_id_to_vec_ids[group_id].begin(), _group_id_to_vec_ids[group_id].end());
            }
         }
         std::cout << "  - 树 (根标签/组ID: " << tree_info.root_label_id << "/" << root_group_id << "): 最大深度=" << tree_max_depth[root_group_id] << ", 深度阈值(>" << deep_threshold << "), 深度样本数=" << per_tree_deep_ids[root_group_id].size() << std::endl;
      }

      // 4. 按比例分配N个查询
      std::vector<int> query_allocations;
      size_t total_top_coverage = 0;
      for (const auto &tree_info : sorted_trees)
      {
         total_top_coverage += tree_info.coverage_count;
      }
      if (total_top_coverage > 0)
      {
         int allocated_queries = 0;
         for (size_t i = 0; i < sorted_trees.size(); ++i)
         {
            if (i == sorted_trees.size() - 1)
            {
               query_allocations.push_back(N - allocated_queries);
            }
            else
            {
               double proportion = static_cast<double>(sorted_trees[i].coverage_count) / total_top_coverage;
               int num = static_cast<int>(N * proportion);
               query_allocations.push_back(num);
               allocated_queries += num;
            }
         }
      }

      // --- 查询生成阶段 ---
      std::cout << "步骤3: 开始为K近邻场景生成查询..." << std::endl;

      std::ofstream query_vec_file(output_prefix + "/" + dataset + "_query.fvecs", std::ios::binary);
      std::ofstream query_label_file(output_prefix + "/" + dataset + "_query_labels.txt");
      std::ofstream ground_truth_file(output_prefix + "/" + dataset + "_groundtruth.txt");
      uint32_t dim = _base_storage->get_dim();

      std::unordered_map<IdxType, IdxType> old_to_new_map;
      for (IdxType new_id = 0; new_id < _num_points; ++new_id)
      {
         old_to_new_map[_new_to_old_vec_ids[new_id]] = new_id;
      }

      std::mt19937 rng(std::random_device{}());
      int queries_generated = 0;

      for (size_t i = 0; i < sorted_trees.size(); ++i)
      {
         IdxType root_group_id = sorted_trees[i].root_group_id;
         LabelType root_label_id = sorted_trees[i].root_label_id;
         int queries_to_gen_for_this_tree = query_allocations[i];
         const auto &local_deep_pool = per_tree_deep_ids[root_group_id];

         if (local_deep_pool.size() < K)
         {
            std::cout << "警告: 树 (根标签 " << root_label_id << ") 的深度样本数 (" << local_deep_pool.size() << ") 小于 K (" << K << ")，跳过此树。" << std::endl;
            continue;
         }

         std::cout << "正在为 树 (根标签 " << root_label_id << ") 生成 " << queries_to_gen_for_this_tree << " 个查询..." << std::endl;

         for (int j = 0; j < queries_to_gen_for_this_tree; ++j)
         {
            if (queries_generated >= N)
               break;

            std::uniform_int_distribution<size_t> dist(0, local_deep_pool.size() - 1);
            IdxType seed_original_id = local_deep_pool[dist(rng)];

            std::vector<float> seed_vec(dim);
            IdxType seed_new_id = old_to_new_map.at(seed_original_id);
            memcpy(seed_vec.data(), _base_storage->get_vector(seed_new_id), dim * sizeof(float));

            std::priority_queue<std::pair<float, IdxType>> top_k_queue;
            for (IdxType candidate_original_id : local_deep_pool)
            {
               if (candidate_original_id == seed_original_id)
                  continue;

               IdxType candidate_new_id = old_to_new_map.at(candidate_original_id);
               float distance = _distance_handler->compute(reinterpret_cast<const char *>(seed_vec.data()), _base_storage->get_vector(candidate_new_id), dim);

               if (top_k_queue.size() < (size_t)K - 1)
               {
                  top_k_queue.push({distance, candidate_original_id});
               }
               else if (distance < top_k_queue.top().first)
               {
                  top_k_queue.pop();
                  top_k_queue.push({distance, candidate_original_id});
               }
            }

            std::vector<IdxType> ground_truth_ids;
            ground_truth_ids.push_back(seed_original_id);
            while (!top_k_queue.empty())
            {
               ground_truth_ids.push_back(top_k_queue.top().second);
               top_k_queue.pop();
            }

            std::vector<float> centroid_vec(dim, 0.0f);
            for (IdxType gt_id : ground_truth_ids)
            {
               IdxType new_id = old_to_new_map.at(gt_id);
               const float *vec = reinterpret_cast<const float *>(_base_storage->get_vector(new_id));
               for (uint32_t d = 0; d < dim; ++d)
                  centroid_vec[d] += vec[d];
            }
            for (uint32_t d = 0; d < dim; ++d)
               centroid_vec[d] /= K;
            // 此处添加噪声和归一化 ?

            query_vec_file.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
            query_vec_file.write(reinterpret_cast<const char *>(centroid_vec.data()), dim * sizeof(float));

            query_label_file << root_label_id << "\n";

            for (size_t gt_idx = 0; gt_idx < ground_truth_ids.size(); ++gt_idx)
            {
               ground_truth_file << ground_truth_ids[gt_idx] << (gt_idx == ground_truth_ids.size() - 1 ? "" : "\t");
            }
            ground_truth_file << "\n";

            queries_generated++;
         }
         if (queries_generated >= N)
            break;
      }

      query_vec_file.close();
      query_label_file.close();
      ground_truth_file.close();

      std::cout << "\n--- 查询生成结束：共生成 " << queries_generated << " 个查询任务 ---" << std::endl;
      std::cout << "查询向量已保存到: " << output_prefix + "/" + dataset + "_query.fvecs" << std::endl;
      std::cout << "查询标签已保存到: " << output_prefix + "/" + dataset + "_query_labels.txt" << std::endl;
      std::cout << "Ground Truth已保存到: " << output_prefix + "/" + dataset + "_groundtruth.txt" << std::endl;
   }

   // fxy_add: 极端方法2的查询任务生成低覆盖率的查询任务（基于 count ∈ [K, max_K]）
   void UniNavGraph::generate_queries_method2_low_coverage(
       std::string &output_prefix,
       std::string dataset,
       int query_n,
       std::string &base_label_file,
       int num_of_per_query_labels, // 每个查询的标签数量
       int K,                       // 标签组合出现次数的下界
       int max_K,                   // 标签组合出现次数的上界
       int min_K)                   // 有效标签组合的最小数量,即符合罕见度要求的“查询模板”个数
   {
      // ==================== 阶段1：统计标签频率 ====================
      std::unordered_map<int, int> label_counts;

      // 第一次扫描：仅统计标签频率
      {
         std::ifstream label_file(base_label_file);
         if (!label_file.is_open())
         {
            std::cerr << "Failed to open label file: " << base_label_file << std::endl;
            return;
         }

         std::string line;
         while (std::getline(label_file, line))
         {
            std::stringstream ss(line);
            std::string label_str;
            while (std::getline(ss, label_str, ','))
            {
               int label = std::stoi(label_str);
               label_counts[label]++;
            }
         }
         label_file.close();
      }

      if (label_counts.empty())
      {
         std::cerr << "No labels found, using default label 1" << std::endl;
         label_counts[1] = 1;
      }

      // ==================== 阶段2：准备低频标签数据 ====================
      // 计算频率阈值（只保留出现次数<=max_K的标签）
      const int freq_threshold = max_K * 2; // 经验值，可根据数据调整

      // 第二次扫描：仅缓存包含低频标签的行
      std::vector<std::vector<int>> low_freq_label_sets;
      {
         std::ifstream label_file(base_label_file);
         std::string line;

         while (std::getline(label_file, line))
         {
            std::vector<int> labels;
            std::stringstream ss(line);
            std::string label_str;
            bool has_low_freq_label = false;

            while (std::getline(ss, label_str, ','))
            {
               int label = std::stoi(label_str);
               if (label_counts[label] <= freq_threshold)
               {
                  has_low_freq_label = true;
               }
               labels.push_back(label);
            }

            if (has_low_freq_label)
            {
               std::sort(labels.begin(), labels.end());
               low_freq_label_sets.push_back(labels);
            }
         }
         label_file.close();
      }

      // ==================== 阶段3：生成有效标签组合 ====================
      // 按频率排序标签（低频优先）
      std::vector<std::pair<int, int>> label_freq_pairs(label_counts.begin(), label_counts.end());
      std::sort(label_freq_pairs.begin(), label_freq_pairs.end(),
                [](const auto &a, const auto &b)
                { return a.second < b.second; });

      std::vector<int> sorted_labels;
      for (const auto &p : label_freq_pairs)
      {
         sorted_labels.push_back(p.first);
      }

      struct LabelSetInfo
      {
         std::vector<int> labels;
         int count;
      };
      std::vector<LabelSetInfo> valid_combinations;
      std::unordered_set<std::string> seen_combinations;

      // 生成候选组合（仅使用低频标签）
      for (size_t start_idx = 0; start_idx < sorted_labels.size() &&
                                 valid_combinations.size() < min_K;
           ++start_idx)
      {
         if (label_counts[sorted_labels[start_idx]] > freq_threshold)
            continue;

         for (int k = 1; k <= num_of_per_query_labels &&
                         k <= sorted_labels.size() - start_idx;
              ++k)
         {
            std::vector<int> candidate;
            for (int i = 0; i < k; ++i)
            {
               candidate.push_back(sorted_labels[start_idx + i]);
               if (label_counts[sorted_labels[start_idx + i]] > freq_threshold)
               {
                  candidate.clear();
                  break;
               }
            }
            if (candidate.empty())
               continue;

            // 去重检查
            std::sort(candidate.begin(), candidate.end());
            std::string key;
            for (int l : candidate)
               key += std::to_string(l) + ",";
            if (seen_combinations.count(key))
               continue;
            seen_combinations.insert(key);

            // 统计组合出现次数（使用缓存的低频数据）
            int occurrence = 0;
            for (const auto &labels : low_freq_label_sets)
            {
               bool match = true;
               for (int l : candidate)
               {
                  if (!std::binary_search(labels.begin(), labels.end(), l))
                  {
                     match = false;
                     break;
                  }
               }
               if (match)
                  occurrence++;
            }

            if (occurrence >= K && occurrence <= max_K)
            {
               valid_combinations.push_back({candidate, occurrence});
               if (valid_combinations.size() >= min_K)
                  break;
            }
         }
      }

      // ==================== 阶段4：生成查询文件 ====================
      if (valid_combinations.empty())
      {
         std::cerr << "Error: No valid combinations in range [" << K << ", " << max_K << "]" << std::endl;
         return;
      }

      std::ofstream txt_file(output_prefix + "/" + dataset + "_query_labels.txt");
      std::ofstream coverage_file(output_prefix + "/" + dataset + "_query_and_coverage_labels.txt");
      std::ofstream fvec_file(output_prefix + "/" + dataset + "_query.fvecs", std::ios::binary);

      if (!txt_file.is_open() || !fvec_file.is_open())
      {
         std::cerr << "Failed to open output files" << std::endl;
         return;
      }

      uint32_t dim = _base_storage->get_dim();
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<int> combo_dis(0, valid_combinations.size() - 1);
      std::uniform_int_distribution<ANNS::IdxType> vec_dis(0, _base_storage->get_num_points() - 1);
      std::unordered_set<ANNS::IdxType> used_vec_ids;

      for (int i = 0; i < query_n; ++i)
      {
         const auto &combo = valid_combinations[combo_dis(gen)];

         // 获取唯一向量ID
         ANNS::IdxType vec_id;
         do
         {
            vec_id = vec_dis(gen);
         } while (used_vec_ids.count(vec_id) &&
                  used_vec_ids.size() < _base_storage->get_num_points());

         if (used_vec_ids.size() >= _base_storage->get_num_points())
         {
            std::cerr << "Warning: Not enough unique vectors" << std::endl;
            break;
         }
         used_vec_ids.insert(vec_id);

         // 写入标签
         for (size_t j = 0; j < combo.labels.size(); ++j)
         {
            txt_file << combo.labels[j];
            coverage_file << combo.labels[j];
            if (j != combo.labels.size() - 1)
            {
               txt_file << ",";
               coverage_file << ",";
            }
         }
         coverage_file << " count:" << combo.count;
         txt_file << "\n";
         coverage_file << "\n";

         // 写入向量
         const char *vec_data = _base_storage->get_vector(vec_id);
         fvec_file.write((char *)&dim, sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));
      }

      // 关闭文件
      txt_file.close();
      coverage_file.close();
      fvec_file.close();

      std::cout << "Generated " << std::min(query_n, (int)valid_combinations.size())
                << " queries with count in [" << K << ", " << max_K << "]\n";
      std::cout << "Unique combinations found: " << valid_combinations.size() << "\n";
   }

   // fxy_add: 人造数据：极端方法2的查询任务生成：根据LNG树的深度信息生成查询任务
   void UniNavGraph::generate_queries_method2_high_coverage_human(
       std::string &output_prefix,
       std::string dataset,
       int query_n,
       std::string &base_label_file,
       std::string &base_label_info_file)
   {
      // Step 1: 读取LNG树信息文件获取总层数
      std::ifstream info_file(base_label_info_file);
      if (!info_file.is_open())
      {
         std::cerr << "Failed to open LNG info file: " << base_label_info_file << std::endl;
         return;
      }

      int total_layers = 0;
      std::string line;
      while (std::getline(info_file, line))
      {
         if (line.find("总层数:") != std::string::npos)
         {
            size_t pos = line.find(":");
            if (pos != std::string::npos)
            {
               total_layers = std::stoi(line.substr(pos + 1));
               break;
            }
         }
      }
      info_file.close();

      if (total_layers == 0)
      {
         std::cerr << "Error: Could not determine total layers from LNG info file" << std::endl;
         return;
      }

      // 计算顶层深度
      // int max_depth_top = std::max(1, total_layers / 10);
      int max_depth_top = 3;
      std::cout << "Total layers: " << total_layers << ", Top layers to use: " << max_depth_top << std::endl;

      // Step 2: 读取base_labels文件获取顶层标签
      std::ifstream label_file(base_label_file);
      if (!label_file.is_open())
      {
         std::cerr << "Failed to open base label file: " << base_label_file << std::endl;
         return;
      }

      std::vector<std::vector<int>> top_layer_labels;
      int lines_read = 0;
      while (std::getline(label_file, line) && lines_read < max_depth_top)
      {
         std::vector<int> labels;
         std::stringstream ss(line);
         std::string label_str;

         while (std::getline(ss, label_str, ','))
         {
            labels.push_back(std::stoi(label_str));
         }

         if (!labels.empty())
         {
            top_layer_labels.push_back(labels);
            lines_read++;
         }
      }
      label_file.close();

      if (top_layer_labels.empty())
      {
         std::cerr << "Error: No valid labels found in base label file" << std::endl;
         return;
      }

      // Step 3: 准备输出文件
      std::ofstream txt_file(output_prefix + "/" + dataset + "_query_labels.txt");
      std::ofstream fvec_file(output_prefix + "/" + dataset + "_query.fvecs", std::ios::binary);

      if (!txt_file.is_open() || !fvec_file.is_open())
      {
         std::cerr << "Failed to open output files" << std::endl;
         return;
      }

      uint32_t dim = _base_storage->get_dim();
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<ANNS::IdxType> vec_dis(0, _base_storage->get_num_points() - 1);
      std::unordered_set<ANNS::IdxType> used_vec_ids;

      // Step 4: 生成查询任务
      int queries_generated = 0;
      while (queries_generated < query_n)
      {
         // 循环使用顶层标签
         const auto &labels = top_layer_labels[queries_generated % top_layer_labels.size()];

         // 写入标签文件
         for (size_t j = 0; j < labels.size(); ++j)
         {
            txt_file << labels[j];
            if (j != labels.size() - 1)
            {
               txt_file << ",";
            }
         }
         txt_file << std::endl;

         // 随机选择一个不重复的向量
         ANNS::IdxType vec_id;
         do
         {
            vec_id = vec_dis(gen);
         } while (used_vec_ids.count(vec_id) > 0 &&
                  used_vec_ids.size() < _base_storage->get_num_points());

         used_vec_ids.insert(vec_id);
         const char *vec_data = _base_storage->get_vector(vec_id);

         // 写入向量文件
         fvec_file.write((char *)&dim, sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));

         queries_generated++;
      }

      txt_file.close();
      fvec_file.close();

      std::cout << "Generated " << queries_generated << " queries based on LNG depth" << std::endl;
      std::cout << "Labels written to: " << output_prefix + "/" + dataset + "_query_labels.txt" << std::endl;
      std::cout << "Vectors written to: " << output_prefix + "/" + dataset + "_query.fvecs" << std::endl;
   }

   // ===================================end：生成query task========================================
}