#ifndef UNG_H
#define UNG_H
#include "trie.h"
#include "graph.h"
#include "storage.h"
#include "distance.h"
#include "search_cache.h"
#include "label_nav_graph.h"
#include "vamana/vamana.h"
#include <unordered_map>
#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <roaring/roaring.h>
#include <roaring/roaring.hh>

using BitsetType = boost::dynamic_bitset<>;

namespace ANNS
{
   struct QueryStats
   {
      float recall;
      double time_ms;
      double flag_time_ms;
      double descendants_merge_time_ms; // descendants合并耗时
      double coverage_merge_time_ms;    // coverage合并耗时
      double entry_group_total_coverage;
      size_t num_distance_calcs;
      size_t num_entry_points;
      size_t num_lng_descendants;
      bool is_global_search;
   };
   class UniNavGraph
   {
   public:
      UniNavGraph(IdxType num_nodes) : _label_nav_graph(std::make_shared<LabelNavGraph>(num_nodes)) {} // 修改构造函数以初始化 _label_nav_graph
      UniNavGraph() = default;
      ~UniNavGraph() = default;

      void build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                 std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                 IdxType max_degree, IdxType Lbuild, float alpha);

      void search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler,
                  uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                  IdxType K, std::pair<IdxType, float> *results, std::vector<float> &num_cmps,
                  std::vector<std::bitset<10000001>> &bitmap);
      void search_hybrid(std::shared_ptr<IStorage> query_storage,
                         std::shared_ptr<DistanceHandler> distance_handler,
                         uint32_t num_threads, IdxType Lsearch,
                         IdxType num_entry_points, std::string scenario,
                         IdxType K, std::pair<IdxType, float> *results,
                         std::vector<float> &num_cmps,
                         std::vector<QueryStats> &query_stats,
                         std::vector<std::bitset<10000001>> &bitmaps,
                         bool is_ori_ung);

      // I/O
      void save(std::string index_path_prefix, std::string results_path_prefix);
      void load(std::string index_path_prefix, const std::string &data_type);

      // query generator
      void query_generate(std::string &output_prefix, int n, float keep_prob, bool stratified_sampling, bool verify);
      void generate_multiple_queries(std::string dataset,
                                     UniNavGraph &index,
                                     const std::string &base_output_path,
                                     int num_sets,
                                     int n_per_set,
                                     float keep_prob,
                                     bool stratified_sampling,
                                     bool verify);
      void generate_queries_method1_high_coverage(std::string &output_prefix, std::string dataset, int query_n, std::string &base_label_file, float coverage_threshold);
      void generate_queries_method1_low_coverage(
          std::string &output_prefix,
          std::string dataset,
          int query_n,
          std::string &base_label_file,
          int num_of_per_query_labels,
          float coverage_threshold,
          int K);
      void generate_queries_method2_high_coverage(int N, int K, int top_M_trees, std::string dataset, const std::string &output_prefix, const std::string &base_label_tree_roots);
      void generate_queries_method2_high_coverage_human(
          std::string &output_prefix,
          std::string dataset,
          int query_n,
          std::string &base_label_file,
          std::string &base_label_info_file);
      void generate_queries_method2_low_coverage(
          std::string &output_prefix,
          std::string dataset,
          int query_n,
          std::string &base_label_file,
          int num_of_per_query_labels,
          int K,
          int max_K,
          int min_K);
      void load_bipartite_graph(const std::string &filename);
      bool compare_graphs(const ANNS::UniNavGraph &g1, const ANNS::UniNavGraph &g2);
      IdxType _num_points;
      std::vector<std::vector<IdxType>> _vector_attr_graph; // 邻接表表示的图
      std::unordered_map<LabelType, AtrType> _attr_to_id;   // 属性到ID的映射
      std::unordered_map<AtrType, LabelType> _id_to_attr;   // ID到属性的映射
      AtrType _num_attributes;                              // 唯一属性数量

      std::pair<std::bitset<10000001>, double> compute_attribute_bitmap(const std::vector<LabelType> &query_attributes) const; // 构建bitmap

      // 求search中flag需要的数据结构
      std::vector<BitsetType> _lng_descendants_bits; // 每个 group 的后代集合
      std::vector<BitsetType> _covered_sets_bits;    // 每个 group 的覆盖集合
      std::vector<roaring::Roaring> _lng_descendants_rb;
      std::vector<roaring::Roaring> _covered_sets_rb;

   private:
      // data
      std::shared_ptr<IStorage> _base_storage,
          _query_storage;
      std::shared_ptr<DistanceHandler> _distance_handler;
      std::shared_ptr<Graph> _graph;
      // IdxType _num_points;

      // trie index and vector groups
      IdxType _num_groups;
      TrieIndex _trie_index;
      std::vector<IdxType> _new_vec_id_to_group_id;
      std::vector<std::vector<IdxType>> _group_id_to_vec_ids;
      std::vector<std::vector<LabelType>> _group_id_to_label_set;
      void build_trie_and_divide_groups();

      // label navigating graph
      std::shared_ptr<LabelNavGraph> _label_nav_graph = nullptr;
      void get_min_super_sets(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                              bool avoid_self = false, bool need_containment = true);
      void cal_f_coverage_ratio();
      void build_label_nav_graph();
      size_t count_all_descendants(IdxType group_id) const;
      void print_lng_descendants_num(const std::string &filename) const;
      void get_descendants_info();

      // prepare vector storage for each group
      std::vector<IdxType> _new_to_old_vec_ids;
      std::vector<std::pair<IdxType, IdxType>> _group_id_to_range;
      std::vector<std::shared_ptr<IStorage>> _group_storages;
      void prepare_group_storages_graphs();

      // graph indices for each graph
      std::string _index_name;
      std::vector<std::shared_ptr<Graph>> _group_graphs;
      std::vector<IdxType> _group_entry_points;
      void build_graph_for_all_groups();
      void build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points);
      std::vector<std::shared_ptr<Vamana>> _vamana_instances;

      std::shared_ptr<Graph> _global_graph;
      std::shared_ptr<Vamana> _global_vamana; // 全局 Vamana 实例
      IdxType _global_vamana_entry_point;     // 全局 Vamana 实例的入口点
      void build_global_vamana_graph();

      // build attr_id_graph（数据预处理）
      // std::vector<std::vector<IdxType>> _vector_attr_graph; // 邻接表表示的图
      // std::unordered_map<LabelType, AtrType> _attr_to_id;   // 属性到ID的映射
      // std::unordered_map<AtrType, LabelType> _id_to_attr;   // ID到属性的映射
      // AtrType _num_attributes;                              // 唯一属性数量
      void build_vector_and_attr_graph();
      size_t count_graph_edges() const;
      void save_bipartite_graph_info() const;
      void save_bipartite_graph(const std::string &filename);
      uint32_t compute_checksum() const;
      // void load_bipartite_graph(const std::string &filename);

      // 处理flag的相关函数
      void initialize_lng_descendants_coverage_bitsets();
      void initialize_roaring_bitsets();

      // index parameters for each graph
      IdxType _max_degree,
          _Lbuild;
      float _alpha;
      uint32_t _num_threads;
      std::string _scenario;

      // cross-group edges
      IdxType _num_cross_edges;
      std::vector<SearchQueue> _cross_group_neighbors;
      void build_cross_group_edges();

      // obtain the final unified navigating graph
      void add_offset_for_uni_nav_graph();

      // obtain entry_points
      std::vector<IdxType> get_entry_points(const std::vector<LabelType> &query_label_set,
                                            IdxType num_entry_points, VisitedSet &visited_set);
      void get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet &visited_set,
                                           IdxType group_id, std::vector<IdxType> &entry_points);

      // search in graph
      IdxType iterate_to_fixed_point(const char *query, std::shared_ptr<SearchCache> search_cache,
                                     IdxType target_id, const std::vector<IdxType> &entry_points,
                                     bool clear_search_queue = true, bool clear_visited_set = true);
      // search in global graph
      IdxType iterate_to_fixed_point_global(const char *query, std::shared_ptr<SearchCache> search_cache,
                                            IdxType target_id, const std::vector<IdxType> &entry_points,
                                            bool clear_search_queue = true, bool clear_visited_set = true);

      // statistics
      float _index_time = 0, _label_processing_time = 0, _build_graph_time = 0, _build_vector_attr_graph_time = 0, _cal_descendants_time = 0, _cal_coverage_ratio_time = 0;
      float _build_LNG_time = 0, _build_cross_edges_time = 0;
      float _index_size;
      IdxType _graph_num_edges, _LNG_num_edges;
      void statistics();
   };
}

#endif // UNG_H