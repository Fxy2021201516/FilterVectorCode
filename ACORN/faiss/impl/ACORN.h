/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <queue>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

// #include <nlohmann/json.hpp>
// // #include <format>
// // for convenience
// using json = nlohmann::json;

namespace faiss
{

   struct VisitedTable;
   struct DistanceComputer; // from AuxIndexStructures
   struct ACORNStats;

   struct SearchParametersACORN : SearchParameters
   {
      int efSearch = 16;
      bool check_relative_distance = true;

      ~SearchParametersACORN() {}
   };

   struct ACORN
   {
      /// internal storage of vectors (32 bits: this is expensive)
      using storage_idx_t = int32_t;

      typedef std::pair<float, storage_idx_t>
          Node; // for heaps with distance, storage

      // stores storage_id of a node
      typedef storage_idx_t NeighNode;

      /** Heap structure that allows fast
       */
      struct MinimaxHeap
      {
         int n;
         int k;
         int nvalid;

         std::vector<storage_idx_t> ids;
         std::vector<float> dis;
         typedef faiss::CMax<float, storage_idx_t> HC;

         explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

         void push(storage_idx_t i, float v);

         float max() const;

         int size() const;

         void clear();

         int pop_min(float *vmin_out = nullptr);

         int count_below(float thresh);
      };

      /// to sort pairs of (id, distance) from nearest to fathest or the reverse
      struct NodeDistCloser
      {
         float d;
         int id;
         NodeDistCloser(float d, int id) : d(d), id(id) {}
         bool operator<(const NodeDistCloser &obj1) const
         {
            return d < obj1.d;
         }
      };

      struct NodeDistFarther
      {
         float d;
         int id;
         NodeDistFarther(float d, int id) : d(d), id(id) {}
         bool operator<(const NodeDistFarther &obj1) const
         {
            return d > obj1.d;
         }
      };

      /// assignment probability to each layer (sum=1)
      std::vector<double> assign_probas;

      /// number of neighbors stored per layer (cumulative), should not
      /// be changed after first add
      std::vector<int> cum_nneighbor_per_level;

      /// level of each vector (base level = 1), size = ntotal
      std::vector<int> levels;

      // added to reference during hybrid construction
      std::vector<storage_idx_t> nb_per_level;

      /// offsets[i] is the offset in the neighbors array where vector i is stored
      /// size ntotal + 1
      std::vector<size_t> offsets;

      /// neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i
      /// for all levels. this is where all storage goes.
      std::vector<NeighNode> neighbors; // changed to add metadata

      /// entry point in the search structure (one of the points with maximum
      /// level
      storage_idx_t entry_point;

      faiss::RandomGenerator rng;

      /// multiplier of M for max edges per vertex
      int gamma;

      int M;

      int M_beta; // parameter for compression

      /// maximum level
      int max_level;

      /// expansion factor at construction time
      int efConstruction;

      /// expansion factor at search time
      int efSearch;

      /// during search: do we check whether the next best distance is good
      /// enough?
      bool check_relative_distance = true;

      /// number of entry points in levels > 0.
      int upper_beam;

      /// use bounded queue during exploration
      bool search_bounded_queue = true;

      // methods that initialize the tree sizes

      /// initialize the assign_probas and cum_nneighbor_per_level to
      /// have 2*M links on level 0 and M links on levels > 0
      void set_default_probas(int M, float levelMult, int M_beta, int gamma = 1);

      /// set nb of neighbors for this level (before adding anything)
      void set_nb_neighbors(int level_no, int n);

      // methods that access the tree sizes

      // added for laion arbitrary predicate test
      std::vector<std::string> load_metadata_strings();

      /// nb of neighbors for this level
      int nb_neighbors(int layer_no) const;

      /// cumumlative nb up to (and excluding) this level
      int cum_nb_neighbors(int layer_no) const;

      /// range of entries in the neighbors table of vertex no at layer_no
      void neighbor_range(idx_t no, int layer_no, size_t *begin, size_t *end)
          const;

      /// only mandatory parameter: nb of neighbors
      // explicit HNSW(int M = 32);
      explicit ACORN(int M, int gamma, std::vector<int> &metadata, int M_beta);
      explicit ACORN(
          int M,
          int gamma,
          std::vector<std::vector<int>> &metadata_multi,
          int M_beta);

      /// pick a random level for a new point
      int random_level();

      /// add n random levels to table (for debugging...)
      void fill_with_random_links(size_t n);

      void add_links_starting_from(
          DistanceComputer &ptdis,
          storage_idx_t pt_id,
          storage_idx_t nearest,
          float d_nearest,
          int level,
          omp_lock_t *locks,
          VisitedTable &vt,
          std::vector<storage_idx_t> ep_per_level = {});

      // void hybrid_add_links_starting_from(
      //         DistanceComputer& ptdis,
      //         storage_idx_t pt_id,
      //         storage_idx_t nearest,
      //         float d_nearest,
      //         int level,
      //         omp_lock_t* locks,
      //         VisitedTable& vt,
      //         std::vector<storage_idx_t> ep_per_level = {});

      /** add point pt_id on all levels <= pt_level and build the link
       * structure for them. */
      void add_with_locks(
          DistanceComputer &ptdis,
          int pt_level,
          int pt_id,
          std::vector<omp_lock_t> &locks,
          VisitedTable &vt);

      /// search interface for 1 point, single thread
      ACORNStats search(
          DistanceComputer &qdis,
          int k,
          idx_t *I,
          float *D,
          VisitedTable &vt,
          const SearchParametersACORN *params = nullptr) const;

      /**************************************************************
       * ACORN HYBRID INDEX
       **************************************************************/
      /// search interface for 1 point, single thread
      const std::vector<int> &metadata;
      const std::vector<std::vector<int>> &metadata_multi;
      std::vector<std::string> metadata_strings;

      std::vector<int> empty_metadata; // 空的 std::vector<int>
      std::vector<std::vector<int>>
          empty_metadata_multi; // 空的 std::vector<std::vector<int>>
      // std::vector<std::string> metadata_strings_vec;

      ACORNStats hybrid_search(
          DistanceComputer &qdis,
          int k,
          idx_t *I,
          float *D,
          VisitedTable &vt,
          char *filter_map,
          bool if_bfs_filter,
          // int filter,
          // Operation op,
          // std::string regex,
          const SearchParametersACORN *params = nullptr) const;

      /**************************************************************
      **************************************************************/

      void reset();

      void clear_neighbor_tables(int level);

      void print_neighbor_stats(int level) const;
      void print_edges(int level) const;
      // void write_filtered_edges_to_json(int level, int filter) const;
      void print_edges_filtered(int level, int filter, Operation op) const;
      void print_neighbor_stats(
          bool edge_list,
          bool filtered_edge_list = false,
          int filter = -1,
          Operation op = EQUAL) const; // overloaded

      int prepare_level_tab(size_t n, bool preset_levels = false);

      void shrink_neighbor_list( // 缩小邻居列表，去除不必要的元素
          DistanceComputer &qdis,
          std::priority_queue<NodeDistFarther> &input,
          std::vector<NodeDistFarther> &output,
          int max_size,
          int gamma = 1,
          storage_idx_t q_id = 0,
          std::vector<int> q_attr = {});
   };

   struct ACORNStats
   {
      size_t n1, n2, n3;
      size_t ndis;
      size_t nreorder;

      // added for timing
      double candidates_loop;
      double neighbors_loop;
      double tuple_unwrap;
      double skips;
      double visits;

      ACORNStats(
          size_t n1 = 0,
          size_t n2 = 0,
          size_t n3 = 0,
          size_t ndis = 0,
          size_t nreorder = 0,
          double candidates_loop = 0.0,
          double neighbors_loop = 0.0,
          double tuple_unwrap = 0.0,
          double skips = 0.0,
          double visits = 0.0)
          : n1(n1),
            n2(n2),
            n3(n3),
            ndis(ndis),
            nreorder(nreorder),
            candidates_loop(candidates_loop),
            neighbors_loop(neighbors_loop),
            tuple_unwrap(tuple_unwrap),
            skips(skips),
            visits(visits) {}

      void reset()
      {
         n1 = n2 = n3 = 0;
         ndis = 0;
         nreorder = 0;

         // added
         candidates_loop = 0.0;
         neighbors_loop = 0.0;
         tuple_unwrap = 0.0;
         skips = 0.0;
         visits = 0.0;
      }

      void combine(const ACORNStats &other)
      {
         n1 += other.n1;
         n2 += other.n2;
         n3 += other.n3;
         ndis += other.ndis;
         nreorder += other.nreorder;

         // added
         candidates_loop += other.candidates_loop;
         neighbors_loop += other.neighbors_loop;
         tuple_unwrap += other.tuple_unwrap;
         skips = other.skips;
         visits = other.visits;
      }
   };

   // global var that collects them all
   FAISS_API extern ACORNStats acorn_stats;

} // namespace faiss
