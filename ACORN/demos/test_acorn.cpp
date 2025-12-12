#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sys/stat.h>
#include <sys/time.h>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <set>
#include <thread>
#include <unordered_set> // +++ 新增，用于 recall 计算

#include "../faiss/Index.h"
#include "../faiss/IndexACORN.h"
#include "../faiss/IndexFlat.h"
#include "../faiss/impl/ACORN.h"
#include "../faiss/index_io.h"
#include "../faiss/impl/platform_macros.h"
#include "utils.cpp"

// +++ 新增类型别名，与 UNG 框架保持一致
namespace ANNS
{
   using IdxType = uint32_t;
}

// 辅助函数：获取文件大小（字节）
long get_file_size(const std::string &filename)
{
   struct stat stat_buf;
   int rc = stat(filename.c_str(), &stat_buf);
   return rc == 0 ? stat_buf.st_size : -1;
}

// 存储单次查询的结果
struct QueryResult
{
   int query_id;
   double acorn_time_ms;
   double acorn_qps;
   float acorn_recall;
   size_t acorn_n3;
   double acorn_1_time_ms;
   double acorn_1_qps;
   float acorn_1_recall;
   size_t acorn_1_n3;
   double filter_time_ms;
};

// 存储多次 repeat 聚合后的平均结果
struct AggregatedResult
{
   double total_acorn_time_s = 0.0;
   float total_acorn_recall = 0.0;
   size_t total_acorn_n3 = 0;

   double total_acorn_1_time_s = 0.0;
   float total_acorn_1_recall = 0.0;
   size_t total_acorn_1_n3 = 0;

   double total_filter_time_s = 0.0;
};

void load_gt_file(const std::string &filename, std::pair<ANNS::IdxType, float> *gt, uint32_t num_queries, uint32_t K)
{
   std::ifstream fin(filename, std::ios::binary);
   fin.read(reinterpret_cast<char *>(gt), num_queries * K * sizeof(std::pair<ANNS::IdxType, float>));
   std::cout << "Ground truth loaded from " << filename << std::endl;
}

// // +++ 新增：从 search_UNG_index.cpp 移植过来的 Recall 计算函数
// float calculate_single_query_recall(
//     const std::pair<ANNS::IdxType, float> *gt_for_one_query,
//     const faiss::idx_t *result_ids,
//     ANNS::IdxType K)
// {
//    std::unordered_set<ANNS::IdxType> gt_set;
//    for (int i = 0; i < K; ++i)
//    {
//       if (gt_for_one_query[i].first != -1)
//       {
//          gt_set.insert(gt_for_one_query[i].first);
//       }
//    }
//    if (gt_set.empty())
//    {
//       return 1.0f; // 如果GT为空（例如，没有满足过滤条件的点），则认为召回率为1
//    }
//    int correct = 0;
//    for (int i = 0; i < K; ++i)
//    {
//       if (result_ids[i] != -1 && gt_set.count(result_ids[i]))
//       {
//          correct++;
//       }
//    }
//    return static_cast<float>(correct) / gt_set.size();
// }

// 修改后的 Recall 计算函数：支持处理距离平局 (Ties)
float calculate_single_query_recall(
    const std::pair<ANNS::IdxType, float> *gt_for_one_query, // GT: ID + Distance
    const faiss::idx_t *result_ids,                          // Algo: ID
    const float *result_dists,                               // Algo: Distance [新增参数]
    ANNS::IdxType K)
{
    // 1. 获取 GT 中第 K 个结果的距离阈值 (GT 是排好序的，最后一个是最大的)
    // 注意：GT pair 是 <ID, Distance>
    float gt_threshold = gt_for_one_query[K - 1].second;
    
    // 设置一个极小的浮点误差容忍度 (Epsilon)
    // 防止浮点数精度问题导致 0.0000001 > 0.0
    float epsilon = 1e-5;

    int correct = 0;
    
    // 建立 GT ID 的集合，用于处理那些距离虽然不相等但 ID 命中的情况 (常规 Recall)
    std::unordered_set<ANNS::IdxType> gt_set;
    for (int i = 0; i < K; ++i) {
        if (gt_for_one_query[i].first != -1) {
            gt_set.insert(gt_for_one_query[i].first);
        }
    }

    if (gt_set.empty()) return 1.0f;

    for (int i = 0; i < K; ++i)
    {
        if (result_ids[i] == -1) continue;

        // 情况 A: ID 直接命中
        if (gt_set.count(result_ids[i])) {
            correct++;
        }
        // 情况 B: ID 没命中，但距离足够好 (处理 Ties)
        // 如果算法找到的距离 <= GT 第 K 个距离，说明它也是一个合法的 Top-K 答案
        else if (result_dists[i] <= gt_threshold + epsilon) {
            correct++;
        }
    }
    
    // 归一化，最大不超过 1.0
    return std::min(1.0f, static_cast<float>(correct) / K);
}

int main(int argc, char *argv[])
{
   std::cout << "====================\nSTART: ACORN Test Suite --" << std::endl;
   double t0 = elapsed();

   // --- 参数定义 ---
   size_t d = 0;
   int M, M_beta, gamma, k = 10;
   std::string dataset;
   size_t N = 0;
   int nthreads, repeat_num = 0;
   // --- 修改：dis_output_path 替换为 gt_bin_path
   std::string base_path, base_label_path, query_path, csv_dir, avg_csv_dir, gt_bin_path;
   std::vector<int> efs_list;
   bool if_bfs_filter = true;
   std::string mode, index_path_acorn, index_path_acorn1;

   // --- 参数解析 ---
   {
      if (argc < 20)
      {
         fprintf(stderr, "Usage: %s <mode> <N> <gamma> <dataset> <M> <M_beta> <base_path> <base_label_path> <query_path> <csv_dir> <avg_csv_dir> <gt_bin_path> <nthreads> <repeat_num> <if_bfs_filter> <efs_list> <index_path_acorn> <index_path_acorn1><k> \n", argv[0]);
         fprintf(stderr, "       <mode> must be 'build' or 'search'\n");
         exit(1);
      }
      mode = argv[1];
      if (mode != "build" && mode != "search")
      {
         fprintf(stderr, "Invalid mode: %s. Must be 'build' or 'search'.\n", mode.c_str());
         exit(1);
      }
      N = strtoul(argv[2], NULL, 10);
      gamma = atoi(argv[3]);
      dataset = argv[4];
      M = atoi(argv[5]);
      M_beta = atoi(argv[6]);
      base_path = argv[7];
      base_label_path = argv[8];
      query_path = argv[9];
      csv_dir = argv[10];
      avg_csv_dir = argv[11];
      gt_bin_path = argv[12];
      nthreads = atoi(argv[13]);
      repeat_num = atoi(argv[14]);
      // if_bfs_filter = atoi(argv[15]);
      std::string bfs_arg = std::string(argv[15]);
      std::transform(bfs_arg.begin(), bfs_arg.end(), bfs_arg.begin(), ::tolower); // 转为小写以兼容 True/TRUE
      if (bfs_arg == "true" || bfs_arg == "1") {
         if_bfs_filter = true;
      } else {
         if_bfs_filter = false;
      }

      char *efs_list_str = argv[16];
      index_path_acorn = argv[17];
      index_path_acorn1 = argv[18];
      k = atoi(argv[19]);

      printf("Running in '%s' mode\n", mode.c_str());
      printf("DEBUG: N=%zu, gamma=%d, M=%d, M_beta=%d, threads=%d, repeat=%d\n", N, gamma, M, M_beta, nthreads, repeat_num);
      printf("DEBUG: if_bfs_filter parsed as: %s (Raw int value: %d)\n", if_bfs_filter ? "TRUE" : "FALSE", if_bfs_filter);

      char *token = strtok(efs_list_str, ",");
      while (token != NULL)
      {
         efs_list.push_back(atoi(token));
         token = strtok(NULL, ",");
      }
      std::sort(efs_list.begin(), efs_list.end());
   }

   // --- 加载元数据 ---
   std::cout << "Loading metadata for base vectors from: " << base_label_path << std::endl;
   std::vector<std::vector<int>> metadata = load_ab_muti(dataset, gamma, "rand", N, base_label_path); // TODO
   metadata.resize(N);
   assert(N == metadata.size());
   for (auto &inner_vector : metadata)
   {
      std::sort(inner_vector.begin(), inner_vector.end());
   }
   printf("[%.3f s] Loaded metadata, %ld vectors found\n", elapsed() - t0, metadata.size());

   // ==================== BUILD 模式 ====================
   if (mode == "build")
   {
      omp_set_num_threads(nthreads);
      //omp_set_num_threads(nthreads);
      printf("Using 32 threads to build index\n");
      std::cout << "Building ACORN index with parameters: "
                << "N=" << N << ", gamma=" << gamma
                << ", M=" << M << ", M_beta=" << M_beta
                << ", dataset=" << dataset << std::endl;

      size_t nb, d2;
      float *xb = nullptr;
      {
         printf("[%.3f s] Loading database for building...\n", elapsed() - t0);
         // std::string filename = base_path + "/" + dataset + "_base.fvecs";
         // xb = fvecs_read(filename.c_str(), &d2, &nb);
         xb = fvecs_read(base_path.c_str(), &d2, &nb);
         d = d2;
         printf("Loaded base vectors from: %s\n", base_path.c_str());
         printf("[%.3f s] Base data loaded, dim: %zu, nb: %zu\n", elapsed() - t0, d, nb);
         if (N > nb)
         {
            fprintf(stderr, "Error: N=%zu is larger than database size nb=%zu\n", N, nb);
            exit(1);
         }
      }

      faiss::IndexACORNFlat hybrid_index(d, M, gamma, metadata, M_beta);
      hybrid_index.verbose = true;
      faiss::IndexACORNFlat hybrid_index_gamma1(d, M, 1, metadata, M * 2);
      hybrid_index_gamma1.verbose = true;

      std::cout << "Index parameters: "
                << "d=" << d << ", M=" << M
                << ", gamma=" << gamma << ", M_beta=" << M_beta
                << std::endl;

      // 构建 ACORN
      printf("[%.3f s] Adding %zu vectors to ACORN index...\n", elapsed() - t0, N);
      double t_start_acorn = elapsed();
      hybrid_index.add(N, xb);
      double acorn_build_time_s = elapsed() - t_start_acorn;
      printf("[%.3f s] ACORN build time: %.3f s\n", elapsed() - t0, acorn_build_time_s);

      std::cout << "ACORN index built. Now building ACORN-1 index with gamma=1 and M_beta=" << M * 2 << std::endl;

      // 构建 ACORN-1
      printf("[%.3f s] Adding %zu vectors to ACORN-1 index...\n", elapsed() - t0, N);
      double t_start_acorn1 = elapsed();
      hybrid_index_gamma1.add(N, xb);
      double acorn_1_build_time_s = elapsed() - t_start_acorn1;
      printf("[%.3f s] ACORN-1 build time: %.3f s\n", elapsed() - t0, acorn_1_build_time_s);

      delete[] xb;

      // 保存索引
      printf("[%.3f s] Saving ACORN index to %s\n", elapsed() - t0, index_path_acorn.c_str());
      faiss::write_index(&hybrid_index, index_path_acorn.c_str());

      printf("[%.3f s] Saving ACORN-1 index to %s\n", elapsed() - t0, index_path_acorn1.c_str());
      faiss::write_index(&hybrid_index_gamma1, index_path_acorn1.c_str());

      // --- ACORN 索引大小计算 ---
      long acorn_total_size_bytes = get_file_size(index_path_acorn);
      long acorn_vectors_size_bytes = (long)N * d * sizeof(float);// 计算理论向量大小 (N * d * 4 bytes)
      long acorn_index_only_size_bytes = acorn_total_size_bytes - acorn_vectors_size_bytes;// 计算纯索引结构大小

      // --- ACORN 逻辑内存大小计算 (新增, 模拟UNG的.size()统计) ---
      size_t acorn_index_only_logical_memory_bytes = hybrid_index.acorn.get_logical_memory_usage();
      size_t acorn_vectors_memory_bytes = (size_t)acorn_vectors_size_bytes; // 保持一致
      size_t acorn_total_logical_memory_bytes = acorn_index_only_logical_memory_bytes + acorn_vectors_memory_bytes;

      std::ofstream meta_file(index_path_acorn + ".meta");
      meta_file << "build_time_s:" << acorn_build_time_s << std::endl;
      // 磁盘大小 (Disk size)
      meta_file << "total_size_bytes:" << acorn_total_size_bytes << std::endl;
      meta_file << "index_only_size_bytes:" << acorn_index_only_size_bytes << std::endl;
      // 逻辑内存大小 (Logical Memory size - 新增)
      meta_file << "total_logical_memory_bytes:" << acorn_total_logical_memory_bytes << std::endl;
      meta_file << "index_only_logical_memory_bytes:" << acorn_index_only_logical_memory_bytes << std::endl;
      meta_file.close();

      printf("[%.3f s] ACORN index saved. \n"
            "    Disk Size (from file):         Total=%.2f MB, Index-Only=%.2f MB.\n"
            "    Logical Memory (like UNG): Total=%.2f MB, Index-Only=%.2f MB.\n",
            elapsed() - t0,
            (double)acorn_total_size_bytes / (1024.0 * 1024.0),
            (double)acorn_index_only_size_bytes / (1024.0 * 1024.0),  
            (double)acorn_total_logical_memory_bytes / (1024.0 * 1024.0),
            (double)acorn_index_only_logical_memory_bytes / (1024.0 * 1024.0)
      );

      // --- ACORN-1 索引大小计算 ---
      long acorn_1_total_size_bytes = get_file_size(index_path_acorn1);
      long acorn_1_vectors_size_bytes = (long)N * d * sizeof(float);
      long acorn_1_index_only_size_bytes = acorn_1_total_size_bytes - acorn_1_vectors_size_bytes;
      // --- ACORN-1 逻辑内存大小计算 (新增, 模拟UNG的.size()统计) ---
      size_t acorn_1_index_only_logical_memory_bytes = hybrid_index_gamma1.acorn.get_logical_memory_usage();
      size_t acorn_1_vectors_memory_bytes = (size_t)acorn_1_vectors_size_bytes; // 保持一致
      size_t acorn_1_total_logical_memory_bytes = acorn_1_index_only_logical_memory_bytes + acorn_1_vectors_memory_bytes;

      // 保存元数据文件
      std::ofstream meta_file1(index_path_acorn1 + ".meta");
      meta_file1 << "build_time_s:" << acorn_1_build_time_s << std::endl;
      // 磁盘大小 (Disk size)
      meta_file1 << "total_size_bytes:" << acorn_1_total_size_bytes << std::endl;
      meta_file1 << "index_only_size_bytes:" << acorn_1_index_only_size_bytes << std::endl;
      // 逻辑内存大小 (Logical Memory size - 新增)
      meta_file1 << "total_logical_memory_bytes:" << acorn_1_total_logical_memory_bytes << std::endl;
      meta_file1 << "index_only_logical_memory_bytes:" << acorn_1_index_only_logical_memory_bytes << std::endl;
      meta_file1.close();

      printf("[%.3f s] ACORN-1 index saved. \n"
            "    Disk Size (from file):         Total=%.2f MB, Index-Only=%.2f MB.\n"
           "    Logical Memory (like UNG): Total=%.2f MB, Index-Only=%.2f MB.\n",
            elapsed() - t0,
            (double)acorn_1_total_size_bytes / (1024.0 * 1024.0),
            (double)acorn_1_index_only_size_bytes / (1024.0 * 1024.0), // <-- 修正点 2
            (double)acorn_1_total_logical_memory_bytes / (1024.0 * 1024.0),
            (double)acorn_1_index_only_logical_memory_bytes / (1024.0 * 1024.0)
            // <-- 删除了多余的 acorn_1_vectors_size_bytes 变量
      );

      // 保存倒排索引
      std::string inverted_index_path = index_path_acorn + ".inverted_index";
      build_and_save_inverted_index(metadata, N, inverted_index_path);

      printf("[%.3f s] Build finished successfully.\n", elapsed() - t0);
      return 0;
   }
   // ==================== SEARCH 模式 ====================
   else if (mode == "search")
   {
      omp_set_num_threads(nthreads);
      printf("Using %d threads for search\n", nthreads);
      std::cout << "Searching ACORN index with parameters: "
                << "N=" << N << ", gamma=" << gamma
                << ", M=" << M << ", M_beta=" << M_beta
                << ", dataset=" << dataset
                << ", k= " << k << std::endl;

      // --- 加载构建阶段的元数据 (build time, index size) ---
      double acorn_build_time_s = 0.0, acorn_1_build_time_s = 0.0;
      long acorn_index_size_bytes = 0, acorn_1_index_size_bytes = 0;

      std::ifstream meta_file(index_path_acorn + ".meta");
      std::string line;
      if (meta_file.is_open())
      {
         while (std::getline(meta_file, line))
         {
            std::stringstream ss(line);
            std::string key;
            std::getline(ss, key, ':');
            if (key == "build_time_s")
               ss >> acorn_build_time_s;
            if (key == "total_size_bytes")
               ss >> acorn_index_size_bytes; // 修正：读取total size
         }
         meta_file.close();
      }
      else
      {
         printf("Warning: Could not open meta file for ACORN index: %s.meta\n", index_path_acorn.c_str());
      }

      std::ifstream meta_file1(index_path_acorn1 + ".meta");
      if (meta_file1.is_open())
      {
         while (std::getline(meta_file1, line))
         {
            std::stringstream ss(line);
            std::string key;
            std::getline(ss, key, ':');
            if (key == "build_time_s")
               ss >> acorn_1_build_time_s;
            if (key == "total_size_bytes")
               ss >> acorn_1_index_size_bytes; // 修正：读取total size
         }
         meta_file1.close();
      }
      else
      {
         printf("Warning: Could not open meta file for ACORN-1 index: %s.meta\n", index_path_acorn1.c_str());
      }

      // --- 加载索引 ---
      faiss::IndexACORNFlat *hybrid_index = nullptr;
      faiss::IndexACORNFlat *hybrid_index_gamma1 = nullptr;

      printf("[%.3f s] Loading ACORN index from: %s\n", elapsed() - t0, index_path_acorn.c_str());
      hybrid_index = dynamic_cast<faiss::IndexACORNFlat *>(faiss::read_index(index_path_acorn.c_str()));
      if (!hybrid_index)
      {
         fprintf(stderr, "Error: Failed to load index from %s.\n", index_path_acorn.c_str());
         exit(1);
      }
      d = hybrid_index->d;

      printf("[%.3f s] Loading ACORN-1 index from: %s\n", elapsed() - t0, index_path_acorn1.c_str());
      hybrid_index_gamma1 = dynamic_cast<faiss::IndexACORNFlat *>(faiss::read_index(index_path_acorn1.c_str()));
      if (!hybrid_index_gamma1)
      {
         fprintf(stderr, "Error: Failed to load index from %s.\n", index_path_acorn1.c_str());
         exit(1);
      }

      // 重新关联元数据
      printf("[%.3f s] Re-associating metadata with loaded indexes...\n", elapsed() - t0);
      hybrid_index->set_metadata(metadata);
      hybrid_index_gamma1->set_metadata(metadata);
      printf("[%.3f s] Indexes loaded and ready for search.\n", elapsed() - t0);

      // --- 加载查询数据 ---
      size_t nq;
      float *xq;
      std::vector<std::vector<int>> aq;
      {
         printf("[%.3f s] Loading query vectors and attributes\n", elapsed() - t0);
         size_t d2;
         std::string filename = query_path + "/" + dataset + "_query.fvecs";
         xq = fvecs_read(filename.c_str(), &d2, &nq);
         assert(d == d2 || !"Query dimension mismatch!");
         std::string last_value = query_path.substr(query_path.find_last_of('_') + 1);
         aq = load_aq_multi(dataset, gamma, 0, N, query_path);
#pragma omp parallel for
         for (size_t i = 0; i < aq.size(); ++i)
         {
            std::sort(aq[i].begin(), aq[i].end());
         }
         std::cout << "Query data loaded. nq=" << nq << std::endl;
         printf("[%.3f s] Loaded %zu queries\n", elapsed() - t0, nq);
      }

      // --- 准备结果存储 ---
      int efs_cnt = efs_list.size();
      std::vector<std::vector<std::vector<QueryResult>>> all_query_results(repeat_num, std::vector<std::vector<QueryResult>>(efs_cnt, std::vector<QueryResult>(nq)));
      for (int r = 0; r < repeat_num; ++r)
         for (int e = 0; e < efs_cnt; ++e)
            for (int i = 0; i < nq; ++i)
               all_query_results[r][e][i].query_id = i;

      std::vector<AggregatedResult> final_avg_results(efs_cnt);

      // +++ 新增：加载二进制 Ground Truth 文件 +++
      std::cout << "\n[INFO] Loading binary ground truth from: " << gt_bin_path << std::endl;
      double t_load_gt_start = elapsed();
      auto gt_data = new std::pair<ANNS::IdxType, float>[nq * k];
      load_gt_file(gt_bin_path, gt_data, nq, k);

      double t_load_gt_end = elapsed();
      printf("[INFO] Binary ground truth loaded in %.3f s.\n\n", t_load_gt_end - t_load_gt_start);

      // +++ 检查连通性 +++
      printf("\n--- Checking Graph Connectivity ---\n");
      hybrid_index->acorn.check_connectivity(0); // 检查 Layer 0
      printf("-----------------------------------\n\n");

      // --- 计算 filter map ---
      std::string inverted_index_path = index_path_acorn + ".inverted_index";
      std::unordered_map<int, std::vector<int>> inverted_index = load_inverted_index(inverted_index_path);
      hybrid_index->set_inverted_index(inverted_index);
      hybrid_index_gamma1->set_inverted_index(inverted_index);
      printf("[%.3f s] Inverted index set into ACORN instances.\n", elapsed() - t0);
      printf("[%.3f s] Generating filter map using loaded index...\n", elapsed() - t0);
      double t_filter_para_0 = elapsed();
      std::vector<char> filter_ids_map_para = generate_filter_map_from_index(inverted_index, nq, N, aq);
      double filter_para_time_s = elapsed() - t_filter_para_0;
      printf("[%.3f s] Filter map created in %.4f s\n", elapsed() - t0, filter_para_time_s);

      // --- 多次重复执行搜索 ---
      for (int repeat = 0; repeat < repeat_num; repeat++)
      {
         std::cout << "=============== Repeat " << repeat + 1 << "/" << repeat_num << " ===============" << std::endl;
         for (int efs_id = 0; efs_id < efs_cnt; efs_id++)
         {
            int current_efs = efs_list[efs_id];

            // --- 搜索 ACORN ---
            std::vector<faiss::idx_t> nns2(k * nq);
            std::vector<float> dis2(k * nq);
            std::vector<double> query_times(nq);
            std::vector<double> query_qps(nq);
            std::vector<size_t> query_n3(nq);

            std::cout << "--- ACORN | efs = " << current_efs << " ---" << std::endl;
            hybrid_index->acorn.efSearch = current_efs;
            faiss::acorn_stats.reset();
            double t1_acorn = elapsed();
            hybrid_index->search_old_bitmap(nq, xq, k, dis2.data(), nns2.data(), aq, &query_times, &query_qps, &query_n3, if_bfs_filter);
            //hybrid_index->search(nq, xq, k, dis2.data(), nns2.data(), filter_ids_map_para.data(), &query_times, &query_qps, &query_n3, if_bfs_filter);
            double search_time_acorn_s = elapsed() - t1_acorn;
            const faiss::ACORNStats &acorn_search_stats = faiss::acorn_stats;

            // +++ 新增：使用新的函数计算 ACORN Recall +++
            std::vector<float> recalls_acorn(nq);
#pragma omp parallel for
            for (size_t i = 0; i < nq; ++i)
            {
               // [调试代码] 仅针对第 0 个查询进行深度诊断
               if (i == 0) {
                  printf("\n================ [DEBUG DIAGNOSIS START] ================\n");
                  printf("Query ID: %zu\n", i);
                  
                  // 1. 打印算法找到的前 10 个结果 (ID 和 距离)
                  printf("--- Algorithm Search Results ---\n");
                  printf("%-10s %-15s %-15s\n", "Rank", "ID", "Distance");
                  for (int z = 0; z < k; ++z) {
                     printf("%-10d %-15lld %-15.6f\n", z, nns2[i * k + z], dis2[i * k + z]);
                  }

                  // 2. 打印 Ground Truth 里的前 10 个结果 (ID 和 距离)
                  // 注意：gt_data 里的 float 通常是距离
                  printf("\n--- Ground Truth (Standard Answer) ---\n");
                  printf("%-10s %-15s %-15s\n", "Rank", "ID", "Distance");
                  for (int z = 0; z < k; ++z) {
                     ANNS::IdxType gt_id = (gt_data + i * k)[z].first;
                     float gt_dist = (gt_data + i * k)[z].second; 
                     printf("%-10d %-15d %-15.6f\n", z, gt_id, gt_dist);
                  }
                  
                  // 3. [核心检查] 检查 GT 中的 ID 在当前的 Filter Map 中是否被允许？
                  // 如果 filter_map 判定 GT ID 为 false，那算法永远搜不到它，这就是 Filter 生成逻辑的问题
                  // 我们需要临时重新生成一下 map 来检查 (因为 filter_ids_map_para 可能不可见或被销毁)
                  // 这里简单调用一下生成逻辑 (会有一点性能损耗，但在 debug 时无所谓)
                  // 注意：需要 access inverted_index，确保变量名对应
                  if (hybrid_index != nullptr) {
                        printf("\n--- Filter Validity Check for GT Items ---\n");
                        std::vector<char> debug_filter = generate_single_filter_map(
                           hybrid_index->inverted_index, hybrid_index->ntotal, aq[i]);
                        
                        for (int z = 0; z < k; ++z) {
                           ANNS::IdxType gt_id = (gt_data + i * k)[z].first;
                           // 检查边界
                           if (gt_id >= 0 && gt_id < debug_filter.size()) {
                              bool allowed = debug_filter[gt_id];
                              printf("GT ID %d -> Filter Allowed? %s\n", gt_id, allowed ? "YES" : "NO [FATAL]");
                           }
                        }
                  }
                  
                  printf("================ [DEBUG DIAGNOSIS END] ==================\n\n");
               }

               recalls_acorn[i] = calculate_single_query_recall(gt_data + i * k, nns2.data() + i * k, dis2.data() + i * k,k);
            }

            // --- 搜索 ACORN-1 ---
            std::vector<faiss::idx_t> nns3(k * nq);
            std::vector<float> dis3(k * nq);
            std::vector<double> query_times3(nq);
            std::vector<double> query_qps3(nq);
            std::vector<size_t> query_n33(nq);

            printf("--- ACORN-1 | efs = %d ---\n", current_efs);
            hybrid_index_gamma1->acorn.efSearch = current_efs;
            faiss::acorn_stats.reset();
            double t1_acorn1 = elapsed();
            hybrid_index_gamma1->search_old_bitmap(nq, xq, k, dis3.data(), nns3.data(), aq, &query_times3, &query_qps3, &query_n33, if_bfs_filter);
            double search_time_acorn1_s = elapsed() - t1_acorn1;
            const faiss::ACORNStats &acorn1_search_stats = faiss::acorn_stats;

            // --- 使用新的方法计算 Recall ---
            float recall_mean_acorn = std::accumulate(recalls_acorn.begin(), recalls_acorn.end(), 0.0f) / nq;

            std::vector<float> recalls_acorn1(nq);
#pragma omp parallel for
            for (size_t i = 0; i < nq; ++i)
            {
               recalls_acorn1[i] = calculate_single_query_recall(gt_data + i * k, nns3.data() + i * k, dis3.data() + i * k,k);
            }
            float recall_mean_acorn1 = std::accumulate(recalls_acorn1.begin(), recalls_acorn1.end(), 0.0f) / nq;

            std::cout << "  ACORN   | Time: " << search_time_acorn_s << " s, QPS: " << nq / search_time_acorn_s << ", Recall: " << recall_mean_acorn << std::endl;
            std::cout << "  ACORN-1 | Time: " << search_time_acorn1_s << " s, QPS: " << nq / search_time_acorn1_s << ", Recall: " << recall_mean_acorn1 << std::endl;

            // --- 存储单次结果 ---
            for (int i = 0; i < nq; i++)
            {
               all_query_results[repeat][efs_id][i].acorn_time_ms = query_times[i] * 1000.0;
               all_query_results[repeat][efs_id][i].acorn_qps = query_qps[i];
               all_query_results[repeat][efs_id][i].acorn_recall = recalls_acorn[i];
               all_query_results[repeat][efs_id][i].acorn_n3 = query_n3[i];
               all_query_results[repeat][efs_id][i].acorn_1_time_ms = query_times3[i] * 1000.0;
               all_query_results[repeat][efs_id][i].acorn_1_qps = query_qps3[i];
               all_query_results[repeat][efs_id][i].acorn_1_recall = recalls_acorn1[i];
               all_query_results[repeat][efs_id][i].acorn_1_n3 = query_n33[i];
            }

            // --- 累加结果用于最终平均 ---
            final_avg_results[efs_id].total_acorn_time_s += search_time_acorn_s;
            final_avg_results[efs_id].total_acorn_recall += recall_mean_acorn;
            final_avg_results[efs_id].total_acorn_n3 += (acorn_search_stats.n1 > 0) ? ((size_t)acorn_search_stats.n3) : 0;

            final_avg_results[efs_id].total_acorn_1_time_s += search_time_acorn1_s;
            final_avg_results[efs_id].total_acorn_1_recall += recall_mean_acorn1;
            final_avg_results[efs_id].total_acorn_1_n3 += (acorn1_search_stats.n1 > 0) ? ((size_t)acorn1_search_stats.n3) : 0;
         }
      }

      // --- 生成动态文件名 ---
      std::string query_num_str = query_path.substr(query_path.find_last_of('_') + 1);
      std::stringstream efs_str_stream;
      efs_str_stream << efs_list.front() << "-" << efs_list.back();
      if (efs_list.size() > 1)
      {
         efs_str_stream << "_" << (efs_list[1] - efs_list[0]);
      }

      std::stringstream file_suffix_stream;
      file_suffix_stream << dataset << "_query" << query_num_str
                         << "_M" << M << "_gamma" << gamma
                         << "_threads" << nthreads << "_repeat" << repeat_num
                         << "_ifbfs" << if_bfs_filter << "_efs" << efs_str_stream.str()
                         << ".csv";

      std::string output_filename = file_suffix_stream.str();
      std::string full_csv_path = csv_dir + "/" + output_filename;
      std::string full_avg_csv_path = avg_csv_dir + "/" + "avg_" + output_filename;

      printf("\n[%.3f s] Writing detailed results to: %s\n", elapsed() - t0, full_csv_path.c_str());
      printf("[%.3f s] Writing average results to: %s\n", elapsed() - t0, full_avg_csv_path.c_str());

      // --- 写详细CSV文件 (per query) ---
      std::ofstream csv_file(full_csv_path);
      csv_file << "repeat,efs,QueryID,acorn_Time_ms,acorn_QPS,acorn_Recall,acorn_n3_visited,"
               << "acorn_build_time_ms,acorn_index_size_MB,"
               << "acorn_1_Time_ms,acorn_1_QPS,acorn_1_Recall,acorn_1_n3_visited,"
               << "acorn_1_build_time_ms,acorn_1_index_size_MB\n";
      for (int repeat = 0; repeat < repeat_num; repeat++)
      {
         for (int efs_id = 0; efs_id < efs_list.size(); efs_id++)
         {
            for (const auto &result : all_query_results[repeat][efs_id])
            {
               csv_file << repeat << ","
                        << efs_list[efs_id] << ","
                        << result.query_id << ","
                        << result.acorn_time_ms << "," << result.acorn_qps << ","
                        << result.acorn_recall << "," << result.acorn_n3 << ","
                        << acorn_build_time_s * 1000.0 << ","
                        << (double)acorn_index_size_bytes / (1024.0 * 1024.0) << ","
                        << result.acorn_1_time_ms << "," << result.acorn_1_qps << ","
                        << result.acorn_1_recall << "," << result.acorn_1_n3 << ","
                        << acorn_1_build_time_s * 1000.0 << ","
                        << (double)acorn_1_index_size_bytes / (1024.0 * 1024.0) << "\n";
            }
         }
      }
      csv_file.close();

      // --- 写平均CSV文件 (per efs) ---
      std::ofstream avg_csv_file(full_avg_csv_path);
      avg_csv_file << "efs,acorn_Time_ms,acorn_QPS,acorn_Recall,acorn_n3_visited_avg,"
                   << "acorn_build_time_ms,acorn_index_size_MB,"
                   << "acorn_1_Time_ms,acorn_1_QPS,acorn_1_Recall,acorn_1_n3_visited_avg,"
                   << "acorn_1_build_time_ms,acorn_1_index_size_MB\n"; // 移除了 FilterMapTime
      for (int efs_id = 0; efs_id < efs_list.size(); efs_id++)
      {
         const auto &aggregated = final_avg_results[efs_id];
         avg_csv_file << efs_list[efs_id] << ","
                      << (aggregated.total_acorn_time_s * 1000.0) / repeat_num << ","
                      << (nq * repeat_num) / aggregated.total_acorn_time_s << ","
                      << aggregated.total_acorn_recall / repeat_num << ","
                      << (double)aggregated.total_acorn_n3 / (nq * repeat_num) << ","
                      << acorn_build_time_s * 1000.0 << ","
                      << (double)acorn_index_size_bytes / (1024.0 * 1024.0) << ","
                      << (aggregated.total_acorn_1_time_s * 1000.0) / repeat_num << ","
                      << (nq * repeat_num) / aggregated.total_acorn_1_time_s << ","
                      << aggregated.total_acorn_1_recall / repeat_num << ","
                      << (double)aggregated.total_acorn_1_n3 / (nq * repeat_num) << ","
                      << acorn_1_build_time_s * 1000.0 << ","
                      << (double)acorn_1_index_size_bytes / (1024.0 * 1024.0) << "\n";
      }
      avg_csv_file.close();

      delete[] gt_data;
      delete[] xq;
      delete hybrid_index;
      delete hybrid_index_gamma1;

      printf("\n[%.3f s] -----SEARCH DONE-----\n", elapsed() - t0);
      return 0;
   }

   return 0;
}