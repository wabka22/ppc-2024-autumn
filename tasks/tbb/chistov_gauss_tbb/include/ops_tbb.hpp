#ifndef TASKS_EXAMPLES_TEST_TBB_OPS_TBB_H_
#define TASKS_EXAMPLES_TEST_TBB_OPS_TBB_H_

#include <string>
#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <tbb/blocked_range.h>

#include "core/task/include/task.hpp"

namespace chistov_gauss_tbb {
class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> image, result_image;
  std::vector<double> kernel;
  size_t height, width;
};

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> image, result_image;
  std::vector<double> kernel;
  size_t height, width;
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> image, result_image;
  std::vector<double> kernel;
  size_t height, width;
};

}  // namespace chistov_gauss_tbb

#endif
