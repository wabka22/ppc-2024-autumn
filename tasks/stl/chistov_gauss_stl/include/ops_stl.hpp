#ifndef TASKS_EXAMPLES_TEST_STD_OPS_STD_H_
#define TASKS_EXAMPLES_TEST_STD_OPS_STD_H_


#include <string>
#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/task/include/task.hpp"

namespace chistov_gauss_stl {
class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
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
