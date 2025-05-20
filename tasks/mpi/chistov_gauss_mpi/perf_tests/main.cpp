#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/chistov_gauss_mpi/include/ops_mpi.hpp"

namespace chistov_gauss_mpi_test {

void CompareImages(const std::vector<double>& output_image, const std::vector<double>& expected_image, size_t width,
                   size_t height, double tolerance = 1e-6) {
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ASSERT_NEAR(output_image[(i * width) + j], expected_image[(i * width) + j], tolerance);
    }
  }
}

std::vector<double> GenerateRandomImage(size_t width, size_t height, double min_val = 0.0, double max_val = 255.0) {
  std::vector<double> image(width * height);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min_val, max_val);

  for (size_t i = 0; i < width * height; ++i) {
    image[i] = dis(gen);
  }

  return image;
}

}  // namespace chistov_gauss_mpi_test

TEST(SEQ, test_task_run) {
  const size_t width = 7000;
  const size_t height = 7000;

  std::vector<double> input_image(width * height, 255.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_image(width * height, 255.0);
  std::vector<double> kernel = {1, 2, 1};

  for (size_t i = 0; i < height; ++i) {
    expected_image[(i * width)] = 191.25;
    expected_image[((i + 1) * width) - 1] = 191.25;
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  // Create Task
  auto test_task_sequential = std::make_shared<chistov_gauss_mpi::TestTaskSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->task_run(perf_attr, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);

  chistov_gauss_mpi_test::CompareImages(output_image, expected_image, width, height);
}

TEST(OMP, test_task_run) {
  const size_t width = 7000;
  const size_t height = 7000;

  std::vector<double> input_image(width * height, 255.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_image(width * height, 255.0);
  std::vector<double> kernel = {1, 2, 1};

  for (size_t i = 0; i < height; ++i) {
    expected_image[(i * width)] = 191.25;
    expected_image[((i + 1) * width) - 1] = 191.25;
  }

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_omp->inputs_count.emplace_back(input_image.size());
  task_data_omp->inputs_count.emplace_back(width);
  task_data_omp->inputs_count.emplace_back(height);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_omp->outputs_count.emplace_back(output_image.size());

  // Create Task
  auto test_task = std::make_shared<chistov_gauss_mpi::TestTaskOpenMP>(task_data_omp);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->task_run(perf_attr, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);

  chistov_gauss_mpi_test::CompareImages(output_image, expected_image, width, height);
}

TEST(MPI, test_task_run) {
  boost::mpi::communicator world;
  const size_t width = 6000;
  const size_t height = 6000;

  std::vector<double> input_image(width * height, 255.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_image(width * height, 255.0);
  std::vector<double> kernel = {1, 2, 1};
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_mpi->inputs_count.emplace_back(input_image.size());
  task_data_mpi->inputs_count.emplace_back(width);
  task_data_mpi->inputs_count.emplace_back(height);
  // Create task_data
  if (world.rank() == 0) {
    for (size_t i = 0; i < height; ++i) {
      expected_image[(i * width)] = 191.25;
      expected_image[((i + 1) * width) - 1] = 191.25;
    }
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
    task_data_mpi->outputs_count.emplace_back(output_image.size());
  }
  // Create Task
  auto test_task = std::make_shared<chistov_gauss_mpi::TestTaskMPI>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->task_run(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);
    chistov_gauss_mpi_test::CompareImages(output_image, expected_image, width, height);
  }
}
