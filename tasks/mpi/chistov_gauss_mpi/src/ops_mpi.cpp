// Copyright 2023 Nesterov Alexander
#include "mpi/chistov_gauss_mpi/include/ops_mpi.hpp"
//#include <boost/mpi.hpp>
#include <algorithm>

bool chistov_gauss_mpi::TestTaskSequential::pre_processing() {
  kernel.assign(reinterpret_cast<double *>(taskData->inputs[1]), reinterpret_cast<double *>(taskData->inputs[1]) + 3);
  width = static_cast<size_t>(taskData->inputs_count[1]);
  height = static_cast<size_t>(taskData->inputs_count[2]);
  result_image = std::vector<double>(width * height, 0);
  return true;
}

bool chistov_gauss_mpi::TestTaskSequential::validation() {
  if (taskData->inputs[0] == nullptr || taskData->inputs_count[0] == 0) {
    return false;
  }

  image.assign(reinterpret_cast<double *>(taskData->inputs[0]),
               reinterpret_cast<double *>(taskData->inputs[0]) + taskData->inputs_count[0]);

  for (size_t i = 0; i < taskData->inputs_count[1] * taskData->inputs_count[2]; ++i) {
    if (image[i] < 0 || image[i] > 255) {
      return false;
    }
  }

  return taskData->inputs_count[0] > 0 &&
         taskData->inputs_count[0] == taskData->inputs_count[1] * taskData->inputs_count[2] &&
         taskData->outputs_count[0] == taskData->inputs_count[1] * taskData->inputs_count[2] &&
         taskData->inputs_count[1] >= 3 && taskData->inputs_count[2] >= 3;
}

bool chistov_gauss_mpi::TestTaskSequential::run() {
  double sum = kernel[0] + kernel[1] + kernel[2];
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      double value = 0.0;
      for (ptrdiff_t k = -1; k <= 1; ++k) {
        ptrdiff_t tmp = static_cast<ptrdiff_t>(j) + k;

        if (tmp >= 0 && tmp < static_cast<ptrdiff_t>(width)) {
          value += image[(i * width) + tmp] * kernel[k + 1];
        }
      }

      result_image[(i * width) + j] = value / sum;
    }
  }

  return true;
}

bool chistov_gauss_mpi::TestTaskSequential::post_processing() {
  std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(taskData->outputs[0]));
  return true;
}

bool chistov_gauss_mpi::TestTaskOpenMP::pre_processing() {
  kernel.assign(reinterpret_cast<double *>(taskData->inputs[1]), reinterpret_cast<double *>(taskData->inputs[1]) + 3);
  width = static_cast<size_t>(taskData->inputs_count[1]);
  height = static_cast<size_t>(taskData->inputs_count[2]);
  result_image = std::vector<double>(width * height, 0);
  return true;
}

bool chistov_gauss_mpi::TestTaskOpenMP::validation() {
  if (taskData->inputs[0] == nullptr || taskData->inputs_count[0] == 0) {
    return false;
  }

  image.assign(reinterpret_cast<double *>(taskData->inputs[0]),
               reinterpret_cast<double *>(taskData->inputs[0]) + taskData->inputs_count[0]);

  for (size_t i = 0; i < taskData->inputs_count[1] * taskData->inputs_count[2]; ++i) {
    if (image[i] < 0 || image[i] > 255) {
      return false;
    }
  }

  return taskData->inputs_count[0] > 0 &&
         taskData->inputs_count[0] == taskData->inputs_count[1] * taskData->inputs_count[2] &&
         taskData->outputs_count[0] == taskData->inputs_count[1] * taskData->inputs_count[2] &&
         taskData->inputs_count[1] >= 3 && taskData->inputs_count[2] >= 3;
}

bool chistov_gauss_mpi::TestTaskOpenMP::run() {
  double sum_inv = 1.0 / (kernel[0] + kernel[1] + kernel[2]);
  int h = static_cast<int>(height);
  int w = static_cast<int>(width);

#pragma omp parallel firstprivate(sum_inv) shared(w, h) num_threads(1)
  {
#pragma omp for
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        double pixel_0 = (j > 0) ? image[i * width + (j - 1)] * kernel[0] : 0.0;
        double pixel_1 = image[i * width + j] * kernel[1];
        double pixel_2 = (j < width - 1) ? image[i * width + (j + 1)] * kernel[2] : 0.0;

        result_image[i * width + j] = (pixel_0 + pixel_1 + pixel_2) * sum_inv;
      }
    }
  }

  return true;
}

bool chistov_gauss_mpi::TestTaskOpenMP::post_processing() {
  std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(taskData->outputs[0]));
  return true;
}

bool chistov_gauss_mpi::TestTaskMPI::pre_processing() {
  internal_order_test();

  kernel.assign(reinterpret_cast<double *>(taskData->inputs[1]), reinterpret_cast<double *>(taskData->inputs[1]) + 3);
  width = static_cast<size_t>(taskData->inputs_count[1]);
  height = static_cast<size_t>(taskData->inputs_count[2]);
  result_image = std::vector<double>(width * height, 0);
  sum_inv = 1.0 / (kernel[0] + kernel[1] + kernel[2]);

  return true;
}

bool chistov_gauss_mpi::TestTaskMPI::validation() {
  internal_order_test();

  image.assign(reinterpret_cast<double *>(taskData->inputs[0]),
               reinterpret_cast<double *>(taskData->inputs[0]) + taskData->inputs_count[0]);


  if (world.rank() == 0) {
    if (taskData->inputs[0] == nullptr || taskData->inputs_count[0] == 0) {
      return false;
    }

    for (size_t i = 0; i < taskData->inputs_count[1] * taskData->inputs_count[2]; ++i) {
      if (image[i] < 0 || image[i] > 255) {
        return false;
      }
    }

    return taskData->inputs_count[0] > 0 &&
           taskData->inputs_count[0] == taskData->inputs_count[1] * taskData->inputs_count[2] &&
           taskData->outputs_count[0] == taskData->inputs_count[1] * taskData->inputs_count[2] &&
           taskData->inputs_count[1] >= 3 && taskData->inputs_count[2] >= 3;
  }

  return true;
}

std::pair<int, int> get_row_range(int rank, int total_rows, int num_procs) {
  int rows_per_proc = total_rows / num_procs;
  int remaining_rows = total_rows % num_procs;
  int start = rank * rows_per_proc + std::min(rank, remaining_rows);
  int end = start + rows_per_proc + (rank < remaining_rows ? 1 : 0);
  return {start, end};
}

bool chistov_gauss_mpi::TestTaskMPI::run() {
  internal_order_test();

  int rank = world.rank();
  int size = world.size();

  const int h = static_cast<int>(height);
  const int w = static_cast<int>(width);

  auto [start_row, end_row] = get_row_range(rank, h, size);
  int local_rows = end_row - start_row;

  std::vector<double> local_result(local_rows * w);

#pragma omp parallel for schedule(static) 
  for (int i = 0; i < local_rows; ++i) {
    int global_row = start_row + i;
    const double *row_ptr = &image[global_row * w];
    double *out_ptr = &local_result[i * w];

    double k0 = kernel[0], k1 = kernel[1], k2 = kernel[2];

#pragma omp simd
    for (int j = 0; j < w; ++j) {
      double p0 = (j > 0) ? row_ptr[j - 1] : 0.0;
      double p1 = row_ptr[j];
      double p2 = (j < w - 1) ? row_ptr[j + 1] : 0.0;
      out_ptr[j] = (p0 * k0 + p1 * k1 + p2 * k2) * sum_inv;
    }
  }

  if (rank == 0) {
    result_image.resize(h * w);

    std::copy(local_result.begin(), local_result.end(), result_image.begin() + start_row * w);

    for (int i = 1; i < size; ++i) {
      auto [s_row, e_row] = get_row_range(i, h, size);
      int rows = e_row - s_row;
      world.recv(i, 0, &result_image[s_row * w], rows * w);
    }

  } else {
    world.send(0, 0, local_result.data(), local_rows * w);
  }

  return true;
}

bool chistov_gauss_mpi::TestTaskMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(taskData->outputs[0]));
  }
  return true;
}