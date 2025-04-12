// Copyright 2023 Nesterov Alexander
#include "tbb/chistov_gauss_tbb/include/ops_tbb.hpp"
#include <algorithm>
#include <tbb/tbb.h>

bool chistov_gauss_tbb::TestTaskSequential::pre_processing() {
  kernel.assign(reinterpret_cast<double *>(taskData->inputs[1]), reinterpret_cast<double *>(taskData->inputs[1]) + 3);
  width = static_cast<size_t>(taskData->inputs_count[1]);
  height = static_cast<size_t>(taskData->inputs_count[2]);
  result_image = std::vector<double>(width * height, 0);
  return true;
}

bool chistov_gauss_tbb::TestTaskSequential::validation() {
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

bool chistov_gauss_tbb::TestTaskSequential::run() {
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

bool chistov_gauss_tbb::TestTaskSequential::post_processing() {
  std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(taskData->outputs[0]));
  return true;
}

bool chistov_gauss_tbb::TestTaskOpenMP::pre_processing() {
  kernel.assign(reinterpret_cast<double *>(taskData->inputs[1]), reinterpret_cast<double *>(taskData->inputs[1]) + 3);
  width = static_cast<size_t>(taskData->inputs_count[1]);
  height = static_cast<size_t>(taskData->inputs_count[2]);
  result_image = std::vector<double>(width * height, 0);
  return true;
}

bool chistov_gauss_tbb::TestTaskOpenMP::validation() {
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

bool chistov_gauss_tbb::TestTaskOpenMP::run() {
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

bool chistov_gauss_tbb::TestTaskOpenMP::post_processing() {
  std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(taskData->outputs[0]));
  return true;
}

 bool chistov_gauss_tbb::TestTaskTBB::pre_processing() {
  kernel.assign(reinterpret_cast<double *>(taskData->inputs[1]), reinterpret_cast<double *>(taskData->inputs[1]) +3); 
  width = static_cast<size_t>(taskData->inputs_count[1]); 
  height =static_cast<size_t>(taskData->inputs_count[2]); 
  result_image = std::vector<double>(width * height, 0);

  return true;
}

 bool chistov_gauss_tbb::TestTaskTBB::validation() {
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

bool chistov_gauss_tbb::TestTaskTBB::run() {
  double inv_kernel_sum = 1.0 / std::accumulate(kernel.begin(), kernel.end(), 0.0);

 auto functor = [&](const tbb::blocked_range<int> &range) {
    for (int row = range.begin(); row < range.end(); ++row) {
      size_t row_offset = row * width;
      for (size_t col = 0; col < width; ++col) {
        double left_pixel = (col > 0) ? image[row_offset + (col - 1)] * kernel[0] : 0.0;
        double center_pixel = image[row_offset + col] * kernel[1];
        double right_pixel = (col < width - 1) ? image[row_offset + (col + 1)] * kernel[2] : 0.0;

        result_image[row_offset + col] = (left_pixel + center_pixel + right_pixel) * inv_kernel_sum;
      }
    }
  };

  const int int_height = static_cast<int>(height);
  oneapi::tbb::task_arena arena(16);

  arena.execute([&] { tbb::parallel_for(tbb::blocked_range<int>(0, int_height), functor); });

  return true;
}

 bool chistov_gauss_tbb::TestTaskTBB::post_processing() {
  std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(taskData->outputs[0]));
  return true;
}
