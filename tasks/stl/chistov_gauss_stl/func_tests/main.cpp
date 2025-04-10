// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "stl/chistov_gauss_stl/include/ops_stl.hpp"

 namespace chistov_gauss_stl_test {

 void CompareImages(const std::vector<double>& output_image, const std::vector<double>& expected_image, size_t width,
                    size_t height, double tolerance = 1e-6) {
   std::cout << "Output Image: " << std::endl;
   for (size_t i = 0; i < output_image.size(); ++i) {
     std::cout << output_image[i] << " ";
     if ((i + 1) % width == 0) {
       std::cout << std::endl;
     }
   }

     std::cout << "Expected Image: " << std::endl;
   for (size_t i = 0; i < output_image.size(); ++i) {
       std::cout << expected_image[i] << " ";
     if ((i + 1) % width == 0) {
       std::cout << std::endl;
     }
   }


   for (size_t i = 0; i < height; ++i) {
     for (size_t j = 0; j < width; ++j) {
       ASSERT_NEAR(output_image[(i * width) + j], expected_image[(i * width) + j], tolerance);
     }
   }
 }

 void RunImageFilterTest(std::shared_ptr<ppc::core::TaskData> task_data_tbb) {
   chistov_gauss_stl::TestTaskSTL image_filter_sequential(task_data_tbb);
   ASSERT_EQ(image_filter_sequential.validation(), true);
   image_filter_sequential.pre_processing();
   image_filter_sequential.run();
   image_filter_sequential.post_processing();
 }

 std::shared_ptr<ppc::core::TaskData> CreateTaskData(std::vector<double>& input_image, std::vector<double>&
 output_image,
     std::vector<double>& kernel, size_t width, size_t height) {
   auto task_data = std::make_shared<ppc::core::TaskData>();
   task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
   task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
   task_data->inputs_count.emplace_back(input_image.size());
   task_data->inputs_count.emplace_back(width);
   task_data->inputs_count.emplace_back(height);
   task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
   task_data->outputs_count.emplace_back(output_image.size());
   return task_data;
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

 }  // namespace chistov_gauss_stl_test

TEST(chistov_gauss_stl, test_empty_image) {
  constexpr size_t width = 5;
  constexpr size_t height = 5;
  std::vector<double> input_image;
  std::vector<double> output_image;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->inputs_count.emplace_back(width);
  task_data_tbb->inputs_count.emplace_back(height);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  chistov_gauss_stl::TestTaskSTL test_task_tbb(task_data_tbb);
  ASSERT_FALSE(test_task_tbb.validation());
}

TEST(chistov_gauss_stl, test_invalid_pixel_values) {
  constexpr size_t width = 3;
  constexpr size_t height = 3;
  std::vector<double> input_image = {-1, 50, 200, 255, 300, 0, 128, 255, -10};
  std::vector<double> output_image;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->inputs_count.emplace_back(width);
  task_data_tbb->inputs_count.emplace_back(height);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  chistov_gauss_stl::TestTaskSTL test_task_tbb(task_data_tbb);
  ASSERT_FALSE(test_task_tbb.validation());
}

TEST(chistov_gauss_stl, test_invalid_output_size) {
  constexpr size_t width = 3;
  constexpr size_t height = 3;
  std::vector<double> input_image = {10, 20, 30, 40, 50, 60, 70, 80, 90};
  std::vector<double> output_image(8);

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->inputs_count.emplace_back(width);
  task_data_tbb->inputs_count.emplace_back(height);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  chistov_gauss_stl::TestTaskSTL test_task_tbb(task_data_tbb);
  ASSERT_FALSE(test_task_tbb.validation());
}

TEST(chistov_gauss_stl, test_too_small_image) {
  constexpr size_t width = 1;
  constexpr size_t height = 1;
  std::vector<double> input_image;
  std::vector<double> output_image;

  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_tbb->inputs_count.emplace_back(input_image.size());
  task_data_tbb->inputs_count.emplace_back(width);
  task_data_tbb->inputs_count.emplace_back(height);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_tbb->outputs_count.emplace_back(output_image.size());

  chistov_gauss_stl::TestTaskSTL test_task_tbb(task_data_tbb);
  ASSERT_FALSE(test_task_tbb.validation());
}

TEST(chistov_gauss_stl, test_small_image) {
  constexpr size_t width = 3;
  constexpr size_t height = 3;

  std::vector<double> input_image = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};

  std::vector<double> output_image(width * height, 0.0);

std::vector<double> expected_image = {10.0, 20.0, 20.0, 32.5, 50.0, 42.5, 55.0, 80.0, 65.0};

  std::vector<double> kernel = {1, 2, 1};

  auto task_data_tbb = chistov_gauss_stl_test::CreateTaskData(input_image, output_image, kernel, width, height);
  chistov_gauss_stl_test::RunImageFilterTest(task_data_tbb);
  chistov_gauss_stl_test::CompareImages(output_image, expected_image, width, height);
}

TEST(chistov_gauss_stl, test_non_standard_kernel) {
  constexpr size_t width = 3;
  constexpr size_t height = 3;

  std::vector<double> input_image = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};

  std::vector<double> output_image(width * height, 0.0);

 std::vector<double> expected_image = {10.0, 20.0, 20.0, 32.5, 50.0, 42.5, 55.0, 80.0, 65.0};

  std::vector<double> kernel = {2, 4, 2};

  auto task_data_tbb = chistov_gauss_stl_test::CreateTaskData(input_image, output_image, kernel, width, height);
  chistov_gauss_stl_test::RunImageFilterTest(task_data_tbb);
  chistov_gauss_stl_test::CompareImages(output_image, expected_image, width, height);
}

TEST(chistov_gauss_stl, test_different_width_and_height) {
  constexpr size_t width = 4;
  constexpr size_t height = 5;

  std::vector<double> input_image = {10.0,  20.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0,  100.0,
                                     110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0};

  std::vector<double> output_image(width * height, 0.0);

  std::vector<double> expected_image = {10.0,  20.0, 30.0,  27.5,  40.0,  60.0,  70.0,  57.5,  70.0,  100.0,
                                        110.0, 87.5, 100.0, 140.0, 150.0, 117.5, 130.0, 180.0, 190.0, 147.5};

  std::vector<double> kernel = {1, 2, 1};

  auto task_data_tbb = chistov_gauss_stl_test::CreateTaskData(input_image, output_image, kernel, width, height);
  chistov_gauss_stl_test::RunImageFilterTest(task_data_tbb);
  chistov_gauss_stl_test::CompareImages(output_image, expected_image, width, height);
}

TEST(chistov_gauss_stl, test_zero_values_image) {
  constexpr size_t width = 5;
  constexpr size_t height = 5;
  std::vector<double> input_image(width * height, 0.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_image(width * height, 0.0);
  std::vector<double> kernel = {1, 2, 1};

  auto task_data_tbb = chistov_gauss_stl_test::CreateTaskData(input_image, output_image, kernel, width, height);
  chistov_gauss_stl_test::RunImageFilterTest(task_data_tbb);
  chistov_gauss_stl_test::CompareImages(output_image, expected_image, width, height);
}

TEST(chistov_gauss_stl, test_max_pixel_values_image) {
  constexpr size_t width = 10;
  constexpr size_t height = 10;
  std::vector<double> input_image(width * height, 255.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_image(width * height, 255.0);
  std::vector<double> kernel = {1, 2, 1};

    for (size_t i = 0; i < height; ++i) {
    expected_image[(i * width)] = 191.25;
      expected_image[((i + 1) * width) - 1] = 191.25;
  }

  auto task_data_tbb = chistov_gauss_stl_test::CreateTaskData(input_image, output_image, kernel, width, height);
  chistov_gauss_stl_test::RunImageFilterTest(task_data_tbb);
  chistov_gauss_stl_test::CompareImages(output_image, expected_image, width, height);
}