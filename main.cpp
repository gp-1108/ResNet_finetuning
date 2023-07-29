#include <torch/script.h> // One-stop header.
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <memory>

int main(int argc, const char *argv[])
{
  if (argc < 2)
  {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error &e)
  {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // Load the image
  cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);

  // Convert the image to RGB format
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  // Resize the image to the desired size
  cv::Size desiredSize(224, 224);
  cv::resize(image, image, desiredSize);

  // Convert the image to floating-point representation and normalize the pixel values
  image.convertTo(image, CV_32F, 1.0 / 255.0);
  cv::Scalar mean(0.485, 0.456, 0.406);
  cv::Scalar std(0.229, 0.224, 0.225);
  cv::subtract(image, mean, image);
  cv::divide(image, std, image);

  // Transpose the image to match the PyTorch tensor format (C, H, W)
  cv::transpose(image, image);

  // Create a tensor from the preprocessed image
  torch::Tensor inputTensor = torch::from_blob(image.data, {1, image.rows, image.cols, image.channels()}, torch::kFloat32);
  inputTensor = inputTensor.permute({0, 3, 1, 2});

  std::cout << inputTensor.sizes() << std::endl;

  // std::cout << "ok\n";

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  // inputs.push_back(torch::ones({1, 3, 224, 224}));
  inputs.push_back(inputTensor);

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  for (int i = 0; i < output.size(0); ++i)
  {
    double value = output[i].item<double>();
    std::cout << "Element " << i << ": " << value << std::endl;
  }
}