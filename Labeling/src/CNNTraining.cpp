#include <torch/torch.h>
#include <string>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <windows.h>
#include "DataLoaderTotal.h"
#include "MarkerFileInput.h"
#include "VGG8.h"


namespace fs = std::filesystem;

int learningRate(int epoch) {
  if (epoch < 50)
    return 0.6;
  if (epoch < 70)
    return 0.06;
  return 0.006;
}

int main() {
  LoadLibrary("torch_cuda.dll");
  if (!torch::cuda::is_available()) {
    std::cout << "Cuda not available\n";
    return 1;
  }

 

  std::string pathImages, pathMarkers;
  pathImages = "./ImageDataBatch";
  pathMarkers = "./MarkerDataBatch";
  std::vector<std::string> listImages, listMarkers;

  for (const auto& entry : fs::directory_iterator(pathImages))
    listImages.push_back(entry.path().string());

  for(const auto& entry : fs::directory_iterator(pathMarkers))
    listMarkers.push_back(entry.path().string());

  auto dataset = CustomDataset(listImages, listMarkers).map(torch::data::transforms::Stack<>());
  //std::cout << "Size of markers = " << dataset.size() << std::endl;
  int batchSize = 256;
  auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    std::move(dataset),
    batchSize);


  auto net = VGG8();
  net->to(device);


  torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.6));
  //torch::optim::Adam optimizer(net->parameters(), 0.6);

  int n_epochs = 76;
  for (int epoch = 1; epoch < n_epochs; ++epoch) {
    if (epoch == 51) {
        auto options = static_cast<torch::optim::SGDOptions&>(optimizer.defaults());
        options.lr(0.06);
    }
    if (epoch == 71) {
        auto options = static_cast<torch::optim::SGDOptions&>(optimizer.defaults());
        options.lr(0.006);
    }
    for (auto& batch : *dataLoader) {
      auto data = batch.data;
      //std::cout << "Datatype : " << data.type() << std::endl;
      auto target = batch.target.squeeze();
      //std::cout << target.slice(0) << std::endl;
      optimizer.zero_grad();
      
      auto output = net->forward(data).to(torch::kCUDA);
      //auto loss = torch::nn::functional::cross_entropy(output, target);
      //std::cout << target.slice(0) << std::endl;
      //std::cout << output.slice(0) << std::endl;
      //auto loss = torch::nn::functional::l1_loss(output, target);
      //auto loss = torch::nn::functional::smooth_l1_loss(output, target);
      auto loss = torch::nn::functional::nll_loss(output, target);
      //auto loss = torch::nn::functional::mse_loss(output, target);
      loss.backward();
      optimizer.step();
      std::cout << "Train Epoch " << epoch << " with loss " << loss.item<float>() << std::endl;
    }
    std::cout << "Epoch " << epoch << "\n";
    if (epoch == 10) {
        std::string file = "model10.pt";
        torch::save(net, file);
    }
    if (epoch == 20) {
        std::string file = "model20.pt";
        torch::save(net, file);
    }
  }
  torch::save(net, "model-final.pt");

  return 0;
}