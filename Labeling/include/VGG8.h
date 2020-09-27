#pragma once
#include <torch/torch.h>

struct VGG8Impl : public torch::nn::Module {
  VGG8Impl() {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)));
    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3)));
    conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3)));
    conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3)));

    bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
    bn3 = register_module("bn3", torch::nn::BatchNorm2d(128));
    bn4 = register_module("bn4", torch::nn::BatchNorm2d(128));
    bn5 = register_module("bn5", torch::nn::BatchNorm2d(128));

    fc1 = register_module("fc1", torch::nn::Linear(128 * 9 * 9, 2048));
    fc2 = register_module("fc2", torch::nn::Linear(2048, 57));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(bn1(conv1(x)));
    x = torch::relu(bn2(conv2(x)));
    x = torch::max_pool2d(x, 2);

    x = torch::relu(bn3(conv3(x)));
    x = torch::relu(bn4(conv4(x)));
    x = torch::relu(bn5(conv5(x)));
    x = torch::max_pool2d(x, 2);
    x = x.view({ -1, 128 * 9 * 9 });
    x = torch::relu(fc1(x));
    x = fc2(x);
    x = x.view({ -1, 19, 3 });
    //  19,3 });

    return x;
  }

  torch::nn::Conv2d conv1 = nullptr, conv2 = nullptr, conv3 = nullptr, conv4 = nullptr, conv5 = nullptr;
  torch::nn::BatchNorm2d bn1 = nullptr, bn2 = nullptr, bn3 = nullptr, bn4 = nullptr, bn5 = nullptr;
  torch::nn::Linear fc1 = nullptr, fc2 = nullptr;


};
TORCH_MODULE(VGG8);
