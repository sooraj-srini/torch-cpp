#include <torch/torch.h>
#include <iostream>

struct Net : torch::nn::Module {
    Net(int N, int M):W(register_module("W", torch::nn::Linear(N, M))) {}
    
    torch::Tensor forward(torch::Tensor input){
      return W(input);
    }
    torch::nn::Linear W;
}; 

int main() {
  torch::Device device = torch::kCUDA;
  Net net(5, 6);
  net.to(device);
  torch::Tensor input = torch::rand({5}, device);
  std::cout << input << std::endl;
  std::cout << net.forward(input) << std::endl;
}