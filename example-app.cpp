#include <torch/torch.h>
#include <iostream>
#include <string>
#include <map>
struct Net : torch::nn::Module {
    Net(int N, int M):W(register_module("W", torch::nn::Linear(N, M))) {}
    
    torch::Tensor forward(torch::Tensor input){
      return W(input);
    }
    torch::nn::Linear W;
};

struct DLGN : torch::nn::Module { 
  int num_layers;
  int beta;
  std::string mode;
  torch::nn::ModuleList gating_layers{nullptr}, value_layers{nullptr};
  // torch::nn::Linear(())
  DLGN(int input_dim, int output_dim, int* hidden_nodes, int num_layers, int beta) {
    gating_layers = torch::nn::ModuleList();
    value_layers = torch::nn::ModuleList();

    this->num_layers = num_layers;
    this->beta = beta;
    this->mode = "pwc";

    torch::nn::Linear input_gating_layer = torch::nn::Linear(input_dim, hidden_nodes[0]);
    torch::nn::Linear input_value_layer = torch::nn::Linear(input_dim, hidden_nodes[0]);
    gating_layers->push_back(input_gating_layer);
    value_layers->push_back(input_value_layer);
    for(int i=0; i< num_layers; i++){
      auto gl = torch::nn::Linear(hidden_nodes[i], hidden_nodes[i+1]);
      auto vl = torch::nn::Linear(hidden_nodes[i], hidden_nodes[i+1]);
      gating_layers->push_back(gl);
      value_layers->push_back(vl);
    }
    auto output_value_layer = torch::nn::Linear(hidden_nodes[num_layers - 1], output_dim);
    value_layers->push_back(output_value_layer);
  }

  // void set_parameters(DLGN to_copy, std::map<std::string, int> parameters_masks){
  //   for (const auto &pair : to_copy.named_parameters()) {
  //     std::cout << pair.key() << ": " << pair.value() << std::endl;
  //     std::string name = pair.key();
  //     torch::Tensor param = pair.value();

  //     if (parameters_masks.count(name) == 0){
  //       this->named_parameters()[name] = param.clone();
  //     } else {
  //       bool mask = parameters_masks[name] > 0;

  //     }
      
  //   }
  // }

  void forward(torch::Tensor x){
    
  }
};

int main() {
  torch::Device device = torch::kCUDA;
  Net net(5, 6);
  net.to(device);
  torch::Tensor input = torch::rand({5}, device);
  std::cout << input << std::endl;
  std::cout << net.forward(input) << std::endl;
}