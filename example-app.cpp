#include <fstream>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>

struct Net : torch::nn::Module
{
  Net(int N, int M) : W(register_module("W", torch::nn::Linear(N, M))) {}

  torch::Tensor forward(torch::Tensor input)
  {
    return W(input);
  }
  torch::nn::Linear W;
};

std::vector<char> get_the_bytes(std::string filename)
{
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));

  input.close();
  return bytes;
}
bool debug = false;

struct DLGNImpl : torch::nn::Module
{
  int num_layers;
  int beta;
  std::string mode;
  torch::nn::ModuleList gating_layers{nullptr}, value_layers{nullptr};
  // torch::nn::Linear(())
  DLGNImpl(int input_dim, int output_dim, int *hidden_nodes, int num_layers, int beta)
  {
    gating_layers = torch::nn::ModuleList();
    value_layers = torch::nn::ModuleList();

    this->num_layers = num_layers;
    this->beta = beta;
    this->mode = "pwc";
    torch::nn::Linear input_gating_layer = torch::nn::Linear(input_dim, hidden_nodes[0]);
    torch::nn::Linear input_value_layer = torch::nn::Linear(input_dim, hidden_nodes[0]);
    gating_layers->push_back(input_gating_layer);
    value_layers->push_back(input_value_layer);
    for (int i = 0; i < num_layers - 1; i++)
    {
      auto gl = torch::nn::Linear(hidden_nodes[i], hidden_nodes[i + 1]);
      auto vl = torch::nn::Linear(hidden_nodes[i], hidden_nodes[i + 1]);
      gating_layers->push_back(gl);
      value_layers->push_back(vl);
    }
    auto output_value_layer = torch::nn::Linear(hidden_nodes[num_layers - 1], output_dim);
    value_layers->push_back(output_value_layer);

    register_module("gating_layer", gating_layers);
    register_module("value_layer", value_layers);
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

  std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> forward(torch::Tensor x)
  {
    torch::Device device(torch::kCPU);

    for (const auto &el : this->parameters())
    {
      if (el.is_cuda())
      {
        device = torch::kCUDA;
      }
      else
      {
        device = torch::kCPU;
      }
    }
    auto values = std::vector<torch::Tensor>(1, torch::ones(x.sizes()).to(device));
    auto gate_scores = std::vector<torch::Tensor>(1, x);
    for (int i = 0; i < this->num_layers; i++)
    {
      if (debug)
        std::cout << "is it here ??? " << i << std::endl;
      gate_scores.push_back(gating_layers[i]->as<torch::nn::Linear>()->forward(gate_scores.back()));
      torch::Tensor curr_gate_on_off = torch::sigmoid(beta * gate_scores.back());
      values.push_back(value_layers[i]->as<torch::nn::Linear>()->forward(values.back()) * curr_gate_on_off);
    }
    values.push_back(value_layers[num_layers]->as<torch::nn::Linear>()->forward(values.back()));

    return {values, gate_scores};
  }
};

TORCH_MODULE(DLGN);

DLGN trainDLGN(DLGN dlgn_obj, torch::Tensor train_tensor, torch::Tensor train_labels_tensor, int num_epochs)
{
  if (debug)
    std::cout << "training began" << std::endl;
  auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  if (debug)
    std::cout << "Device chosen " << device << std::endl;
  dlgn_obj->to(device);

  // auto train_tensor = torch::tensor(train_data, device);
  // auto train_labels_tensor = torch::tensor(train_labels, device);
  train_labels_tensor = train_labels_tensor.to(torch::kInt64);
  train_labels_tensor = train_labels_tensor.to(device);
  train_tensor = train_tensor.to(device);

  auto optimizer = torch::optim::Adam(dlgn_obj->parameters(), torch::optim::AdamOptions(1e-3));
  auto criterion = torch::nn::CrossEntropyLoss();

  int num_batches = 10;
  auto num_data = train_tensor.size(0);
  int batch_size = num_data / num_batches;
  for (int epoch = 1; epoch <= num_epochs; epoch++)
  {
    std::cout << "Starting epoch " << epoch << std::endl;
    for (int batch_num = 0; batch_num < num_data; batch_num += batch_size)
    {
      if (batch_num + batch_size > num_data)
        break;
      if (debug)
        std::cout << "In the batch " << batch_num << " till " << batch_num + batch_size << std::endl;
      auto batch_data = train_tensor.index({torch::indexing::Slice(batch_num, batch_num + batch_size)}).to(device);
      auto batch_labels = train_labels_tensor.index({torch::indexing::Slice(batch_num, batch_num + batch_size)}).reshape(batch_size).to(device);
      if (debug)
        std::cout << "Batch data size " << batch_data.size(0) << " " << batch_data.size(1) << std::endl;
      if (debug)
        std::cout << "Created the batches on device " << batch_data.device() << std::endl;
      auto pair = dlgn_obj->forward(batch_data);
      if (debug)
        std::cout << "Forward process done now " << std::endl;
      auto values = pair.first;
      if (debug)
        std::cout << "what does value look like?" << values.back() << std::endl;
      auto outputs = torch::cat({-1 * values.back(), values.back()}, 1);
      if (debug)
        std::cout << "Outputs have been created thank god " << std::endl;
      if (debug)
        std::cout << "Analyzing outputs: " << outputs.size(0) << " " << batch_labels.size(0) << std::endl;
      if (debug)
        std::cout << " Output is " << outputs << std::endl
                  << "Train_data is " << batch_labels << std::endl;
      if (debug)
        std::cout << "Output is in this device btw " << outputs.device() << std::endl;
      optimizer.zero_grad();
      auto loss = criterion(outputs, batch_labels);
      std::cout << "Current loss: " << loss << std::endl;
      loss.backward();
      optimizer.step();
    }
  }
  return dlgn_obj;
}

int main()
{
  torch::Device device = torch::kCUDA;
  Net net(5, 6);
  net.to(device);
  torch::Tensor input = torch::rand({5}, device);
  std::cout << input << std::endl;
  std::cout << net.forward(input) << std::endl;

  // DLGN test

  // train the dlgn with some random data just to see how it does
  std::vector<char> f = get_the_bytes("covertype_features.pt");
  std::cout << "here" << std::endl;
  torch::IValue x = torch::pickle_load(f);
  torch::Tensor train_data = x.toTensor().to(torch::kFloat32);
  f = get_the_bytes("covertype_labels.pt");
  x = torch::pickle_load(f);
  torch::Tensor train_labels = x.toTensor();
  std::cout << "Size of tensor" << train_data.size(0) << "  " << train_data.size(1) << "\n";

  int layers[3] = {50, 50, 50};
  DLGN dlgn(10, 1, layers, 3, 5);
  dlgn->to(device);
  input = torch::randn({2, 10}, device);
  std::cout << "Reached dlgn time??" << std::endl;
  std::cout << dlgn->forward(input) << std::endl;
  trainDLGN(dlgn, train_data, train_labels, 2000);
}