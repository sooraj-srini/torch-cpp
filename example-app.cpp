#include <fstream>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>

using namespace torch::indexing;

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
    auto gate_old = x;
    auto value_old = torch::ones(x.sizes()).to(device);
    for (int i = 0; i < this->num_layers; i++)
    {
      // gate_scores.push_back(gating_layers[i]->as<torch::nn::Linear>()->forward(gate_scores.back()));
      gate_old = (gating_layers[i]->as<torch::nn::Linear>()->forward(gate_old));
      // torch::Tensor curr_gate_on_off = torch::sigmoid(beta * gate_scores.back());
      torch::Tensor curr_gate_on_off = torch::sigmoid(beta * gate_old);
      // values.push_back(value_layers[i]->as<torch::nn::Linear>()->forward(values.back()) * curr_gate_on_off);
      value_old = (value_layers[i]->as<torch::nn::Linear>()->forward(value_old) * curr_gate_on_off);
    }
    // values.push_back(value_layers[num_layers]->as<torch::nn::Linear>()->forward(values.back()));
    value_old = (value_layers[num_layers]->as<torch::nn::Linear>()->forward(value_old));

    // return {values, gate_scores};
    return {{value_old}, {gate_old}};
  }
};

TORCH_MODULE(DLGN);

DLGN trainDLGN(DLGN dlgn_obj, torch::Tensor train_tensor, torch::Tensor train_labels_tensor, int num_epochs)
{
  auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
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
    std::cout << "Starting epoch " << epoch << "\n";
    for (int batch_num = 0; batch_num < num_data; batch_num += batch_size)
    {
      if (batch_num + batch_size > num_data)
        break;
      auto batch_data = train_tensor.index({torch::indexing::Slice(batch_num, batch_num + batch_size)});
      auto batch_labels = train_labels_tensor.index({torch::indexing::Slice(batch_num, batch_num + batch_size)}).reshape(batch_size);
      auto pair = dlgn_obj->forward(batch_data);
      auto values = pair.first;
      auto outputs = torch::cat({-1 * values.back(), values.back()}, 1);
      optimizer.zero_grad();
      auto loss = criterion(outputs, batch_labels);
      loss.backward();
      optimizer.step();
    }
  }
  return dlgn_obj;
}


//works only when the tensors are on the same device
torch::Tensor inference(DLGN dlgn_obj, torch::Tensor features, torch::Device device) {
  features = features.to(device);
  auto pair = dlgn_obj->forward(features);
  auto values = pair.first;
  auto preds = values.back();
  auto prediction = torch::sign(preds.index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), 0}));
  prediction = torch::add(prediction, 1);
  prediction = torch::floor(torch::div(prediction, 2));
  return prediction;
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
  std::vector<char> f = get_the_bytes("/root/sem/torch-cpp/electricity_features.pt");
  torch::IValue x = torch::pickle_load(f);
  torch::Tensor data = x.toTensor().to(torch::kFloat32);
  f = get_the_bytes("/root/sem/torch-cpp/electricity_labels.pt");
  x = torch::pickle_load(f);
  torch::Tensor labels = x.toTensor();
  labels = labels.to(torch::kInt64);
  int num_data = data.size(0);

  //Split the data into train validation and test
  auto train_data = data.index({Slice(None, int(0.7 * num_data))});
  auto train_labels = labels.index({Slice(None, int(0.7 * num_data))});
  auto valid_data = data.index({Slice(int(0.7 * num_data), int(0.79 * num_data))});
  auto valid_labels= labels.index({Slice(int(0.7 * num_data), int(0.79 * num_data))});
  auto test_data = data.index({Slice(int(0.79 * num_data), None)});
  auto test_labels = labels.index({Slice(int(0.79 * num_data), None)});

  int layers[3] = {50, 50, 50};
  DLGN dlgn(data.size(1), 1, layers, 3, 5);
  dlgn->to(device);
  trainDLGN(dlgn, train_data, train_labels, 2048);

  auto train_preds = inference(dlgn, train_data, device).to(torch::kInt64).to(torch::kCPU);
  int train_correct = torch::sum(torch::eq(train_labels, train_preds)).item<int>();
  float train_accuracy = train_correct / (1.0 * train_labels.size(0));
  std::cout << "Out of " << train_labels.size(0) << " train data, the model has gotten " << train_correct << " correct. " <<std::endl;
  std::cout << "Train Data Percentage Accuracy is " << train_accuracy * 100 << std::endl;
  
  
  auto test_preds = inference(dlgn, test_data, device).to(torch::kInt64).to(torch::kCPU);
  int test_correct = torch::sum(torch::eq(test_labels, test_preds)).item<int>();
  float test_accuracy = test_correct / (1.0 * test_labels.size(0));
  std::cout << "Out of " << test_labels.size(0) << " test data, the model has gotten " << test_correct << " correct. " <<std::endl;
  std::cout << "Test Data Percentage Accuracy is " << test_accuracy * 100 << std::endl;
}