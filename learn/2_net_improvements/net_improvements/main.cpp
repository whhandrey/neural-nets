#include "Net/Net.hpp"
#include "Data/Loader/Loader.hpp"
#include <iostream>

int main() {
    auto net = net::Builder()
        .AddInputLayer(784)
        .AddLayer(30, activation::Type::Sigmoid)
        .AddLayer(10, activation::Type::Sigmoid)
        .AddCost(cost::Type::MSE)
        .Build();

    const int num_epochs = 30;
    const int mini_batch_size = 10;
    const float learning_rate = 3.0f;

    auto set = data::loader::Load();

    net->StochasticGradDesc(set, num_epochs, mini_batch_size, learning_rate);
    std::cout << "\nDone" << std::endl;
}
