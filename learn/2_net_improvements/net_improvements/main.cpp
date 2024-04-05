#include "Net/Net.hpp"
#include "Data/Loader/Loader.hpp"
#include <iostream>

int main() {
    const int num_epochs = 30;
    const int mini_batch_size = 10;
    const float learning_rate = 3.0f;

    auto set = data::loader::Load();

    net::Net net({ 784, 30, 10 });
    net.StochasticGradDesc(set, num_epochs, mini_batch_size, learning_rate);

    std::cout << "\nDone" << std::endl;
}
