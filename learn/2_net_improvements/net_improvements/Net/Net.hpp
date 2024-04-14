#pragma once
#include "Data/Data.hpp"
#include "Cost.hpp"
#include "Activation.hpp"

namespace net {
    class Net {
    public:
        virtual ~Net() = default;
        virtual Eigen::VectorXf FeedForward(Eigen::VectorXf x) = 0;
        virtual void StochasticGradDesc(data::Set& input, int epochs, int mini_batch_size, float learning_rate) = 0;
    };

    class Builder {
    public:
        Builder& AddInputLayer(int num_neurons);
        Builder& AddLayer(int num_neurons, activation::Type type);
        Builder& AddCost(cost::Type type);

        std::unique_ptr<Net> Build();

    private:
        std::vector<int> m_sizes;
        std::vector<activation::Ptr> m_activations;
        cost::Ptr m_cost;
    };
}
