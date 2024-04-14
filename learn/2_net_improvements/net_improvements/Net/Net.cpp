#include "Net.hpp"
#include "Activation.hpp"
#include "Cost.hpp"
#include <random>
#include <numeric>
#include <iostream>

namespace net {
    using VecVecXf = std::vector<Eigen::VectorXf>;
    using VecMatrixXf = std::vector<Eigen::MatrixXf>;

    using Ref = std::reference_wrapper<const data::PairXY>;
    using VecXfRef = std::reference_wrapper<const Eigen::VectorXf>;
    using VecVecXfRef = std::vector<VecXfRef>;

    struct grad_pair {
        VecMatrixXf m_dc_dw;
        VecVecXf m_dc_db;

        bool empty() const {
            return m_dc_dw.empty() && m_dc_db.empty();
        }

        int size() {
            return int(m_dc_dw.size());
        }
    };

    template <class Cont>
    void ShuffleInput(Cont& container) {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::shuffle(container.begin(), container.end(), gen);
    }

    std::vector<Ref> MiniBatch(const std::vector<data::PairXY>& input, int mini_batch_index, int mini_batch_size) {
        const auto beginIt = std::next(input.begin(), mini_batch_index * mini_batch_size);
        const auto endIt = std::min(input.end(), beginIt + mini_batch_size);

        return { beginIt, endIt };
    }

    std::vector<VecXfRef> AllActivations(const Eigen::VectorXf& input, const VecVecXf& activations) {
        std::vector<VecXfRef> output;
        output.reserve(activations.size() + 1);

        output.push_back(std::cref(input));
        output.insert(output.end(), activations.begin(), activations.end());

        return output;
    }

    std::vector<VecVecXfRef> AllActivations(const std::vector<Ref>& input, const std::vector<VecVecXf>& activations) {
        std::vector<VecVecXfRef> output;
        output.reserve(input.size());

        std::transform(input.begin(), input.end(), activations.begin(), std::back_inserter(output), [](const auto& data, const auto& act) {
            return AllActivations(data.get().first, act);
        });

        return output;
    }

    grad_pair SumGrads(grad_pair&& batch_grads, grad_pair grads) {
        if (batch_grads.empty()) {
            return grads;
        }

        for (int i = 0; i < int(batch_grads.size()); ++i) {
            batch_grads.m_dc_dw[i] += grads.m_dc_dw[i];
            batch_grads.m_dc_db[i] += grads.m_dc_db[i];
        }

        return batch_grads;
    }
}

namespace init {
    void FillData(float* data, int size) {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Define a standard Gaussian distribution with mean = 0 and std.dev. = 1
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
    }

    Eigen::MatrixXf GenerateMatrix(int rows, int cols) {
        Eigen::MatrixXf mtx(rows, cols);

        FillData(mtx.data(), rows * cols);
        return mtx;
    }

    Eigen::VectorXf GenerateVector(int num) {
        Eigen::VectorXf vec(num);

        FillData(vec.data(), num);
        return vec;
    }

    decltype(auto) DefaultWeightsAndBiases(const std::vector<int>& sizes) {
        std::vector<Eigen::MatrixXf> weights;
        weights.reserve(sizes.size() - 1);

        std::vector<Eigen::VectorXf> biases;
        biases.reserve(sizes.size() - 1);

        for (size_t i = 0; i < sizes.size() - 1; ++i) {
            weights.push_back(init::GenerateMatrix(sizes[i + 1], sizes[i]));
            biases.push_back(init::GenerateVector(sizes[i + 1]));
        }

        return std::make_tuple(weights, biases);
    }
}

namespace net {
    class NetImpl : public Net {
    public:
        NetImpl(VecMatrixXf&& weights, VecVecXf&& biases, std::vector<activation::Ptr>&& acts, cost::Ptr cost);

        Eigen::VectorXf FeedForward(Eigen::VectorXf x) override;
        void StochasticGradDesc(data::Set& input, int epochs, int mini_batch_size, float learning_rate) override;

    private:
        VecVecXf Activations(const Ref& ref);
        std::vector<VecVecXf> ActivationsBatch(const std::vector<Ref>& input);

        std::vector<float> LossBatch(const std::vector<Ref>& input, const std::vector<VecVecXfRef>& activations);

        grad_pair BackProp(const Eigen::VectorXf& y, const std::vector<VecXfRef>& activations);
        grad_pair BackPropBatch(const std::vector<Ref>& input, const std::vector<VecVecXfRef>& activations, float learning_rate);

        // batch_grads expected to be moved here
        void GradDescent(grad_pair batch_grads);

        int Evaluate(const std::vector<data::PairXY>& test_data);
        void PrintEvalResult(int num_matches, int y_count, int epoch);

    private:
        const int m_numLayers;
        cost::Ptr m_cost;

        std::vector<Eigen::MatrixXf> m_weights;
        std::vector<Eigen::VectorXf> m_biases;
        std::vector<activation::Ptr> m_activations;
    };

    NetImpl::NetImpl(VecMatrixXf&& weights, VecVecXf&& biases, std::vector<activation::Ptr>&& acts, cost::Ptr cost)
        : m_numLayers(int(acts.size()) + 1)
        , m_weights{ std::move(weights) }
        , m_biases{ std::move(biases) }
        , m_activations{ std::move(acts) }
        , m_cost{ std::move(cost) }
    {
    }

    Eigen::VectorXf NetImpl::FeedForward(Eigen::VectorXf x) {

        for (int i = 0; i < m_numLayers - 1; ++i) {
            x = m_activations[i]->f(m_weights[i] * x + m_biases[i]);
        }

        return x;
    }

    void NetImpl::StochasticGradDesc(data::Set& input, int epochs, int mini_batch_size, float learning_rate) {

        auto& training_data = input.m_training_data;
        for (int i = 0; i < epochs; ++i) {
            net::ShuffleInput(training_data);

            int mini_batches = int(training_data.size()) / mini_batch_size;
            for (int j = 0; j < mini_batches; ++j) {
                const auto mini_batch = net::MiniBatch(training_data, j, mini_batch_size);

                const auto activations = ActivationsBatch(mini_batch);
                const auto activations_ref = AllActivations(mini_batch, activations);

                //const auto loss_batch = LossBatch(mini_batch, activations_ref);

                GradDescent(BackPropBatch(mini_batch, activations_ref, learning_rate));
            }

            if (!input.m_test_data.empty()) {
                PrintEvalResult(Evaluate(input.m_test_data), int(input.m_test_data.size()), i);
            }
            else {
                std::cout << "Epoch: " << i << " completed" << std::endl;
            }
        }
    }

    VecVecXf NetImpl::Activations(const Ref& ref) {
        VecVecXf output;
        output.reserve(m_numLayers - 1);

        VecXfRef x = ref.get().first;
        for (int i = 0; i < m_numLayers - 1; ++i) {
            output.push_back(m_activations[i]->f(m_weights[i] * x.get() + m_biases[i]));
            x = output.back();
        }

        return output;
    }

    std::vector<VecVecXf> NetImpl::ActivationsBatch(const std::vector<Ref>& input) {
        std::vector<VecVecXf> output;
        output.reserve(input.size());

        for (const auto& data : input) {
            output.push_back(Activations(data));
        }

        return output;
    }

    std::vector<float> NetImpl::LossBatch(const std::vector<Ref>& input, const std::vector<VecVecXfRef>& activations) {
        std::vector<float> loss_batch;
        loss_batch.reserve(input.size());

        for (int i = 0; i < int(input.size()); ++i) {
            loss_batch.push_back(m_cost->loss(input[i].get().second, activations[i].back()));
        }

        return loss_batch;
    }

    grad_pair NetImpl::BackProp(const Eigen::VectorXf& y, const std::vector<VecXfRef>& activations) {

        Eigen::VectorXf dc_da = m_cost->dc_da(y, activations.back().get());

        std::vector<Eigen::MatrixXf> dc_dw_vec;
        dc_dw_vec.reserve(activations.size() - 1);

        std::vector<Eigen::VectorXf> dc_db_vec;
        dc_db_vec.reserve(activations.size() - 1);

        for (int layer = m_numLayers - 2; layer >= 0; --layer) {
            const auto z = m_weights[layer] * activations[layer].get() + m_biases[layer];

            const auto da_dz = m_activations[layer]->f_prime(z);
            const auto dc_dz = Eigen::VectorXf{ dc_da.array() * da_dz.array() };

            dc_db_vec.push_back(dc_dz);

            const auto& a_prev = activations[layer];
            dc_dw_vec.push_back(dc_dz * a_prev.get().transpose());

            dc_da = m_weights[layer].transpose() * dc_dz;
        }

        return { dc_dw_vec, dc_db_vec };
    }

    grad_pair NetImpl::BackPropBatch(const std::vector<Ref>& input, const std::vector<VecVecXfRef>& activations, float learning_rate) {
        grad_pair batch_grads;

        for (int i = 0; i < int(input.size()); ++i) {
            batch_grads = SumGrads(std::move(batch_grads), BackProp(input[i].get().second, activations[i]));
        }

        float ratio = learning_rate / float(input.size());

        for (int i = 0; i < int(batch_grads.size()); ++i) {
            batch_grads.m_dc_dw[i] *= ratio;
            batch_grads.m_dc_db[i] *= ratio;
        }

        return batch_grads;
    }

    void NetImpl::GradDescent(grad_pair batch_grads) {
        int last_index = m_numLayers - 2;

        for (int i = 0; i < m_numLayers - 1; ++i) {
            m_weights[i] -= batch_grads.m_dc_dw[last_index];
            m_biases[i] -= batch_grads.m_dc_db[last_index];

            --last_index;
        }
    }

    int NetImpl::Evaluate(const std::vector<data::PairXY>& test_data) {
        std::vector<Eigen::VectorXf> output;
        output.reserve(test_data.size());

        std::transform(test_data.begin(), test_data.end(), std::back_inserter(output), [this](const auto& data) {
            return FeedForward(data.first);
        });

        return std::inner_product(test_data.begin(), test_data.end(), output.begin(), 0, std::plus<>(), [](const auto& test, const auto& out) {
            Eigen::Index max_y_idx;
            test.second.maxCoeff(&max_y_idx);

            Eigen::Index max_out_idx;
            out.maxCoeff(&max_out_idx);

            return int(max_y_idx == max_out_idx);
        });
    }

    void NetImpl::PrintEvalResult(int num_matches, int y_count, int epoch) {
        std::cout << "Epoch: " << epoch << " : " << "num_matches: " << num_matches << '/' << y_count << std::endl;
    }

    Builder& Builder::AddInputLayer(int num_neurons) {
        m_sizes.push_back(num_neurons);
        return *this;
    }

    Builder& Builder::AddLayer(int num_neurons, activation::Type type) {
        m_sizes.push_back(num_neurons);
        m_activations.push_back(activation::Create(type));

        return *this;
    }

    Builder& Builder::AddCost(cost::Type type) {
        m_cost = cost::Create(type);
        return *this;
    }

    std::unique_ptr<Net> Builder::Build() {
        auto [weights, biases] = init::DefaultWeightsAndBiases(m_sizes);

        return std::make_unique<NetImpl>(
            std::move(weights),
            std::move(biases),
            std::move(m_activations),
            std::move(m_cost)
        );
    }
}
