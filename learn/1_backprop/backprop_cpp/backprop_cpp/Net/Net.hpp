#pragma once

#include <vector>
#include <tuple>
#include "../Data/Data.hpp"

namespace net {
    using VecVecXf = std::vector<Eigen::VectorXf>;
    using VecMatrixXf = std::vector<Eigen::MatrixXf>;

    using Ref = std::reference_wrapper<const data::PairXY>;
    using VecXfRef = std::reference_wrapper<const Eigen::VectorXf>;
    using VecVecXfRef = std::vector<VecXfRef>;

    class Net {
    public:
        Net(const std::vector<int>& sizes);

    public:
        Eigen::VectorXf FeedForward(const Eigen::VectorXf& x);
        void StochasticGradDesc(data::Set& input, int epochs, int mini_batch_size, float learning_rate);

    private:
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

        VecVecXf Activations(const Ref& ref);
        std::vector<VecVecXf> ActivationsBatch(const std::vector<Ref>& input);

        std::vector<float> LossBatch(const std::vector<Ref>& input, const std::vector<VecVecXfRef>& activations);

        grad_pair BackProp(const Eigen::VectorXf& y, const std::vector<VecXfRef>& activations);
        grad_pair BackPropBatch(const std::vector<Ref>& input, const std::vector<VecVecXfRef>& activations, float learning_rate);

        grad_pair SumGrads(grad_pair&& batch_grads, grad_pair grads);

        // batch_grads expected to be moved here
        void GradDescent(grad_pair batch_grads);

        int Evaluate(const std::vector<data::PairXY>& test_data);

        void PrintEvalResult(int num_matches, int y_count, int epoch);

    private:
        const int m_numLayers;

        std::vector<Eigen::MatrixXf> m_weights;
        std::vector<Eigen::VectorXf> m_biases;
    };
}
