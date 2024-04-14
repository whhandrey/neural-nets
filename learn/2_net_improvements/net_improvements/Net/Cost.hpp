#pragma once
#include <Eigen/Dense>

namespace cost {
    enum class Type {
        MSE,
        CrossEntropy
    };

    class Cost {
    public:
        virtual ~Cost() = default;
        virtual float loss(const Eigen::VectorXf& y, const Eigen::VectorXf& a) = 0;
        virtual Eigen::VectorXf dc_da(const Eigen::VectorXf& y, const Eigen::VectorXf& a) = 0;
    };

    using Ptr = std::unique_ptr<Cost>;

    Ptr Create(Type type);
}
