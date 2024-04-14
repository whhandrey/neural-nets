#pragma once
#include <Eigen/Dense>

namespace activation {
    enum class Type {
        Sigmoid
    };

    class Activation {
    public:
        virtual ~Activation() = default;

        virtual Eigen::VectorXf f(const Eigen::VectorXf& z) = 0;
        virtual Eigen::VectorXf f_prime(const Eigen::VectorXf& z) = 0;
    };

    using Ptr = std::unique_ptr<Activation>;

    Ptr Create(Type type);
}
