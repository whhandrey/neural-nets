#pragma once
#include <Eigen/Dense>
#include "Activation.hpp"

namespace activation {
    Eigen::VectorXf sigmoid(const Eigen::VectorXf& z) {
        return 1.0f / (1.0f + Eigen::exp(-z.array()));
    }

    Eigen::VectorXf sigmoid_prime_impl(Eigen::VectorXf s) {
        return s.array() * (1.0f - s.array());
    }

    Eigen::VectorXf sigmoid_prime(const Eigen::VectorXf& z) {
        return sigmoid_prime_impl(sigmoid(z));
    }
}
