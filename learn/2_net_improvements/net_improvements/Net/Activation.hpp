#pragma once
#include <Eigen/Dense>

namespace activation {
    Eigen::VectorXf sigmoid(const Eigen::VectorXf& z);
    Eigen::VectorXf sigmoid_prime(const Eigen::VectorXf& z);
}
