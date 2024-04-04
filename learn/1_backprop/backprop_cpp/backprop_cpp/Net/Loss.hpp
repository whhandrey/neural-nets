#pragma once
#include <Eigen/Dense>

namespace loss {
    float mse(const Eigen::VectorXf& y, const Eigen::VectorXf& a);
}
