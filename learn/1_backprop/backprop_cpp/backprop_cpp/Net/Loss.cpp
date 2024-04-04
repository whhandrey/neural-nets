#pragma once
#include "Loss.hpp"

namespace loss {
    float mse(const Eigen::VectorXf& y, const Eigen::VectorXf& a) {
        return (y - a).squaredNorm() / y.size();
    }
}
