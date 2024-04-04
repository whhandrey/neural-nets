#pragma once
#include <vector>
#include <Eigen/Dense>

namespace data {
    using PairXY = std::pair<Eigen::VectorXf, Eigen::VectorXf>;

    struct Set {
        std::vector<PairXY> m_training_data;
        std::vector<PairXY> m_validation_data;
        std::vector<PairXY> m_test_data;
    };
}
