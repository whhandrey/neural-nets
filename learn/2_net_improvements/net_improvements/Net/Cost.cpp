#pragma once
#include "Cost.hpp"

namespace cost {
    namespace impl {
        class MSE : public Cost {
            float loss(const Eigen::VectorXf& y, const Eigen::VectorXf& a) override {
                return (y - a).squaredNorm() / y.size();
            }

            Eigen::VectorXf dc_da(const Eigen::VectorXf& y, const Eigen::VectorXf& a) override {
                return a - y;
            }
        };

        class CrossEntropy : public Cost {
        public:
            float loss(const Eigen::VectorXf& y, const Eigen::VectorXf& a) override {
                return 0.0f;
            }

            Eigen::VectorXf dc_da(const Eigen::VectorXf& y, const Eigen::VectorXf& a) override {
                return {};
            }
        };
    }

    Ptr Create(Type type) {
        if (type == Type::MSE) {
            return std::make_unique<impl::MSE>();
        }

        if (type == Type::CrossEntropy) {
            return std::make_unique<impl::CrossEntropy>();
        }

        throw std::logic_error("Cost::Create(): Incorrect type has been passed as parameter");
    }
}
