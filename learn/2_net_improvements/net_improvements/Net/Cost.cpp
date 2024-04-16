#pragma once
#include "Cost.hpp"

namespace cost {
    namespace impl {
        constexpr float eps = 1e-10f;

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
                const auto first_term = y.array() * Eigen::log(a.array() + eps);
                const auto second_term = (1.0f - y.array()) * Eigen::log( (1.0f - a.array()) + eps );

                return -(first_term + second_term).mean();
            }

            Eigen::VectorXf dc_da(const Eigen::VectorXf& y, const Eigen::VectorXf& a) override {
                return (a - y).array() / ( a.array() * (1.0f - a.array()) + eps );
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
