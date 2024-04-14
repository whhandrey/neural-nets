#pragma once
#include <Eigen/Dense>
#include "Activation.hpp"

namespace activation {
    namespace impl {
        class Sigmoid : public Activation {
        public:
            Eigen::VectorXf f(const Eigen::VectorXf& z) override {
                return 1.0f / (1.0f + Eigen::exp(-z.array()));
            }

            Eigen::VectorXf f_prime(const Eigen::VectorXf& z) override {
                return sigmoid_prime_impl(f(z));
            }

        private:
            Eigen::VectorXf sigmoid_prime_impl(Eigen::VectorXf s) {
                return s.array() * (1.0f - s.array());
            }
        };
    }

    Ptr Create(Type type) {
        if (type == Type::Sigmoid) {
            return std::make_unique<impl::Sigmoid>();
        }

        throw std::logic_error("Activation::Create(): Incorrect type has been passed as parameter");
    }
}
