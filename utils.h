//
//  MachineLearning.h
//  Core Lib
//
//  Created by Rasmus Anthin on 2022-11-28.
//

#pragma once
#include "../StlUtils.h"
#include "../Rand.h"
#include "../StringHelper.h"
#include <optional>
#include <memory>


namespace ml
{

  namespace ann
  {
  
    enum class PhiType
    {
      BinaryStep,
      Heaviside_BinaryStep,
      Linear,
      Sigmoid,
      Tanh,
      Parametric_Tanh,
      ReLU,
      Parametric_ReLU,
      Leaky_ReLU,
      Parametric_Leaky_ReLU,
      ELU,
      Swish,
      GELU,
      SELU,
    };
  
    float phi(float z, PhiType type, float a = 1.f, float k = 1.f, float l = 1.1f)
    {
      switch (type)
      {
        case PhiType::BinaryStep: return z < 0 ? 0 : 1;
        case PhiType::Heaviside_BinaryStep: return z <= 0 ? 0 : 1;
        case PhiType::Linear: return z;
        case PhiType::Sigmoid: return 1./(1 + std::exp(-z));
        case PhiType::Tanh: return std::tanh(z);
        case PhiType::Parametric_Tanh: return a*std::tanh(k*z);
        case PhiType::ReLU: return std::max(0.f, z);
        case PhiType::Parametric_ReLU: return std::max(0.f, k*z + l);
        case PhiType::Leaky_ReLU: return std::max(0.1f*z, z);
        case PhiType::Parametric_Leaky_ReLU: return std::max(a*(k*z + l), k*z + l);
        case PhiType::ELU: return z < 0 ? a*(std::exp(z) - 1) : z;
        case PhiType::Swish: return z*phi(z, PhiType::Sigmoid);
        //case PhiType::GELU: return 0.5*z*(1 + std::tanh(M_2_SQRTPI*M_SQRT1_2*(z + 0.044715*math::cube(z))));
        case PhiType::GELU: return 0.5*z*(1 + std::erf(z/M_SQRT2));
        case PhiType::SELU: return l*phi(z, PhiType::ELU, a, k, l);
      }
    }
  
    float phi_diff(float z, PhiType type, float a = 1.f, float k = 1.f, float l = 1.1f)
    {
      switch (type)
      {
        case PhiType::BinaryStep: return 0.f; // Actually inf at 0- and 0 everywhere else.
        case PhiType::Heaviside_BinaryStep: return 0.f; // Actually inf at 0+ and 0 everywhere else.
        case PhiType::Linear: return 1.f;
        case PhiType::Sigmoid:
        {
          auto s = phi(z, type, a, k, l);
          return s * (1 - s);
        }
        case PhiType::Tanh:
        {
          auto th = phi(z, type, a, k, l);
          return 1 - math::sq(th);
        }
        case PhiType::Parametric_Tanh: return a*k/math::sq(std::cosh(k*z));
        case PhiType::ReLU: return z < 0 ? 0 : 1;
        case PhiType::Parametric_ReLU: return z < -l/k ? 0 : k;
        case PhiType::Leaky_ReLU: return z < 0 ? 0.1 : 1;
        case PhiType::Parametric_Leaky_ReLU: return z < -l/k ? a*k : k;
        case PhiType::ELU: return z < 0 ? phi(z, type, a, k, l) + a : 1;
        case PhiType::Swish:
        {
          auto sw = phi(z, type, a, k, l);
          auto sig = phi(z, PhiType::Sigmoid, a, k, l);
          return sw + sig * (1 - sw);
        }
        case PhiType::GELU:
        {
          //auto z3 = math::cube(z);
          //auto sech = [](float v) { return std::sqrt(1 - math::sq(std::tanh(v))); };
          //auto b = 0.797885f*z + 0.0356774f*z3;
          //return 0.5f + (0.398942f*z + 0.0535161f*z3)*math::sq(sech(b)) + 0.5f*std::tanh(b);
          
          // 1/2*(erf(z/sqrt(2)) + 1) + exp(-z^2/2)*z/(sqrt(2*pi))
          static const auto c_1_sqrt_2pi = M_2_SQRTPI/M_SQRT2;
          return 0.5f*(1 + std::erf(z/M_SQRT2)) + (std::exp(-math::sq(z)*0.5f)*z)*c_1_sqrt_2pi;
        }
        case PhiType::SELU: return l*phi_diff(z, PhiType::ELU, a, k, l);
      }
    }
  
    // softmax() behaves like softmax when w > 0 and softmin when w < 0.
    template<typename Cont>
    Cont softmax(const Cont& c, float w = 1)
    {
      Cont ret = c;
      Cont ec = c;
      for (auto& v : ec)
        v = std::exp(w * v);
      auto ecs = stlutils::sum(ec);
      auto N = c.size();
      for (size_t i = 0; i < N; ++i)
        ret[i] = ec[i]/ecs;
      return ret;
    }
  
    // softmin() behaves like softmin when w > 0 and softmax when w < 0.
    template<typename Cont>
    Cont softmin(const Cont& c, float w = 1) { return softmax(c, -w); }
  
    template<typename Cont>
    Cont hardmax(const Cont& c)
    {
      Cont ret = c;
      std::fill(ret.begin(), ret.end(), 0);
      for (auto idx : stlutils::argmax(c))
        ret[idx] = 1;
      return ret;
    }
  
    template<typename Cont>
    Cont hardmin(const Cont& c)
    {
      Cont ret = c;
      std::fill(ret.begin(), ret.end(), 0);
      for (auto idx : stlutils::argmin(c))
        ret[idx] = 1;
      return ret;
    }
  
    class Input
    {
      float perceptron_signal = 0.f;
      const float* dendrite_output = nullptr;
      bool set = false;
      
    public:
      Input() = default;
      Input(const float signal) : perceptron_signal(signal), set(true) {}
      Input(const float* output) : dendrite_output(output), set(true) {}
      
      std::optional<float> get() const
      {
        if (!set)
          return {};
        if (dendrite_output != nullptr)
          return *dendrite_output;
        return perceptron_signal;
      }
    };
  
    //
  
    enum class GenWeightsType { Zero, MinusOne, One, Two, Rnd_0_p1, Rndn_0_p1, Rnd_m1_p1, Rndn_m1_p1 };
    float generate_weight(GenWeightsType type)
    {
      switch (type)
      {
        case GenWeightsType::Zero: return 0;
        case GenWeightsType::MinusOne: return -1;
        case GenWeightsType::One: return 1;
        case GenWeightsType::Two: return 2;
        case GenWeightsType::Rnd_0_p1: return rnd::rand_float(0, 1);
        case GenWeightsType::Rndn_0_p1: return rnd::randn_clamp(0.5, 1, 0, 1);
        case GenWeightsType::Rnd_m1_p1: return rnd::rand_float(-1, 1);
        case GenWeightsType::Rndn_m1_p1: return rnd::randn_clamp(0, 2, -1, 1);
      }
    }
  
    std::vector<std::vector<float>> generate_weights(GenWeightsType type, size_t Ni, size_t No)
    {
      std::vector<std::vector<float>> w(No);
      for (auto& w_o : w)
      {
        for (size_t i_idx = 0; i_idx < Ni; ++i_idx)
          w_o.emplace_back(generate_weight(type));
      }
      return w;
    }
  
    std::vector<float> generate_weights(GenWeightsType type, size_t No)
    {
      std::vector<float> b(No);
      for (auto& b_o : b)
        b_o = generate_weight(type);
      return b;
    }
  
  }

}
