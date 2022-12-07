//
//  ann_dense.h
//  Core Lib
//
//  Created by Rasmus Anthin on 2022-12-01.
//

#pragma once
#include "utils.h"


namespace ml
{

  namespace ann
  {
  
    enum class BackPropType { Output, Hidden };
  
    class Neuron
    {
      const size_t Nw = 0;
      std::vector<Input> inputs;
      std::vector<float> weights;
      float bias = 0.f;
      float z = 0.f;
      PhiType phi_type;
      float phi_param_a = 1.f;
      float phi_param_k = 1.f;
      float phi_param_l = 1.1f;
      float y = 0.f;
      
      std::vector<float> w_diff_prev;
      float b_diff_prev = 0.f;
      
    public:
      Neuron(const std::vector<float>& w, float b, PhiType af_type)
        : Nw(w.size()), weights(w), bias(b), phi_type(af_type)
      {
        w_diff_prev.assign(Nw, 0);
      }
      
      void set_inputs(const std::vector<Input>& x)
      {
        assert(x.size() == Nw);
        inputs = x;
      }
      
      void set_phi_params(float a, float k, float l)
      {
        phi_param_a = a;
        phi_param_k = k;
        phi_param_l = l;
      }
      
      float update_forward()
      {
        std::vector<float> x, w;
        for (int i = 0; i < Nw; ++i)
        {
          auto ival = inputs[i].get();
          if (ival.has_value())
          {
            x.emplace_back(ival.value());
            w.emplace_back(weights[i]);
          }
        }
        z = stlutils::dot(x, w);
        z += bias;
        y = phi(z, phi_type, phi_param_a, phi_param_k, phi_param_l);
        return y;
      }
      
      // Back-prop
      // type : output or hidden. If output, then y_trg is the target output,
      //        if hidden, then it is the combined gradient from the next layer.
      // y_trg_or_grad : target output or combined gradient from behind the next layer.
      // eta : learning rate (0.1).
      // mu : momentum term (0.05).
      // r : random term for simulated annealing-ish behaviour (0).
      // diff = -eta * grad + mu * diff_prev + r
      // Returns gradient.
      std::vector<float> update_backward(BackPropType type, float y_trg_or_grad,
                                         float eta = 0.1f, float mu = 0.05f, float r = 0.f)
      {
        // dC/dw1 = dC/dy * dy/dz * dz/dw
        auto y_trg = y_trg_or_grad;
        auto grad_in = y_trg_or_grad;
        auto dC_dy = type == BackPropType::Output ? -(y_trg - y) : grad_in; // y_trg : dC_dy
        auto dy_dz = phi_diff(z, phi_type, phi_param_a, phi_param_k, phi_param_l);
        std::vector<float> dz_dw(Nw); // z = w0*x0 + w1*x1 + b => dz/dw0 = x0, dz/dw1 = x1, dz/db = 1.
        for (size_t i = 0; i < Nw; ++i)
          dz_dw[i] = inputs[i].get().value_or(0);
        auto dz_db = 1.f;
        auto dC_dz = dC_dy * dy_dz;
        auto dC_dw = stlutils::mult_scalar(dz_dw, dC_dz);
        auto dC_db = dC_dz * dz_db;
      
        auto w_diff = stlutils::mult_scalar(dC_dw, -eta);
        w_diff = stlutils::add(w_diff, stlutils::mult_scalar(w_diff_prev, mu));
        w_diff = stlutils::add_scalar(w_diff, r);
        auto b_diff = -eta * dC_db + mu * b_diff_prev + r;
      
        weights = stlutils::add(weights, w_diff);
        bias += b_diff;
      
        w_diff_prev = w_diff;
        b_diff_prev = b_diff;
      
        // Return dC/dz*w
        auto grad_out = stlutils::mult_scalar(weights, dC_dz);
        return grad_out;
      }
      
      // Forward-prop followed by a back-prop.
      // type : output or hidden. If output, then y_trg is the target output,
      //        if hidden, then it is the combined gradient from the next layer.
      // y_trg_or_grad : target output or combined gradient from behind the next layer.
      // eta : learning rate (0.1).
      // mu : momentum term (0.05).
      // r : random term for simulated annealing-ish behaviour (0).
      // diff = -eta * grad + mu * diff_prev + r
      // Returns gradient.
      std::vector<float> train(BackPropType type, float y_trg_or_grad,
                               float eta = 0.1f, float mu = 0.05f, float r = 0.f)
      {
        update_forward();
        return update_backward(type, y_trg_or_grad, eta, mu, r);
      }
    
      const float* output() const { return &y; }
    
      void print() const
      {
        // x0 = 1.4241414 | w0 = -0.233455  \
        // x1 = -0.444    | w1 = 2.333      |
        // x2 = 0.2       | w2 = -1.14      |---->[z = -0.1203]---->(Leaky_ReLU)---->[y = -0.01203]
        // x3 = 2.47      | w3 = 0.13444    |
        //                  b = 1.1455      /
      
        const auto nan = std::numeric_limits<float>::quiet_NaN();
      
        std::vector<std::string> wb_col(Nw + 1), x_col(Nw);
        for (size_t w_idx = 0; w_idx < Nw; ++w_idx)
        {
          wb_col[w_idx] = "w" + std::to_string(w_idx) + " = " + std::to_string(weights[w_idx]);
          x_col[w_idx] = "x" + std::to_string(w_idx) + " = " + std::to_string(inputs[w_idx].get().value_or(nan));
        }
        wb_col[Nw] = "b = " + std::to_string(bias);
        
        size_t wb_col_width = 0, x_col_width = 0;
        for (size_t w_idx = 0; w_idx < Nw; ++w_idx)
        {
          math::maximize(wb_col_width, wb_col[w_idx].size());
          math::maximize(x_col_width, x_col[w_idx].size());
        }
        math::maximize(wb_col_width, wb_col[Nw].size());
        
        for (auto& str : wb_col)
          str = str::adjust_str(str, str::Adjustment::Left, static_cast<int>(wb_col_width));
        for (auto& str : x_col)
          str = str::adjust_str(str, str::Adjustment::Left, static_cast<int>(x_col_width));
        
        std::vector<std::string> table(Nw + 1);
        for (size_t w_idx = 0; w_idx < Nw; ++w_idx)
        {
          table[w_idx] = x_col[w_idx] + " | " + wb_col[w_idx];
          table[w_idx] += " ";
          table[w_idx] += w_idx == 0ul ? "\\" : "|";
          if (w_idx == Nw/2)
          {
            table[w_idx] += "---->[z = " + std::to_string(z) + "]";
            table[w_idx] += "---->(";
            switch (phi_type)
            {
              case PhiType::BinaryStep: table[w_idx] += "BinaryStep"; break;
              case PhiType::Heaviside_BinaryStep: table[w_idx] += "Heaviside_BinaryStep"; break;
              case PhiType::Linear: table[w_idx] += "Linear"; break;
              case PhiType::Sigmoid: table[w_idx] += "Sigmoid"; break;
              case PhiType::Tanh: table[w_idx] += "Tanh"; break;
              case PhiType::Parametric_Tanh: table[w_idx] += "Parametric_Tanh"; break;
              case PhiType::ReLU: table[w_idx] += "ReLU"; break;
              case PhiType::Parametric_ReLU: table[w_idx] += "Parametric_ReLU"; break;
              case PhiType::Leaky_ReLU: table[w_idx] += "Leaky_ReLU"; break;
              case PhiType::Parametric_Leaky_ReLU: table[w_idx] += "Parametric_Leaky_ReLU"; break;
              case PhiType::ELU: table[w_idx] += "ELU"; break;
              case PhiType::Swish: table[w_idx] += "Swish"; break;
              case PhiType::GELU: table[w_idx] += "GELU"; break;
              case PhiType::SELU: table[w_idx] += "SELU"; break;
            }
            table[w_idx] += ")";
            table[w_idx] += "---->[y = " + std::to_string(y) + "]";
          }
        }
        table[Nw] = str::rep_char(' ', static_cast<int>(x_col_width)) + " | " + wb_col[Nw] + " /";
    
        std::cout << str::rep_char('-', static_cast<int>(table[0].size()) - 1) << '\n';
        for (const auto& str : table)
          std::cout << str << '\n';
        std::cout << str::rep_char('-', static_cast<int>(table[0].size()) - 1) << '\n';
      }
    };

    //

    class NeuralLayer
    {
      const size_t Ni = 0;
      const size_t No = 0;
      std::vector<std::unique_ptr<Neuron>> neurons;
    
    public:
      NeuralLayer(const std::vector<std::vector<float>>& w,
                  const std::vector<float>& b,
                  PhiType af_type)
        : No(w.size()), Ni(w[0].size())
      {
        for (size_t n_idx = 0; n_idx < No; ++n_idx)
          neurons.emplace_back(std::make_unique<Neuron>(w[n_idx], b[n_idx], af_type));
      }
    
      void set_inputs(const std::vector<Input>& x)
      {
        for (auto& n : neurons)
          n->set_inputs(x);
      }
    
      void set_phi_params(float a, float k, float l)
      {
        for (auto& n : neurons)
          n->set_phi_params(a, k, l);
      }
    
      void update_forward()
      {
        for (auto& n : neurons)
          n->update_forward();
      }
    
      // Back-prop
      // type : output or hidden. If output, then y_trg is the target output,
      //        if hidden, then it is the combined gradient from the next layer.
      // y_trg_or_grad : target output or combined gradient from behind the next layer.
      // eta : learning rate (0.1).
      // mu : momentum term (0.05).
      // r : random term for simulated annealing-ish behaviour (0).
      // diff = -eta * grad + mu * diff_prev + r
      // Returns gradient.
      std::vector<std::vector<float>> update_backward(BackPropType type,
                                                      const std::vector<float>& y_trg_or_grad,
                                                      float eta = 0.1f, float mu = 0.05f, float r = 0.f)
      {
        assert(y_trg_or_grad.size() == No);
        std::vector<std::vector<float>> grad(No);
        for (size_t n_idx = 0; n_idx < No; ++n_idx)
          grad[n_idx] = neurons[n_idx]->update_backward(type, y_trg_or_grad[n_idx], eta, mu, r);
        return grad;
      }
    
      // Forward-prop followed by a back-prop.
      // type : output or hidden. If output, then y_trg is the target output,
      //        if hidden, then it is the combined gradient from the next layer.
      // y_trg_or_grad : target output or combined gradient from behind the next layer.
      // eta : learning rate (0.1).
      // mu : momentum term (0.05).
      // r : random term for simulated annealing-ish behaviour (0).
      // diff = -eta * grad + mu * diff_prev + r
      // Returns gradient.
      std::vector<std::vector<float>> train(BackPropType type,
                                            const std::vector<float>& y_trg_or_grad,
                                            float eta = 0.1f, float mu = 0.05f, float r = 0.f)
      {
        update_forward();
        return update_backward(type, y_trg_or_grad, eta, mu, r);
      }
    
      const std::vector<Input> output() const
      {
        std::vector<Input> ret(No);
        for (size_t n_idx = 0; n_idx < No; ++n_idx)
          ret[n_idx] = neurons[n_idx]->output();
        return ret;
      }
      
      const std::vector<float> result() const
      {
        std::vector<float> ret(No);
        for (size_t n_idx = 0; n_idx < No; ++n_idx)
          ret[n_idx] = *neurons[n_idx]->output();
        return ret;
      }
    
      const Neuron* operator[](size_t n_idx) const
      {
        assert(n_idx < No);
        return neurons[n_idx].get();
      }
    
      size_t num_inputs() const { return Ni; }
      size_t num_outputs() const { return No; }
    
      void print_result() const
      {
        std::cout << "[ ";
        std::cout << *neurons.front()->output();
        for (size_t n_idx = 1; n_idx < No; ++n_idx)
          std::cout << ", " << *neurons[n_idx]->output();
        std::cout << " ]\n";
      }
    
      void print() const
      {
        for (const auto& n : neurons)
          n->print();
      }
    };
  
    //
  
    class RBFLayer
    {
      const size_t Ni = 0;
      const size_t No = 0;
      std::vector<std::vector<float>> weights;
      std::vector<float> biases;
      std::vector<Input> inputs;
      std::vector<float> outputs;
      
      std::vector<std::vector<float>> w_diff_prev;
      
    public:
      RBFLayer(const std::vector<std::vector<float>>& w,
               const std::vector<float>& b)
        : No(w.size()), Ni(w[0].size())
        , weights(w), biases(b)
        , outputs(No)
      {}
      
      void set_inputs(const std::vector<Input>& x)
      {
        inputs = x;
      }
      
      void update_forward()
      {
        const auto nan = std::numeric_limits<float>::quiet_NaN();
        for (size_t o_idx = 0; o_idx < No; ++o_idx)
        {
          auto& o = outputs[o_idx];
          o = 0.f;
          const auto& w_o = weights[o_idx];
          for (size_t i_idx = 0; i_idx < Ni; ++i_idx)
            o += math::sq(inputs[i_idx].get().value_or(nan) - w_o[i_idx]);
        }
      }
      
      // Back-prop
      // y_trg : target output.
      // eta : learning rate (0.1).
      // mu : momentum term (0.05).
      // r : random term for simulated annealing-ish behaviour (0).
      // diff = -eta * grad + mu * diff_prev + r
      // Returns gradient.
      std::vector<std::vector<float>> update_backward(const std::vector<float>& y_trg,
                                                      float eta = 0.1f, float mu = 0.05f, float r = 0.f)
      {
        const auto nan = std::numeric_limits<float>::quiet_NaN();
        assert(y_trg.size() == No);
        std::vector<std::vector<float>> grad(No);
        // dC/dwi = dC/df * df/dwi
        for (size_t o_idx = 0; o_idx < No; ++o_idx)
        {
          auto err_diff = -(y_trg[o_idx] - outputs[o_idx]);
          auto dC_df = err_diff;
          auto& w_o = weights[o_idx];
          std::vector<float> df_dw(Ni);
          for (size_t i_idx = 0; i_idx < Ni; ++i_idx)
            df_dw[i_idx] = 2*(inputs[i_idx].get().value_or(nan) - w_o[i_idx]);
          auto dC_dw = stlutils::mult_scalar(df_dw, dC_df);
          
          auto w_diff = stlutils::mult_scalar(dC_dw, -eta);
          w_diff = stlutils::add(w_diff, stlutils::mult_scalar(w_diff_prev[o_idx], mu));
          w_diff = stlutils::add_scalar(w_diff, r);
          w_o = stlutils::add(w_o, w_diff);
          w_diff_prev[o_idx] = w_diff;
          grad[o_idx] = dC_dw;
        }
        return grad;
      }
      
      // Forward-prop followed by a back-prop.
      // y_trg : target output.
      // eta : learning rate (0.1).
      // mu : momentum term (0.05).
      // r : random term for simulated annealing-ish behaviour (0).
      // diff = -eta * grad + mu * diff_prev + r
      // Returns gradient.
      std::vector<std::vector<float>> train(const std::vector<float>& y_trg,
                                            float eta = 0.1f, float mu = 0.05f, float r = 0.f)
      {
        update_forward();
        return update_backward(y_trg, eta, mu, r);
      }
      
      const std::vector<Input> output() const
      {
        std::vector<Input> ret(No);
        for (size_t o_idx = 0; o_idx < No; ++o_idx)
          ret[o_idx] = outputs[o_idx];
        return ret;
      }
      
      const std::vector<float> result() const
      {
        return outputs;
      }
    
      size_t num_inputs() const { return Ni; }
      size_t num_outputs() const { return No; }
    
      void print_result() const
      {
        std::cout << "[ ";
        std::cout << outputs.front();
        for (size_t o_idx = 1; o_idx < No; ++o_idx)
          std::cout << ", " << outputs[o_idx];
        std::cout << " ]\n";
      }
    };

    //

    class NeuralNetwork
    {
      const size_t Nl = 0;
      std::vector<std::unique_ptr<NeuralLayer>> layers;
    
    public:
      NeuralNetwork(const std::vector<std::vector<std::vector<float>>>& w,
                  const std::vector<std::vector<float>>& b,
                  const std::vector<PhiType>& af_type)
        : Nl(w.size())
      {
        for (size_t l_idx = 0; l_idx < Nl; ++l_idx)
          layers.emplace_back(std::make_unique<NeuralLayer>(w[l_idx], b[l_idx], af_type[l_idx]));
        for (size_t l_idx = 1; l_idx < Nl; ++l_idx)
          layers[l_idx]->set_inputs(layers[l_idx - 1]->output());
      }
    
      void set_inputs(const std::vector<Input>& x)
      {
        layers.front()->set_inputs(x);
      }
    
      void set_phi_params(float a, float k, float l)
      {
        for (auto& ll : layers)
          ll->set_phi_params(a, k, l);
      }
    
      void update_forward()
      {
        for (auto& l : layers)
          l->update_forward();
      }
    
      // Back-prop
      // type : output or hidden. If output, then y_trg is the target output,
      //        if hidden, then it is the combined gradient from the next layer.
      //        Normally set to Output.
      // y_trg_or_grad : target output or combined gradient from behind the next layer.
      // eta : learning rate (0.1).
      // mu : momentum term (0.05).
      // r : random term for simulated annealing-ish behaviour (0).
      // diff = -eta * grad + mu * diff_prev + r
      // Returns gradient.
      std::vector<std::vector<float>> update_backward(BackPropType type,
                                                      const std::vector<float>& y_trg_or_grad,
                                                      float eta = 0.1f, float mu = 0.05f, float r = 0.f)
      {
        size_t No = num_outputs();
        assert(y_trg_or_grad.size() == No);
        auto* l = layers.back().get();
        auto grad = l->update_backward(type, y_trg_or_grad, eta, mu, r);
        for (int l_idx = static_cast<int>(Nl) - 2; l_idx >= 0; --l_idx)
        {
          size_t Ni = l->num_inputs();
          size_t No = l->num_outputs();
          std::vector<float> grad_flat(Ni, 0);
          l = layers[l_idx].get();
          for (size_t o_idx = 0; o_idx < No; ++o_idx)
            for (size_t i_idx = 0; i_idx < Ni; ++i_idx)
              grad_flat[i_idx] += grad[o_idx][i_idx];
          grad = l->update_backward(BackPropType::Hidden, grad_flat, eta, mu, r);
        }
        return grad;
      }
    
      // Forward-prop followed by a back-prop.
      // type : output or hidden. If output, then y_trg is the target output,
      //        if hidden, then it is the combined gradient from the next layer.
      //        Normally set to Output.
      // y_trg_or_grad : target output or combined gradient from behind the next layer.
      // eta : learning rate (0.1).
      // mu : momentum term (0.05).
      // r : random term for simulated annealing-ish behaviour (0).
      // diff = -eta * grad + mu * diff_prev + r
      // Returns gradient.
      std::vector<std::vector<float>> train(BackPropType type,
                                            const std::vector<float>& y_trg_or_grad,
                                            float eta = 0.1f, float mu = 0.05f, float r = 0.f)
      {
        update_forward();
        return update_backward(type, y_trg_or_grad, eta, mu, r);
      }
    
      const std::vector<Input> output() const
      {
        return layers.back()->output();
      }
      
      const std::vector<float> result() const
      {
        return layers.back()->result();
      }
    
      const NeuralLayer* operator[](size_t l_idx) const
      {
        assert(l_idx < Nl);
        return layers[l_idx].get();
      }
    
      size_t num_inputs() const { return layers.front()->num_inputs(); }
      size_t num_outputs() const { return layers.back()->num_outputs(); }
    
      void print_result() const
      {
        layers.back()->print_result();
      }
    
      void print() const
      {
    
        for (size_t l_idx = 0; l_idx < Nl; ++l_idx)
        {
          const auto& l = layers[l_idx];
          std::cout << str::rep_char('=', 4) << " Layer " << l_idx << " " << str::rep_char('=', 4) << '\n';
          l->print();
        }
      }
    };
  
  }

}
