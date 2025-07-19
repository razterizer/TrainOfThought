//
//  LeNet.h
//  Core Lib
//
//  Created by Rasmus Anthin on 2022-12-03.
//

#pragma once
#include "MNIST/MNIST_Loader.h"


namespace ml
{

  namespace ann
  {
  
    class LeNet5
    {
      std::vector<std::vector<float>> input;
      
      std::vector<std::vector<std::vector<float>>> W_kernel_C1; // 5x5
      std::vector<std::vector<std::vector<float>>> W_kernel_C3; // 5x5
      std::vector<std::vector<std::vector<float>>> W_kernel_C5; // 5x5
      std::vector<float> bias_C1;
      std::vector<float> bias_C3;
      std::vector<float> bias_C5;
      std::vector<float> W_S2;
      std::vector<float> W_S4;
      std::vector<float> bias_S2;
      std::vector<float> bias_S4;
      
      std::array<std::array<bool, 6>, 16> map_C3_to_S2;
      
      std::vector<std::vector<std::vector<float>>> feature_maps_C1;
      std::vector<std::vector<std::vector<float>>> feature_maps_S2_raw;
      std::vector<std::vector<std::vector<float>>> feature_maps_S2;
      std::vector<std::vector<std::vector<float>>> feature_maps_C3;
      std::vector<std::vector<std::vector<float>>> feature_maps_S4_raw;
      std::vector<std::vector<std::vector<float>>> feature_maps_S4;
      std::vector<std::vector<std::vector<float>>> feature_maps_C5;
      NeuralNetwork dense_network;
      std::array<float, 10> output;
      
      const float phi_a = 0;
      
      std::vector<std::vector<std::vector<float>>> W_diff_C5_prev;
      std::vector<float> b_diff_C5_prev;
      std::vector<float> W_diff_S4_prev;
      std::vector<float> b_diff_S4_prev;
      std::vector<std::vector<std::vector<float>>> W_diff_C3_prev;
      std::vector<float> b_diff_C3_prev;
      
    public:
      LeNet5()
        : dense_network
          {
            {
              // Ni*No = 120x84
              generate_weights(GenWeightsType::Rndn_m1_p1, 120, 84),
              // Ni*No = 84x10
              generate_weights(GenWeightsType::Rndn_m1_p1, 84, 10),
            },
            { // No = 84, 10
              generate_weights(GenWeightsType::Rndn_m1_p1, 84),
              generate_weights(GenWeightsType::Rndn_m1_p1, 10),
            },
            { PhiType::Parametric_Tanh, PhiType::Linear }
          }
        , phi_a { 1.7159f }
      {
        // C1 : 6*(5*5 + 1) = 156 trainable params.
        // S2 : 6*(1 + 1) = 12 trainable params.
        // C3 : 60*5*5 + 16*1 = 1'516 trainable params.
        // S4 : 16*(1 + 1) = 32 trainable params.
        // C5 : 120*(16*5*5 + 1) = 120*400 = 48'120 trainable params.
        // F6 : 84*(120 + 1) = 10'164 trainable params.
        // O  : 10*84 = 840 trainable params. RBFN if dealing with ASCII and not just numbers.
        // Tot : 60'840 trainable params.
        
        for (size_t l_idx = 0; l_idx < 6; ++l_idx)
          W_kernel_C1.emplace_back(generate_weights(GenWeightsType::Rndn_m1_p1, 5, 5));
        bias_C1 = generate_weights(GenWeightsType::Rndn_m1_p1, 6);
        for (size_t l_idx = 0; l_idx < 60; ++l_idx)
          W_kernel_C3.emplace_back(generate_weights(GenWeightsType::Rndn_m1_p1, 5, 5));
        bias_C3 = generate_weights(GenWeightsType::Rndn_m1_p1, 16);
        size_t Nc5wl = 120*16;
        for (size_t l_idx = 0; l_idx < Nc5wl; ++l_idx)
          W_kernel_C5.emplace_back(generate_weights(GenWeightsType::Rndn_m1_p1, 5, 5));
        bias_C5 = generate_weights(GenWeightsType::Rndn_m1_p1, 120);
        W_S2 = generate_weights(GenWeightsType::Rndn_m1_p1, 6);
        W_S4 = generate_weights(GenWeightsType::Rndn_m1_p1, 16);
        bias_S2 = generate_weights(GenWeightsType::Rndn_m1_p1, 6);
        bias_S4 = generate_weights(GenWeightsType::Rndn_m1_p1, 16);
        
        for (int i = 0; i < 16; ++i)
          for (int j = 0; j < 6; ++j)
            map_C3_to_S2[i][j] = 0;
        for (int i = 0; i <= 5; ++i)
          for (int j = 0; j < 3; ++j)
            map_C3_to_S2[i][(i+j)%6] = true;
        for (int i = 6; i <= 11; ++i)
          for (int j = 0; j < 4; ++j)
            map_C3_to_S2[i][(i+j)%6] = true;
        for (int i = 12; i <= 14; ++i)
          for (int j = 0; j < 5; ++j)
            map_C3_to_S2[i][(i+j)%6] = j != 2;
        for (int j = 0; j < 6; ++j)
          map_C3_to_S2[15][j] = true;
        
        stlutils::resize(feature_maps_C1, 6, 28, 28);
        stlutils::resize(feature_maps_S2_raw, 6, 14, 14);
        stlutils::resize(feature_maps_S2, 6, 14, 14);
        stlutils::resize(feature_maps_C3, 16, 10, 10);
        stlutils::resize(feature_maps_S4_raw, 16, 5, 5);
        stlutils::resize(feature_maps_S4, 16, 5, 5);
        stlutils::resize(feature_maps_C5, 120, 1, 1);
        
        dense_network.set_phi_params(phi_a, 1.f, 1.1f);
        
        //
        
        stlutils::resize(W_diff_C5_prev, Nc5wl, 5, 5); // 120*16*(5x5)
        b_diff_C5_prev.resize(Nc5wl);
        W_diff_S4_prev.resize(16);
        b_diff_S4_prev.resize(16);
        stlutils::resize(W_diff_C3_prev, 60, 5, 5);
        b_diff_C3_prev.resize(60);
      }
      
      void set_input(const std::vector<std::vector<unsigned char>>& x)
      {
        assert(x.size() == 32);
        assert(x.back().size() == 32);
        stlutils::resize(input, 32, 32);
        for (size_t r_idx = 0; r_idx < 32; ++r_idx)
          for (size_t c_idx = 0; c_idx < 32; ++c_idx)
            input[r_idx][c_idx] = math::value_to_param(x[r_idx][c_idx]/255.f, -0.1f, 1.175f);
      }
      
      void update_forward()
      {
        for (size_t l_idx = 0; l_idx < 6; ++l_idx)
        {
          auto& kernel = W_kernel_C1[l_idx];
          auto bias = bias_C1[l_idx];
          feature_maps_C1[l_idx] = conv_2d(input, kernel, bias, ConvRange::Valid, ConvType::Correlation);
        }
        for (size_t l_idx = 0; l_idx < 6; ++l_idx)
        {
          // a b
          // c d
          // "(a + b + c + d)*trainable_coeff + bias"
          //
          auto& fm_l_raw = feature_maps_S2_raw[l_idx];
          auto& fm_l = feature_maps_S2[l_idx];
          fm_l_raw = pooling_2d(feature_maps_C1[l_idx], PoolingType::Sum, { 2, 2 }, { 2, 2 });
          fm_l = fm_l_raw;
          fm_l = stlutils::mult_scalar(fm_l, W_S2[l_idx]);
          fm_l = stlutils::add_scalar(fm_l, bias_S2[l_idx]);
          fm_l = activation_2d(fm_l, PhiType::Parametric_Tanh, phi_a);
        }
        int s2c3_w_idx = 0;
        for (size_t c3l_idx = 0; c3l_idx < 16; ++c3l_idx)
        {
          auto& C3_l = feature_maps_C3[c3l_idx];
          const auto& S2_conn = map_C3_to_S2[c3l_idx];
          for (size_t s2l_idx = 0; s2l_idx < 6; ++s2l_idx)
          {
            if (S2_conn[s2l_idx])
            {
              const auto& S2_l = feature_maps_S2[s2l_idx];
              auto conv_s2 = conv_2d(S2_l, W_kernel_C3[s2c3_w_idx++], 0, ConvRange::Valid, ConvType::Correlation);
              C3_l = stlutils::add(C3_l, conv_s2);
            }
          }
          C3_l = stlutils::add_scalar(C3_l, bias_C3[c3l_idx]);
        }
        for (size_t l_idx = 0; l_idx < 16; ++l_idx)
        {
          auto& fm_l_raw = feature_maps_S4_raw[l_idx];
          auto& fm_l = feature_maps_S4[l_idx];
          fm_l_raw = pooling_2d(feature_maps_C3[l_idx], PoolingType::Sum, { 2, 2 }, { 2, 2 });
          fm_l = fm_l_raw;
          fm_l = stlutils::mult_scalar(fm_l, W_S4[l_idx]);
          fm_l = stlutils::add_scalar(fm_l, bias_S4[l_idx]);
          fm_l = activation_2d(fm_l, PhiType::Parametric_Tanh, phi_a);
        }
        for (size_t c5l_idx = 0; c5l_idx < 120; ++c5l_idx)
        {
          auto& C5_l = feature_maps_C5[c5l_idx];
          for (size_t s4l_idx = 0; s4l_idx < 16; ++s4l_idx)
          {
            const auto& S4_l = feature_maps_S4[s4l_idx];
            auto conv_s4 = conv_2d(S4_l, W_kernel_C5[c5l_idx*16 + s4l_idx], 0, ConvRange::Valid, ConvType::Correlation);
            C5_l = stlutils::add(C5_l, conv_s4);
          }
          C5_l = stlutils::add_scalar(C5_l, bias_C5[c5l_idx]);
        }
        const auto& C5_flat = flatten_3d_to_1d_input(feature_maps_C5);
        dense_network.set_inputs(C5_flat);
        dense_network.update_forward();
        //output
        dense_network.print_result();
        output = stlutils::to_array<float, 10>(dense_network.result());
        std::cout << str::row_vector(softmax(output));
        std::cout << str::row_vector(stlutils::argmax(output));
      }
      
      void update_backward(unsigned char label,
                           float eta = 0.1f, float mu = 0.05f, float r = 0.f)
      {
        // Fully Connected Step F6 : dC/dw_F6.
        assert(label < 10);
        std::vector<float> y_trg(10, 0);
        y_trg[label] = 1;
        auto grad_F6 = dense_network.update_backward(BackPropType::Output, y_trg, eta, mu, r);
        
        size_t Ni = dense_network.num_inputs();
        size_t No = dense_network.num_outputs();
        std::vector<float> grad_F6_flat(Ni, 0);
        for (size_t o_idx = 0; o_idx < No; ++o_idx)
          for (size_t i_idx = 0; i_idx < Ni; ++i_idx)
            grad_F6_flat[i_idx] += grad_F6[o_idx][i_idx]; // 120
        auto grad_F6_full = unflatten_1d_to_3d(grad_F6_flat, Ni, 1, 1);
        
        // Convolutional Step C5 : dC/dw_C5 = conv_2d(input_C5, dC/dy_C5)
        size_t Nc5wl = 120*16;
        std::vector<std::vector<std::vector<float>>> dC_dw_C5(Nc5wl);
        std::vector<std::vector<std::vector<float>>> W_diff_C5(Nc5wl);
        std::vector<float> b_diff_C5(Nc5wl, 0);
        std::vector<std::vector<std::vector<float>>> dC_dx_C5(Nc5wl);
        for (size_t c5l_idx = 0; c5l_idx < 120; ++c5l_idx)
        {
          //auto& C5_l = feature_maps_C5[c5l_idx];
          for (size_t s4l_idx = 0; s4l_idx < 16; ++s4l_idx)
          {
            const auto& S4_l = feature_maps_S4[s4l_idx];
            size_t s4c5_w_idx = c5l_idx*16 + s4l_idx;
            dC_dw_C5[s4c5_w_idx] = conv_2d(S4_l, grad_F6_full[c5l_idx], 0, ConvRange::Valid, ConvType::Correlation);
            W_diff_C5[s4c5_w_idx] = stlutils::mult_scalar(dC_dw_C5[s4c5_w_idx], -eta);
            W_diff_C5[s4c5_w_idx] = stlutils::add(W_diff_C5[s4c5_w_idx], stlutils::mult_scalar(W_diff_C5[s4c5_w_idx], mu));
            W_diff_C5[s4c5_w_idx] = stlutils::add_scalar(W_diff_C5[s4c5_w_idx], r);
            // Update weights
            W_kernel_C5[s4c5_w_idx] = stlutils::add(W_kernel_C5[s4c5_w_idx], W_diff_C5[s4c5_w_idx]);
            W_diff_C5_prev[s4c5_w_idx] = W_diff_C5[s4c5_w_idx];
            // Next gradient
            dC_dx_C5[s4c5_w_idx] = conv_2d(W_kernel_C5[s4c5_w_idx], grad_F6_full[c5l_idx], 0, ConvRange::Valid, ConvType::Convolution);
          }
          b_diff_C5[c5l_idx] = -eta * grad_F6_flat[c5l_idx] + mu * b_diff_C5_prev[c5l_idx] + r;
          // Update bias
          bias_C5[c5l_idx] += b_diff_C5[c5l_idx];
          b_diff_C5_prev[c5l_idx] = b_diff_C5[c5l_idx];
        }
        
        // Pooling Step S4 : dC/dw_S4.
        std::vector<float> dC_dw_S4(16);
        std::vector<float> dC_dy_S4(16);
        std::vector<float> dy_dz_S4(16);
        std::vector<float> dz_dw_S4(16);
        std::vector<float> W_diff_S4(16);
        std::vector<float> b_diff_S4(16);
        std::vector<float> dC_dx_S4(16);
        for (size_t s4l_idx = 0; s4l_idx < 16; ++s4l_idx)
        {
          // dC/dw = dC/dy * dy/dz * dz/dw
          dy_dz_S4[s4l_idx] = stlutils::sum(activation_diff_2d(feature_maps_S4[s4l_idx], PhiType::Parametric_Tanh, phi_a));
          dz_dw_S4[s4l_idx] = stlutils::sum(feature_maps_S4_raw[s4l_idx]);
          dC_dy_S4[s4l_idx] = 0;
          for (size_t c5l_idx = 0; c5l_idx < 120; ++c5l_idx)
          {
            size_t c5s4_idx = c5l_idx*16 + s4l_idx;
            dC_dy_S4[s4l_idx] += stlutils::sum(dC_dx_C5[c5s4_idx]); // dC/dx_C5 : 120*16*(5x5)
          }
          dC_dw_S4[s4l_idx] = dC_dy_S4[s4l_idx] * dy_dz_S4[s4l_idx] * dz_dw_S4[s4l_idx];
          W_diff_S4[s4l_idx] = -eta * dC_dw_S4[s4l_idx] + mu * W_diff_S4_prev[s4l_idx] + r;
          // Update weights
          W_S4[s4l_idx] += W_diff_S4[s4l_idx];
          W_diff_S4[s4l_idx] = W_diff_S4[s4l_idx];
          
          b_diff_S4[s4l_idx] = -eta * dC_dy_S4[s4l_idx] + mu * b_diff_S4_prev[s4l_idx] + r;
          // Update biases
          bias_S4[s4l_idx] += b_diff_S4[s4l_idx];
          b_diff_S4_prev[s4l_idx] = b_diff_S4[s4l_idx];
          
          // dC_dx = dC/dz * w
          dC_dx_S4[s4l_idx] = dC_dy_S4[s4l_idx] * dy_dz_S4[s4l_idx] * W_S4[s4l_idx];
        }
        
        // Convolutional Step C3 : dC/dw_C3 = conv_2d(input_C3, dC/dy_C3)
        size_t Nc3wl = 60;
        //std::vector<std::vector<std::vector<float>>> grad_S4_full = unflatten_1d_to_3d(dC_dx_S4, 16, 1, 1); //Should be 16x10x10
        std::vector<std::vector<std::vector<float>>> dC_dw_C3(Nc3wl);
        std::vector<std::vector<std::vector<float>>> W_diff_C3(Nc3wl);
        std::vector<float> b_diff_C3(Nc3wl, 0);
        std::vector<std::vector<std::vector<float>>> dC_dx_C3(Nc3wl);
        for (size_t c3l_idx = 0; c3l_idx < 16; ++c3l_idx)
        {
          const auto& S2_conn = map_C3_to_S2[c3l_idx];
          auto dC_dy_C3_l = stlutils::mult_scalar(feature_maps_C3[c3l_idx], dC_dx_S4[c3l_idx]); // #TODO: Check if this is valid!
          for (size_t s2l_idx = 0; s2l_idx < 6; ++s2l_idx)
          {
            if (S2_conn[s2l_idx])
            {
              const auto& S2_l = feature_maps_S2[s2l_idx];
              size_t s2c3_w_idx = c3l_idx*6 + s2l_idx;
              dC_dw_C3[s2c3_w_idx] = conv_2d(S2_l, dC_dy_C3_l, 0, ConvRange::Valid, ConvType::Correlation);
              W_diff_C3[s2c3_w_idx] = stlutils::mult_scalar(dC_dw_C3[s2c3_w_idx], -eta);
              W_diff_C3[s2c3_w_idx] = stlutils::add(W_diff_C3[s2c3_w_idx], stlutils::mult_scalar(W_diff_C3[s2c3_w_idx], mu));
              W_diff_C3[s2c3_w_idx] = stlutils::add_scalar(W_diff_C3[s2c3_w_idx], r);
              // Update weights
              W_kernel_C3[s2c3_w_idx] = stlutils::add(W_kernel_C3[s2c3_w_idx], W_diff_C3[s2c3_w_idx]);
              W_diff_C3_prev[s2c3_w_idx] = W_diff_C3[s2c3_w_idx];
              // Next gradient
              dC_dx_C3[s2c3_w_idx] = conv_2d(W_kernel_C3[s2c3_w_idx], dC_dy_C3_l, 0, ConvRange::Valid, ConvType::Convolution);
            }
          }
          b_diff_C3[c3l_idx] = -eta * dC_dx_S4[c3l_idx] + mu * b_diff_C3_prev[c3l_idx] + r;
          // Update bias
          bias_C3[c3l_idx] += b_diff_C3[c3l_idx];
          b_diff_C3_prev[c3l_idx] = b_diff_C3[c3l_idx];
        }
        
        
        // // dC/dw1 = dC/dy * dy/dz * dz/dw
        // auto dC_dy = -(y_trg - y);
        // auto dy_dz = phi_diff(z, phi_type, phi_param_a, phi_param_k, phi_param_l);
        // std::vector<float> dz_dw(Nw); // z = w0*x0 + w1*x1 + b => dz/dw0 = x0, dz/dw1 = x1, dz/db = 1.
        // for (size_t i = 0; i < Nw; ++i)
        //   dz_dw[i] = inputs[i].get().value_or(0);
        // auto dz_db = 1.f;
        // auto dC_dz = dC_dy * dy_dz;
        // auto dC_dw = stlutils::mult_scalar(dz_dw, dC_dz);
        // auto dC_db = dC_dz * dz_db;
      
        // auto w_diff = stlutils::mult_scalar(dC_dw, -eta);
        // w_diff = stlutils::add(w_diff, stlutils::mult_scalar(w_diff_prev, mu));
        // w_diff = stlutils::add_scalar(w_diff, r);
        // auto b_diff = -eta * dC_db + mu * b_diff_prev + r;
      
        // weights = stlutils::add(weights, w_diff);
        // bias += b_diff;
      
        // w_diff_prev = w_diff;
        // b_diff_prev = b_diff;
      
        // // Return dC/dz*w
        // auto grad_out = stlutils::mult_scalar(weights, dC_dz);
        // return grad_out;
      }
      
      void train(const mnist::MNIST_Unit& training_ex,
                 float eta = 0.1f, float mu = 0.05f, float r = 0.f)
      {
        set_input(training_ex.image);
        update_forward();
        update_backward(training_ex.label, eta, mu, r);
      }
    };
  
  }

}
