//
//  LeNet.h
//  Core Lib
//
//  Created by Rasmus Anthin on 2022-12-03.
//

#pragma once

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
      float bias_S2 = 0.f;
      float bias_S4 = 0.f;
      
      std::array<std::array<bool, 6>, 16> map_C3_to_S2;
      
      std::vector<std::vector<std::vector<float>>> feature_maps_C1;
      std::vector<std::vector<std::vector<float>>> feature_maps_S2;
      std::vector<std::vector<std::vector<float>>> feature_maps_C3;
      std::vector<std::vector<std::vector<float>>> feature_maps_S4;
      std::vector<std::vector<std::vector<float>>> feature_maps_C5;
      NeuralNetwork dense_network;
      std::array<float, 10> output;
      
      const float phi_a = 0;
      
    public:
      LeNet5(const std::vector<std::vector<int>>& x)
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
        // O  : 10*84 = 840 trainable params.
        // Tot : 60'840 trainable params.
        
        assert(x.size() == 32);
        assert(x.back().size() == 32);
        stlutils::resize(input, 32, 32);
        for (size_t r_idx = 0; r_idx < 32; ++r_idx)
          for (size_t c_idx = 0; c_idx < 32; ++c_idx)
            input[r_idx][c_idx] = math::value_to_param(x[r_idx][c_idx]/255.f, -0.1f, 1.175f);
        
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
        bias_S2 = generate_weight(GenWeightsType::Rndn_m1_p1);
        bias_S4 = generate_weight(GenWeightsType::Rndn_m1_p1);
        
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
        stlutils::resize(feature_maps_S2, 6, 14, 14);
        stlutils::resize(feature_maps_C3, 16, 10, 10);
        stlutils::resize(feature_maps_S4, 16, 5, 5);
        stlutils::resize(feature_maps_C5, 120, 1, 1);
        
        dense_network.set_phi_params(phi_a, 1.f, 1.1f);
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
          auto& fm_l = feature_maps_S2[l_idx];
          fm_l = pooling_2d(feature_maps_C1[l_idx], PoolingType::Sum, { 2, 2 }, { 2, 2 });
          fm_l = stlutils::mult_scalar(fm_l, W_S2[l_idx]);
          fm_l = stlutils::add_scalar(fm_l, bias_S2);
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
          auto& fm_l = feature_maps_S4[l_idx];
          fm_l = pooling_2d(feature_maps_C3[l_idx], PoolingType::Sum, { 2, 2 }, { 2, 2 });
          fm_l = stlutils::mult_scalar(fm_l, W_S4[l_idx]);
          fm_l = stlutils::add_scalar(fm_l, bias_S4);
          fm_l = activation_2d(fm_l, PhiType::Parametric_Tanh, phi_a);
        }
        int s4c5_w_idx = 0;
        for (size_t c5l_idx = 0; c5l_idx < 120; ++c5l_idx)
        {
          auto& C5_l = feature_maps_C5[c5l_idx];
          for (size_t s4l_idx = 0; s4l_idx < 16; ++s4l_idx)
          {
            const auto& S4_l = feature_maps_S4[s4l_idx];
            auto conv_s4 = conv_2d(S4_l, W_kernel_C5[s4c5_w_idx++], 0, ConvRange::Valid, ConvType::Correlation);
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
    };
  
  }

}
