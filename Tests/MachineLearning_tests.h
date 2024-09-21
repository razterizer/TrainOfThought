//
//  MachineLearning_tests.h
//  Core Lib
//
//  Created by Rasmus Anthin on 2022-12-01.
//

#pragma once
#include "../MachineLearning/ann_dense.h"
#include "../MachineLearning/ann_cnn.h"
#include "../MachineLearning/LeNet.h"
#include "../MachineLearning/MNIST/MNIST_Loader.h"
#include <unistd.h>
#include <assert.h>

namespace ml
{

  namespace ann
  {
  
    void unit_tests()
    {
      Neuron n({ 1.1, 1, 0.7, 1, 1 }, 0.5, PhiType::GELU);
      n.set_inputs({ -0.61, 0.75, 2.6, -1.5, -1 });
      std::cout << '\n';
      n.update_forward();
      std::cout << *n.output() << '\n';
      for (int i = 0; i < 12; ++i)
      {
        n.train(BackPropType::Output, 0.57f);
        std::cout << i << " : " << *n.output() << '\n';
      }
      n.print();
      
      std::cout << "== Neural Layer ==\n";
      
      NeuralLayer nl { { { 1.1, 1, 0.7, 1, 1 }, { 0.3, 1.2, 3.2, 0.75, 0.1 } }, { 0.5, -0.21 }, PhiType::GELU };
      nl.set_inputs({ -0.61, 0.75, 2.6, -1.5, -1 });
      for (int i = 0; i < 20; ++i)
        nl.train(BackPropType::Output, { 1, 0 });
      nl.print();
      
      NeuralNetwork nn
      {
        {
          // Ni*No = 5x4
          generate_weights(GenWeightsType::Rndn_m1_p1, 5, 4),
          // Ni*No = 4x2
          generate_weights(GenWeightsType::Rndn_m1_p1, 4, 2),
        },
        { // No = 4, 2
          generate_weights(GenWeightsType::Zero, 4),
          generate_weights(GenWeightsType::Zero, 2)
        },
        { PhiType::Leaky_ReLU, PhiType::Tanh }
      };
      
      nn.set_inputs({ -0.61, 0.75, 2.6, -1.5, -1 });
      for (int i = 0; i < 1'000; ++i)
        nn.train(BackPropType::Output, { 1, 0 }, 0.02, 0.5);
      nn.print();
      nn.print_result();
      
      auto smax = softmax(nn.result(), 3);
      auto smin = softmin(nn.result(), 3);
      auto hmax = hardmax(nn.result());
      auto hmin = hardmin(nn.result());
      auto max_idx = stlutils::argmax(nn.result());
      auto min_idx = stlutils::argmin(nn.result());
      std::cout << "smax = \n" << str::column_vector(smax, str::BracketType::MatrixStyle);
      std::cout << "smin = \n" << str::column_vector(smin, str::BracketType::MatrixStyle);
      std::cout << "hmax = \n" << str::column_vector(hmax, str::BracketType::MatrixStyle);
      std::cout << "hmin = \n" << str::column_vector(hmin, str::BracketType::MatrixStyle);
      std::cout << "argmax(nn.result()) = " << str::row_vector(max_idx);
      std::cout << "argmin(nn.result()) = " << str::row_vector(min_idx);
      
      // CNN
      
      {
        auto x = std::vector<float> { 1, 0, -2, 7, 3, 9, 1 };
        auto p0_1d = pooling_1d(x, PoolingType::Max, 2, 2, false);
        assert((p0_1d == std::vector<float> { 1, 7, 9 }));
        
        auto p1_1d = pooling_1d(x, PoolingType::Max, 2, 2, true);
        assert((p1_1d == std::vector<float> { 1, 7, 9, 1 }));
        
        auto p2_1d = pooling_1d(x, PoolingType::Min, 3, 1, false);
        assert((p2_1d == std::vector<float> { -2, -2, -2, 3, 1 }));
        
        auto p3_1d = pooling_1d(x, PoolingType::Min, 3, 1, true);
        assert((p3_1d == std::vector<float> { -2, -2, -2, 3, 1 }));
        
        auto p4_1d = pooling_1d(x, PoolingType::Mean, 2, 1, false);
        assert((p4_1d == std::vector { 0.5f, -1.f, 2.5f, 5.f, 6.f, 5.f }));
        
        auto af_p4_1d = activation_1d(p4_1d, PhiType::ReLU);
        assert((af_p4_1d == std::vector { 0.5f, 0.f, 2.5f, 5.f, 6.f, 5.f }));
        
        auto kernel = std::vector<float> { 1, 2, 3 };
        auto c0_1d = conv_1d(x, kernel, 1.f, ConvRange::Full);
        assert((c0_1d == std::vector<float> { 2, 3, 2, 4, 12, 37, 29, 30, 4 }));
        
        auto f = std::vector<float> { 0, 0, -1, 2, 1, 0, 1, 2, 3, 0, 0 };
        auto g = std::vector<float> { 0, 1, -1, 1, 0 };
        auto c1_1d = conv_1d(f, g, 0, ConvRange::Full);
        assert((c1_1d == std::vector<float> { 0, 0, 0, -1, 3, -2, 1, 2, 1, 2, -1, 3, 0, 0, 0 }));
        auto c2_1d = conv_1d(f, g, 0, ConvRange::Same);
        assert((c2_1d == std::vector<float> { 0, -1, 3, -2, 1, 2, 1, 2, -1, 3, 0 }));
        auto c3_1d = conv_1d(f, g, 0, ConvRange::Valid);
        assert((c3_1d == std::vector<float> { 3, -2, 1, 2, 1, 2, -1 }));
        g = std::vector<float> { 0, 1, 2, 3 };
        auto c4_1d = conv_1d(f, g, 0, ConvRange::Full);
        assert((c4_1d == std::vector<float> { 0, 0, 0, -1, 0, 2, 8, 4, 4, 10, 12, 9, 0, 0 }));
        auto c5_1d = conv_1d(f, g, 0, ConvRange::Same);
        assert((c5_1d == std::vector<float> { 0, -1, 0, 2, 8, 4, 4, 10, 12, 9, 0 }));
        auto c6_1d = conv_1d(f, g, 0, ConvRange::Valid);
        assert((c6_1d == std::vector<float> { -1, 0, 2, 8, 4, 4, 10, 12 }));
        g = std::vector<float> { 1, -1, -2 };
        auto c7_1d = conv_1d(f, g, 0, ConvRange::Full);
        assert((c7_1d == std::vector<float> { 0, 0, -1, 3, 1, -5, -1, 1, -1, -7, -6, 0, 0 }));
        auto c8_1d = conv_1d(f, g, 0, ConvRange::Same);
        assert((c8_1d == std::vector<float> { 0, -1, 3, 1, -5, -1, 1, -1, -7, -6, 0 }));
        auto c9_1d = conv_1d(f, g, 0, ConvRange::Valid);
        assert((c9_1d == std::vector<float> { -1, 3, 1, -5, -1, 1, -1, -7, -6 }));
        g = std::vector<float> { -1, 5 };
        auto c10_1d = conv_1d(f, g, 0, ConvRange::Full);
        assert((c10_1d == std::vector<float> { 0, 0, 1, -7, 9, 5, -1, 3, 7, 15, 0, 0 }));
        auto c11_1d = conv_1d(f, g, 0, ConvRange::Same);
        assert((c11_1d == std::vector<float> { 0, 1, -7, 9, 5, -1, 3, 7, 15, 0, 0 }));
        auto c12_1d = conv_1d(f, g, 0, ConvRange::Valid);
        assert((c12_1d == std::vector<float> { 0, 1, -7, 9, 5, -1, 3, 7, 15, 0 }));
        g = std::vector<float> { 4 };
        auto c13_1d = conv_1d(f, g, 0, ConvRange::Full);
        assert((c13_1d == std::vector<float> { 0, 0, -4, 8, 4, 0, 4, 8, 12, 0, 0 }));
        auto c14_1d = conv_1d(f, g, 0, ConvRange::Same);
        assert((c14_1d == std::vector<float> { 0, 0, -4, 8, 4, 0, 4, 8, 12, 0, 0 }));
        auto c15_1d = conv_1d(f, g, 0, ConvRange::Valid);
        assert((c15_1d == std::vector<float> { 0, 0, -4, 8, 4, 0, 4, 8, 12, 0, 0 }));
        f = std::vector<float> { 0, 0, -1, 2, 1, 0, 1, 2, 3, 0, 0, 1 };
        g = std::vector<float> { 0, 1, -1, 1, 0 };
        auto c16_1d = conv_1d(f, g, 0, ConvRange::Full);
        assert((c16_1d == std::vector<float> { 0, 0, 0, -1, 3, -2, 1, 2, 1, 2, -1, 3, 1, -1, 1, 0 }));
        auto c17_1d = conv_1d(f, g, 0, ConvRange::Same);
        assert((c17_1d == std::vector<float> { 0, -1, 3, -2, 1, 2, 1, 2, -1, 3, 1, -1 }));
        auto c18_1d = conv_1d(f, g, 0, ConvRange::Valid);
        assert((c18_1d == std::vector<float> { 3, -2, 1, 2, 1, 2, -1, 3 }));
        g = std::vector<float> { 1, -1 };
        auto c19_1d = conv_1d(f, g, 0, ConvRange::Full);
        assert((c19_1d == std::vector<float> { 0, 0, -1, 3, -1, -1, 1, 1, 1, -3, 0, 1, -1 }));
        auto c20_1d = conv_1d(f, g, 0, ConvRange::Same);
        assert((c20_1d == std::vector<float> { 0, -1, 3, -1, -1, 1, 1, 1, -3, 0, 1, -1 }));
        auto c21_1d = conv_1d(f, g, 0, ConvRange::Valid);
        assert((c21_1d == std::vector<float> { 0, -1, 3, -1, -1, 1, 1, 1, -3, 0, 1 }));
        f = std::vector<float> { -7, 0, -1, 2 };
        g = std::vector<float> { 0, 1, -1, 1, 0 };
        auto c22_1d = conv_1d(f, g, 0, ConvRange::Full);
        assert((c22_1d == std::vector<float> { 0, -7, 7, -8, 3, -3, 2, 0 }));
        auto c23_1d = conv_1d(f, g, 0, ConvRange::Same);
        assert((c23_1d == std::vector<float> { 7, -8, 3, -3 }));
        auto c24_1d = conv_1d(f, g, 0, ConvRange::Valid);
        assert((c24_1d == std::vector<float> {}));
        f = std::vector<float> { -7, 0, -1 };
        auto c25_1d = conv_1d(f, g, 0, ConvRange::Full);
        assert((c25_1d == std::vector<float> { 0, -7, 7, -8, 1, -1, 0 }));
        auto c26_1d = conv_1d(f, g, 0, ConvRange::Same);
        assert((c26_1d == std::vector<float> { 7, -8, 1 }));
        auto c27_1d = conv_1d(f, g, 0, ConvRange::Valid);
        assert((c27_1d == std::vector<float> {}));
      }
      
      {
        auto x = std::vector<std::vector<float>>
        {
          { 1, 0, -2, 7, 3, 9, 1 },
          { 2, 3, -2, 4, 0, 0, 7 },
          { 8, 2, 2, -14, -7, -5, 0},
          { -1, 0, 0, 4, 12, -8, 6 },
          { 0, 5, -3, 11, 2, 10, -11 },
          { -11, 1, 4, 3, 0, 0, 5 },
          { 2, 0, 7, 1, 0, 5, -2 }
        };
        auto p0_2d = pooling_2d(x, PoolingType::Max, { 2, 2 }, { 2, 2 }, false);
        assert((p0_2d == std::vector<std::vector<float>>
        {
          { 3, 7, 9 },
          { 8, 4, 12 },
          { 5, 11, 10 }
        }));
        auto p1_2d = pooling_2d(x, PoolingType::Max, { 2, 2 }, { 2, 2 }, true);
        assert((p1_2d == std::vector<std::vector<float>>
        {
          { 3, 7, 9, 7 },
          { 8, 4, 12, 6 },
          { 5, 11, 10, 5 },
          { 2, 7, 5, 0 }
        }));
        auto p2_2d = pooling_2d(x, PoolingType::Min, { 3, 3 }, { 1, 1 }, false);
        assert((p2_2d == std::vector<std::vector<float>>
        {
          { -2, -14, -14, -14, -7 },
          { -2, -14, -14, -14, -8 },
          { -3, -14, -14, -14, -11 },
          { -11, -3, -3, -8, -11 },
          { -11, -3, -3, 0, -11 }
        }));
        auto p3_2d = pooling_2d(x, PoolingType::Mean, { 3, 3 }, { 3, 3 }, false);
        assert(math::fuz_eq(p3_2d[0][0], 1.5556f, 1e-4f));
        assert(math::fuz_eq(p3_2d[0][1], -0.3333f, 1e-4f));
        assert(math::fuz_eq(p3_2d[1][0], -0.5556f, 1e-4f));
        assert(math::fuz_eq(p3_2d[1][1], 3.7778f, 1e-4f));
        
        auto af_p3_2d = activation_2d(p3_2d, PhiType::ReLU);
        assert(math::fuz_eq(af_p3_2d[0][0], 1.5556f, 1e-4f));
        assert(math::is_eps(af_p3_2d[0][1], 1e-4f));
        assert(math::is_eps(af_p3_2d[1][0], 1e-4f));
        assert(math::fuz_eq(af_p3_2d[1][1], 3.7778f, 1e-4f));
        
        auto k = std::vector<std::vector<float>>
        {
          { 0, -1, 0 },
          { -1, 5, -1 },
          { 0, -1, 0 }
        };
        auto c0_2d = conv_2d(x, k, 0, ConvRange::Full);
        assert((c0_2d == std::vector<std::vector<float>>
        {
          {  0,  -1,   0,   2,  -7,  -3,  -9,  -1,  0 },
          { -1,   3,  -2, -15,  30,  -1,  41, -11, -1 },
          { -2,  -2,  13, -17,  29,   0, -11,  34, -7 },
          { -8,  37,  -3,  24, -73, -28, -10,  -8,  0 },
          {  1, -13,  -6,  -3,  11,  69, -63,  49, -6 },
          {  0,   7,  27, -35,  49, -23,  67, -76, 11 },
          { 11, -58,   7,  12,  -1,  -5, -20,  38, -5 },
          { -2,  21, -10,  30,  -5,  -6,  27, -20,  2 },
          {  0,  -2,   0,  -7,  -1,   0,  -5,   2,  0 }
        }));
        auto c1_2d = conv_2d(x, k, 0, ConvRange::Same);
        assert((c1_2d == std::vector<std::vector<float>>
        {
          {   3,  -2, -15,  30,  -1,  41, -11 },
          {  -2,  13, -17,  29,   0, -11,  34 },
          {  37,  -3,  24, -73, -28, -10,  -8 },
          { -13,  -6,  -3,  11,  69, -63,  49 },
          {   7,  27, -35,  49, -23,  67, -76 },
          { -58,   7,  12,  -1,  -5, -20,  38 },
          {  21, -10,  30,  -5,  -6,  27, -20 },
        }));
        auto c2_2d = conv_2d(x, k, 0, ConvRange::Valid);
        assert((c2_2d == std::vector<std::vector<float>>
        {
          { 13, -17,  29,   0, -11 },
          { -3,  24, -73, -28, -10 },
          { -6,  -3,  11,  69, -63 },
          { 27, -35,  49, -23,  67 },
          {  7,  12,  -1,  -5, -20 },
        }));
        auto c3_2d = conv_2d(x, k, 73, ConvRange::Valid);
        assert((c3_2d == std::vector<std::vector<float>>
        {
          {  86, 56, 102,  73,  62 },
          {  70, 97,   0,  45,  63 },
          {  67, 70,  84, 142,  10 },
          { 100, 38, 122,  50, 140 },
          {  80, 85,  72,  68,  53 },
        }));
      }
      
      // LeNet5
      
      char cwd[PATH_MAX];
         if (getcwd(cwd, sizeof(cwd)) != NULL) {
             printf("Current working dir: %s\n", cwd);
         } else {
             perror("getcwd() error");
             //return;
         }
      
      mnist::MNIST_Loader mnist({ 2, 2 });
      mnist.print(mnist::MNIST_Subset::Training, 0);
      mnist.print(mnist::MNIST_Subset::Training, 0, true);
      mnist.print(mnist::MNIST_Subset::Training, 1, true);
      mnist.print(mnist::MNIST_Subset::Training, 2, true);
      LeNet5 lenet5;
      auto ex = mnist.get(mnist::MNIST_Subset::Training, 2);
      if (!ex.empty())
      {
        lenet5.train(ex);
      }
    }
  
  }

}
