//
//  ann_cnn.h
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
  
    std::vector<float> activation_1d(const std::vector<float>& x, PhiType phi_type, float a = 1.f, float k = 1.f, float l = 1.1f)
    {
      auto N = x.size();
      std::vector<float> ret(N);
      for (size_t e_idx = 0; e_idx < N; ++e_idx)
        ret[e_idx] = phi(x[e_idx], phi_type, a, k, l);
      return ret;
    }
  
    enum class PoolingType { Min, Mean, Max };
  
    // #FIXME: Also treat 'valid' and 'same' ranges and perhaps remove use_padding altogether if 'same' covers this case, else add also 'pad' mode.
    std::vector<float> pooling_1d(const std::vector<float>& x, PoolingType type, int filter_size, int stride, bool use_padding = false)
    {
      auto xx = x;
      auto Ni = xx.size();
      //auto No = (Ni - filter_size + 1)/stride;
      //auto No = (Ni - filter_size + 2*padding)/stride + 1;
      auto No = (Ni - filter_size)/stride + 1;
      auto rest = static_cast<float>(Ni - filter_size)/stride + 1 - No;
      if (use_padding && rest != 0) // #FIXME: Bad code smell.
      {
        xx.emplace_back(0);
        No++;
      }
      std::vector<float> ret(No);
      for (size_t o_idx = 0; o_idx < No; ++o_idx)
      {
        float cell = 0.f;
        switch (type)
        {
          case PoolingType::Min: cell = math::get_max<float>(); break;
          case PoolingType::Max: cell = math::get_min<float>(); break;
          case PoolingType::Mean: cell = 0; break;
        }
        for (size_t f_idx = 0; f_idx < filter_size; ++f_idx)
        {
          size_t i_idx = o_idx * stride + f_idx;
          switch (type)
          {
            case PoolingType::Min: math::minimize(cell, xx[i_idx]); break;
            case PoolingType::Max: math::maximize(cell, xx[i_idx]); break;
            case PoolingType::Mean: cell += xx[i_idx]; break;
          }
        }
        if (type == PoolingType::Mean && filter_size > 0)
          cell /= filter_size;
        ret[o_idx] = cell;
      }
      return ret;
    }
  
  }

}
