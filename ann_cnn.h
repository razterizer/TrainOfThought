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
  
    std::vector<std::vector<float>> activation_2d(const std::vector<std::vector<float>>& x, PhiType phi_type, float a = 1.f, float k = 1.f, float l = 1.1f)
    {
      auto N = x.size();
      std::vector<std::vector<float>> ret(N);
      for (size_t r_idx = 0; r_idx < N; ++r_idx)
        ret[r_idx] = activation_1d(x[r_idx], phi_type, a, k, l);
      return ret;
    }
  
    enum class PoolingType { Min, Mean, Max, Sum };
  
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
          case PoolingType::Sum:
          case PoolingType::Mean: cell = 0; break;
        }
        for (size_t f_idx = 0; f_idx < filter_size; ++f_idx)
        {
          size_t i_idx = o_idx * stride + f_idx;
          switch (type)
          {
            case PoolingType::Min: math::minimize(cell, xx[i_idx]); break;
            case PoolingType::Max: math::maximize(cell, xx[i_idx]); break;
            case PoolingType::Sum:
            case PoolingType::Mean: cell += xx[i_idx]; break;
          }
        }
        if (type == PoolingType::Mean && filter_size > 0)
          cell /= filter_size;
        ret[o_idx] = cell;
      }
      return ret;
    }
  
    // #FIXME: Also treat 'valid' and 'same' ranges and perhaps remove use_padding altogether if 'same' covers this case, else add also 'pad' mode.
    std::vector<std::vector<float>> pooling_2d(const std::vector<std::vector<float>>& x, PoolingType type, std::array<int, 2> filter_size, std::array<int, 2> stride, bool use_padding = false)
    {
      auto xx = x;
      auto Nir = xx.size();
      auto Nic = xx.back().size();
      //auto No = (Ni - filter_size + 1)/stride;
      //auto No = (Ni - filter_size + 2*padding)/stride + 1;
      auto Nor = (Nir - filter_size[0])/stride[0] + 1;
      auto Noc = (Nic - filter_size[1])/stride[1] + 1;
      auto rest_r = static_cast<float>(Nir - filter_size[0])/stride[0] + 1 - Nor;
      auto rest_c = static_cast<float>(Nic - filter_size[1])/stride[1] + 1 - Noc;
      if (use_padding && rest_r != 0) // #FIXME: Bad code smell.
      {
        xx.emplace_back(stlutils::repval(0.f, Nic));
        Nor++;
      }
      if (use_padding && rest_c != 0)
      {
        for (auto& row : xx)
          row.emplace_back(0);
        Noc++;
      }
      std::vector<std::vector<float>> ret(Nor);
      for (auto& row : ret)
        row.resize(Noc);
      for (size_t or_idx = 0; or_idx < Nor; ++or_idx)
      {
        for (size_t oc_idx = 0; oc_idx < Noc; ++oc_idx)
        {
          float cell = 0.f;
          switch (type)
          {
            case PoolingType::Min: cell = math::get_max<float>(); break;
            case PoolingType::Max: cell = math::get_min<float>(); break;
            case PoolingType::Sum:
            case PoolingType::Mean: cell = 0; break;
          }
          for (size_t fr_idx = 0; fr_idx < filter_size[0]; ++fr_idx)
          {
            size_t ir_idx = or_idx * stride[0] + fr_idx;
            for (size_t fc_idx = 0; fc_idx < filter_size[1]; ++fc_idx)
            {
              size_t ic_idx = oc_idx * stride[1] + fc_idx;
              switch (type)
              {
                case PoolingType::Min: math::minimize(cell, xx[ir_idx][ic_idx]); break;
                case PoolingType::Max: math::maximize(cell, xx[ir_idx][ic_idx]); break;
                case PoolingType::Sum:
                case PoolingType::Mean: cell += xx[ir_idx][ic_idx]; break;
              }
            }
          }
          int Nf = filter_size[0] * filter_size[1];
          if (type == PoolingType::Mean && Nf > 0)
            cell /= Nf;
          ret[or_idx][oc_idx] = cell;
        }
      }
      return ret;
    }
  
    enum class ConvType { Full, Same, Valid };
  
    std::vector<float> conv_1d(const std::vector<float>& x,
                               const std::vector<float>& w_kernel, float bias = 0.f,
                               ConvType type = ConvType::Full)
    {
      auto Ni = static_cast<int>(x.size());
      auto Nk = static_cast<int>(w_kernel.size());
      int No = 0;
      int pad_L = 0;
      int pad_R = 0;
      switch (type)
      {
        case ConvType::Full:
          No = Ni + Nk - 1;
          pad_L = Nk - 1;
          pad_R = pad_L;
          break;
        case ConvType::Same:
          No = Ni;
          if (Nk % 2 == 1)
          {
            pad_L = std::max(0, Nk/2);
            pad_R = pad_L;
          }
          else
          {
            pad_L = std::max(0, Nk/2 - 1);
            pad_R = pad_L + 1;
          }
          break;
        case ConvType::Valid:
          No = std::max(0, Ni - Nk + 1);
          break;
      }
      auto pad_vec_L = stlutils::repval<float>(0, pad_L);
      auto pad_vec_R = stlutils::repval<float>(0, pad_R);
      auto xx = stlutils::cat(pad_vec_L, x, pad_vec_R);
      auto kk = w_kernel;
      std::reverse(kk.begin(), kk.end());
      std::vector<float> y(No);
      for (int o_idx = 0; o_idx < No; ++o_idx)
        y[o_idx] = stlutils::dot(stlutils::subset(xx, o_idx, o_idx + Nk - 1), kk) + bias;
      return y;
    }
  
    std::vector<std::vector<float>> conv_2d(const std::vector<std::vector<float>>& x,
                                            const std::vector<std::vector<float>>& w_kernel,
                                            float bias = 0.f,
                                            ConvType type = ConvType::Full)
    {
      auto Nir = static_cast<int>(x.size());
      auto Nic = static_cast<int>(x.back().size());
      auto Nkr = static_cast<int>(w_kernel.size());
      auto Nkc = static_cast<int>(w_kernel.back().size());
      int Nor = 0;
      int Noc = 0;
      int pad_L = 0;
      int pad_R = 0;
      int pad_D = 0;
      int pad_U = 0;
      switch (type)
      {
        case ConvType::Full:
          Nor = Nir + Nkr - 1;
          pad_D = Nkr - 1;
          pad_U = pad_D;
          Noc = Nic + Nkc - 1;
          pad_L = Nkc - 1;
          pad_R = pad_L;
          break;
        case ConvType::Same:
        {
          Nor = Nir;
          Noc = Nic;
          auto set_padding = [](int& pad_lo, int& pad_hi, int Nk)
          {
            if (Nk % 2 == 1)
            {
              pad_lo = std::max(0, Nk/2);
              pad_hi = pad_lo;
            }
            else
            {
              pad_lo = std::max(0, Nk/2 - 1);
              pad_hi = pad_lo + 1;
            }
          };
          set_padding(pad_D, pad_U, Nkr);
          set_padding(pad_L, pad_R, Nkc);
          break;
        }
        case ConvType::Valid:
          Nor = std::max(0, Nir - Nkr + 1);
          Noc = std::max(0, Nic - Nkc + 1);
          break;
      }
      auto pad_vec_L = stlutils::repval<float>(0, pad_L);
      auto pad_vec_R = stlutils::repval<float>(0, pad_R);
      auto pad_vec_UD = std::vector<std::vector<float>> { stlutils::repval<float>(0, Nic) };
      auto xx = x;
      for (int p = 0; p < pad_U; ++p)
        xx = stlutils::cat(pad_vec_UD, xx);
      for (int p = 0; p < pad_D; ++p)
        xx = stlutils::cat(xx, pad_vec_UD);
      for (auto& row : xx)
        row = stlutils::cat(pad_vec_L, row, pad_vec_R);
      
      auto kk = w_kernel;
      std::reverse(kk.begin(), kk.end());
      for (auto& row : kk)
        std::reverse(row.begin(), row.end());
      
      std::vector<std::vector<float>> y(Nor);
      for (auto& row : y)
        row.resize(Noc);
      for (int or_idx = 0; or_idx < Nor; ++or_idx)
      {
        auto xx_r = stlutils::subset(xx, or_idx, or_idx + Nkr - 1);
        for (int oc_idx = 0; oc_idx < Noc; ++oc_idx)
        {
          auto xx_rc = stlutils::subset_columns(xx_r, oc_idx, oc_idx + Nkc - 1);
          y[or_idx][oc_idx] = stlutils::dot(xx_rc, kk) + bias;
        }
      }
      return y;
    }
  
    // Function for transferring data from convolutional network domain to regular dense network domain.
    std::vector<Input> flatten_2d_to_1d_input(const std::vector<std::vector<float>>& x)
    {
      auto Nr = x.size();
      auto Nc = x.back().size();
      std::vector<Input> ret(Nr * Nc);
      for (size_t r_idx = 0; r_idx < Nr; ++r_idx)
        for (size_t c_idx = 0; c_idx < Nc; ++c_idx)
          ret[r_idx*Nc + c_idx] = x[r_idx][c_idx];
      return ret;
    }
  }

}
