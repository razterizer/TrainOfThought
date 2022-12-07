//
//  MNIST_loader.h
//  Core Lib
//
//  Created by Rasmus Anthin on 2022-12-04.
//

/*
 TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

 [offset] [type]          [value]          [description]
 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
 0004     32 bit integer  60000            number of items
 0008     unsigned byte   ??               label
 0009     unsigned byte   ??               label
 ........
 xxxx     unsigned byte   ??               label
 The labels values are 0 to 9.

 TRAINING SET IMAGE FILE (train-images-idx3-ubyte):

 [offset] [type]          [value]          [description]
 0000     32 bit integer  0x00000803(2051) magic number
 0004     32 bit integer  60000            number of images
 0008     32 bit integer  28               number of rows
 0012     32 bit integer  28               number of columns
 0016     unsigned byte   ??               pixel
 0017     unsigned byte   ??               pixel
 ........
 xxxx     unsigned byte   ??               pixel
 Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

 TEST SET LABEL FILE (t10k-labels-idx1-ubyte):

 [offset] [type]          [value]          [description]
 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
 0004     32 bit integer  10000            number of items
 0008     unsigned byte   ??               label
 0009     unsigned byte   ??               label
 ........
 xxxx     unsigned byte   ??               label
 The labels values are 0 to 9.

 TEST SET IMAGE FILE (t10k-images-idx3-ubyte):

 [offset] [type]          [value]          [description]
 0000     32 bit integer  0x00000803(2051) magic number
 0004     32 bit integer  10000            number of images
 0008     32 bit integer  28               number of rows
 0012     32 bit integer  28               number of columns
 0016     unsigned byte   ??               pixel
 0017     unsigned byte   ??               pixel
 ........
 xxxx     unsigned byte   ??               pixel
 Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
   
 */

#pragma once
#include "wichtounet_mnist_reader/mnist_reader.h"

namespace mnist
{

  class MNIST_Loader
  {
    std::vector<unsigned char> training_labels;
    std::vector<std::vector<std::vector<unsigned char>>> training_images;
    std::vector<unsigned char> test_labels;
    std::vector<std::vector<std::vector<unsigned char>>> test_images;
    const int Nr0 = 28;
    const int Nc0 = 28;
    int pad_r = 0;
    int pad_c = 0;
    int Nr = Nr0;
    int Nc = Nc0;
    
  public:
    // img_padding (row, col) : To get 32x32 images use padding { 2, 2 }.
    MNIST_Loader(const std::array<int, 2>& img_padding = { 0, 0 },
                 const std::string& path = "/Users/rasmusanthin/Documents/xcode/resources/mnist")
    {
      auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(path);
      
      training_labels = dataset.training_labels;
      test_labels = dataset.test_labels;
      auto Nti = dataset.training_images.size();
      pad_r = img_padding[0];
      pad_c = img_padding[1];
      Nr = Nr0 + 2*pad_r;
      Nc = Nc0 + 2*pad_c;
      stlutils::resize(training_images, Nti, Nr, Nc);
      
      for (size_t img_idx = 0; img_idx < Nti; ++img_idx)
      {
        const auto& img_src = dataset.training_images[img_idx];
        auto& img_dst = training_images[img_idx];
        for (size_t r_idx = 0; r_idx < Nr0; ++r_idx)
          for (size_t c_idx = 0; c_idx < Nc0; ++c_idx)
          {
            auto px_idx = r_idx * Nc0 + c_idx;
            img_dst[r_idx + pad_r][c_idx + pad_c] = img_src[px_idx];
          }
      }
    }
    
    enum class MNIST_Subset { Training, Test };
    
    void print(MNIST_Subset subset, size_t img_idx, bool shaded = false) const
    {
      const std::vector<unsigned char>* labels = nullptr;
      const std::vector<std::vector<std::vector<unsigned char>>>* images = nullptr;
      switch (subset)
      {
        case MNIST_Subset::Training:
          labels = &training_labels;
          images = &training_images;
          break;
        case MNIST_Subset::Test:
          labels = &test_labels;
          images = &training_images;
          break;
      }
      
      if (labels == nullptr || images == nullptr)
        return;
      
      const auto label = (*labels)[img_idx];
      const auto& img = (*images)[img_idx];
      
      std::cout << "=== MNIST Image " << img_idx << " --- label: " << static_cast<int>(label) << " ===\n";
      for (size_t r_idx = 0; r_idx < Nr; ++r_idx)
      {
        std::cout << " ";
        if (shaded)
          for (size_t c_idx = 0; c_idx < Nc; ++c_idx)
          {
            auto px = img[r_idx][c_idx] / 255.f;
            if (px == 0)
              std::cout << " ";
            else if (px < 0.25f)
              std::cout << "+";
            else if (px < 0.5f)
              std::cout << "@";
            else if (px < 0.75f)
              std::cout << "*";
            else if (px <= 1.f)
              std::cout << "#";
          }
        else
          for (size_t c_idx = 0; c_idx < Nc; ++c_idx)
            std::cout << str::adjust_str(std::to_string(static_cast<int>(img[r_idx][c_idx])), str::Adjustment::Right, 3) << " ";
        std::cout << "\n";
      }
      std::cout << str::rep_char('=', 100) << "\n";
    }
  };

}

