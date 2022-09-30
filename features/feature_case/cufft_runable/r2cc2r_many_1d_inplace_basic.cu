// ===--- r2cc2r_many_1d_inplace_basic.cu --------------------*- CUDA -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include "cufft.h"
#include "cufftXt.h"
#include "common.h"
#include <cstring>
#include <iostream>


bool r2cc2r_many_1d_inplace_basic() {
  cufftHandle plan_fwd;
  cufftCreate(&plan_fwd);
  float forward_idata_h[24];
  set_value(forward_idata_h, 10);
  set_value(forward_idata_h + 12, 10);

  float* data_d;
  cudaMalloc(&data_d, sizeof(float) * 24);
  cudaMemcpy(data_d, forward_idata_h, sizeof(float) * 24, cudaMemcpyHostToDevice);

  int n[1] = {10};
  size_t workSize;
  cufftMakePlanMany(plan_fwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_R2C, 2, &workSize);
  cufftExecR2C(plan_fwd, data_d, (float2*)data_d);
  cudaDeviceSynchronize();
  float2 forward_odata_h[12];
  cudaMemcpy(forward_odata_h, data_d, sizeof(float) * 24, cudaMemcpyDeviceToHost);

  float2 forward_odata_ref[12];
  forward_odata_ref[0] =  float2{45,0};
  forward_odata_ref[1] =  float2{-5,15.3884};
  forward_odata_ref[2] =  float2{-5,6.88191};
  forward_odata_ref[3] =  float2{-5,3.63271};
  forward_odata_ref[4] =  float2{-5,1.6246};
  forward_odata_ref[5] =  float2{-5,0};
  forward_odata_ref[6] =  float2{45,0};
  forward_odata_ref[7] =  float2{-5,15.3884};
  forward_odata_ref[8] =  float2{-5,6.88191};
  forward_odata_ref[9] =  float2{-5,3.63271};
  forward_odata_ref[10] = float2{-5,1.6246};
  forward_odata_ref[11] = float2{-5,0};

  cufftDestroy(plan_fwd);

  if (!compare(forward_odata_ref, forward_odata_h, 12)) {
    std::cout << "forward_odata_h:" << std::endl;
    print_values(forward_odata_h, 12);
    std::cout << "forward_odata_ref:" << std::endl;
    print_values(forward_odata_ref, 12);

    cudaFree(data_d);

    return false;
  }

  cufftHandle plan_bwd;
  cufftCreate(&plan_bwd);
  cufftMakePlanMany(plan_bwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2R, 2, &workSize);
  cufftExecC2R(plan_bwd, (float2*)data_d, data_d);
  cudaDeviceSynchronize();
  float backward_odata_h[24];
  cudaMemcpy(backward_odata_h, data_d, sizeof(float) * 24, cudaMemcpyDeviceToHost);

  float backward_odata_ref[24];
  backward_odata_ref[0] =  0;
  backward_odata_ref[1] =  10;
  backward_odata_ref[2] =  20;
  backward_odata_ref[3] =  30;
  backward_odata_ref[4] =  40;
  backward_odata_ref[5] =  50;
  backward_odata_ref[6] =  60;
  backward_odata_ref[7] =  70;
  backward_odata_ref[8] =  80;
  backward_odata_ref[9] =  90;
  backward_odata_ref[10] = -5;
  backward_odata_ref[11] = 0;
  backward_odata_ref[12] = 0;
  backward_odata_ref[13] = 10;
  backward_odata_ref[14] = 20;
  backward_odata_ref[15] = 30;
  backward_odata_ref[16] = 40;
  backward_odata_ref[17] = 50;
  backward_odata_ref[18] = 60;
  backward_odata_ref[19] = 70;
  backward_odata_ref[20] = 80;
  backward_odata_ref[21] = 90;
  backward_odata_ref[22] = -5;
  backward_odata_ref[23] = 0;

  cudaFree(data_d);
  cufftDestroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  if (!compare(backward_odata_ref, backward_odata_h, indices)) {
    std::cout << "backward_odata_h:" << std::endl;
    print_values(backward_odata_h, indices);
    std::cout << "backward_odata_ref:" << std::endl;
    print_values(backward_odata_ref, indices);
    return false;
  }
  return true;
}

#ifdef DEBUG_FFT
int main() {
#define FUNC r2cc2r_many_1d_inplace_basic
  bool res = FUNC();
  cudaDeviceSynchronize();
  if (!res) {
    std::cout << "Fail" << std::endl;
    return -1;
  }
  std::cout << "Pass" << std::endl;
  return 0;
}
#endif
