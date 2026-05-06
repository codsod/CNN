#ifndef __INT_COMMON_H__
#define __INT_COMMON_H__

// standard library
#include <cassert> //assert宏
#include <ctime>
#include <iostream>
#include <cstdint>
#include <random>
#include <fstream>
#include <numeric>

// hls library
#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_vector.h>

// user defined library
#include "adapter.h"
#include "utils.h"

using namespace std;

// some hyper parameters
constexpr int N                 = 1;
constexpr int IN_CHANNELS       = 1;
constexpr int IN_DIM            = 15;
constexpr int T                 = 410;
constexpr int BR0_CHANNELS      = 13;
constexpr int BR1_CHANNELS      = 13;
constexpr int BR2_CHANNELS      = 14;
constexpr int BR_DIM2           = T;
constexpr int SPATIAL_VEC       = 32;
constexpr int SPATIAL_CHUNKS    = (BR_DIM2 + SPATIAL_VEC - 1) / SPATIAL_VEC;
constexpr int CONCAT_CHANNELS   = (BR0_CHANNELS + BR1_CHANNELS + BR2_CHANNELS);
constexpr int LV_DIM1           = 1;
constexpr int T_pool            = 19;
constexpr int FlatDim           = (CONCAT_CHANNELS * LV_DIM1 * T_pool);
constexpr int FC_CLS_OUT        = 7;
constexpr int FC_HUE_OUT        = 2;

// y
constexpr int BR0_KERNAL        = 16;
constexpr int BR1_KERNAL        = 32;
constexpr int BR2_KERNAL        = 64;
// X
constexpr int SPATIAL_KERNAL    = 15;

constexpr int K_POOL            = 40;
constexpr int S_POOL            = 20;

// data types
constexpr int DW_X              = 8;
constexpr int DW_LUT            = 8;

// define data types
typedef ap_uint<DW_X             > X_T;
typedef ap_uint<DW_LUT           > LUT_T;
typedef int8_t                     W_T;
typedef int32_t                    B_T;
typedef int16_t                    I_DEQUAN_T;
typedef int16_t                    ZP_T;
typedef int64_t                    O_QUAN_T;
typedef float                      SCALE_T;


#endif
