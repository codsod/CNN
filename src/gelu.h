#ifndef __GELU_H__
#define __GELU_H__

#include "common.h"

// =============================================================================
// GELU 激活函数查找表 (LUT)，三路分支各一张表
// 仅需 GELU 时 include "gelu.h" 并用 GeluLUT<..., BRANCH>；多激活混用请用 lut.h
// =============================================================================
const LUT_T gelu_lut0[256] = {
    #include "ref/gelu/ms_conv1_branch0_gelu_lut_lut.txt"
};
const LUT_T gelu_lut1[256] = {
    #include "ref/gelu/ms_conv1_branch1_gelu_lut_lut.txt"
};
const LUT_T gelu_lut2[256] = {
    #include "ref/gelu/ms_conv1_branch2_gelu_lut_lut.txt"
};
// 注：BIND_STORAGE 需在函数内使用，此处省略；HLS 会对 const 数组推断 ROM/BRAM

// =============================================================================
// GELU 专用 LUT 模块，BRANCH=0,1,2 对应三路，编译期选表便于 HLS 推断 ROM
// =============================================================================
template<
    typename if_t,
    typename of_t,
    int N,
    int CHANNELS,
    int DIM1,
    int DIM2,
    int BRANCH
>
class GeluLUT {
public:
    GeluLUT() = default;
    static constexpr int VEC = SPATIAL_VEC;
    static constexpr int CHUNKS = (DIM2 + VEC - 1) / VEC;

    void do_lut_func(
        hls::stream<hls::vector<if_t, VEC>>& i_stream,
        hls::stream<hls::vector<of_t, VEC>>& o_stream)
    {
#pragma HLS INLINE off
        n_loop: for (int n = 0; n < N; n++) {
            channels_loop: for (int c = 0; c < CHANNELS; c++) {
                rows_loop: for (int row = 0; row < DIM1; row++) {
                    chunks_loop: for (int ck = 0; ck < CHUNKS; ck++) {
                        hls::vector<if_t, VEC> in = i_stream.read();
                        hls::vector<of_t, VEC> out;

                        elem_loop: for (int k = 0; k < VEC; k++) {
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL factor=8
                            int t = ck * VEC + k;
                            ap_uint<8> idx = in[k];
                            if (t < DIM2) {
                                if constexpr (BRANCH == 0)
                                    out[k] = gelu_lut0[idx];
                                else if constexpr (BRANCH == 1)
                                    out[k] = gelu_lut1[idx];
                                else
                                    out[k] = gelu_lut2[idx];
                            } else {
                                out[k] = (of_t)0;
                            }
                        }
                        o_stream.write(out);
                    }
                }
            }
        }
    }
};

#endif
