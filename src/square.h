#ifndef __SQUARE_H__
#define __SQUARE_H__

#include "common.h"

// =============================================================================
// SQUARE 激活函数查找表 (LUT)，用于 FPGA HLS 部署时以查表替代 x^2
// 仅需 SQUARE 时 include "square.h" 并用 SquareLUT；多激活混用请用 lut.h
// =============================================================================
const LUT_T square_lut[256] = {
    #include "ref/square/lut_square_lut.txt"
};
// 注：BIND_STORAGE 需在函数内使用，此处省略；HLS 会对 const 数组推断 ROM/BRAM

// =============================================================================
// SQUARE 专用 LUT 模块（无 mode 分支，便于 HLS 推断 ROM）
// =============================================================================
template<
    typename if_t,
    typename of_t,
    int N,
    int CHANNELS,
    int DIM1,
    int DIM2
>
class SquareLUT {
public:
    SquareLUT() = default;

    void do_lut_func(
        hls::stream<hls::vector<if_t, DIM1 * DIM2>>& i_stream,
        hls::stream<hls::vector<of_t, DIM1 * DIM2>>& o_stream)
    {
#pragma HLS INLINE off
        n_loop: for (int n = 0; n < N; n++) {
            channels_loop: for (int c = 0; c < CHANNELS; c++) {
#pragma HLS PIPELINE II=1
                hls::vector<if_t, DIM1 * DIM2> in = i_stream.read();
                hls::vector<of_t, DIM1 * DIM2> out;

                elem_loop: for (int k = 0; k < DIM1 * DIM2; k++) {
#pragma HLS UNROLL
                    ap_uint<8> idx = in[k];
                    out[k] = square_lut[idx];
                }
                o_stream.write(out);
            }
        }
    }
};

#endif
