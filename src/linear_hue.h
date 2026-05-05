#ifndef __LINEAR_HUE_H__
#define __LINEAR_HUE_H__

#include "linear_common.h"

// =============================================================================
// FC 色调头 (HUE)：权重、bias、输出 scale/zp，仅色调分支使用
// =============================================================================
constexpr W_T HUE_WEIGHT[FC_HUE_OUT][FlatDim] = {
    #include "ref/fc/hue/fc_hue_weight_int8.txt"
};
constexpr SCALE_T HUE_WEIGHT_SCALES = {
    #include "ref/fc/hue/fc_hue_weight_scales.txt"
};
constexpr B_T HUE_BIAS[FC_HUE_OUT] = {
    #include "ref/fc/hue/fc_hue_bias_int32.txt"
};
constexpr SCALE_T HUE_OUTPUT_SCALE = 0.040061067790;
constexpr ZP_T    HUE_OUTPUT_ZP    = 120;




template<
    typename if_t,
    typename of_t,
    typename i_dequan_t,
    typename o_quan_t,
    int N,
    int CHANNELS,
    int DIM1,
    int DIM2
>
class LinearHue {
public:
    i_dequan_t dequan_input[N][FlatDim];
    o_quan_t  dequan_output[N][FC_HUE_OUT];

    LinearHue() = default;

    void do_linear_func(
        hls::stream<hls::vector<if_t, DIM1 * DIM2>>& i_stream,
        hls::stream<hls::vector<of_t, FC_HUE_OUT>>& o_stream)
    {
        hls::vector<if_t, DIM1 * DIM2> in[N][CHANNELS];
        hls::vector<of_t, FC_HUE_OUT> out[N];

        batch_loop: for (int n = 0; n < N; n++) {
            outdim_loop: for (int out_dim = 0; out_dim < FC_HUE_OUT; out_dim++) {
                dequan_output[n][out_dim] = 0;
                channels_loop: for (int channels = 0; channels < CHANNELS; channels++) {
                    in[n][channels] = (out_dim == 0) ? i_stream.read() : in[n][channels];
#pragma HLS PIPELINE II=1
                    dim1_loop: for (int dim1 = 0; dim1 < DIM1; dim1++) {
                        dim2_loop: for (int dim2 = 0; dim2 < DIM2; dim2++) {
#pragma HLS UNROLL
                            int idx = channels * DIM1 * DIM2 + dim1 * DIM2 + dim2;
                            dequan_input[n][idx] = (i_dequan_t)((in[n][channels][dim1 * DIM2 + dim2] - FC_INPUT_ZP));
                            dequan_output[n][out_dim] += (o_quan_t)(dequan_input[n][idx] * HUE_WEIGHT[out_dim][idx]);
                        }
                    }
                }
                dequan_output[n][out_dim] = dequan_output[n][out_dim] + HUE_BIAS[out_dim];
                out[n][out_dim] = (of_t)((dequan_output[n][out_dim] * FC_INPUT_SCALE * HUE_WEIGHT_SCALES / HUE_OUTPUT_SCALE) + HUE_OUTPUT_ZP);
            }
            o_stream.write(out[n]);
        }
    }
};

#endif
