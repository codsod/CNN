#ifndef __CONV_BR_H__
#define __CONV_BR_H__

#include "common.h"
#include "conv2d.h"
#include "gelu.h"

// =============================================================================
// 三路时序卷积分支 (Conv1d + 重量化 + GELU LUT)，每路独立权重/scale，输入共享
// 输入: N*T 个 vector<X_T, IN_DIM>；输出: N*(T+1) 个 vector<X_T, OUT_CH*IN_DIM>
// =============================================================================
template<int BRANCH_ID>
class ConvBrBranch {
public:
    static constexpr int IN_CH = IN_CHANNELS;
    static constexpr int IN_D = IN_DIM;
    static constexpr int T_len = T;
    static constexpr int PAD = (BRANCH_ID == 0) ? (BR0_KERNAL / 2) : (BRANCH_ID == 1) ? (BR1_KERNAL / 2) : (BR2_KERNAL / 2);  // 输出 T+1 时在 t>=PAD-1 写
    static constexpr int K = (BRANCH_ID == 0) ? BR0_KERNAL : (BRANCH_ID == 1) ? BR1_KERNAL : BR2_KERNAL;
    static constexpr int OUT_CH = (BRANCH_ID == 0) ? BR0_CHANNELS : (BRANCH_ID == 1) ? BR1_CHANNELS : BR2_CHANNELS;
    static constexpr int OUT_LEN = T_len ;

    void do_conv_br(
        hls::stream<hls::vector<X_T, IN_D>>& i_stream,
        hls::stream<hls::vector<X_T, OUT_CH * IN_D>>& o_stream)
    {
        X_T line_buf[IN_D][BR2_KERNAL];  // max K=64 for branch2
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1。  // line_buf优化

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < IN_D; c++) {
                for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
                    line_buf[c][k] = (X_T)QUANTSTUB_ZP;
                }
            }

            for (int t = 0; t < T_len + PAD; t++) {
#pragma HLS PIPELINE 

                hls::vector<X_T, IN_D> new_data;
                if (t < T_len) new_data = i_stream.read();

                for (int c = 0; c < IN_D; c++) {
#pragma HLS UNROLL
                    for (int k = 0; k < K - 1; k++)
                        line_buf[c][k] = line_buf[c][k + 1];
                    line_buf[c][K - 1] = (t < T_len) ? new_data[c] : (X_T)QUANTSTUB_ZP;
                }

                if (t >= PAD ) {
                    hls::vector<X_T, OUT_CH * IN_D> o_vec;

                    for (int oc = 0; oc < OUT_CH; oc++) {

                        for (int c = 0; c < IN_D; c++) {

                            int32_t acc = 0;
                            for (int k = 0; k < K; k++) {
#pragma HLS UNROLL factor=4
                                int16_t x = (int16_t)(line_buf[c][k]) - (int16_t)QUANTSTUB_ZP;
                                int8_t w = (BRANCH_ID == 0) ? CONV_BR0_WEIGHT[oc][0][0][k]
                                          : (BRANCH_ID == 1) ? CONV_BR1_WEIGHT[oc][0][0][k]
                                          : CONV_BR2_WEIGHT[oc][0][0][k];
                                acc += (int32_t)x * (int32_t)w;
                            }
                            int32_t bias = (BRANCH_ID == 0) ? CONV_BR0_BIAS[oc]
                                         : (BRANCH_ID == 1) ? CONV_BR1_BIAS[oc]
                                         : CONV_BR2_BIAS[oc];
                            float w_scale = (BRANCH_ID == 0) ? CONV_BR0_WEIGHT_SCALES
                                           : (BRANCH_ID == 1) ? CONV_BR1_WEIGHT_SCALES
                                           : CONV_BR2_WEIGHT_SCALES;
                            float out_scale = (BRANCH_ID == 0) ? CONV_BR0_OUTPUT_SCALE
                                            : (BRANCH_ID == 1) ? CONV_BR1_OUTPUT_SCALE
                                            : CONV_BR2_OUTPUT_SCALE;
                            ZP_T out_zp = (BRANCH_ID == 0) ? CONV_BR0_OUTPUT_ZP
                                          : (BRANCH_ID == 1) ? CONV_BR1_OUTPUT_ZP
                                          : CONV_BR2_OUTPUT_ZP;
                            float r_scale = (float)QUANTSTUB_SCALE * w_scale / out_scale;
                            float scaled = (float)(acc + bias) * r_scale + (float)out_zp;
                            int v = (int)(scaled + (scaled >= 0 ? 0.5f : -0.5f));
                            uint8_t idx = (uint8_t)clamp(v, 0, 255);

                            if constexpr (BRANCH_ID == 0)
                                o_vec[oc * IN_D + c] = gelu_lut0[idx];
                            else if constexpr (BRANCH_ID == 1)
                                o_vec[oc * IN_D + c] = gelu_lut1[idx];
                            else
                                o_vec[oc * IN_D + c] = gelu_lut2[idx];
                        }
                    }
                    o_stream.write(o_vec);
                }
            }
        }
    }
};

#endif

