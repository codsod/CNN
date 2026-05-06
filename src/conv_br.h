#ifndef __CONV_BR_H__
#define __CONV_BR_H__

#include "common.h"
#include "conv2d.h"
#include "gelu.h"

static constexpr int CONV_BR_R_SHIFT = 26;
static constexpr int64_t CONV_BR_R_ONE = 1LL << CONV_BR_R_SHIFT;

static constexpr int64_t CONV_BR0_R_MULT =
    (int64_t)((double)QUANTSTUB_SCALE * (double)CONV_BR0_WEIGHT_SCALES /
                  (double)CONV_BR0_OUTPUT_SCALE * (double)CONV_BR_R_ONE +
              0.5);
static constexpr int64_t CONV_BR1_R_MULT =
    (int64_t)((double)QUANTSTUB_SCALE * (double)CONV_BR1_WEIGHT_SCALES /
                  (double)CONV_BR1_OUTPUT_SCALE * (double)CONV_BR_R_ONE +
              0.5);
static constexpr int64_t CONV_BR2_R_MULT =
    (int64_t)((double)QUANTSTUB_SCALE * (double)CONV_BR2_WEIGHT_SCALES /
                  (double)CONV_BR2_OUTPUT_SCALE * (double)CONV_BR_R_ONE +
              0.5);

static int conv_br_requant_fixed(int32_t acc, int32_t bias, int64_t mult, ZP_T out_zp)
{
#pragma HLS INLINE
    int64_t prod = (int64_t)(acc + bias) * mult;
    int64_t round = 1LL << (CONV_BR_R_SHIFT - 1);
    int64_t abs_prod = (prod >= 0) ? prod : -prod;
    int64_t q = (abs_prod + round) >> CONV_BR_R_SHIFT;
    return (int)((prod >= 0) ? q : -q) + (int)out_zp;
}

// =============================================================================
// 三路时序卷积分支 (Conv1d + 重量化 + GELU LUT)，每路独立权重/scale，输入共享
// 输入: N*T 个 vector<X_T, IN_DIM>；输出: N*T*OUT_CH 个 vector<X_T, IN_DIM>
// =============================================================================

class ConvBrBranch0 {
public:
    static constexpr int IN_CH = IN_CHANNELS;
    static constexpr int IN_D = IN_DIM;
    static constexpr int T_len = T;
    static constexpr int PAD = (BR0_KERNAL / 2);  // 输出 T+1 时在 t>=PAD-1 写
    static constexpr int K = BR0_KERNAL ;
    static constexpr int OUT_CH = BR0_CHANNELS ;
    static constexpr int OUT_LEN = T_len ;

    void do_conv_br(
        hls::stream<hls::vector<X_T, IN_D>>& i_stream,
        hls::stream<hls::vector<X_T, IN_D>>& o_stream)
    {
        X_T line_buf[IN_D][BR0_KERNAL];  // max K=64 for branch0
#pragma HLS ARRAY_PARTITION variable = line_buf complete dim = 1
#pragma HLS ARRAY_PARTITION variable = line_buf cyclic factor = 2 dim = 2

        const ZP_T out_zp = CONV_BR0_OUTPUT_ZP;

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
                    for (int oc = 0; oc < OUT_CH; oc++) {
                        hls::vector<X_T, IN_D> o_vec;
                        int32_t bias = CONV_BR0_BIAS[oc];

                        for (int c = 0; c < IN_D; c++) {

                            int32_t acc = 0;
                            for (int k = 0; k < K; k++) {
#pragma HLS UNROLL factor=2
                                int16_t x = (int16_t)(line_buf[c][k]) - (int16_t)QUANTSTUB_ZP;
                                int8_t w = CONV_BR0_WEIGHT[oc][0][0][k];
                                acc += (int32_t)x * (int32_t)w;
                            }

                            int v = conv_br_requant_fixed(acc, bias, CONV_BR0_R_MULT, out_zp);
                            uint8_t idx = (uint8_t)clamp(v, 0, 255);

                            o_vec[c] = gelu_lut0[idx];
                        }
                        o_stream.write(o_vec);
                    }
                }
            }
        }
    }
};

class ConvBrBranch1 {
public:
    static constexpr int IN_CH = IN_CHANNELS;
    static constexpr int IN_D = IN_DIM;
    static constexpr int T_len = T;
    static constexpr int PAD = (BR1_KERNAL / 2);  // 输出 T+1 时在 t>=PAD-1 写
    static constexpr int K = BR1_KERNAL;
    static constexpr int OUT_CH = BR1_CHANNELS ;
    static constexpr int OUT_LEN = T_len ;

    void do_conv_br(
        hls::stream<hls::vector<X_T, IN_D>>& i_stream,
        hls::stream<hls::vector<X_T, IN_D>>& o_stream)
    {
        X_T line_buf[IN_D][BR1_KERNAL];  // max K=64 for branch1
#pragma HLS ARRAY_PARTITION variable = line_buf complete dim = 1
#pragma HLS ARRAY_PARTITION variable = line_buf cyclic factor = 4 dim = 2
        const ZP_T out_zp = CONV_BR1_OUTPUT_ZP;

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
                    for (int oc = 0; oc < OUT_CH; oc++) {
                        hls::vector<X_T, IN_D> o_vec;
                        int32_t bias = CONV_BR1_BIAS[oc];

                        for (int c = 0; c < IN_D; c++) {

                            int32_t acc = 0;
                            for (int k = 0; k < K; k++) {
#pragma HLS UNROLL factor=4
                                int16_t x = (int16_t)(line_buf[c][k]) - (int16_t)QUANTSTUB_ZP;
                                int8_t w = CONV_BR1_WEIGHT[oc][0][0][k];
                                acc += (int32_t)x * (int32_t)w;
                            }
                            

                            int v = conv_br_requant_fixed(acc, bias, CONV_BR1_R_MULT, out_zp);
                            uint8_t idx = (uint8_t)clamp(v, 0, 255);

                            o_vec[c] = gelu_lut1[idx];

                        }
                        o_stream.write(o_vec);
                    }
                }
            }
        }
    }
};

class ConvBrBranch2 {
public:
    static constexpr int IN_CH = IN_CHANNELS;
    static constexpr int IN_D = IN_DIM;
    static constexpr int T_len = T;
    static constexpr int PAD =(BR2_KERNAL / 2);  // 输出 T+1 时在 t>=PAD-1 写
    static constexpr int K = BR2_KERNAL;
    static constexpr int OUT_CH = BR2_CHANNELS;
    static constexpr int OUT_LEN = T_len ;

    void do_conv_br(
        hls::stream<hls::vector<X_T, IN_D>>& i_stream,
        hls::stream<hls::vector<X_T, IN_D>>& o_stream)
    {
        X_T line_buf[IN_D][BR2_KERNAL];  // max K=64 for branch2
#pragma HLS ARRAY_PARTITION variable = line_buf complete dim = 1
#pragma HLS ARRAY_PARTITION variable = line_buf cyclic factor = 8 dim = 2
        const ZP_T out_zp = CONV_BR2_OUTPUT_ZP;

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
                    for (int oc = 0; oc < OUT_CH; oc++) {
                        hls::vector<X_T, IN_D> o_vec;
                        int32_t bias = CONV_BR2_BIAS[oc];

                        for (int c = 0; c < IN_D; c++) {

                            int32_t acc = 0;
                            for (int k = 0; k < K; k++) {
#pragma HLS UNROLL factor=8
                                int16_t x = (int16_t)(line_buf[c][k]) - (int16_t)QUANTSTUB_ZP;
                                int8_t w = CONV_BR2_WEIGHT[oc][0][0][k];
                                acc += (int32_t)x * (int32_t)w;
                            }                            
                            int v = conv_br_requant_fixed(acc, bias, CONV_BR2_R_MULT, out_zp);
                            uint8_t idx = (uint8_t)clamp(v, 0, 255);

                            o_vec[c] = gelu_lut2[idx];
                        }
                        o_stream.write(o_vec);
                    }
                }
            }
        }
    }
};

#endif
