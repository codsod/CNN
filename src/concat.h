#ifndef __CONCAT_H__
#define __CONCAT_H__

#include "common.h"

// BRx GELU output domain
constexpr SCALE_T CONV_BR0_GELU_OUTPUT_SCALE = 0.027724377811;
constexpr ZP_T CONV_BR0_GELU_OUTPUT_ZP = 6;
constexpr SCALE_T CONV_BR1_GELU_OUTPUT_SCALE = 0.035352136940;
constexpr ZP_T CONV_BR1_GELU_OUTPUT_ZP = 5;
constexpr SCALE_T CONV_BR2_GELU_OUTPUT_SCALE = 0.035856030881;
constexpr ZP_T CONV_BR2_GELU_OUTPUT_ZP = 5;

// 真实的 concat(QFunctional) 统一量化域
constexpr SCALE_T CONCAT_OUTPUT_SCALE = 0.034158173949;
constexpr ZP_T CONCAT_OUTPUT_ZP = 5;

// 银行家舍入
inline int round_to_even_concat(float x)
{
    int base = (int)x;
    float frac = x - (float)base;

    if (x < 0 && frac != 0.0f)
    {
        base -= 1;
        frac = x - (float)base;
    }

    if (frac < 0.5f)
        return base;
    if (frac > 0.5f)
        return base + 1;

    return (base & 1) ? (base + 1) : base;
}

// =============================================================================
// Concat：三路 branch 输出先重标定到统一量化域，再按通道维拼成一路，供 CONV_SPATIAL 消费
// 输入：br0 的 411 个 vector<X_T, BR0_CH*IN_DIM>，br1 同，br2 的 411 个 vector<X_T, BR2_CH*IN_DIM>
// 输出：CONCAT_CHANNELS*IN_DIM 个 vector<X_T, BR_DIM2>，顺序为 br0 ch0 row0..row14, br1..., br2...
// =============================================================================
template <typename data_t>
class Concat
{
public:
    static constexpr int IN_D = IN_DIM;
    static constexpr int BR_D = BR_DIM2;
    static constexpr int CH_OUT = CONCAT_CHANNELS;
    static constexpr int OUT_VEC_LEN = BR_D;

    static data_t requant_to_concat_domain(data_t x, SCALE_T in_scale, ZP_T in_zp)
    {
        float scaled =
            ((int)x - (int)in_zp) * ((float)in_scale / (float)CONCAT_OUTPUT_SCALE) +
            (float)CONCAT_OUTPUT_ZP;
        int v = round_to_even_concat(scaled);
        return (data_t)clamp(v, 0, 255);
    }

    void do_concat(
        hls::stream<hls::vector<data_t, BR0_CHANNELS * IN_D>> &i_stream_0,
        hls::stream<hls::vector<data_t, BR1_CHANNELS * IN_D>> &i_stream_1,
        hls::stream<hls::vector<data_t, BR2_CHANNELS * IN_D>> &i_stream_2,
        hls::stream<hls::vector<data_t, OUT_VEC_LEN>> &o_stream)
    {
        data_t buf0[BR0_CHANNELS][IN_D][BR_D];
        data_t buf1[BR1_CHANNELS][IN_D][BR_D];
        data_t buf2[BR2_CHANNELS][IN_D][BR_D];
#pragma HLS BIND_STORAGE variable = buf0 type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = buf1 type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = buf2 type = ram_2p impl = bram

        for (int n = 0; n < N; n++)
        {
            for (int t = 0; t < BR_D; t++)
            {
#pragma HLS PIPELINE II = 1
                hls::vector<data_t, BR0_CHANNELS * IN_D> v0 = i_stream_0.read();
                hls::vector<data_t, BR1_CHANNELS * IN_D> v1 = i_stream_1.read();
                hls::vector<data_t, BR2_CHANNELS * IN_D> v2 = i_stream_2.read();

                for (int oc = 0; oc < BR0_CHANNELS; oc++)
                {
                    for (int c = 0; c < IN_D; c++)
                        buf0[oc][c][t] = requant_to_concat_domain(
                            v0[oc * IN_D + c], CONV_BR0_GELU_OUTPUT_SCALE, CONV_BR0_GELU_OUTPUT_ZP);
                }
                for (int oc = 0; oc < BR1_CHANNELS; oc++)
                {
                    for (int c = 0; c < IN_D; c++)
                        buf1[oc][c][t] = requant_to_concat_domain(
                            v1[oc * IN_D + c], CONV_BR1_GELU_OUTPUT_SCALE, CONV_BR1_GELU_OUTPUT_ZP);
                }
                for (int oc = 0; oc < BR2_CHANNELS; oc++)
                {
                    for (int c = 0; c < IN_D; c++)
                        buf2[oc][c][t] = requant_to_concat_domain(
                            v2[oc * IN_D + c], CONV_BR2_GELU_OUTPUT_SCALE, CONV_BR2_GELU_OUTPUT_ZP);
                }
            }

            for (int ch = 0; ch < BR0_CHANNELS; ch++)
            {
                for (int r = 0; r < IN_D; r++)
                {
                    hls::vector<data_t, OUT_VEC_LEN> out_vec;
                    for (int t = 0; t < BR_D; t++)
#pragma HLS PIPELINE
                        out_vec[t] = buf0[ch][r][t];
                    o_stream.write(out_vec);
                }
            }
            for (int ch = 0; ch < BR1_CHANNELS; ch++)
            {
                for (int r = 0; r < IN_D; r++)
                {
                    hls::vector<data_t, OUT_VEC_LEN> out_vec;
                    for (int t = 0; t < BR_D; t++)
#pragma HLS PIPELINE
                        out_vec[t] = buf1[ch][r][t];
                    o_stream.write(out_vec);
                }
            }
            for (int ch = 0; ch < BR2_CHANNELS; ch++)
            {
                for (int r = 0; r < IN_D; r++)
                {
                    hls::vector<data_t, OUT_VEC_LEN> out_vec;
                    for (int t = 0; t < BR_D; t++)
#pragma HLS PIPELINE
                        out_vec[t] = buf2[ch][r][t];
                    o_stream.write(out_vec);
                }
            }
        }
    }
};

#endif
