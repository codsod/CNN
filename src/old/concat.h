#ifndef __CONCAT_H__
#define __CONCAT_H__

#include "common.h"


// =============================================================================
// Concat：三路 branch 输出按通道维拼成一路，供 CONV_SPATIAL 消费
// 输入：br0 的 411 个 vector<X_T, BR0_CH*IN_DIM>，br1 同，br2 的 411 个 vector<X_T, BR2_CH*IN_DIM>
// 输出：CONCAT_CHANNELS 个 vector<X_T, IN_DIM*BR_DIM2>，顺序为 br0 ch0..ch12, br1 ch0..ch12, br2 ch0..ch13
// =============================================================================
template<typename data_t>
class Concat {
public:
    static constexpr int IN_D   = IN_DIM;
    static constexpr int BR_D   = BR_DIM2;
    static constexpr int CH_OUT = CONCAT_CHANNELS;
    static constexpr int OUT_VEC_LEN = IN_D * BR_D;  // 15*411

    void do_concat(
        hls::stream<hls::vector<data_t, BR0_CHANNELS * IN_D>>& i_stream_0,
        hls::stream<hls::vector<data_t, BR1_CHANNELS * IN_D>>& i_stream_1,
        hls::stream<hls::vector<data_t, BR2_CHANNELS * IN_D>>& i_stream_2,
        hls::stream<hls::vector<data_t, OUT_VEC_LEN>>& o_stream)
    {
        data_t buf0[BR0_CHANNELS][IN_D][BR_D];
        data_t buf1[BR1_CHANNELS][IN_D][BR_D];
        data_t buf2[BR2_CHANNELS][IN_D][BR_D];
#pragma HLS ARRAY_PARTITION variable=buf0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=buf1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=buf2 complete dim=0

        for (int n = 0; n < N; n++) {
            for (int t = 0; t < BR_D; t++) {
#pragma HLS PIPELINE II=1
                hls::vector<data_t, BR0_CHANNELS * IN_D> v0 = i_stream_0.read();
                hls::vector<data_t, BR1_CHANNELS * IN_D> v1 = i_stream_1.read();
                hls::vector<data_t, BR2_CHANNELS * IN_D> v2 = i_stream_2.read();

                for (int oc = 0; oc < BR0_CHANNELS; oc++) {
                    for (int c = 0; c < IN_D; c++)
                        buf0[oc][c][t] = v0[oc * IN_D + c];
                }
                for (int oc = 0; oc < BR1_CHANNELS; oc++) {
                    for (int c = 0; c < IN_D; c++)
                        buf1[oc][c][t] = v1[oc * IN_D + c];
                }
                for (int oc = 0; oc < BR2_CHANNELS; oc++) {
                    for (int c = 0; c < IN_D; c++)
                        buf2[oc][c][t] = v2[oc * IN_D + c];
                }
            }

            for (int ch = 0; ch < BR0_CHANNELS; ch++) {
                hls::vector<data_t, OUT_VEC_LEN> out_vec;
                for (int r = 0; r < IN_D; r++)
                    for (int t = 0; t < BR_D; t++)
                        out_vec[r * BR_D + t] = buf0[ch][r][t];
                o_stream.write(out_vec);
            }
            for (int ch = 0; ch < BR1_CHANNELS; ch++) {
                hls::vector<data_t, OUT_VEC_LEN> out_vec;
                for (int r = 0; r < IN_D; r++)
                    for (int t = 0; t < BR_D; t++)
                        out_vec[r * BR_D + t] = buf1[ch][r][t];
                o_stream.write(out_vec);
            }
            for (int ch = 0; ch < BR2_CHANNELS; ch++) {
                hls::vector<data_t, OUT_VEC_LEN> out_vec;
                for (int r = 0; r < IN_D; r++)
                    for (int t = 0; t < BR_D; t++)
                        out_vec[r * BR_D + t] = buf2[ch][r][t];
                o_stream.write(out_vec);
            }
        }
    }
};

#endif