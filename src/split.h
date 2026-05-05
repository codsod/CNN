#ifndef __SPLIT_H__
#define __SPLIT_H__

#include "common.h"

// =============================================================================
// Split：单路 stream 复制为两路相同输出（如 log 输出 -> linear_cls / linear_hue 输入）
// 读入 N*CHANNELS 个 vector，依次写入 o_stream_0 与 o_stream_1
// =============================================================================
template<
    typename data_t,
    int N,
    int CHANNELS,
    int DIM1,
    int DIM2
>
class Split {
public:
    static constexpr int VEC_LEN = DIM1 * DIM2;

    Split() = default;

    void do_split_func(
        hls::stream<hls::vector<data_t, VEC_LEN>>& i_stream,
        hls::stream<hls::vector<data_t, VEC_LEN>>& o_stream_0,
        hls::stream<hls::vector<data_t, VEC_LEN>>& o_stream_1)
    {
#pragma HLS INLINE off
        n_loop: for (int n = 0; n < N; n++) {
            channels_loop: for (int c = 0; c < CHANNELS; c++) {
#pragma HLS PIPELINE II=1
                hls::vector<data_t, VEC_LEN> vec = i_stream.read();
                o_stream_0.write(vec);
                o_stream_1.write(vec);
            }
        }
    }
};

#endif
