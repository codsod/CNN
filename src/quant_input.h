#ifndef __QUANT_INPUT_H__
#define __QUANT_INPUT_H__

#include "common.h"

// =============================================================================
// 输入量化：fp32 -> quint8，与 QuantStub 一致，供 CONV_BR 等后续使用
// 公式: q = clamp(round(x / scale) + zp, 0, 255)
// =============================================================================
constexpr SCALE_T INPUT_QUANT_SCALE = 3.542000532150;
constexpr ZP_T    INPUT_QUANT_ZP   = 106;

template<typename of_t>
inline of_t quantize_one(float x) {
    float q = x / (float)INPUT_QUANT_SCALE + (float)INPUT_QUANT_ZP;
    int v = (int)(q + (q >= 0 ? 0.5f : -0.5f));
    return (of_t)clamp(v, 0, 255);
}

template<typename of_t, int VEC_LEN>
class QuantInput {
public:
    QuantInput() = default;

    void do_quantize(
        hls::stream<hls::vector<float, VEC_LEN>>& i_stream,
        hls::stream<hls::vector<of_t, VEC_LEN>>& o_stream)
    {
#pragma HLS INLINE off
        for (int n = 0; n < N; n++) {
            for (int t = 0; t < T; t++) {
#pragma HLS PIPELINE II=1
                hls::vector<float, VEC_LEN> in = i_stream.read();
                hls::vector<of_t, VEC_LEN> out;
                for (int k = 0; k < VEC_LEN; k++) {
#pragma HLS UNROLL
                    out[k] = quantize_one<of_t>(in[k]);
                }
                o_stream.write(out);
            }
        }
    }
};

#endif
