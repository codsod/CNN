

#ifndef __INT_CONV2D_H__
#define __INT_CONV2D_H__

#include "common.h"

// QuantStub
constexpr SCALE_T QUANTSTUB_SCALE = 3.542000532150;
constexpr ZP_T QUANTSTUB_ZP = 106;

// BR0
constexpr W_T CONV_BR0_WEIGHT[BR0_CHANNELS][IN_CHANNELS][1][BR0_KERNAL] = {
#include "ref/conv_br/br0/ms_conv1_branch0_conv_weight_int8.txt"
}; // 208
constexpr SCALE_T CONV_BR0_WEIGHT_SCALES = {
#include "ref/conv_br/br0/ms_conv1_branch0_conv_weight_scales.txt"
}; // 1
constexpr B_T CONV_BR0_BIAS[BR0_CHANNELS] = {
#include "ref/conv_br/br0/ms_conv1_branch0_conv_bias_int32.txt"
}; // 13
constexpr SCALE_T CONV_BR0_OUTPUT_SCALE = 0.057105172426;
constexpr ZP_T CONV_BR0_OUTPUT_ZP = 130;

// BR1
constexpr W_T CONV_BR1_WEIGHT[BR1_CHANNELS][IN_CHANNELS][1][BR1_KERNAL] = {
#include "ref/conv_br/br1/ms_conv1_branch1_conv_weight_int8.txt"
}; // 416
constexpr SCALE_T CONV_BR1_WEIGHT_SCALES = {
#include "ref/conv_br/br1/ms_conv1_branch1_conv_weight_scales.txt"
}; // 1
constexpr B_T CONV_BR1_BIAS[BR1_CHANNELS] = {
#include "ref/conv_br/br1/ms_conv1_branch1_conv_bias_int32.txt"
}; // 13
constexpr SCALE_T CONV_BR1_OUTPUT_SCALE = 0.061033196747;
constexpr ZP_T CONV_BR1_OUTPUT_ZP = 125;

// BR2
constexpr W_T CONV_BR2_WEIGHT[BR2_CHANNELS][IN_CHANNELS][1][BR2_KERNAL] = {
#include "ref/conv_br/br2/ms_conv1_branch2_conv_weight_int8.txt"
}; // 896
constexpr SCALE_T CONV_BR2_WEIGHT_SCALES = {
#include "ref/conv_br/br2/ms_conv1_branch2_conv_weight_scales.txt"
}; // 1
constexpr B_T CONV_BR2_BIAS[BR2_CHANNELS] = {
#include "ref/conv_br/br2/ms_conv1_branch2_conv_bias_int32.txt"
}; // 14
constexpr SCALE_T CONV_BR2_OUTPUT_SCALE = 0.064764507115;
constexpr ZP_T CONV_BR2_OUTPUT_ZP = 127;

// spatial (conv2: 40x15x411 -> 40x1x411, kernel 15x1)
constexpr SCALE_T CONV_SPATIAL_INPUT_SCALE = 0.034158173949;
constexpr ZP_T CONV_SPATIAL_INPUT_ZP = 5;
constexpr W_T CONV_SPATIAL_WEIGHT[CONCAT_CHANNELS][CONCAT_CHANNELS][SPATIAL_KERNAL][1] = {
#include "ref/conv_spatial/conv2_weight_int8.txt"
};
constexpr SCALE_T CONV_SPATIAL_WEIGHT_SCALES = {
#include "ref/conv_spatial/conv2_weight_scales.txt"
};
constexpr B_T CONV_SPATIAL_BIAS[CONCAT_CHANNELS] = {
#include "ref/conv_spatial/conv2_bias_int32.txt"
};
constexpr SCALE_T CONV_SPATIAL_OUTPUT_SCALE = 0.059495572001;
constexpr ZP_T CONV_SPATIAL_OUTPUT_ZP = 132;

static constexpr int CONV_SPATIAL_R_SHIFT = 26;
static constexpr int64_t CONV_SPATIAL_R_ONE = 1LL << CONV_SPATIAL_R_SHIFT;
static constexpr int64_t CONV_SPATIAL_R_MULT =
    (int64_t)((double)CONV_SPATIAL_INPUT_SCALE * (double)CONV_SPATIAL_WEIGHT_SCALES /
                  (double)CONV_SPATIAL_OUTPUT_SCALE * (double)CONV_SPATIAL_R_ONE +
              0.5);
static constexpr int CONV_SPATIAL_ROW_TILE = 5;
static_assert(SPATIAL_KERNAL % CONV_SPATIAL_ROW_TILE == 0, "SPATIAL_KERNAL must be divisible by tile");

// 银行家舍入
inline int round_to_even(float x)
{
    int base = (int)x;
    float frac = x - (float)base;

    if (x < 0 && frac != 0.0f)
    {
        // 对负数修正，保证 base 是 floor(x)
        base -= 1;
        frac = x - (float)base;
    }

    if (frac < 0.5f)
        return base;
    if (frac > 0.5f)
        return base + 1;

    return (base & 1) ? (base + 1) : base;
}

static int conv_spatial_requant_fixed(int32_t acc)
{
#pragma HLS INLINE
    int64_t prod = (int64_t)acc * CONV_SPATIAL_R_MULT;
    bool neg = prod < 0;
    uint64_t mag = neg ? (uint64_t)(-prod) : (uint64_t)prod;
    uint64_t q = mag >> CONV_SPATIAL_R_SHIFT;
    uint64_t rem = mag & (CONV_SPATIAL_R_ONE - 1);
    uint64_t half = (uint64_t)1 << (CONV_SPATIAL_R_SHIFT - 1);

    if (rem > half || (rem == half && (q & 1)))
        q++;

    int64_t signed_q = neg ? -(int64_t)q : (int64_t)q;
    return (int)(signed_q + (int64_t)CONV_SPATIAL_OUTPUT_ZP);
}

// =============================================================================
// 空间卷积 ConvSpatial：输入 (N, CONCAT_CHANNELS, IN_DIM, BR_DIM2)，核 (SPATIAL_KERNAL,1)
// 输出 (N, CONCAT_CHANNELS, 1, BR_DIM2)，即每通道每时刻一个标量
// =============================================================================
template <typename if_t, typename of_t>
class ConvSpatial
{
public:
    static constexpr int IN_D = IN_DIM;
    static constexpr int BR_D = BR_DIM2;
    static constexpr int IN_VEC = SPATIAL_VEC; // 32 per chunk
    static constexpr int OUT_VEC = SPATIAL_VEC;
    static constexpr int OUT_TT = SPATIAL_CHUNKS;

    void do_conv_spatial(
        hls::stream<hls::vector<if_t, IN_VEC>> &i_stream,
        hls::stream<hls::vector<of_t, OUT_VEC>> &o_stream)
    {
        if_t buf[CONCAT_CHANNELS][IN_D][BR_D];
#pragma HLS BIND_STORAGE variable = buf type = ram_2p impl = bram
#pragma HLS ARRAY_PARTITION variable = buf cyclic factor=5 dim=2

        for (int n = 0; n < N; n++)
        {
            for (int ic = 0; ic < CONCAT_CHANNELS; ic++)
            {
                for (int row = 0; row < IN_D; row++)
                {
                    for (int ck = 0; ck < OUT_TT; ck++)
                    {
                        hls::vector<if_t, IN_VEC> v = i_stream.read();
                        for (int k = 0; k < IN_VEC; k++)
                        {
#pragma HLS PIPELINE II = 1
                            int t_idx = ck * IN_VEC + k;
                            if (t_idx < BR_D)
                                buf[ic][row][t_idx] = v[k];
                        }
                    }
                }
            }

            for (int oc = 0; oc < CONCAT_CHANNELS; oc++)
            {
                for (int ck = 0; ck < OUT_TT; ck++)
                {
                    hls::vector<of_t, OUT_VEC> out_vec;
                    for (int k = 0; k < OUT_VEC; k++)
                    {
                        int t = ck * OUT_VEC + k;
                        int32_t acc = CONV_SPATIAL_BIAS[oc];
                        if (t < BR_D)
                        {
                            for (int ic = 0; ic < CONCAT_CHANNELS; ic++)
                            {
                                for (int row0 = 0; row0 < SPATIAL_KERNAL; row0 += CONV_SPATIAL_ROW_TILE)
                                {
#pragma HLS PIPELINE II = 1
                                    for (int rt = 0; rt < CONV_SPATIAL_ROW_TILE; rt++)
                                    {
#pragma HLS UNROLL
                                        int row = row0 + rt;
                                        int16_t x = (int16_t)buf[ic][row][t] - (int16_t)CONV_SPATIAL_INPUT_ZP;
                                        int8_t w = CONV_SPATIAL_WEIGHT[oc][ic][row][0];
                                        acc += (int32_t)x * (int32_t)w;
                                    }
                                }
                            }
                            int v = conv_spatial_requant_fixed(acc);
                            out_vec[k] = (of_t)clamp(v, 0, 255);
                        }
                        else
                        {
                            out_vec[k] = (of_t)0;
                        }
                    }
                    o_stream.write(out_vec);
                }
            }
        }
    }
};

#endif
