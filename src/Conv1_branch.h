#ifndef __CONV1_BRANCH_H__
#define __CONV1_BRANCH_H__

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include <hls_vector.h>
#include "utils.h"
#include <algorithm>
#include "common.h"

constexpr uint8_t gelu_lut0         [] = {
    #include "./ref/ms_conv1_branch0_gelu_lut_lut.txt"
};

constexpr uint8_t gelu_lut1         [] = {
    #include "./ref/ms_conv1_branch1_gelu_lut_lut.txt"
};

constexpr uint8_t gelu_lut2         [] = {
    #include "./ref/ms_conv1_branch2_gelu_lut_lut.txt"
};

/**
 * @tparam if_t     输入特征类型 (int8_t)
 * @tparam w_t      权重类型 (int8_t)
 * @tparam b_t      偏置类型 (int32_t)
 * @tparam o_t      卷积输出类型 (int32_t)
 * @tparam lut_t    LUT输入输出类型 (int8_t)
 * @tparam N        batch
 * @tparam CHANS    电极通道数 (15)
 * @tparam T        输入时间步长
 * @tparam OUT_CH   输出通道数 (13，14)
 * @tparam K        卷积核大小 (16, 32, 或 64)
 */
template<
    class if_t,
    class w_t,
    class b_t,
    class o_t,
    class lut_t,
    int N,
    int CHANS,
    int T,    
    int OUT_CH,
    int K,
    int ID
>
class CONV1_BRANCH_CORE {
public:
    CONV1_BRANCH_CORE() {}

    void do_conv1(
        hls::stream<hls::vector<if_t, CHANS          >>& i_stream,    // 逐个时间点读入 15 通道数据
        hls::stream<hls::vector<w_t, OUT_CH*K        >>& w_stream,
        hls::stream<hls::vector<b_t, OUT_CH          >>& b_stream,
        hls::stream<hls::vector<lut_t, OUT_CH * CHANS>>& o_stream,    // 每一个branch输出
        const int input_zp ,                                           // 输入零点
        const int weight_zp ,                                           // 输入零点
        const float R_scale,                                          // 重量化系数，不需要自己算
        const int Conv1_out_zp                                              // 输出零点
    ) {
        //输入维度: (N, T, 1, CHANS)
        // Line Buffer: 缓存 K 个时间点的数据，每行 15 个通道
        // 使用 ARRAY_PARTITION 使得 15 个通道可以并行访问
        if_t line_buf[CHANS][K];
        #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=0
        w_t local_w[OUT_CH][K];
        #pragma HLS ARRAY_PARTITION variable=local_w complete dim=0  
        b_t local_b[OUT_CH];
        #pragma HLS ARRAY_PARTITION variable=local_b complete dim=0

        // 从流中预载权重和偏置
        hls::vector<w_t, OUT_CH * K> w_vec = w_stream.read();
        hls::vector<b_t, OUT_CH    > b_vec = b_stream.read();
        
        for(int oc=0; oc<OUT_CH; oc++) {
            #pragma HLS unroll
            local_b[oc] = b_vec[oc];
            for(int k=0; k<K; k++) {
                #pragma HLS unroll
                local_w[oc][k] = w_vec[oc * K + k];
            }
        }

        const int PAD = K / 2;
        
        for (int n = 0; n < N; n++) {

            for(int c=0; c<CHANS; c++) {
                for(int k=0; k<K; k++) {
                    #pragma HLS unroll
                    line_buf[c][k] = (if_t)input_zp; 
                }
            }

        // 卷积滑动窗口逻辑
            for (int t = 0; t < T + PAD; t++) {
                #pragma HLS pipeline II=1

                // 1. 更新滑动窗口 (Shift left & Read new)
                hls::vector<if_t, CHANS> new_data;
                if (t < T) new_data = i_stream.read();

                for (int c = 0; c < CHANS; c++) {
                    #pragma HLS unroll
                    for (int k = 0; k < K - 1; k++) {
                        line_buf[c][k] = line_buf[c][k + 1];
                    }
                    // 如果超过 T 则补 input_zp (量化域的"零")
                    line_buf[c][K - 1] = (t < T) ? new_data[c] : (if_t)input_zp;
                }

                //最终产生的输出矩阵的维度是 (N, T+1, OUT_CH, CHANS)
                // 2. 产生输出 (当窗口填满 PAD 之后开始输出)
                if (t >= PAD - 1) {
                    hls::vector<lut_t, OUT_CH * CHANS> o_vec;

                    // 遍历 13 个卷积核 (输出特征)
                    for (int oc = 0; oc < OUT_CH; oc++) {
                        #pragma HLS unroll

                        // 遍历 15 个电极通道
                        for (int c = 0; c < CHANS; c++) {
                            #pragma HLS unroll
                        
                            // --- 卷积计算 (MAC) ---
                            o_t acc = 0;
                            for (int k = 0; k < K; k++) {
                                #pragma HLS unroll
                                acc += (o_t)((line_buf[c][k]-input_zp) * (local_w[oc][k]-weight_zp));
                            }

                            // --- 加偏置 ---
                            o_t biased_acc = acc + local_b[oc];

                            // --- 重量化 (Re-quantization) ---
                            lut_t scaled = (lut_t)biased_acc * R_scale + Conv1_out_zp;
                            // 使用 +0.5f 模拟 round
                            lut_t quantized = (lut_t)(scaled + (scaled >= 0 ? 0.5f : -0.5f));
                            uint8_t idx = clamp(quantized+128, 0, 255);
                            
                            if(ID == 0)  {
                                // --- GELU LUT 查表 ---
                                o_vec[oc * CHANS + c] = gelu_lut0[idx];
                            } else if (ID == 1) {
                                // --- GELU LUT 查表 ---
                                o_vec[oc * CHANS + c] = gelu_lut1[idx];
                            } else
                            {
                            // --- GELU LUT 查表 ---
                            o_vec[oc * CHANS + c] = gelu_lut2[idx];
                            }
                        }
                    }
                    o_stream.write(o_vec);
                }
            }
        }
    }
};

#endif