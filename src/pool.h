#ifndef __INT_POOL_H__
#define __INT_POOL_H__

#include "common.h"

template<
    class if_t,
    class of_t,
    

    int N,
    int CHANNELS,
    int DIM1,
    int DIM2,
    int OUTDIM,
    int K,
    int S
>
class POOL{
public:
    static constexpr int VEC = SPATIAL_VEC;
    static constexpr int CHUNKS = (DIM2 + VEC - 1) / VEC;
    static constexpr int SEGMENTS = OUTDIM + 1;
    static_assert(K == 40, "POOL rounding assumes K == 40");
    static_assert(K == 2 * S, "POOL segment accumulation assumes K == 2*S");

    POOL(){};

    void do_pool_func(hls::stream<hls::vector<if_t, VEC> >& i_stream,hls::stream<hls::vector<of_t, DIM1 * OUTDIM>>& o_stream){
        n_loop:for(int n = 0 ; n < N ; n++){
            channels_loop:for(int chan = 0 ;chan < CHANNELS ; chan ++){
                hls::vector<of_t, DIM1 * OUTDIM> out;
                #pragma HLS RESET variable=out off

                dim1_loop:for (int dim1 = 0; dim1 < DIM1; dim1++) {
                    uint16_t seg_sum[SEGMENTS];
                    #pragma HLS ARRAY_PARTITION variable=seg_sum complete dim=1
                    #pragma HLS RESET variable=seg_sum off

                    init_seg_loop:for (int seg = 0; seg < SEGMENTS; seg++) {
                        #pragma HLS UNROLL
                        seg_sum[seg] = 0;
                    }

                    load_chunks_loop:for (int ck = 0; ck < CHUNKS; ck++) {
                        hls::vector<if_t, VEC> in = i_stream.read();
                        load_elem_loop:for (int k = 0; k < VEC; k++) {
#pragma HLS PIPELINE II=1
                            int t = ck * VEC + k;
                            int seg = t / S;
                            if (t < DIM2 && seg < SEGMENTS)
                                seg_sum[seg] += (uint16_t)in[k];
                        }
                    }

                    outdim_loop:for (int outdim = 0 ;outdim < OUTDIM ;outdim++){
                        #pragma HLS PIPELINE II=1
                        uint16_t sum = seg_sum[outdim] + seg_sum[outdim + 1];

                        // 取整方式须与量化 trace 一致。当前 ref（quant/acc0304.ipynb VerificationProbe）
                        // 导出 trace_pool1 时 PyTorch 量化 AvgPool 实际为截断，故用 sum/K。
                        // 若改用四舍五入与 pool1_meta（同 scale）对齐，可改为: (sum + K/2) / K
                        // 实际算法中采取的是银行家舍入，关键区别在于0.5的处理：银行家舍入在0.5时向最近的偶数舍入，而普通四舍五入在0.5时总是向上舍入。
                        uint16_t q = ((uint32_t)sum * 3277) >> 17; // exact floor(sum / 40) for sum <= 10200
                        uint16_t r = sum - ((q << 5) + (q << 3));


                        if (2* r > K) {
                            out[dim1 * OUTDIM + outdim] = q + 1;
                        } else if (2* r < K) {
                            out[dim1 * OUTDIM + outdim] = q;
                        } else {
                            out[dim1 * OUTDIM + outdim] = (q & 1) ? (q + 1) : q;
                        }

                        // out[dim1 * OUTDIM + outdim] = sum/K;
                    }
                }
                o_stream.write(out);
            }
        }
    }
};

#endif
