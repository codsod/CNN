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

    POOL(){};

    void do_pool_func(hls::stream<hls::vector<if_t, DIM1 * DIM2> >& i_stream,hls::stream<hls::vector<of_t, DIM1 * OUTDIM>>& o_stream){
        
        n_loop:for(int n = 0 ; n < N ; n++){
            channels_loop:for(int chan = 0 ;chan < CHANNELS ; chan ++){
                #pragma HLS PIPELINE II=1
                hls::vector<if_t, DIM1 * DIM2> in = i_stream.read();
                hls::vector<of_t, DIM1 * OUTDIM> out;
                dim1_loop:for (int dim1 = 0; dim1 < DIM1; dim1++) {
                    outdim_loop:for (int outdim = 0 ;outdim < OUTDIM ;outdim++){
                        uint16_t sum=0;
                        k_loop:for(int k = 0; k < K ; k++){
                            #pragma HLS UNROLL factor=4
                            sum += in[dim1 * DIM2 + outdim * S + k];
                        }
                        // 取整方式须与量化 trace 一致。当前 ref（quant/acc0304.ipynb VerificationProbe）
                        // 导出 trace_pool1 时 PyTorch 量化 AvgPool 实际为截断，故用 sum/K。
                        // 若改用四舍五入与 pool1_meta（同 scale）对齐，可改为: (sum + K/2) / K
                        // 实际算法中采取的是银行家舍入，关键区别在于0.5的处理：银行家舍入在0.5时向最近的偶数舍入，而普通四舍五入在0.5时总是向上舍入。
                        uint16_t q = sum / K;
                        uint16_t r = sum % K;

                        if (r > K / 2) {
                            out[dim1 * OUTDIM + outdim] = q + 1;
                        } else if (r < K / 2) {
                            out[dim1 * OUTDIM + outdim] = q;
                        } else {
                            out[dim1 * OUTDIM + outdim] = (q & 1) ? (q + 1) : q;
                        }

                    }
                }
                o_stream.write(out);
            }
        }
    }
};

#endif