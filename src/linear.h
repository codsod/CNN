#ifndef __INT_LINEAR_H__
#define __INT_LINEAR_H__

#include "common.h"

// sed -i 's/$/,/' *.txt 
// 上述指令实现对ref文件换行变成,

// 模块使用，硬件
constexpr SCALE_T FC_INPUT_SCALE = 0.035311974585;
constexpr ZP_T FC_INPUT_ZP = 162;
// CLS
constexpr W_T CLS_WEIGHT[FC_CLS_OUT][FlatDim] = { 
    #include "ref/fc/cls/fc_cls_weight_int8.txt"
}; //5320=FC_OUT*FlatDim
constexpr SCALE_T CLS_WEIGHT_SCALES = { 
    #include "ref/fc/cls/fc_cls_weight_scales.txt"
}; //1
constexpr B_T CLS_BIAS[FC_CLS_OUT] = { 
    #include "ref/fc/cls/fc_cls_bias_int32.txt"
}; //7
constexpr SCALE_T CLS_OUTPUT_SCALE = 0.061297483742;
constexpr ZP_T CLS_OUTPUT_ZP = 102;

// HUE
const W_T HUE_WEIGHT[FC_HUE_OUT][FlatDim] = { 
    #include "ref/fc/hue/fc_hue_weight_int8.txt"
}; //1520
constexpr SCALE_T HUE_WEIGHT_SCALES = { 
    #include "ref/fc/hue/fc_hue_weight_scales.txt"
}; //1
constexpr B_T HUE_BIAS[FC_HUE_OUT] = { 
    #include "ref/fc/hue/fc_hue_bias_int32.txt"
}; //7
constexpr SCALE_T HUE_OUTPUT_SCALE = 0.040244147182;
constexpr ZP_T HUE_OUTPUT_ZP = 120;

template<
    class if_t,
    class of_t,
    class i_dequan_t,
    class o_quan_t,

    int N,
    int CHANNELS,
    int DIM1,
    int DIM2,
    int FC_OUT,
    bool mode_cls
>
class LINEAR{
public:
    const W_T (*WEIGHT)[FlatDim] = mode_cls ? CLS_WEIGHT : HUE_WEIGHT;
    const B_T* BIAS = mode_cls ? CLS_BIAS : HUE_BIAS;
    const SCALE_T WEIGHT_SCALES = mode_cls ? CLS_WEIGHT_SCALES : HUE_WEIGHT_SCALES;
    const SCALE_T OUTPUT_SCALE = mode_cls ? CLS_OUTPUT_SCALE : HUE_OUTPUT_SCALE;
    const ZP_T OUTPUT_ZP = mode_cls ? CLS_OUTPUT_ZP : HUE_OUTPUT_ZP;

    // some internal buffers
    i_dequan_t dequan_input[N][FlatDim];
    o_quan_t dequan_output[N][FC_OUT];

    LINEAR(){}

    void do_linear_func(hls::stream<hls::vector<if_t, DIM1 * DIM2> >& i_stream,hls::stream<hls::vector<of_t, FC_OUT>>& o_stream){
        hls::vector<if_t, DIM1 * DIM2> in[N][CHANNELS];
        hls::vector<of_t, FC_OUT> out[N];
        //#pragma HLS ARRAY_RESHAPE variable=in dim=2 type=complete
        //#pragma HLS ARRAY_RESHAPE variable=in dim=3 type=complete
        batch_loop:for (int n = 0; n < N; n++) {
            outdim_loop:for (int out_dim = 0; out_dim < FC_OUT; out_dim++) {
                dequan_output[n][out_dim] = 0;
                channels_loop:for (int channels = 0; channels < CHANNELS; channels++) {
                    in[n][channels] = out_dim==0 ? i_stream.read():in[n][channels];
                    #pragma HLS PIPELINE II=1
                    dim1_loop:for (int dim1 = 0; dim1 < DIM1; dim1++) {
                        dim2_loop:for (int dim2 = 0; dim2 < DIM2; dim2++) {
                            #pragma HLS UNROLL
                            int idx = channels*DIM1*DIM2 + dim1 * DIM2 + dim2;
                            dequan_input[n][idx] = (i_dequan_t)((in[n][channels][dim1*DIM2 + dim2] - FC_INPUT_ZP));
                            //std::cout<<"dequan_input["<<n<<"]["<<idx<<"] = "<< (int)dequan_input[n][idx] <<std::endl;
                            dequan_output[n][out_dim] +=  (o_quan_t)(dequan_input[n][idx] * WEIGHT[out_dim][idx]);
                        }
                    }
                }
                dequan_output[n][out_dim] = dequan_output[n][out_dim] + BIAS[out_dim];      
                //dequan_output_float[n][out_dim] = dequan_output[n][out_dim]* FC_INPUT_SCALE * WEIGHT_SCALES ;
                out[n][out_dim] = (of_t)((dequan_output[n][out_dim] * FC_INPUT_SCALE * WEIGHT_SCALES / OUTPUT_SCALE) + OUTPUT_ZP) ;
                // std::cout<<"out["<<n*FC_OUT + out_dim<<"] = "<< (int)out[n][out_dim] <<std::endl;
            }
            o_stream.write(out[n]);
        }  
    }
};
// 需要添加do_linear来实现上下级stream之间的转换

#endif