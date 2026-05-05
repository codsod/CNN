#include "src/split.h"
#include "src/linear_cls.h"
#include "src/linear_hue.h"

Split<X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool> split_inst;
LinearCLS<X_T, X_T, I_DEQUAN_T, O_QUAN_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool> linear_cls_inst;
LinearHue<X_T, X_T, I_DEQUAN_T, O_QUAN_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool> linear_hue_inst;

void top(hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>>& i_stream,
         hls::stream<hls::vector<X_T, FC_CLS_OUT>>& cls_stream,
         hls::stream<hls::vector<X_T, FC_HUE_OUT>>& hue_stream)
{
#pragma HLS interface ap_ctrl_chain port=return
#pragma HLS interface axis port=i_stream
#pragma HLS interface axis port=cls_stream
#pragma HLS interface axis port=hue_stream
#pragma HLS aggregate variable=i_stream compact=bit
#pragma HLS aggregate variable=cls_stream compact=bit
#pragma HLS aggregate variable=hue_stream compact=bit

    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> i_stream_cls;
    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> i_stream_hue;

#pragma HLS dataflow
    split_inst.do_split_func(i_stream, i_stream_cls, i_stream_hue);
    linear_cls_inst.do_linear_func(i_stream_cls, cls_stream);
    linear_hue_inst.do_linear_func(i_stream_hue, hue_stream);
}

void test_layer(){
    
    //仿真测试使用
    const X_T LINEAR_IN[N][CONCAT_CHANNELS][LV_DIM1][T_pool] = {
        #include "../src/ref/trace/trace_lut_log.txt"
    };//760=40*19
    const X_T CLS_OUT[N][1][1][FC_CLS_OUT] = {
        #include "../src/ref/trace/trace_fc_cls.txt"
    };//7
    const X_T HUE_OUT[N][1][1][FC_HUE_OUT] = {
        #include "../src/ref/trace/trace_fc_hue.txt"
    };//7
    
    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> i_stream;
    hls::stream<hls::vector<X_T, FC_CLS_OUT>> cls_stream;
    hls::stream<hls::vector<X_T, FC_HUE_OUT>> hue_stream;

    i_stream_load<X_T,N,CONCAT_CHANNELS,LV_DIM1,T_pool>(i_stream,LINEAR_IN);

    top(i_stream,cls_stream,hue_stream);
    
    std::cout<<"======================linear_cls_out======================"<< std::endl;
    o_stream_compare<X_T,N,1,1,FC_CLS_OUT>(cls_stream,CLS_OUT);
    std::cout<<"======================linear_hue_out======================"<< std::endl;
    o_stream_compare<X_T,N,1,1,FC_HUE_OUT>(hue_stream,HUE_OUT);
}

int main(){
    test_layer();
    return 0;
}