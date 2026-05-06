#include "src/split.h"

//===========================================================================================================================//
// 仅测试 Split：单路输入 -> 复制到两路输出，两路均与 trace_lut_log 对比
//===========================================================================================================================//

Split<X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool> split_inst;

void top(hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>>& i_stream,
         hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>>& o_stream_0,
         hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>>& o_stream_1)
{
#pragma HLS interface ap_ctrl_chain port=return
#pragma HLS interface axis port=i_stream
#pragma HLS interface axis port=o_stream_0
#pragma HLS interface axis port=o_stream_1
#pragma HLS aggregate variable=i_stream compact=bit
#pragma HLS aggregate variable=o_stream_0 compact=bit
#pragma HLS aggregate variable=o_stream_1 compact=bit

#pragma HLS dataflow
    split_inst.do_split_func(i_stream, o_stream_0, o_stream_1);
}

#ifndef __SYNTHESIS__
void test_layer()
{
    const X_T SPLIT_INPUT[N][CONCAT_CHANNELS][LV_DIM1][T_pool] = {
#include "../src/ref/trace/trace_lut_log.txt"
    };
    const X_T SPLIT_REF[N][CONCAT_CHANNELS][LV_DIM1][T_pool] = {
#include "../src/ref/trace/trace_lut_log.txt"
    };

    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> i_stream;
    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> o_stream_0;
    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> o_stream_1;

    i_stream_load<X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool>(i_stream, SPLIT_INPUT);
    top(i_stream, o_stream_0, o_stream_1);

    std::cout << "====================== split_out_0 ======================" << std::endl;
    o_stream_compare<X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool>(o_stream_0, SPLIT_REF);
    std::cout << "====================== split_out_1 ======================" << std::endl;
    o_stream_compare<X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool>(o_stream_1, SPLIT_REF);
}

int main()
{
    test_layer();
    return 0;
}
#endif
