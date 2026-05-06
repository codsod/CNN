#include "src/square.h"

//===========================================================================================================================//
// 仅测试 SQUARE 激活层：输入 -> SquareLUT -> 输出，与 trace_lut_square 对比
//===========================================================================================================================//

SquareLUT<X_T, X_T, N, CONCAT_CHANNELS, LV_DIM1, BR_DIM2> square_inst;

void top(hls::stream<hls::vector<X_T, SPATIAL_VEC>>& i_stream,
         hls::stream<hls::vector<X_T, SPATIAL_VEC>>& o_stream)
{
#pragma HLS interface ap_ctrl_chain port=return
#pragma HLS interface axis port=i_stream
#pragma HLS interface axis port=o_stream
#pragma HLS aggregate variable=i_stream compact=bit
#pragma HLS aggregate variable=o_stream compact=bit

#pragma HLS dataflow
    square_inst.do_lut_func(i_stream, o_stream);
}

#ifndef __SYNTHESIS__
void test_layer()
{
    const X_T SQUARE_INPUT[N][CONCAT_CHANNELS][LV_DIM1][BR_DIM2] = {
#include "../src/ref/trace/trace_conv2.txt"
    };
    const X_T SQUARE_OUTPUT[N][CONCAT_CHANNELS][LV_DIM1][BR_DIM2] = {
#include "../src/ref/trace/trace_lut_square.txt"
    };

    hls::stream<hls::vector<X_T, SPATIAL_VEC>> i_stream;
    hls::stream<hls::vector<X_T, SPATIAL_VEC>> o_stream;

    i_stream_load_chunks<X_T, N, CONCAT_CHANNELS, LV_DIM1, BR_DIM2, SPATIAL_VEC>(i_stream, SQUARE_INPUT);
    top(i_stream, o_stream);

    std::cout << "====================== square_test ======================" << std::endl;
    o_stream_compare_chunks<X_T, N, CONCAT_CHANNELS, LV_DIM1, BR_DIM2, SPATIAL_VEC>(o_stream, SQUARE_OUTPUT);
}

int main()
{
    test_layer();
    return 0;
}
#endif
