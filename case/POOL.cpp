#include "src/pool.h"

//===========================================================================================================================//
// 仅测试 POOL 层：输入 -> POOL -> 输出，与 trace_pool1 对比
//===========================================================================================================================//

POOL<X_T, X_T, N, CONCAT_CHANNELS, LV_DIM1, BR_DIM2, T_pool, K_POOL, S_POOL> pool_inst;

void top(hls::stream<hls::vector<X_T, SPATIAL_VEC>>& i_stream,
         hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>>& o_stream)
{
#pragma HLS interface ap_ctrl_chain port=return
#pragma HLS interface axis port=i_stream
#pragma HLS interface axis port=o_stream
#pragma HLS aggregate variable=i_stream compact=bit
#pragma HLS aggregate variable=o_stream compact=bit

#pragma HLS dataflow
    pool_inst.do_pool_func(i_stream, o_stream);
}

#ifndef __SYNTHESIS__
void test_layer()
{
    const X_T POOL_INPUT[N][CONCAT_CHANNELS][LV_DIM1][BR_DIM2] = {
#include "../src/ref/trace/trace_lut_square.txt"
    };
    const X_T POOL_OUTPUT[N][CONCAT_CHANNELS][LV_DIM1][T_pool] = {
#include "../src/ref/trace/trace_pool1.txt"
    };

    hls::stream<hls::vector<X_T, SPATIAL_VEC>> i_stream;
    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> o_stream;

    i_stream_load_chunks<X_T, N, CONCAT_CHANNELS, LV_DIM1, BR_DIM2, SPATIAL_VEC>(i_stream, POOL_INPUT);
    top(i_stream, o_stream);

    std::cout << "====================== pool_test ======================" << std::endl;
    o_stream_compare<X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool>(o_stream, POOL_OUTPUT);
}

int main()
{
    test_layer();
    return 0;
}
#endif
