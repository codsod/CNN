#include "src/log.h"

//===========================================================================================================================//
// 仅测试 LOG 激活层：输入 -> LogLUT -> 输出，与 trace_lut_log 对比
//===========================================================================================================================//

LogLUT<X_T, X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool> log_inst;

void top(hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>>& i_stream,
         hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>>& o_stream)
{
#pragma HLS interface ap_ctrl_chain port=return
#pragma HLS interface axis port=i_stream
#pragma HLS interface axis port=o_stream
#pragma HLS aggregate variable=i_stream compact=bit
#pragma HLS aggregate variable=o_stream compact=bit

#pragma HLS dataflow
    log_inst.do_lut_func(i_stream, o_stream);
}

void test_layer()
{
    const X_T LOG_INPUT[N][CONCAT_CHANNELS][LV_DIM1][T_pool] = {
#include "../src/ref/trace/trace_pool1.txt"
    };
    const X_T LOG_OUTPUT[N][CONCAT_CHANNELS][LV_DIM1][T_pool] = {
#include "../src/ref/trace/trace_lut_log.txt"
    };

    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> i_stream;
    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> o_stream;

    i_stream_load<X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool>(i_stream, LOG_INPUT);
    top(i_stream, o_stream);

    std::cout << "====================== log_test ======================" << std::endl;
    o_stream_compare<X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool>(o_stream, LOG_OUTPUT);
}

int main()
{
    test_layer();
    return 0;
}
