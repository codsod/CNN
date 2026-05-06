#include "src/log.h"

//===========================================================================================================================//
// 复制来自AVER_POOL.cpp,一起测试
#include "src/pool.h"

POOL   <X_T, X_T, N, CONCAT_CHANNELS, LV_DIM1 ,BR_DIM2 ,T_pool,  K_POOL , S_POOL> pool_inst;

void do_pool(hls::stream<hls::vector<X_T, SPATIAL_VEC> >& i_stream,hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>>& o_stream){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=i_stream
    #pragma HLS interface axis port=o_stream

    #pragma HLS aggregate variable=i_stream compact=bit
    #pragma HLS aggregate variable=o_stream compact=bit

    #pragma HLS dataflow
    pool_inst   .do_pool_func(i_stream,o_stream);
}
//===========================================================================================================================//

LogLUT<X_T, X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool> log_inst;

void do_log(hls::stream<hls::vector<X_T, LV_DIM1 * T_pool> >& i_stream,hls::stream<hls::vector<X_T, LV_DIM1 * T_pool> >& o_stream)
{
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=i_stream
    #pragma HLS interface axis port=o_stream

    #pragma HLS aggregate variable=i_stream compact=bit
    #pragma HLS aggregate variable=o_stream compact=bit

    #pragma HLS dataflow
    // std::cout<<"======================square_out======================"<< std::endl;
    log_inst   .do_lut_func(i_stream,o_stream);
}

//

void top(hls::stream<hls::vector<X_T, SPATIAL_VEC> >& i_stream,hls::stream<hls::vector<X_T, LV_DIM1 * T_pool> >& o_stream)
{
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=i_stream
    #pragma HLS interface axis port=o_stream

    #pragma HLS aggregate variable=i_stream compact=bit
    #pragma HLS aggregate variable=o_stream compact=bit

    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool> > m_stream;
    do_pool(i_stream , m_stream);
    do_log(m_stream , o_stream);


}

#ifndef __SYNTHESIS__
void test_layer(){
    // 仿真测试使用
    // 此处需要连同pool一起测试

    const X_T POOL_INPUT[N][CONCAT_CHANNELS][LV_DIM1][BR_DIM2] = {
        #include "../src/ref/trace/trace_lut_square.txt"
    };
    const X_T LOG_OUTPUT[N][CONCAT_CHANNELS][LV_DIM1][T_pool] = {
        #include "../src/ref/trace/trace_lut_log.txt"
    };
    hls::stream<hls::vector<X_T, SPATIAL_VEC> > i_stream;
    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool> > o_stream;

    i_stream_load_chunks<X_T,N,CONCAT_CHANNELS,LV_DIM1,BR_DIM2,SPATIAL_VEC>(i_stream,POOL_INPUT);

    top(i_stream , o_stream);

    std::cout<<"======================pool_log_test======================"<< std::endl;
    o_stream_compare<X_T,N,CONCAT_CHANNELS,LV_DIM1,T_pool>(o_stream,LOG_OUTPUT);
}


int main(){
    test_layer();
    return 0;
}
#endif
