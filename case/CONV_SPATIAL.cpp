#include "src/conv2d.h"
#include "src/concat.h"

//===========================================================================================================================//
// 仅测试空间卷积 ConvSpatial：输入 (N,40,15,411) -> 输出 (N,40,1,411)，与 trace_conv2 对比
// 输入由三路 branch 输出 concat 得到，此处用三份 trace 合并
//===========================================================================================================================//

ConvSpatial<X_T, X_T> conv_spatial_inst;

void top(hls::stream<hls::vector<X_T, SPATIAL_VEC>> &i_stream,
         hls::stream<hls::vector<X_T, BR_DIM2>> &o_stream)
{
#pragma HLS interface ap_ctrl_chain port = return
#pragma HLS interface axis port = i_stream
#pragma HLS interface axis port = o_stream
#pragma HLS aggregate variable = i_stream compact = bit
#pragma HLS aggregate variable = o_stream compact = bit

    conv_spatial_inst.do_conv_spatial(i_stream, o_stream);
}

static void load_conv_spatial_input(
    hls::stream<hls::vector<X_T, SPATIAL_VEC>> &i_stream,
    const X_T arr[N][CONCAT_CHANNELS][IN_DIM][BR_DIM2])
{
    for (int n = 0; n < N; n++)
    {
        for (int ic = 0; ic < CONCAT_CHANNELS; ic++)
        {
            for (int row = 0; row < IN_DIM; row++)
            {
                for (int ck = 0; ck < SPATIAL_CHUNKS; ck++)
                {
                    hls::vector<X_T, SPATIAL_VEC> vec;
                    for (int k = 0; k < SPATIAL_VEC; k++)
                    {
                        int t = ck * SPATIAL_VEC + k;
                        vec[k] = (t < BR_DIM2) ? arr[n][ic][row][t] : (X_T)0;
                    }
                    i_stream.write(vec);
                }
            }
        }
    }
}

void test_layer()
{
    const X_T BR0_OUT[N][BR0_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_0_3.txt"
    };
    const X_T BR1_OUT[N][BR1_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_1_3.txt"
    };
    const X_T BR2_OUT[N][BR2_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_2_3.txt"
    };

    X_T CONV_INPUT[N][CONCAT_CHANNELS][IN_DIM][BR_DIM2];
    for (int n = 0; n < N; n++)
    {
        for (int ic = 0; ic < CONCAT_CHANNELS; ic++)
        {
            for (int r = 0; r < IN_DIM; r++)
            {
                for (int t = 0; t < BR_DIM2; t++)
                {
                    if (ic < BR0_CHANNELS)
                        CONV_INPUT[n][ic][r][t] = concat_lut0[(ap_uint<8>)BR0_OUT[n][ic][r][t]];
                    else if (ic < BR0_CHANNELS + BR1_CHANNELS)
                        CONV_INPUT[n][ic][r][t] = concat_lut1[(ap_uint<8>)BR1_OUT[n][ic - BR0_CHANNELS][r][t]];
                    else
                        CONV_INPUT[n][ic][r][t] = concat_lut2[(ap_uint<8>)BR2_OUT[n][ic - BR0_CHANNELS - BR1_CHANNELS][r][t]];
                }
            }
        }
    }

    const X_T CONV_REF[N][CONCAT_CHANNELS][LV_DIM1][BR_DIM2] = {
#include "../src/ref/trace/trace_conv2.txt"
    };

    hls::stream<hls::vector<X_T, SPATIAL_VEC>> i_stream;
    hls::stream<hls::vector<X_T, BR_DIM2>> o_stream;

    load_conv_spatial_input(i_stream, CONV_INPUT);
    top(i_stream, o_stream);

    std::cout << "====================== conv_spatial_test ======================" << std::endl;
    o_stream_compare<X_T, N, CONCAT_CHANNELS, LV_DIM1, BR_DIM2>(o_stream, CONV_REF);
}

int main()
{
    test_layer();
    return 0;
}
