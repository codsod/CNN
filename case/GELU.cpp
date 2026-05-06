#include "src/gelu.h"

//===========================================================================================================================//
// 仅测试 GELU 激活层：三路 branch 各自 输入 -> GeluLUT -> 输出，与 trace_ms_conv1_branches_*_2 对比
//===========================================================================================================================//

GeluLUT<X_T, X_T, N, BR0_CHANNELS, IN_DIM, BR_DIM2, 0> gelu_br0_inst;
GeluLUT<X_T, X_T, N, BR1_CHANNELS, IN_DIM, BR_DIM2, 1> gelu_br1_inst;
GeluLUT<X_T, X_T, N, BR2_CHANNELS, IN_DIM, BR_DIM2, 2> gelu_br2_inst;

template<typename data_t, int NN, int CHANNELS, int DIM1, int DIM2>
void load_row_stream(
    hls::stream<hls::vector<data_t, DIM2>>& s,
    const data_t arr[NN][CHANNELS][DIM1][DIM2])
{
    for (int n = 0; n < NN; n++) {
        for (int ch = 0; ch < CHANNELS; ch++) {
            for (int row = 0; row < DIM1; row++) {
                hls::vector<data_t, DIM2> vec;
                for (int t = 0; t < DIM2; t++)
                    vec[t] = arr[n][ch][row][t];
                s.write(vec);
            }
        }
    }
}

template<typename data_t, int NN, int CHANNELS, int DIM1, int DIM2>
void compare_row_stream(
    hls::stream<hls::vector<data_t, DIM2>>& s,
    const data_t arr[NN][CHANNELS][DIM1][DIM2])
{
    int num_match = 0;
    for (int n = 0; n < NN; n++) {
        for (int ch = 0; ch < CHANNELS; ch++) {
            for (int row = 0; row < DIM1; row++) {
                hls::vector<data_t, DIM2> vec = s.read();
                for (int t = 0; t < DIM2; t++) {
                    int idx = t + row * DIM2 + ch * DIM1 * DIM2 + n * CHANNELS * DIM1 * DIM2;
                    int hls_data = vec[t];
                    int ref_data = arr[n][ch][row][t];
                    if (hls_data != ref_data) {
                        printf("mismatch at [%5d]:(hls vs ref) %5d vs %5d\n", idx, hls_data, ref_data);
                    } else {
                        num_match++;
                    }
                }
            }
        }
    }
    printf("match numbers :%5d\n", num_match);
}

void top(hls::stream<hls::vector<X_T, SPATIAL_VEC>>& i_stream_0,
         hls::stream<hls::vector<X_T, SPATIAL_VEC>>& i_stream_1,
         hls::stream<hls::vector<X_T, SPATIAL_VEC>>& i_stream_2,
         hls::stream<hls::vector<X_T, SPATIAL_VEC>>& o_stream_0,
         hls::stream<hls::vector<X_T, SPATIAL_VEC>>& o_stream_1,
         hls::stream<hls::vector<X_T, SPATIAL_VEC>>& o_stream_2)
{
#pragma HLS interface ap_ctrl_chain port=return
#pragma HLS interface axis port=i_stream_0
#pragma HLS interface axis port=i_stream_1
#pragma HLS interface axis port=i_stream_2
#pragma HLS interface axis port=o_stream_0
#pragma HLS interface axis port=o_stream_1
#pragma HLS interface axis port=o_stream_2
#pragma HLS aggregate variable=i_stream_0 compact=bit
#pragma HLS aggregate variable=i_stream_1 compact=bit
#pragma HLS aggregate variable=i_stream_2 compact=bit
#pragma HLS aggregate variable=o_stream_0 compact=bit
#pragma HLS aggregate variable=o_stream_1 compact=bit
#pragma HLS aggregate variable=o_stream_2 compact=bit

#pragma HLS dataflow
    gelu_br0_inst.do_lut_func(i_stream_0, o_stream_0);
    gelu_br1_inst.do_lut_func(i_stream_1, o_stream_1);
    gelu_br2_inst.do_lut_func(i_stream_2, o_stream_2);
}

void test_layer()
{
    const X_T BR0_INPUT[N][BR0_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_0_1.txt"
    };
    const X_T BR0_OUTPUT[N][BR0_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_0_3.txt"
    };
    const X_T BR1_INPUT[N][BR1_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_1_1.txt"
    };
    const X_T BR1_OUTPUT[N][BR1_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_1_3.txt"
    };
    const X_T BR2_INPUT[N][BR2_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_2_1.txt"
    };
    const X_T BR2_OUTPUT[N][BR2_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_2_3.txt"
    };

    hls::stream<hls::vector<X_T, SPATIAL_VEC>> i_stream_0, i_stream_1, i_stream_2;
    hls::stream<hls::vector<X_T, SPATIAL_VEC>> o_stream_0, o_stream_1, o_stream_2;

    i_stream_load_chunks<X_T, N, BR0_CHANNELS, IN_DIM, BR_DIM2, SPATIAL_VEC>(i_stream_0, BR0_INPUT);
    i_stream_load_chunks<X_T, N, BR1_CHANNELS, IN_DIM, BR_DIM2, SPATIAL_VEC>(i_stream_1, BR1_INPUT);
    i_stream_load_chunks<X_T, N, BR2_CHANNELS, IN_DIM, BR_DIM2, SPATIAL_VEC>(i_stream_2, BR2_INPUT);

    top(i_stream_0, i_stream_1, i_stream_2, o_stream_0, o_stream_1, o_stream_2);

    std::cout << "====================== gelu_br0_out ======================" << std::endl;
    o_stream_compare_chunks<X_T, N, BR0_CHANNELS, IN_DIM, BR_DIM2, SPATIAL_VEC>(o_stream_0, BR0_OUTPUT);
    std::cout << "====================== gelu_br1_out ======================" << std::endl;
    o_stream_compare_chunks<X_T, N, BR1_CHANNELS, IN_DIM, BR_DIM2, SPATIAL_VEC>(o_stream_1, BR1_OUTPUT);
    std::cout << "====================== gelu_br2_out ======================" << std::endl;
    o_stream_compare_chunks<X_T, N, BR2_CHANNELS, IN_DIM, BR_DIM2, SPATIAL_VEC>(o_stream_2, BR2_OUTPUT);
}

int main()
{
    test_layer();
    return 0;
}
