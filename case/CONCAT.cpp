#include "src/concat.h"

//===========================================================================================================================//
// 仅测试 Concat：三路 branch 输出 stream -> 按通道拼成一路 -> 与合并后的 ref 对比
// 输入格式与 CONV_BR 输出一致；输出格式与 CONV_SPATIAL 输入一致
//===========================================================================================================================//

Concat<X_T> concat_inst;

// 按 CONV_BR 输出顺序：每时刻每通道一个 vector (in_dim)，共 BR_DIM2*CH 个向量
template<typename data_t, int NN, int CH, int IN_D, int BR_D>
void load_conv_br_out_stream(
    hls::stream<hls::vector<data_t, IN_D>>& s,
    const data_t arr[NN][CH][IN_D][BR_D])
{
    for (int n = 0; n < NN; n++) {
        for (int t = 0; t < BR_D; t++) {
            for (int oc = 0; oc < CH; oc++) {
                hls::vector<data_t, IN_D> vec;
                for (int c = 0; c < IN_D; c++)
                    vec[c] = arr[n][oc][c][t];
                s.write(vec);
            }
        }
    }
}

template<typename data_t, int NN, int CHANNELS, int IN_D, int BR_D>
void compare_concat_row_stream(
    hls::stream<hls::vector<data_t, BR_D>>& s,
    const data_t arr[NN][CHANNELS][IN_D][BR_D])
{
    int num_match = 0;
    for (int n = 0; n < NN; n++) {
        for (int ch = 0; ch < CHANNELS; ch++) {
            for (int row = 0; row < IN_D; row++) {
                hls::vector<data_t, BR_D> vec = s.read();
                for (int t = 0; t < BR_D; t++) {
                    int idx = t + row * BR_D + ch * IN_D * BR_D + n * CHANNELS * IN_D * BR_D;
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

void top(hls::stream<hls::vector<X_T, IN_DIM>>& i_stream_0,
         hls::stream<hls::vector<X_T, IN_DIM>>& i_stream_1,
         hls::stream<hls::vector<X_T, IN_DIM>>& i_stream_2,
         hls::stream<hls::vector<X_T, BR_DIM2>>& o_stream)
{
#pragma HLS interface ap_ctrl_chain port=return
#pragma HLS interface axis port=i_stream_0
#pragma HLS interface axis port=i_stream_1
#pragma HLS interface axis port=i_stream_2
#pragma HLS interface axis port=o_stream
#pragma HLS aggregate variable=i_stream_0 compact=bit
#pragma HLS aggregate variable=i_stream_1 compact=bit
#pragma HLS aggregate variable=i_stream_2 compact=bit
#pragma HLS aggregate variable=o_stream compact=bit

    concat_inst.do_concat(i_stream_0, i_stream_1, i_stream_2, o_stream);
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

    X_T CONCAT_REF[N][CONCAT_CHANNELS][IN_DIM][BR_DIM2];
    for (int n = 0; n < N; n++) {
        for (int ic = 0; ic < CONCAT_CHANNELS; ic++) {
            for (int r = 0; r < IN_DIM; r++) {
                for (int t = 0; t < BR_DIM2; t++) {
                    if (ic < BR0_CHANNELS)
                        CONCAT_REF[n][ic][r][t] = concat_lut0[(ap_uint<8>)BR0_OUT[n][ic][r][t]];
                    else if (ic < BR0_CHANNELS + BR1_CHANNELS)
                        CONCAT_REF[n][ic][r][t] = concat_lut1[(ap_uint<8>)BR1_OUT[n][ic - BR0_CHANNELS][r][t]];
                    else
                        CONCAT_REF[n][ic][r][t] = concat_lut2[(ap_uint<8>)BR2_OUT[n][ic - BR0_CHANNELS - BR1_CHANNELS][r][t]];
                }
            }
        }
    }

    hls::stream<hls::vector<X_T, IN_DIM>> i_0;
    hls::stream<hls::vector<X_T, IN_DIM>> i_1;
    hls::stream<hls::vector<X_T, IN_DIM>> i_2;
    hls::stream<hls::vector<X_T, BR_DIM2>> o_stream;

    load_conv_br_out_stream<X_T, N, BR0_CHANNELS, IN_DIM, BR_DIM2>(i_0, BR0_OUT);
    load_conv_br_out_stream<X_T, N, BR1_CHANNELS, IN_DIM, BR_DIM2>(i_1, BR1_OUT);
    load_conv_br_out_stream<X_T, N, BR2_CHANNELS, IN_DIM, BR_DIM2>(i_2, BR2_OUT);

    top(i_0, i_1, i_2, o_stream);

    std::cout << "====================== concat_test ======================" << std::endl;
    compare_concat_row_stream<X_T, N, CONCAT_CHANNELS, IN_DIM, BR_DIM2>(o_stream, CONCAT_REF);
}

int main()
{
    test_layer();
    return 0;
}
