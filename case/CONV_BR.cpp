

#include "src/conv_br.h"
#include <vector>
#include <utility>

//===========================================================================================================================//
// 三路时序卷积分支：同一输入 tee 成三路 -> ConvBrBranch (conv+requant+GELU) -> 三路输出
// 输入 (N,1,IN_DIM,T)；输出 (N, BRx_CHANNELS, IN_DIM, BR_DIM2)
//===========================================================================================================================//

ConvBrBranch0 conv_br0_inst;
ConvBrBranch1 conv_br1_inst;
ConvBrBranch2 conv_br2_inst;

// 将 (N, IN_CHANNELS, IN_DIM, T) 按时间维写入 stream：每时刻一个 vector<X_T, IN_DIM>
template<typename data_t, int NN, int C, int D, int TT>
void load_conv_br_input(
    hls::stream<hls::vector<data_t, D>>& s0,
    hls::stream<hls::vector<data_t, D>>& s1,
    hls::stream<hls::vector<data_t, D>>& s2,
    const data_t arr[NN][C][D][TT])
{
    for (int n = 0; n < NN; n++) {
        for (int t = 0; t < TT; t++) {
            hls::vector<data_t, D> vec;
            for (int c = 0; c < D; c++)
                vec[c] = arr[n][0][c][t];
            s0.write(vec);
            s1.write(vec);
            s2.write(vec);
        }
    }
}

// 按分支输出格式读 stream 并与 ref [N][OUT_CH][IN_D][BR_DIM2] 逐元素比较
template<typename data_t, int NN, int OUT_CH, int IN_D, int TT>
void compare_conv_br_out(
    hls::stream<hls::vector<data_t, IN_D>>& o_stream,
    const data_t ref[NN][OUT_CH][IN_D][TT],
    const char* name)
{
    const int MAX_MISMATCH_PRINT = 32;
    const int MAX_MATCH_RANGE_PRINT = 32;

    auto print_coord = [](int flat_idx) {
        int idx = flat_idx;
        int c = idx % IN_D;
        idx /= IN_D;
        int oc = idx % OUT_CH;
        idx /= OUT_CH;
        int t = idx % TT;
        int n = idx / TT;
        printf("[n=%d oc=%d c=%d t=%d]", n, oc, c, t);
    };

    int match = 0;
    int mismatch = 0;
    int flat_idx = 0;
    int range_start = -1;
    int range_end = -1;
    std::vector<std::pair<int, int>> match_ranges;

    for (int n = 0; n < NN; n++) {
        for (int t = 0; t < TT; t++) {
            for (int oc = 0; oc < OUT_CH; oc++) {
                hls::vector<data_t, IN_D> v = o_stream.read();
                for (int c = 0; c < IN_D; c++) {
                    data_t hls_v = v[c];
                    data_t ref_v = ref[n][oc][c][t];
                    if (hls_v != ref_v) {
                        if (range_start != -1) {
                            match_ranges.push_back({range_start, range_end});
                            range_start = -1;
                            range_end = -1;
                        }
                        if (mismatch < MAX_MISMATCH_PRINT) {
                            printf("mismatch %s [n=%d oc=%d c=%d t=%d]: hls=%d ref=%d flat_idx=%d\n",
                                   name, n, oc, c, t, (int)hls_v, (int)ref_v, flat_idx);
                        }
                        mismatch++;
                    } else {
                        match++;
                        if (range_start == -1) {
                            range_start = flat_idx;
                        }
                        range_end = flat_idx;
                    }
                    flat_idx++;
                }
            }
        }
    }

    if (range_start != -1) {
        match_ranges.push_back({range_start, range_end});
    }

    const int total = NN * OUT_CH * IN_D * TT;
    printf("%s summary: match=%d mismatch=%d total=%d\n", name, match, mismatch, total);
    if (mismatch > MAX_MISMATCH_PRINT) {
        printf("%s mismatch details truncated: printed %d / %d\n", name, MAX_MISMATCH_PRINT, mismatch);
    }

    printf("%s passed ranges (%zu total):\n", name, match_ranges.size());
    for (size_t i = 0; i < match_ranges.size() && i < (size_t)MAX_MATCH_RANGE_PRINT; i++) {
        const auto& range = match_ranges[i];
        printf("  pass_range[%zu] flat_idx=[%d, %d] ", i, range.first, range.second);
        print_coord(range.first);
        printf(" -> ");
        print_coord(range.second);
        printf("\n");
    }
    if ((int)match_ranges.size() > MAX_MATCH_RANGE_PRINT) {
        printf("  ... %zu more pass ranges omitted\n", match_ranges.size() - MAX_MATCH_RANGE_PRINT);
    }
}

void top(hls::stream<hls::vector<X_T, IN_DIM>>& i_stream_0,
         hls::stream<hls::vector<X_T, IN_DIM>>& i_stream_1,
         hls::stream<hls::vector<X_T, IN_DIM>>& i_stream_2,
         hls::stream<hls::vector<X_T, IN_DIM>>& o_stream_0,
         hls::stream<hls::vector<X_T, IN_DIM>>& o_stream_1,
         hls::stream<hls::vector<X_T, IN_DIM>>& o_stream_2)
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
    conv_br0_inst.do_conv_br(i_stream_0, o_stream_0);
    conv_br1_inst.do_conv_br(i_stream_1, o_stream_1);
    conv_br2_inst.do_conv_br(i_stream_2, o_stream_2);
}

void test_layer()
{
    const X_T INPUT[N][IN_CHANNELS][IN_DIM][T] = {
#include "../src/ref/trace/trace_input_0.txt"
    };
    const X_T BR0_OUTPUT[N][BR0_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_0_3.txt"
    };
    const X_T BR1_OUTPUT[N][BR1_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_1_3.txt"
    };
    const X_T BR2_OUTPUT[N][BR2_CHANNELS][IN_DIM][BR_DIM2] = {
#include "../src/ref/trace/trace_ms_conv1_branches_2_3.txt"
    };

    hls::stream<hls::vector<X_T, IN_DIM>> i_0, i_1, i_2;
    hls::stream<hls::vector<X_T, IN_DIM>> o_0;
    hls::stream<hls::vector<X_T, IN_DIM>> o_1;
    hls::stream<hls::vector<X_T, IN_DIM>> o_2;

    load_conv_br_input<X_T, N, IN_CHANNELS, IN_DIM, T>(i_0, i_1, i_2, INPUT);
    top(i_0, i_1, i_2, o_0, o_1, o_2);

    std::cout << "====================== conv_br0 ======================" << std::endl;
    compare_conv_br_out<X_T, N, BR0_CHANNELS, IN_DIM, BR_DIM2>(o_0, BR0_OUTPUT, "br0");
    std::cout << "====================== conv_br1 ======================" << std::endl;
    compare_conv_br_out<X_T, N, BR1_CHANNELS, IN_DIM, BR_DIM2>(o_1, BR1_OUTPUT, "br1");
    std::cout << "====================== conv_br2 ======================" << std::endl;
    compare_conv_br_out<X_T, N, BR2_CHANNELS, IN_DIM, BR_DIM2>(o_2, BR2_OUTPUT, "br2");
}

int main()
{
    test_layer();
    return 0;
}
