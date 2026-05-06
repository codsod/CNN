#include "src/quant_input.h"
#include <fstream>

//===========================================================================================================================//
// 仅测试输入量化：fp32 stream -> QuantInput -> uint8 stream，与 trace_input_0.txt 对比
// 输入从 trace_input_0_fp32.txt 读取（6150 个 float，顺序与 trace_input_0 一致）
//===========================================================================================================================//

QuantInput<X_T, IN_DIM> quant_input_inst;

void top(hls::stream<hls::vector<float, IN_DIM>>& i_stream,
         hls::stream<hls::vector<X_T, IN_DIM>>& o_stream)
{
#pragma HLS interface ap_ctrl_chain port=return
#pragma HLS interface axis port=i_stream
#pragma HLS interface axis port=o_stream
#pragma HLS aggregate variable=i_stream compact=bit
#pragma HLS aggregate variable=o_stream compact=bit

    quant_input_inst.do_quantize(i_stream, o_stream);
}

#ifndef __SYNTHESIS__
void test_layer()
{
    const X_T QUANT_REF[N][IN_CHANNELS][IN_DIM][T] = {
#include "../src/ref/trace/trace_input_0.txt"
    };

    float fp32_buf[N][IN_CHANNELS][IN_DIM][T];
    const char* try_paths[] = {
        "../src/ref/trace/trace_input_0_fp32.txt",
        "src/ref/trace/trace_input_0_fp32.txt",
        "ref/trace/trace_input_0_fp32.txt",
        "../../../../ref/trace/trace_input_0_fp32.txt",
        "../../../../src/ref/trace/trace_input_0_fp32.txt",
        "../../../../../src/ref/trace/trace_input_0_fp32.txt",
        "../../../../../../src/ref/trace/trace_input_0_fp32.txt",
    };
    std::ifstream f;
    for (int i = 0; i < (int)(sizeof(try_paths) / sizeof(try_paths[0])); i++) {
        f.open(try_paths[i]);
        if (f.is_open()) break;
    }
    if (!f.is_open()) {
        std::cerr << "Cannot open trace_input_0_fp32.txt (put it in src/ref/trace/ or ref/trace/)" << std::endl;
        return;
    }
    for (int n = 0; n < N; n++) {
        for (int ch = 0; ch < IN_CHANNELS; ch++) {
            for (int c = 0; c < IN_DIM; c++) {
                for (int t = 0; t < T; t++) {
                    f >> fp32_buf[n][ch][c][t];
                }
            }
        }
    }
    f.close();

    hls::stream<hls::vector<float, IN_DIM>> i_stream;
    hls::stream<hls::vector<X_T, IN_DIM>> o_stream;

    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            hls::vector<float, IN_DIM> vec;
            for (int c = 0; c < IN_DIM; c++)
                vec[c] = fp32_buf[n][0][c][t];
            i_stream.write(vec);
        }
    }

    top(i_stream, o_stream);

    int match = 0;
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            hls::vector<X_T, IN_DIM> v = o_stream.read();
            for (int c = 0; c < IN_DIM; c++) {
                X_T hls_v = v[c];
                X_T ref_v = QUANT_REF[n][0][c][t];
                if (hls_v != ref_v)
                    printf("mismatch [n=%d t=%d c=%d]: hls=%d ref=%d\n", n, t, c, (int)hls_v, (int)ref_v);
                else
                    match++;
            }
        }
    }
    printf("quant_input match: %d / %d\n", match, N * IN_CHANNELS * IN_DIM * T);
}

int main()
{
    test_layer();
    return 0;
}
#endif
