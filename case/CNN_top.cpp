//===========================================================================================================================//
// CNN 整网：quant_input -> conv_br(3) -> concat -> conv_spatial -> square -> pool -> log -> split -> linear_cls / linear_hue
// 测试：输入 trace_input_0_fp32.txt，输出与 trace_fc_cls / trace_fc_hue 对比
//===========================================================================================================================//

#include "src/quant_input.h"
#include <fstream>
#include "src/conv_br.h"
#include "src/concat.h"
#include "src/conv2d.h"
#include "src/square.h"
#include "src/pool.h"
#include "src/log.h"
#include "src/split.h"
#include "src/linear_cls.h"
#include "src/linear_hue.h"

// 模块实例
QuantInput<X_T, IN_DIM> quant_input_inst;
ConvBrBranch0 conv_br0_inst;
ConvBrBranch1 conv_br1_inst;
ConvBrBranch2 conv_br2_inst;
Concat<X_T> concat_inst;
ConvSpatial<X_T, X_T> conv_spatial_inst;
SquareLUT<X_T, X_T, N, CONCAT_CHANNELS, LV_DIM1, BR_DIM2> square_inst;
POOL<X_T, X_T, N, CONCAT_CHANNELS, LV_DIM1, BR_DIM2, T_pool, K_POOL, S_POOL> pool_inst;
LogLUT<X_T, X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool> log_inst;
Split<X_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool> split_inst;
LinearCLS<X_T, X_T, I_DEQUAN_T, O_QUAN_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool> linear_cls_inst;
LinearHue<X_T, X_T, I_DEQUAN_T, O_QUAN_T, N, CONCAT_CHANNELS, LV_DIM1, T_pool> linear_hue_inst;

// 量化输出 1->3 复制，供三路 conv_br 使用
static void tee3_quant_to_conv_br(
    hls::stream<hls::vector<X_T, IN_DIM>>& in,
    hls::stream<hls::vector<X_T, IN_DIM>>& out0,
    hls::stream<hls::vector<X_T, IN_DIM>>& out1,
    hls::stream<hls::vector<X_T, IN_DIM>>& out2)
{
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
#pragma HLS PIPELINE II=1
            hls::vector<X_T, IN_DIM> v = in.read();
            out0.write(v);
            out1.write(v);
            out2.write(v);
        }
    }
}

void top(hls::stream<hls::vector<float, IN_DIM>>& i_stream,
         hls::stream<hls::vector<X_T, FC_CLS_OUT>>& cls_stream,
         hls::stream<hls::vector<X_T, FC_HUE_OUT>>& hue_stream)
{
#pragma HLS interface ap_ctrl_chain port=return
#pragma HLS interface axis port=i_stream
#pragma HLS interface axis port=cls_stream
#pragma HLS interface axis port=hue_stream
#pragma HLS aggregate variable=i_stream compact=bit
#pragma HLS aggregate variable=cls_stream compact=bit
#pragma HLS aggregate variable=hue_stream compact=bit

    hls::stream<hls::vector<X_T, IN_DIM>> quant_out;
    hls::stream<hls::vector<X_T, IN_DIM>> conv_br_i0, conv_br_i1, conv_br_i2;
    hls::stream<hls::vector<X_T, IN_DIM>> concat_i0;
    hls::stream<hls::vector<X_T, IN_DIM>> concat_i1;
    hls::stream<hls::vector<X_T, IN_DIM>> concat_i2;
    hls::stream<hls::vector<X_T, BR_DIM2>> conv_spatial_i;
    hls::stream<hls::vector<X_T, BR_DIM2>> conv_spatial_o;
    hls::stream<hls::vector<X_T, LV_DIM1 * BR_DIM2>> square_o;
    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> pool_o;
    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> log_o;
    hls::stream<hls::vector<X_T, LV_DIM1 * T_pool>> linear_cls_i, linear_hue_i;

#pragma HLS dataflow
    quant_input_inst.do_quantize(i_stream, quant_out);
    tee3_quant_to_conv_br(quant_out, conv_br_i0, conv_br_i1, conv_br_i2);
    conv_br0_inst.do_conv_br(conv_br_i0, concat_i0);
    conv_br1_inst.do_conv_br(conv_br_i1, concat_i1);
    conv_br2_inst.do_conv_br(conv_br_i2, concat_i2);
    concat_inst.do_concat(concat_i0, concat_i1, concat_i2, conv_spatial_i);
    conv_spatial_inst.do_conv_spatial(conv_spatial_i, conv_spatial_o);
    square_inst.do_lut_func(conv_spatial_o, square_o);
    pool_inst.do_pool_func(square_o, pool_o);
    log_inst.do_lut_func(pool_o, log_o);
    split_inst.do_split_func(log_o, linear_cls_i, linear_hue_i);
    linear_cls_inst.do_linear_func(linear_cls_i, cls_stream);
    linear_hue_inst.do_linear_func(linear_hue_i, hue_stream);
}

void test_layer()
{
    const X_T CLS_REF[N][1][1][FC_CLS_OUT] = {
#include "../src/ref/trace/trace_fc_cls.txt"
    };
    const X_T HUE_REF[N][1][1][FC_HUE_OUT] = {
#include "../src/ref/trace/trace_fc_hue.txt"
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
    hls::stream<hls::vector<X_T, FC_CLS_OUT>> cls_stream;
    hls::stream<hls::vector<X_T, FC_HUE_OUT>> hue_stream;

    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            hls::vector<float, IN_DIM> vec;
            for (int c = 0; c < IN_DIM; c++)
                vec[c] = fp32_buf[n][0][c][t];
            i_stream.write(vec);
        }
    }

    top(i_stream, cls_stream, hue_stream);

    std::cout << "====================== CNN_top cls_out ======================" << std::endl;
    o_stream_compare<X_T, N, 1, 1, FC_CLS_OUT>(cls_stream, CLS_REF);
    std::cout << "====================== CNN_top hue_out ======================" << std::endl;
    o_stream_compare<X_T, N, 1, 1, FC_HUE_OUT>(hue_stream, HUE_REF);
}

int main()
{
    test_layer();
    return 0;
}
