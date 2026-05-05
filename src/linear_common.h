#ifndef __LINEAR_COMMON_H__
#define __LINEAR_COMMON_H__

#include "common.h"

// FC 层共用：输入量化参数（log 输出作为 linear 输入）
constexpr SCALE_T FC_INPUT_SCALE = 0.033854458481;
constexpr ZP_T    FC_INPUT_ZP    = 162;

#endif
