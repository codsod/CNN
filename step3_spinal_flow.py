from pre_syn_process import *
from pst_syn_process import *
import os

INSTANCE_DIR = os.path.join(ROOT_DIR, "instances_tp8_syn_kv260/")

case_names = [
    "GEMM_PERMUTE",   # PLEASE use 2022.1
    "DEMUX",
    "ROPE_QK_QUANT",
    "QK_GEMM",
    "SOFTMAX_QUANT",
    "RV_GEMM",
    "SILU_EM_QUANT",
    "RESIDUAL",
    "RMSNORM_QUANT",
    "MUX",
    "M_AXI",
    "KV_CACHE"
]

instances_list = ["proj_" + case_name for case_name in case_names]

backup_verilog  (INSTANCE_DIR, instances_list=instances_list)
backup_log      (INSTANCE_DIR, instances_list=instances_list)

to_spinal(INSTANCE_DIR, case_names=case_names)