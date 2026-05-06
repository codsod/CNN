from pst_syn_process import *
from pre_syn_process import *

import os
# use os to add include path for C++ compiler
os.environ["CPLUS_INCLUDE_PATH"] = os.getcwd()
INSTANCE_DIR = os.path.join(ROOT_DIR, "instances_tp8_sim")

# compile all
case_names = [
    # "LOG",   # PLEASE use 2022.1
    # "POOL",     # pass, two questions are found
    # "SQUARE",   # psss 
    # "LINEAR",   # pass, requantization is needed
    # "GELU",     # pass
    # "CONV_BR",    # pass, two questions are found
    # "CONV_SPATIAL",  # no pass
    # "CONCAT",   # pass
    # "QUANT_INPUT", # pass
    "CNN_top", 
]

create_subprojects  (INSTANCE_DIR, case_names=case_names, overwrite=True)
create_tcls         (INSTANCE_DIR, case_names=case_names, do_csim=True, do_csynth=False, do_cosim=False, do_impl=False, pipeline_styles="flp")

# launch the tcl files
# run_instances(INSTANCE_DIR, case_names=case_names, versions="2024.2", parallel=False)
run_instances(INSTANCE_DIR, case_names=case_names, versions="2022.1", parallel=False)
#run_instances(INSTANCE_DIR, case_names=case_names, versions="2023.2", parallel=False)