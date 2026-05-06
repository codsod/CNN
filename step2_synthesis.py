from pst_syn_process import *
from pre_syn_process import *

import os
# use os to add include path for C++ compiler
os.environ["CPLUS_INCLUDE_PATH"] = os.getcwd()
INSTANCE_DIR = os.path.join(ROOT_DIR, "instances_tp8_syn_kv260")

# compile all
configs = [
    # ("LOG",  "flp", "2022.1"),  # PLEASE use 2022.1
    # ("POOL",  "flp", "2022.1"),   # pass, two questions are found
    # ("SQUARE", "flp", "2022.1"),   # psss 
    # ("LINEAR", "flp", "2022.1"),   # pass, requantization is needed
    # ("GELU", "flp", "2022.1"),     # pass
    # ("CONV_BR", "flp", "2022.1"),    # pass, two questions are found
    # ("CONV_SPATIAL", "flp", "2022.1"),  # no pass
    ("CONCAT", "flp", "2022.1"),   # pass
    # ("QUANT_INPUT", "flp", "2022.1"), # pass
    # ("CNN_top",           "flp",      "2022.1"),  # stp for performance
]
case_names, pipeline_styles, versions = zip(*configs)
create_subprojects(INSTANCE_DIR, case_names=case_names, overwrite=True)
create_tcls       (INSTANCE_DIR, case_names=case_names, do_csim=False, do_csynth=True, do_cosim=False, do_syn=True, do_impl=False, pipeline_styles=pipeline_styles)
# launch the tcl files
run_instances(INSTANCE_DIR, case_names=case_names, versions=versions, parallel=True)