
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

# # 设置Delphes路径
export DELPHES_PATH="/eos/home-y/youpeng/research/whg/MG5_aMC_v2_9_24/Delphes"
export ROOT_INCLUDE_PATH="$DELPHES_PATH:$DELPHES_PATH/external/ExRootAnalysis:$ROOT_INCLUDE_PATH"

# # 设置库路径
export LD_LIBRARY_PATH="$DELPHES_PATH:$DELPHES_PATH/external/ExRootAnalysis:$LD_LIBRARY_PATH"
