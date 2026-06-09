
## conda env
conda create -n lam python=3.10 -y
conda activate lam

## torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128


## set mirror
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


## edit deps 
cd pkg/eaglevl && pip install -e . && cd ../../


## other deps
pip install hf_xet sortedcontainers pynvml tensorboard


## download model weights

unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

cd datasets/robot-detection/data && wget https://yubinux.cn/tmp/images.zip && unzip images.zip && cd ../../../
python download_locate_model.py