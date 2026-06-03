


### create conda env and build env

``` bash
conda create -n lam python=3.10 -y
```
### install requirement pkg

``` bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

cd pkg/yolo

pip install -e .

```