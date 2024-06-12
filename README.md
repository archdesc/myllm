# myllm
Create a large language model from scratch. 基于 https://github.com/jiahe7ay/MINI_LLM 修改的。
可以自己选择自己要的huggingface模型，调节各种参数.

代码里面用的zephyr-7b-beta模型，但是修改了参数，改为了1B模型。
稍微修改一下可以使用modescope模型。


## Installation

```conda create -n myllm python=3.10
conda activate myllm
git clone https://github.com/archdesc/myllm
cd myllm
pip install -r requirement.txt
python z-pre.py
```
