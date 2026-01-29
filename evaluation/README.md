# Set Environment

1. Docker image:
   For MI30X:
   ```
   rocm/ali-private:ubuntu22.04_rocm6.4.3.127_sglang_5f2ee42_vllm_e858fc9_aiter_1b3efa9_20251202
   ```
2. Install aiter main branch:
   ```
   pip uninstall aiter
   git clone -b main git@github.com:ROCm/aiter.git
   cd aiter
   git submodule sync && git submodule update --init --recursive
   # for MI308
   PREBUILD_KERNELS=1 GPU_ARCHS=gfx942 python3 setup.py install
   ```
3. Install sglang 

   ```
   git clone -b main https://github.com/sgl-project/sglang.git
   cd sglang
   pip install --upgrade pip
   cd sgl-kernel
   pip uninstall sgl-kernel
   python setup_rocm.py install
   export PYTHONPATH=<you_sglang_path/sglang/python>
   ```

# Launch server
  ```
export PYTHONPATH=/home/yajizhan/qwen_code/sglang/python
model=/mnt/raid0/models/Qwen3-235B-A22B-FP8-dynamic
export SGLANG_USE_AITER=1
export SGLANG_USE_TRITON_MOE=0
export SGLANG_MOE_PADDING=0
TP=8
EP=1

echo "launching ${model}"
echo "TP=${TP}"
echo "EP=${EP}"

python3 -m sglang.launch_server \
    --model-path ${model} \
    --host localhost \
    --port 9000 \
    --tp-size ${TP} \
    --ep-size ${EP} \
    --trust-remote-code \
    --chunked-prefill-size 16384 \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --max-prefill-tokens 16384 \
    --cuda-graph-max-bs 128 \
    --max-running-requests 128 \
    --mm-attention-backend aiter_attn \
    --disable-cuda-graph

  ```
 
# client
```
python3 requset_simple.py or python3 requset_simple.py
```
You can run the same request twice and check whether there is a diff in the output results.


# Notes
If you want to switch to SGLang triton MoE, launch the service with this environment variable.
```
export SGLANG_USE_AITER=1
export SGLANG_USE_TRITON_MOE=0
```
LayerNorm may throw an error if aiter is not used. 

Modify sglang/python/sglang/srt/layers/layernorm.py:forward_hip

```
def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        return self.forward_native(x, residual, **kwargs)
```
