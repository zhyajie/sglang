export PYTHONPATH=/home/yajizhan/qwen_code/sglang_main/python
export SGLANG_FORCE_TORCH_SDPA=1
export SGLANG_TORCH_PROFILER_DIR=./sglang_qwen_profling
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_PROFILE_RECORD_SHAPES=1
export CUDA_VISIBLE_DEVICES=0
export SGLANG_CACHE_DIT_ENABLED=false
sglang serve  \
	--model-path /mnt/raid0/pretrained_model/Qwen-Image-Edit-2511 \
	--num-gpus 1 \
	--ulysses-degree 1 \
	--tp-size 1 \
	--image-encoder-precision bf16 \
	--vae-precision bf16 \
	--host 0.0.0.0 \
	--port 40000 \
	--text-encoder-cpu-offload False \
   	--vae-cpu-offload False \
    	--image-encoder-cpu-offload False \
    	--use-fsdp-inference False \
	--dit-cpu-offload False \
       	2>&1 | tee /home/yajizhan/dev/qwen_trace.log
    #--lora_path /mnt/raid0/pretrained_model/Qwen-Image-Edit-2511-Lightning \
    	#--enable-torch-compile \
