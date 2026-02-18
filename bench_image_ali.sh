export PYTHONPATH=/home/yajizhan/qwen_code/sglang_main/python
python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
  --backend sglang-image \
  --task image-to-image \
  --port 40000 \
  --dataset vbench \
  --dataset-path /home/yajizhan/dev/benchmark_data_multiImage \
  --num-prompts 5 \
  --max-concurrency 1 \
  --width 768 \
  --height 1024 \
  --num-inference-steps 40 \
  --guidance-scale 4 

