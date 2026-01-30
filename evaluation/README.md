# Set Environment

1. Docker image:
   For MI30X:
   ```
   rocm/ali-private:ubuntu22.04_rocm6.4.3.127_sglang_mha_batch_prefill_20260122
   ```

# Launch server
  ```
  cd /opt/sglang_qwen3/moe
  bash server.sh
  ```
 
# client
```
python3 requset_simple.py or python3 requset_simple.py
```
You can run the same request twice and check whether there is a diff in the output results.


If you want to test the MoE operator unit test
```
python3 test_moe_bit_consistency.py
```