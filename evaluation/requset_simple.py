import requests
import json
import time

text = "你好，请介绍一下人工智能的伦理问题，包括数据隐私、算法偏见、责任归属等方面。"

url = "http://0.0.0.0:9000/generate"
headers = {
    "Content-Type": "application/json"
}
data = {
    "text": text,
    "sampling_params": {
        "temperature": 0.0,
        "max_new_tokens": 4900,
        "stop": [
            "<|im_end|>",
            "<|endoftext|>"
        ],
        "ignore_eos": False,
    },
    "stream": True,
}

start = time.time()
response = requests.post(url, json=data, headers=headers, stream=True)
response.encoding = 'utf-8'

if response.status_code == 200:
    full_text = ""
    
    for line in response.iter_lines(decode_unicode=True):
        if not line or line.strip() == "":
            continue
            
        if line.startswith('data:'):
            json_str = line[5:].strip()
            if not json_str or json_str == "[DONE]":
                continue
                
            try:
                chunk = json.loads(json_str)
                if "text" in chunk:
                    full_text = chunk["text"]
            except json.JSONDecodeError:
                pass
    
    print(f"总时间: {time.time()-start:.4f}秒")
    print(f"\n=== 最终结果 ===")
    print(full_text)
    
else:
    print(f"请求失败，状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
