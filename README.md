

<div align="center">
  <h1 style="font-size: 32px; font-weight: bold;"> MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn Reinforcement Learning </h1>
<br>

<p align="left">
📖 <a href="https://arxiv.org/abs/xxx" target="_blank">Paper</a> · 
⭐ <a href="https://github.com/yellow-binary-tree/MMDuet2" target="_blank">GitHub</a> · 
📊 <a href="https://huggingface.co/datasets/wangyueqian/MMDuet2-data" target="_blank">Dataset</a> · 
🤗 <a href="https://huggingface.co/wangyueqian/MMDuet2" target="_blank">Checkpoints</a>
</p>
</div>


Key Features:
- MMDuet2 is a Video MLLM for **proactive interaction**, which means that it can not only reply right after the user's turn, but also at any approprite and timely moment during the video playback.

- With only a 3B model, MMDuet2 is **lightweight** and **fast** for real-time interaction.

- Responses are neither too sparse nor too dense and repetitive, which was a common issue in previous works.

(Demo Video Here)


## Quick Start: A Real-World Demo on your own laptop camera!
Here we assume you have a GPU server as backend, and a laptop with camera as frontend:

- On the GPU server, create conda environment and start the backend server:
```bash
cd demo/
conda create -n mmduet2-infer python=3.10
conda activate mmduet2-infer
pip install -r requirements.txt
python api_server.py
```

- Download `demo/frontend.py` to laptop and start the frontend:
```bash
pip install requests, opencv-python
python frondend.py --server_url http://xxx.xxx.xxx.xxx:8000   # (your server ip)
```
After starting the frontend, you can type in the terminal to input your text, and type "RESET" to remove all previous frames and messages.


## Training and Inference
- For SFT, follow the instructions in [train/README.md](train/README.md)
- For RL, follow the instructions in [rl/README.md](rl/README.md)
- For proactive inference and evaluation, follow the instructions in [proactive_eval/README.md](proactive_eval/README.md)

- When inference on offline video understanding (Video-MME, LongVideoBench, etc.), MMDuet2 is identical to [Qwen2.5-VL-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct). You can use frameworks including [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) just like working on Qwen2.5-VL.


## Star Count


## Licence


## Acknowledgement
We thank the following projects for their open-source contributions:
- [TimeChat-Online](https://github.com/yaolinli/TimeChat-Online)
- [ms-swift](https://github.com/modelscope/ms-swift)
- [verl](https://github.com/volcengine/verl)
- [SGLang](https://github.com/sgl-project/sglang)
