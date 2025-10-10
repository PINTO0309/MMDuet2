## RL phase
We use [verl](https://github.com/volcengine/verl) with some modifications in the RL phase to train [MMDuet2](../README.md).

For your reference, code files in `./verl/` are modified from commit [83ebd00](https://github.com/volcengine/verl/commit/83ebd00) with changes listed in `./changes/verl_patch.diff`,
and code files in `./sglang/` are modified from commit [dcae1f](https://github.com/sgl-project/sglang/commit/dcae1f) with changes listed in `./changes/sglang_patch.diff`.


## Training Process
### Create conda environment
```bash
conda create --name mmduet2_rl python=3.10
conda activate mmduet2_rl
pip install -r requirements.txt

# install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..

# install sglang and verl in this project folder
cd sglang
pip install -e .
cd ..

cd verl
pip install -e .
cd ..
```

### Download training data
Follow the instructions in [MMDuet2-data](https://huggingface.co/datasets/wangyueqian/MMDuet2-data) to prepare the dataset, and move the `rl` sub folder to `./data`


### Train!
- 2 seconds per frame, max 90 frames: 
```bash
cd verl
bash ./recipe/proactive/run_scripts/train_grpo-half_multi_half_single_question-2_sec-20s_seg_as_example-max_90_frames.sh
```

- 1 second per frame, max 180 frames: 
```bash
cd verl
bash ./recipe/proactive/run_scripts/train_grpo-half_multi_half_single_question-1_sec-20s_seg_as_example-max_180_frames.sh
```
