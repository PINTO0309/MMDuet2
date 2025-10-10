## Proactive Evaluation of MMDuet2
Here we provide a demo of evaluating MMDuet2 on [ProactiveVideoQA](https://github.com/yellow-binary-tree/ProactiveVideoQA).

### Create conda environment
```bash
conda create -n mmduet2-infer python=3.10
conda activate mmduet2-infer
pip install -r requirements.txt
python api_server.py
```

### Download inference data
Follow the instructions in [MMDuet2-data](https://huggingface.co/datasets/wangyueqian/MMDuet2-data) to prepare the dataset, and move the `evaluate` sub folder to `./data/annotations`.

We will use two types of data with the same content but different formats:
- `xxx-proactivevideoqa_format.json` is identical to the annotation files in [ProactiveVideoQA](https://github.com/yellow-binary-tree/ProactiveVideoQA). It contains ground truth answer text and reply spans, which is used to evaluate model outputs.
- `xxx-frame_input_format.json` is converted from `-proactivevideoqa_format.json`, by enumerating the text and frame input of each turn. It is used to provide model input in the inference process.

### Run!
```bash
# first inference
bash ./scripts/inference.sh
# then evaluate
bash ./scripts/evaluate.sh
```