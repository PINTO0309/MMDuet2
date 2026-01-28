import copy
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify
from inference import ProactiveInferenceClient, ProactiveTestArguments
from dataclasses import dataclass
from transformers import HfArgumentParser
from qwen_vl_utils import process_vision_info



@dataclass
class ProactiveTestAPIArguments(ProactiveTestArguments):
    port: int = 8000
    num_frames_per_turn: int = 2


def get_args():
    args, = HfArgumentParser(ProactiveTestAPIArguments).parse_args_into_dataclasses()
    return args


class ProactiveInferenceAPIClient(ProactiveInferenceClient):
    def __init__(self, args=None, model=None, processor=None) -> None:
        super().__init__(args, model, processor)
        self.num_frames_per_turn = args.num_frames_per_turn
        self.image_buffer = list()
        self.text_buffer = ''

    def add_image(self, new_image, debug_print=False):
        print('added image:', new_image.size)
        self.image_buffer.append(new_image)
        if len(self.image_buffer) < self.num_frames_per_turn:
            return {'response': False}

        content = [{'type': 'image', 'image': image} for image in self.image_buffer]
        self.image_buffer = []
        if self.text_buffer:
            content.append({'type': 'text', 'text': self.text_buffer})
            self.text_buffer = ''

        query = {'role': 'user', 'content': content}
        self.history.append(query)
        text = self.processor.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True,
        )

        if debug_print:
            print("DEBUG text before generate:", text)

        new_image_inputs, new_video_inputs = process_vision_info([query])
        if new_image_inputs is not None:
            self.prev_image_inputs.extend(new_image_inputs)
        if new_video_inputs is not None:
            self.prev_video_inputs.extend(new_video_inputs)
        image_inputs = copy.deepcopy(self.prev_image_inputs) if self.prev_image_inputs else None
        video_inputs = copy.deepcopy(self.prev_video_inputs) if self.prev_video_inputs else None

        num_frames = self._recursive_stat_num_frames(new_image_inputs) + self._recursive_stat_num_frames(new_video_inputs)
        self.video_time += num_frames * self.frame_interval
        self.history[-1]['time'] = self.video_time

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:0")

        model_output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            past_key_values=self.past_key_values,
            return_dict_in_generate=True,
            drop_method='none', drop_threshold=1.0, drop_absolute=True,      # no dropping
            do_sample=self.do_sample, temperature=self.temperature, top_k=self.top_k
        )

        self.past_key_values = model_output.past_key_values
        output_token_ids = model_output.sequences
        output_token_ids = output_token_ids[:, inputs.input_ids.size(1):]
        reply_text = self.processor.batch_decode(output_token_ids, skip_special_tokens=True)[0]
        if query.get('must_reply', False):
            reply_text = self.must_reply_prompt + reply_text
        model_reply = {'role': 'assistant', 'content': reply_text, 'time': self.video_time}
        self.history.append(model_reply)

        if debug_print:
            print("kvcache length now:", self.past_key_values.get_seq_length())
        if reply_text.startswith('NO'):
            return {'respsone': False}
        else:
            return {'response': True, **model_reply}

    def add_text(self, new_text):
        print('added text:', new_text)
        if self.text_buffer:
            self.text_buffer += ' '
        self.text_buffer += new_text

    def reset(self):
        super().reset()
        self.image_buffer = list()
        self.text_buffer = ''
        self.history = [{'role': 'system', 'content': self.system_prompt}]


def create_app(client):
    app = Flask(__name__)

    @app.route('/add_image', methods=['POST'])
    def add_image_endpoint():
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        result = client.add_image(image, debug_print=True)
        return jsonify({**result})

    @app.route('/add_text', methods=['POST'])
    def add_text_endpoint():
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        client.add_text(data['text'])
        return jsonify({'status': 'success'})

    @app.route('/reset', methods=['POST'])
    def reset_endpoint():
        client.reset()
        return jsonify({'status': 'success'})

    return app


if __name__ == '__main__':
    args = get_args()
    client = ProactiveInferenceAPIClient(args)
    client.set_fps(frame_interval=2)
    app = create_app(client)
    app.run(host='0.0.0.0', port=args.port, debug=True)
