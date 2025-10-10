# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image

import torch
from pydantic import BaseModel, model_validator, ConfigDict
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")
BASE_CHAT_HISTORY = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "I am a user."}]
logger.warn(f"using system prompt: {SYSTEM_PROMPT}. This system prompt will be added to the first turn of the conversation if the first turn is not a system prompt.")

class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class Message(BaseModel):
    role: str
    content: Union[str, Dict, List]
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    input_ids: List[int]
    prompt_ids: List[int]
    response_ids: List[int]
    attention_mask: List[int]
    prompt_attention_mask: List[int]
    response_attention_mask: List[int]
    position_ids: List[int]
    prompt_position_ids: List[int]
    response_position_ids: List[int]
    loss_mask: List[int]
    prompt_loss_mask: List[int]
    response_loss_mask: List[int]
    reward_scores: Dict[str, float]
    max_prompt_len: int
    max_response_len: int = 8192
    max_model_len: int = 32768
    metrics: Dict[str, List[Any]] = {}

    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    images: List[Union[str, Image]] = list()

    use_inference_chat_template: bool
    enable_tokenization_sanity_check: bool
    generation_prompt_ids: List[int]
    base_conv_wo_gen_prompt_end_pos: int
    base_conv_with_gen_prompt_end_pos: int

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if not (messages := values.get("messages")):
            raise ValueError("messages is required for AsyncRolloutRequest initialization")
        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError("max_prompt_len is required for AsyncRolloutRequest initialization")
        if not (tokenizer := values.pop("tokenizer", None)):
            raise ValueError("tokenizer is required for AsyncRolloutRequest initialization")
        processor = values.pop("processor", None)
        is_qwen2_5_vl = processor is not None and 'Qwen2_5_VL'in processor.__class__.__name__

        # add system prom
        if not len(messages) or messages[0]['role'] != "system":
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        else:
            # if messages[0]['content'] != SYSTEM_PROMPT:
            #     logger.warn(f"System prompt {messages[0]['content']} is not the same as the global system prompt set in workers.rollout.schemas: {SYSTEM_PROMPT=}")
            pass        # if the data has a system prompt, use the system prompt in data.

        values["messages"] = [Message.model_validate(msg) for msg in messages]
        tools = [tool.model_dump() for tool in tool_schemas] if (tool_schemas := values.get("tool_schemas", [])) else None
        if is_qwen2_5_vl:
            from qwen_vl_utils import process_vision_info
            prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=False)
            images, videos = process_vision_info(messages)
            model_input = processor(text=[prompt], images=images, videos=videos, return_tensors='pt')
            tokens_without_prompt = model_input['input_ids'][0].tolist()

            # update_visual_data
            new_images = []
            for msg in values["messages"]:
                for ele in msg.content:
                    if 'image' in ele:
                        new_images.append(ele['image'])
                    elif 'image_url' in ele:
                        new_images.extend(ele['image_url'])
            values['images'] = new_images

            # pixel_values 和 image_grid_thw 在rollout中不会用到，是之后计算probs时用的
            values['pixel_values'] = model_input.get('pixel_values', None)
            values['image_grid_thw'] = model_input.get('image_grid_thw', None)

        else:
            tokens_without_prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=True)

        if not values.get("input_ids") or not values.get("attention_mask"):
            if is_qwen2_5_vl:
                # When using sglang to rollout online interaction data for qwen2_5_vl, the input_ids here are not used during rollout. sglang will preprocess + tokenize based on self.messages.
                # However, when calculating probabilities (including action model and ref model), AutoModelForCausalLM will be directly fed with the input_ids and attention_mask here.
                # Therefore, we must record the input_ids attention_mask loss_mask that correspond to the visual token length.
                # The same applies to recording the corresponding input_ids for the length in self.add_user_response_messages.

                prompt = tokenizer.apply_chat_template(messages, tools=[tool.model_dump() for tool in tool_schemas], add_generation_prompt=True, tokenize=False)
                tokenization_dict_with_prompt = processor(text=[prompt], images=images, videos=videos)
                values["input_ids"], values["attention_mask"] = tokenization_dict_with_prompt["input_ids"][0], tokenization_dict_with_prompt["attention_mask"][0]
            else:
                tokenization_dict_with_prompt = tokenizer.apply_chat_template(messages, tools=[tool.model_dump() for tool in tool_schemas], add_generation_prompt=True, tokenize=True, return_dict=True)
                values["input_ids"], values["attention_mask"] = tokenization_dict_with_prompt["input_ids"], tokenization_dict_with_prompt["attention_mask"]

            if len(values["input_ids"]) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an error for this case in the future.
                logger.warning(f"Prompt {values['batch_data_id']} length {len(values['input_ids'])} greater than max_prompt_len {max_prompt_len} after applied chat template with tools.")

        values["prompt_ids"], values["prompt_attention_mask"] = values["input_ids"], values["attention_mask"]
        values["position_ids"] = values["prompt_position_ids"] = compute_position_id_with_mask(torch.tensor(values["attention_mask"])).tolist()
        values["loss_mask"] = values["prompt_loss_mask"] = [0] * len(values["input_ids"])
        values["generation_prompt_ids"] = values["input_ids"][len(tokens_without_prompt) :]
        values["base_conv_wo_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=False, tokenize=False))
        values["base_conv_with_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=True, tokenize=False))
        return values

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids += new_input_ids
        attention_mask = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask += attention_mask
        self.loss_mask += [int(loss_mask)] * len(new_input_ids)
        self.position_ids += (compute_position_id_with_mask(torch.tensor(attention_mask)) + (self.position_ids[-1] + 1)).tolist()

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def _update_visual_data(self, new_messages, model_input):
        # extract all images from the new messages
        new_images = []
        for msg in new_messages:
            for ele in msg.content:
                if 'image' in ele:
                    new_images.append(ele['image'])
                elif 'image_url' in ele:
                    new_images.extend(ele['image_url'])
        self.images.extend(new_images)

        if model_input is not None:
            new_pixel_values = model_input.get('pixel_values', None)
            new_image_grid_thw = model_input.get('image_grid_thw', None)
            if new_pixel_values is not None:
                self.pixel_values = torch.cat([self.pixel_values, new_pixel_values], dim=0) if self.pixel_values is not None else new_pixel_values
            if new_image_grid_thw is not None:
                self.image_grid_thw = torch.cat([self.image_grid_thw, new_image_grid_thw], dim=0) if self.image_grid_thw is not None else new_image_grid_thw

    def get_image_data(self):
        return self.images

    def get_prompt(self, tokenizer: PreTrainedTokenizer) -> str:
        return tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], add_generation_prompt=True, tokenize=False)

    def get_generation_prompt_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        generation_prompt_ids = [] if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids else self.generation_prompt_ids
        if generation_prompt_ids:
            self._update_input_ids(generation_prompt_ids, attention_mask=True, loss_mask=False)

        if self.use_inference_chat_template:
            return tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=True, tokenize=True)
        else:
            return self.input_ids

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
    ) -> None:
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
        content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, self.messages[-1]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(content[self.base_conv_with_gen_prompt_end_pos :], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)

    def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, contents: list[str]) -> None:
        if not contents:
            return
        self.messages.extend([Message(role="tool", content=content) for content in contents])
        content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, *self.messages[-len(contents) :]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(content[self.base_conv_wo_gen_prompt_end_pos :], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def add_user_response_messages(self, processor: ProcessorMixin, contents: list[str]) -> None:
        if not contents:
            return
        is_qwen2_5_vl = "Qwen2_5_VL" in processor.__class__.__name__

        tokenizer = processor.tokenizer
        self.messages.extend([Message(role="user", content=json.loads(content)) for content in contents])

        # I suspect that apply_chat_template first adds BASE_CHAT_HISTORY, then removes the front part of the token ids,
        # probably to clear out the system turn that some tokenizers forcibly include
        new_messages = self.messages[-len(contents) :]
        if is_qwen2_5_vl:
            from qwen_vl_utils import process_vision_info
            content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, *new_messages], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
            images, videos = process_vision_info([{'role': 'user', 'content': json.loads(content)} for content in contents])
            model_input = processor(text=[content[self.base_conv_wo_gen_prompt_end_pos :]], images=images, videos=videos, return_tensors='pt')
            content_ids = model_input['input_ids'][0].tolist()
            self._update_visual_data(new_messages, model_input)
        else:
            content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, *new_messages], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
            content_ids = tokenizer.encode(content[self.base_conv_wo_gen_prompt_end_pos :], add_special_tokens=False)

        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, float],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        if self.enable_tokenization_sanity_check:
            full_tokens = tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=True)
            # this is intentional for qwen2_5vl tokenizer, no need to warn
            # if self.input_ids != full_tokens:
            #     logger.warning("Inconsistent training and inference tokenization detected. This may lead to unexpected behavior during training. Please review your chat template to determine if this is intentional. For more information, refer to the multiturn README.md.")
            #     logger.info(f"Inference tokenization result:\n{tokenizer.decode(full_tokens, skip_special_tokens=False)}\ntraining content:\n{tokenizer.decode(self.input_ids, skip_special_tokens=False)}")

        # In case we failed to generate the assistant message and the generation prompt ids were already added to input_ids, remove them from the end of input_ids
        if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids:
            self.input_ids = self.input_ids[: -len(self.generation_prompt_ids)]
            self.attention_mask = self.attention_mask[: -len(self.generation_prompt_ids)]
            self.position_ids = self.position_ids[: -len(self.generation_prompt_ids)]
            self.loss_mask = self.loss_mask[: -len(self.generation_prompt_ids)]

        self.response_ids = self.input_ids[len(self.prompt_ids) :]
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
        self.truncate_output_ids(tokenizer)
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
