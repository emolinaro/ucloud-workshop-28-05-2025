import logging
import json
import difflib
import re
import os
import requests
import pytesseract

from PIL import Image, ImageOps
from io import BytesIO
from typing import Union, List, Dict, Optional, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
from label_studio_sdk.label_interface.object_tags import ImageTag, ParagraphsTag
from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag

logger = logging.getLogger(__name__)

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
def chat_completion_call(messages, params, *args, **kwargs):
    client = OpenAI(
        base_url=params.get('base_url', TRITONInteractive.TRITON_ENDPOINT),
        api_key='triton',  # dummy key required by OpenAI class
    )

    request_params = {
        "messages": messages,
        "model": params.get("model", TRITONInteractive.TRITON_MODEL),
        "n": params.get("num_responses", TRITONInteractive.NUM_RESPONSES),
        "temperature": params.get("temperature", TRITONInteractive.TEMPERATURE),
        "max_tokens": params.get("max_tokens", TRITONInteractive.MAX_TOKENS)
    }

    return client.chat.completions.create(**request_params, timeout=30)

def gpt(messages: Union[List[Dict], str], params, *args, **kwargs):
    if isinstance(messages, str):
        messages = [
                {"role": "system", "content": "detailed thinking off\n"},
                {"role": "assistant", "content":"<think>\n</think>\n"},
                {"role": "user", "content": messages}
        ]

    completion = chat_completion_call(messages, params)
    raw_responses = [choice.message.content for choice in completion.choices]

    trimmed_responses = []
    for resp in raw_responses:
        match = list(re.finditer(r'\bassistant\b', resp, flags=re.IGNORECASE))
        trimmed = resp[match[-1].end():].strip() if match else resp.strip()
        trimmed_responses.append(trimmed)

    return trimmed_responses

class TRITONInteractive(LabelStudioMLBase):
    PROMPT_PREFIX = os.getenv("PROMPT_PREFIX", "prompt")
    USE_INTERNAL_PROMPT_TEMPLATE = bool(int(os.getenv("USE_INTERNAL_PROMPT_TEMPLATE", 1)))
    DEFAULT_PROMPT = os.getenv('DEFAULT_PROMPT')
    PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", '**Source Text**:\n\n"{text}"\n\n**Task Directive**:\n\n"{prompt}"')
    PROMPT_TAG = "TextArea"
    SUPPORTED_INPUTS = ("Image", "Text", "HyperText", "Paragraphs")
    NUM_RESPONSES = int(os.getenv("NUM_RESPONSES", 1))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 500))

    TRITON_ENDPOINT = os.getenv("TRITON_ENDPOINT", "http://localhost:8000/v1")
    TRITON_MODEL = os.getenv("TRITON_MODEL")

    def setup(self):
        if self.DEFAULT_PROMPT and os.path.isfile(self.DEFAULT_PROMPT):
            with open(self.DEFAULT_PROMPT) as f:
                self.DEFAULT_PROMPT = f.read()

    def _ocr(self, image_url):
        image = ImageOps.exif_transpose(Image.open(BytesIO(requests.get(image_url).content)))
        return pytesseract.image_to_string(image)

    def _get_text(self, task_data, object_tag):
        data = task_data.get(object_tag.value_name)
        if isinstance(object_tag, ImageTag):
            return self._ocr(data)
        elif isinstance(object_tag, ParagraphsTag):
            return json.dumps(data)
        return data

    def _get_prompts(self, context, prompt_tag) -> List[str]:
        if context and (result := context.get('result')):
            return [item['value']['text'] for item in result if item.get('from_name') == prompt_tag.name]
        elif prompt := self.get(prompt_tag.name):
            return [prompt]
        elif self.DEFAULT_PROMPT:
            if self.USE_INTERNAL_PROMPT_TEMPLATE:
                return []
            return [self.DEFAULT_PROMPT]
        return []

    def _match_choices(self, response: List[str], original_choices: List[str]) -> List[str]:
        predicted_classes = response[0].splitlines()
        return [original_choices[max(range(len(original_choices)), key=lambda i: difflib.SequenceMatcher(None, pred, original_choices[i]).ratio())] for pred in predicted_classes]

    def _find_prompt_tags(self) -> Tuple[ControlTag, ObjectTag]:
        li = self.label_interface
        from_name, to_name, _ = li.get_first_tag_occurence(self.PROMPT_TAG, self.SUPPORTED_INPUTS, name_filter=lambda s: s.startswith(self.PROMPT_PREFIX))
        return li.get_control(from_name), li.get_object(to_name)

    def _find_choices_tag(self, object_tag):
        try:
            li = self.label_interface
            from_name, _, _ = li.get_first_tag_occurence('Choices', self.SUPPORTED_INPUTS, to_name_filter=lambda s: s == object_tag.name)
            return li.get_control(from_name)
        except:
            return None

    def _find_textarea_tag(self, prompt_tag, object_tag):
        try:
            li = self.label_interface
            from_name, _, _ = li.get_first_tag_occurence('TextArea', self.SUPPORTED_INPUTS, name_filter=lambda s: s != prompt_tag.name, to_name_filter=lambda s: s == object_tag.name)
            return li.get_control(from_name)
        except:
            return None

    def _validate_tags(self, choices_tag: str, textarea_tag: str) -> None:
        if not choices_tag and not textarea_tag:
            raise ValueError('No supported tags found: <Choices> or <TextArea>')

    def _generate_normalized_prompt(self, text: str, prompt: str, task_data: Dict, labels: Optional[List[str]]) -> str:
        return self.PROMPT_TEMPLATE.format(text=text, prompt=prompt, labels=labels) if self.USE_INTERNAL_PROMPT_TEMPLATE else prompt.format(labels=labels, **task_data)

    def _generate_response_regions(self, response: List[str], prompt_tag, choices_tag: ControlTag, textarea_tag: ControlTag, prompts: List[str]) -> List:
        regions = []
        if choices_tag and response:
            regions.append(choices_tag.label(self._match_choices(response, choices_tag.labels)))
        if textarea_tag:
            regions.append(textarea_tag.label(text=response))
        regions.append(prompt_tag.label(text=prompts))
        return regions

    def _predict_single_task(self, task_data: Dict, prompt_tag: Any, object_tag: Any, prompt: str, choices_tag: ControlTag, textarea_tag: ControlTag, prompts: List[str]) -> Dict:
        text = self._get_text(task_data, object_tag)
        labels = choices_tag.labels if choices_tag else None
        norm_prompt = self._generate_normalized_prompt(text, prompt, task_data, labels)
        response = gpt(norm_prompt, self.extra_params)
        regions = self._generate_response_regions(response, prompt_tag, choices_tag, textarea_tag, prompts)
        return PredictionValue(result=regions, score=0.1, model_version=str(self.model_version))

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        predictions = []
        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self._get_prompts(context, prompt_tag)
        if prompts:
            prompt = "\n".join(prompts)
            choices_tag = self._find_choices_tag(object_tag)
            textarea_tag = self._find_textarea_tag(prompt_tag, object_tag)
            self._validate_tags(choices_tag, textarea_tag)
            for task in tasks:
                task_data = self.preload_task_data(task, task['data'])
                predictions.append(self._predict_single_task(task_data, prompt_tag, object_tag, prompt, choices_tag, textarea_tag, prompts))
        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **additional_params):
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED'):
            return
        prompt_tag, _ = self._find_prompt_tags()
        prompts = self._get_prompts(data['annotation'], prompt_tag)
        if not prompts:
            return
        prompt = '\n'.join(prompts)
        current_prompt = self.get(prompt_tag.name)
        if current_prompt:
            diff = "\n".join(line for line in difflib.unified_diff(current_prompt.splitlines(), prompt.splitlines(), lineterm="") if line.startswith('+') and not line.startswith('+++'))
            if not diff:
                return
        self.set(prompt_tag.name, prompt)
        self.bump_model_version()

