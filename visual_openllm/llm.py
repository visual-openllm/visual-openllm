from typing import List
import re

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from .chatglm.modeling_chatglm import ChatGLMForConditionalGeneration

ACTION_MAP = {
    "t2i": "Generate Image From User Input Text",
}


def find_all_images(input: str):
    return re.findall(r"(image/.+?\.png)", input)


class PromptParser:
    def __init__(self):
        self._history = []

    def parse_prompt(self, prompt: str):
        _, chat_history, input, agent_scratchpad = [i.strip() for i in prompt.split("==#==")]

        print(f"{chat_history = }")
        print(f"{input = }")

        return chat_history, input, agent_scratchpad

    def parse_output(self, output: str):
        m = re.search(r"<(\w+)>(.+)", output)
        if m:
            action = ACTION_MAP[m.group(1)]
            action_input = m.group(2)
            return action, action_input
        return None


class LLM:
    PROMPT_FORMAT = """Instruction: {input_text}\nAnswer: """

    def __init__(self):
        self.device = "cuda:0"
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", device_map="auto")
        self.model.to(self.device)
        self.model = PeftModel.from_pretrained(self.model, "visual-openllm/visual-openllm-chatglm-6b-rola")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

        self.prompt_parser = PromptParser()

    def __call__(self, input_texts: List[str]) -> List[str]:
        input_text = input_texts[0]
        chat_history, input, agent_scratchpad = self.prompt_parser.parse_prompt(input_text)
        if "Observation: image" in agent_scratchpad:
            images = "\n".join(find_all_images(agent_scratchpad))
            output = f" No\nAI: 这是你要的图片: {images}"
            return [output]

        with torch.no_grad():
            format_input = self.PROMPT_FORMAT.format(input_text=input)
            ids = self.tokenizer.encode(format_input)
            input_ids = torch.LongTensor([ids]).to(self.device)
            out = self.model.generate(input_ids=input_ids, max_length=512, do_sample=False, temperature=0)
            out_text = self.tokenizer.decode(out[0].detach()[len(ids) :])
            result = self.prompt_parser.parse_output(out_text)
            if result:
                action, action_input = result
                output = f" Yes\nAction: {action}\nAction Input: {action_input}"
            else:
                output = f" No\nAI: {out_text}"

            return [output]
