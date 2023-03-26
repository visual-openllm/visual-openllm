from typing import List

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from .chatglm.modeling_chatglm import ChatGLMForConditionalGeneration


class LLM:
    # PROMPT_FORMAT = """Instruction: {input_text}\nAnswer: """

    def __init__(self):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", device_map="auto")
        self.model = PeftModel.from_pretrained(self.model, "visual-openllm/visual-openllm-chatglm-6b-rola")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

    def __call__(self, input_texts: List[str]) -> List[str]:
        input_text = input_texts[0]
        with torch.no_grad():
            # input_text = self.PROMPT_FORMAT.format(input_text=input_text)
            ids = self.tokenizer.encode(input_text)
            input_ids = torch.LongTensor([ids])
            out = self.model.generate(input_ids=input_ids, max_length=150, do_sample=False, temperature=0)
            out_text = self.tokenizer.decode(out[0])
            return [out_text]
