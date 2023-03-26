import uuid
import time
import json

import requests

from openai.api_requestor import APIRequestor
from requests.models import CaseInsensitiveDict

from .llm import LLM


llm_instance = LLM()


def inject(*args, **kwargs):
    print(f"!!! {args = }")
    print(f"!!! {kwargs = }")

    url_path = args[-1]

    if "params" in kwargs and "prompt" in kwargs["params"]:
        params = kwargs["params"]
        prompt = params["prompt"]
        out_texts = llm_instance(prompt)
        result = {
            "id": str(uuid.uuid4()),
            "object": "text_completion" if "/completions" == url_path else "",
            "created": int(time.time()),
            "model": params.get("model") or "",
            "choices": [
                {"text": txt, "index": i, "logprobs": None, "finish_reason": "stop"} for i, txt in enumerate(out_texts)
            ],
            "usage": {"prompt_tokens": 776, "completion_tokens": 47, "total_tokens": 823},
        }
        result_body = json.dumps(result, ensure_ascii=False).encode("utf-8")

        resp = requests.Response()
        resp.headers = CaseInsensitiveDict()
        resp.headers["Content-Type"] = "application/json"
        resp.status_code = 200
        resp._content = result_body
        return resp
    else:
        return requests.Response()


origin_request_raw = APIRequestor.request_raw


def inject_test(*args, **kwargs):
    print(f"!!! {args = }")
    print(f"!!! {kwargs = }")

    resp = origin_request_raw(*args, **kwargs)
    print(f"{resp.json() = }")
    return resp


APIRequestor.request_raw = inject
