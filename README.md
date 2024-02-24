# Visual OpenLLM

一种基于开源模型, 已交互方式连接不同视觉模型的开源工具。

- 基于 ChatGLM + Visual ChatGPT + Stable Diffusion
- 开源版的"文心一言"


## 📝 Changelog

- [2024.2.24] 增加对Chatglm3的支持，新增vqa和pix2pix功能。(thank [@MrChen314](https://github.com/MrChen314) for this contribution )
- [2023.3.27] 代码开源


## Demo：

![](assets/demo.gif)


## 运行:

run with `Chatglm3-6B`(Default)
```
python run.py --load_llm Chatglm3
```
run with `Chatglm`

```
python run.py --load_llm Chatglm
```

## Todo:

- 支持多轮chat
- 支持其他视觉工具
- 支持其他LLM
