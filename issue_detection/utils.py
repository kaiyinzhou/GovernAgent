# -*- coding:utf-8 -*-

import asyncio
import json
import os

from openai import AsyncOpenAI
from transformers import AutoTokenizer
import re
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import AutoTokenizer

tokenizer_kwargs = {
    "cache_dir": None,
    "use_fast": False,
    "revision": "main",
    "use_auth_token": None,
}

# tokenizer = AutoTokenizer.from_pretrained("../tokenizer", **tokenizer_kwargs)

tokenizer = AutoTokenizer.from_pretrained("/home/share/sft/qwen2.5/LLM-70b")

# tokenizer = AutoTokenizer.from_pretrained("/home/zhouky/Llama3/8b")


async def async_query_openai(query, url, model_name):
    # if model_name in {"Qwen2.5-summary"}:
    # qwen2.5-14b
    JinJa = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
    # llama
    # JinJa = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    max_length = 8192
    openai_api_key = "EMPTY"
    openai_api_base = url

    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    messages = [
        {"role": "user", "content": query},
    ]
    query_id = tokenizer.apply_chat_template(messages, tokenize=True, chat_template=JinJa, add_generation_prompt=True)
    # try:

    chat_response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_length - len(query_id),
    )
    return chat_response.model_dump_json()
    # except:
    #     return None


async def async_process_queries(queries, url, model_name):
    results = await asyncio.gather(*(async_query_openai(query, url, model_name) for query in queries))
    return results
