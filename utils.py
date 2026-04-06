# -*- coding:utf-8 -*-

import asyncio
import json
import os
import re

from openai import AsyncOpenAI

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


os.environ["TOKENIZERS_PARALLELISM"] = "true"


class _FallbackTokenizer:
    def __call__(self, text):
        return {"input_ids": list(text)}


def _load_tokenizer():
    if AutoTokenizer is None:
        return _FallbackTokenizer()

    model_path = os.environ.get("UTIL_TOKENIZER_PATH")
    if not model_path:
        return _FallbackTokenizer()

    try:
        return AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except Exception:
        return _FallbackTokenizer()


tokenizer = _load_tokenizer()


async def async_query_openai(query, url, model_name):
    client = AsyncOpenAI(api_key="EMPTY", base_url=url)
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}],
        temperature=0.0,
        top_p=1.0,
    )
    return response.model_dump_json()


async def async_process_queries(queries, url, model_name):
    return await asyncio.gather(*(async_query_openai(query, url, model_name) for query in queries))


def _empty_response():
    return json.dumps({"choices": [{"message": {"content": "[]"}}]})


async def _single_query_safe(query, model_name, client):
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": query}],
            temperature=0.0,
            top_p=1.0,
        )
        return response.model_dump_json()
    except Exception:
        return _empty_response()


async def process_queries_safe(queries, model_name, client):
    results = await asyncio.gather(*(_single_query_safe(query, model_name, client) for query in queries))
    return [res if res else _empty_response() for res in results]


def find_all_start_indices(a, content):
    indices = []
    start = 0
    while start < len(content):
        start = content.find(a, start)
        if start == -1:
            break
        indices.append(start)
        start += 1
    return indices


def find_all_end_indices(b, content):
    indices = []
    start = 0
    while start < len(content):
        start = content.find(b, start)
        if start == -1:
            break
        indices.append(start + len(b) - 1)
        start += 1
    return indices


def _token_len(text):
    try:
        return len(tokenizer(text)["input_ids"])
    except Exception:
        return len(text)


def find_closest_substring(string, start_str, end_str, length, current_offset=None):
    string = string.replace(":", "：")
    start_str = start_str.replace(":", "：")
    end_str = end_str.replace(":", "：")

    starts = find_all_start_indices(start_str, string)
    ends = find_all_end_indices(end_str, string)
    if current_offset is not None:
        starts = [s for s in starts if s >= current_offset]
        ends = [e for e in ends if e >= current_offset]

    best = None
    best_diff = float("inf")

    for start in starts:
        for end in ends:
            if start <= end:
                candidate = string[start:end + 1]
                diff = abs(_token_len(candidate) - length)
                if diff < best_diff:
                    best = (candidate, start, end)
                    best_diff = diff

    if best is None:
        if current_offset is None:
            return None
        return None, current_offset

    candidate, _, end = best
    if current_offset is None:
        return candidate
    return candidate, end + 1
