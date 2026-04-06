# -*- coding:utf-8 -*-

import copy
import json
import random

from transformers import AutoTokenizer

from conflation_revise import build_conflation
from duplication_revise import build_duplication
from incomplete_revise import build_incomplete
from name_error_revise import build_name_error

tokenizer_kwargs = {
    "cache_dir": None,
    "use_fast": False,
    "revision": "main",
    "use_auth_token": None,
}

# tokenizer = AutoTokenizer.from_pretrained("../tokenizer", **tokenizer_kwargs)

tokenizer = AutoTokenizer.from_pretrained("/home/share/sft/qwen2.5/LLM-14b")

random.seed(42)

label_explain = json.load(open("../configs/label_explain.json", 'r', encoding='utf-8'))
section_str = label_explain["Section"]
field_str = label_explain["Field"]


def read_datas(files):
    datas = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                datas.append(data)
    return datas


def build_train_test(duplications, conflations, name_errors, incompletes, rate=0.2):
    # 每类型数据集都分别采样测试集，最后汇总成总测试集
    random.shuffle(duplications)
    random.shuffle(conflations)
    random.shuffle(name_errors)
    random.shuffle(incompletes)

    test_num_dup = int(len(duplications) * rate)
    test_num_con = int(len(conflations) * rate)
    test_num_name = int(len(name_errors) * rate)
    test_num_incom = int(len(incompletes) * rate)
    # 采样测试集
    test_duplications = duplications[:test_num_dup]
    test_conflations = conflations[:test_num_con]
    test_name_errors = name_errors[:test_num_name]
    test_incompletes = incompletes[:test_num_incom]
    # 采样训练集
    train_duplications = duplications[test_num_dup:]
    train_conflations = conflations[test_num_con:]
    train_name_errors = name_errors[test_num_name:]
    train_incompletes = incompletes[test_num_incom:]
    # 汇总训练集
    total_train = train_duplications + train_conflations + train_name_errors + train_incompletes

    new_train_datas = []
    for data in total_train:
        data["meta"] = json.dumps(data["meta"], ensure_ascii=False)
        new_train_datas.append(data)

    # 汇总测试集
    total_test = test_duplications + test_conflations + test_name_errors + test_incompletes
    new_total_test = []
    for data in total_test:
        data["meta"] = json.dumps(data["meta"], ensure_ascii=False)
        new_total_test.append(data)
    new_train_datas, new_total_test = filter_duplicated(new_train_datas, new_total_test)
    return new_train_datas, new_total_test


def filter_length(datas, max_length):
    outs = []
    for data in datas:
        source = data["conversations"][0]["content"]
        target = data["conversations"][1]["content"]
        messages = [
            {"role": "user", "content": source},
            {"role": "user", "content": target},
        ]
        JinJa = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"

        query_id = tokenizer.apply_chat_template(messages, tokenize=True, chat_template=JinJa,
                                                 add_generation_prompt=True)
        if len(query_id) < max_length:
            outs.append(data)
    return outs


def write_datas(datas, outfile):
    with open(outfile, 'w', encoding='utf-8') as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def filter_duplicated(datas1, datas2):
    has_set = set()
    new_outs1 = []
    new_outs2 = []
    for data in datas1:
        content_query = data["conversations"][0]["content"]
        if content_query in has_set:
            continue
        has_set.add(content_query)
        new_outs1.append(data)
    for data in datas2:
        content_query = data["conversations"][0]["content"]
        if content_query in has_set:
            continue
        has_set.add(content_query)
        new_outs2.append(data)
    return datas1, datas2


def note_section_split(datas):
    random.shuffle(datas)
    notes = []
    sections = []
    for data in datas:
        ctype = json.loads(data["meta"])["ctype"]
        if "note" in ctype.lower():
            notes.append(data)
        else:
            sections.append(data)
    return notes, sections


if __name__ == "__main__":
    files = [
        "../train_test_datas/base/test_data.json",
        "../train_test_datas/base/train_data.json",
        "../train_test_datas/base/jiwai_data.json",
    ]
    datas = read_datas(files)
    datas1 = copy.deepcopy(datas)
    duplications = build_duplication(datas1)
    datas2 = copy.deepcopy(datas)
    conflations = build_conflation(datas2, section_str, field_str)
    datas3 = copy.deepcopy(datas)
    name_errors = build_name_error(datas3, section_str, field_str)
    datas4 = copy.deepcopy(datas)
    incompletes = build_incomplete(datas4, section_str, field_str)
    trains, tests = build_train_test(duplications, conflations, name_errors, incompletes, rate=0.2)
    trains = filter_length(trains, 7600)
    tests = filter_length(tests, 7600)
    print("Train samples:", len(trains))
    print("Test samples:", len(tests))
    train_note, train_section = note_section_split(trains)
    test_note, test_section = note_section_split(tests)
    print(len(train_note), len(train_section), len(test_note), len(test_section))

    write_datas(train_note, outfile="./train_test/new/note_trains.json")
    write_datas(train_section, outfile="./train_test/new/section_trains.json")
    write_datas(test_note, outfile="./train_test/new/note_tests.json")
    write_datas(test_section, outfile="./train_test/new/section_tests.json")

    trains = train_note + train_section
    tests = test_note + test_section

    random.shuffle(trains)
    write_datas(trains, outfile="./train_test/new/trains.json")
    write_datas(tests, outfile="./train_test/new/tests.json")
