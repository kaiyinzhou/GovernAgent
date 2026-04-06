# -*- coding:utf-8 -*-

import json
import os

import tqdm

from utils import async_process_queries


def read_datas(files):
    datas = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                datas.append(data)
    return datas

def read_datas(file):
    with open(file, 'r', encoding='utf-8') as f:
        datas = [json.loads(line.strip()) for line in f if line.strip()]
    return datas


if __name__ == "__main__":
    import asyncio
    # 推理测试样本，计算性能
    url = "http://10.0.0.228:8000/v1"
    model_name = "qwen32b-train"

    file = "./train_test/new/section_tests.json"
    datas = read_datas(file)
    print("Total data samples:", len(datas))

    loop = asyncio.get_event_loop()
    batch_size = 100
    base_name = os.path.basename(file)
    output_file = "./predicts_new/{}".format(base_name)
    with open(output_file, 'w', encoding='utf-8') as f:
        batch_datas = [datas[index:index + batch_size] for index in range(0, len(datas), batch_size)]
        for batch_data in tqdm.tqdm(batch_datas, desc="Processing batches"):
            batch_query = []
            for data in batch_data:

                batch_query.append(data["conversations"][0]["content"])
            batch_predicts = loop.run_until_complete(async_process_queries(batch_query, url, model_name))
            for predict, data in zip(batch_predicts, batch_data):
                data["predict"] = json.loads(predict)["choices"][0]["message"]["content"]
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
