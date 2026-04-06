# -*- coding:utf-8 -*-

import collections
import difflib
import json
from typing import List, Dict, Tuple


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def calculate_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def compute_subcontent_metrics(golds: List[Dict], predicts: List[Dict]):
    results = []
    for gold, pred in zip(golds, predicts):
        matched_golds = [0] * len(gold["sub_contents"])
        matched_preds = [0] * len(pred["sub_contents"])

        gold_subs = []
        pred_subs = []

        for sub in gold["sub_contents"]:
            gold_subs.append((sub["sub_meta"]["sec_name"], sub["sub_content"]))

        for sub in pred["sub_contents"]:
            pred_subs.append((sub["sub_meta"]["sec_name"], sub["sub_content"]))
        gold_subs = [gf for gf in gold_subs if gf[1] != '']
        pred_subs = [pf for pf in pred_subs if pf[1] != '']

        tp, fp, fn = 0, 0, 0

        for i, g_sub in enumerate(gold_subs):
            g_sec_name, g_content = g_sub

            best_j = -1
            best_sim = 0.0

            for j, p_sub in enumerate(pred_subs):
                p_sec_name, p_content = p_sub

                if matched_preds[j] == 0 and (g_sec_name == p_sec_name):
                    curr_sim = similarity(g_content, p_content)
                    if curr_sim > best_sim:
                        best_sim = curr_sim
                        best_j = j

            if best_j != -1:
                matched_golds[i] = 1
                matched_preds[best_j] = 1
                tp += best_sim

        fp = len(pred_subs) - sum(matched_preds)
        fn = len(gold_subs) - sum(matched_golds)
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        # print(f1)
        # print(pred_subs)
        # print(gold_subs)
        results.append({
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    return results


def compute_fields_metrics(golds: List[Dict], predicts: List[Dict]):
    results = []
    for gold, pred in zip(golds, predicts):
        gold_fields = []
        pred_fields = []

        for sub in gold["sub_contents"]:
            for k, v in sub['fields'].items():
                gold_fields.append((k, v))

        for sub in pred["sub_contents"]:
            for k, v in sub['fields'].items():
                pred_fields.append((k, v))
        # print(pred_fields[-1])
        gold_fields = [gf for gf in gold_fields if not gf[1] == '']
        pred_fields = [pf for pf in pred_fields if not pf[1] == '']

        matched_g = [0] * len(gold_fields)
        matched_p = [0] * len(pred_fields)

        tp, fp, fn = 0, 0, 0

        for i, (g_key, g_val) in enumerate(gold_fields):
            best_j = -1
            best_sim = 0.0
            for j, (p_key, p_val) in enumerate(pred_fields):

                if matched_p[j] == 0 and g_key == p_key:
                    curr_sim = similarity(g_val, p_val)
                    if curr_sim > best_sim:
                        best_sim = curr_sim
                        best_j = j

            # 超过相似度阈值才进行匹配
            if best_j != -1:
                matched_g[i] = 1
                matched_p[best_j] = 1
                tp += best_sim
        fp = len(pred_fields) - sum(matched_p)
        fn = len(gold_fields) - sum(matched_g)
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        # print(f1)
        # print(gold_fields)
        # print(pred_fields)
        results.append({
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
    return results


def compute_average(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    n = len(metrics)
    if n == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    precision = sum(m['precision'] for m in metrics) / n
    recall = sum(m['recall'] for m in metrics) / n
    f1 = sum(m['f1_score'] for m in metrics) / n

    # return {'precision': precision, 'recall': recall, 'f1_score': f1}
    return "{:.2f}".format(f1 * 100)


def read_jsonl(file_path: str) -> List[Dict]:
    """
    读取JSONL文件

    Args:
        file_path: JSONL文件路径

    Returns:
        数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def rebuild_datas(datas1, datas2):
    """
    重建数据格式，确保每个数据项都包含'sub_contents'和'sub_meta'
    """
    outs1 = collections.defaultdict(list)
    outs2 = collections.defaultdict(list)
    for data1, data2 in zip(datas1, datas2):
        if "hos_name" in data1['meta']:
            hos_name = data1['meta']["hos_name"]
            if data1['meta']["hos_name"] != data2['meta']["hos_name"]:
                data2['meta']["hos_name"] = data1['meta']["hos_name"]
            assert data1['meta']["hos_name"] == data2['meta']["hos_name"], "医院名称不匹配"
            outs1[hos_name].append(data1)
            outs2[hos_name].append(data2)
        else:
            outs1["same"].append(data1)
            outs2["same"].append(data2)

    return outs1, outs2


if __name__ == "__main__":

    # model_name = "qwen-agent-14b-grpo-experience"
    # model_name = "qwen-agent-14b-experience"
    # model_name = "qwen-agent-14b"
    model_name = "qwen-agent-7b"
    # model_name = "qwen-extract-7b"
    # for file_ctype in ["test", "jiwai", "mimic"]:
    for file_ctype in ["mimic"]:
        print(">>>>>>>>>>>>>>>{}<<<<<<<<<<<<<<".format(file_ctype))
        if file_ctype in ["test", "jiwai"]:
            gold_file = "./train_test_datas/base/v1/{}_data.json".format(file_ctype)
        else:
            gold_file = "./datas/note_sample_50_已标注_加噪.json"

        hos_gold_datas, hos_predict_datas = rebuild_datas(
            read_jsonl(gold_file),
            read_jsonl("./predicts/{}/{}_results_wo_base.json".format(model_name, file_ctype))
            # read_jsonl("./predicts/{}/mimic_results_0.json".format(model_name, file_ctype))
        )
        print("加权F1 vs 宏平均F1差异测试 - 使用test_data数据")
        for hos_name, gold_datas in hos_gold_datas.items():
            # 按照意愿类型进行分组
            print("\n" + "=" * 50, hos_name)
            # if hos_name == "wh":
            predict_datas = hos_predict_datas.get(hos_name, [])
            datas = [(gold, predict) for gold, predict in zip(gold_datas, predict_datas)]
            subcontent_metrics = compute_average(compute_subcontent_metrics(gold_datas, predict_datas))
            fields_metrics = compute_average(compute_fields_metrics(gold_datas, predict_datas))

            print("Sub-content metrics:", subcontent_metrics)
            print("Fields metrics:", fields_metrics)
