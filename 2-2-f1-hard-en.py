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


name_maps = {
    "入院记录": "Admission Record",
    "患者基本信息": "Basic Patient Information",
    "记录时间": "Record Time",
    "入院时间": "Admission Time",
    "主诉": "Chief Complaint",
    "现病史": "History of Present Illness",
    "既往史": "Past Medical History",
    "个人史": "Personal History",
    "婚育史": "Obstetric History",
    "月经史": "Menstrual History",
    "家族史": "Family History",
    "体格检查": "Physical Examination",
    "专科检查": "Specialty Examination",
    "辅助检查": "Auxiliary Examinations",
    "初步诊断": "Preliminary Diagnosis",
    "入院诊断": "Admission Diagnosis",
    "治则治法": "Therapeutic Principle and Method",
    "首次病程": "Initial Progress Note",
    "病例特点": "Case Characteristics",
    "诊断依据": "Diagnostic Basis",
    "鉴别诊断": "Differential Diagnosis",
    "诊疗计划": "Management Plan",
    "日常病程": "Daily Progress Note",
    "日常医嘱": "Daily Orders",
    "治疗计划": "Treatment Plan",
    "非结构化大文本": "Unstructured Long Text",
    "会诊记录": "Consultation Record",
    "会诊类型": "Consultation Type",
    "申请科室": "Requesting Department",
    "会诊科室": "Consulting Department",
    "申请时间": "Request Time",
    "会诊时间": "Consultation Time",
    "目前诊断": "Current Diagnosis",
    "简要病史及诊疗经过": "Brief History and Treatment Course",
    "会诊原因及目的": "Reason and Purpose for Consultation",
    "会诊意见": "Consultation Opinion",
    "手术记录": "Operation Record",
    "手术时间": "Operation Time",
    "术前诊断": "Preoperative Diagnosis",
    "术后诊断": "Postoperative Diagnosis",
    "手术名称": "Operation Name",
    "手术级别": "Operation Level",
    "手术医师": "Surgeon",
    "助手医师": "Assistant Surgeon",
    "护士": "Nurse",
    "麻醉师": "Anesthesiologist",
    "麻醉方法": "Anesthesia Method",
    "麻醉剂": "Anesthetic Agent",
    "手术经过": "Operation Process",
    "手术中用药": "Intraoperative Medications",
    "术后首程": "Postoperative Initial Note",
    "术后处理措施": "Postoperative Management Measures",
    "术后注意事项": "Postoperative Precautions",
    "出院记录": "Discharge Summary",
    "科室": "Department",
    "出院时间": "Discharge Time",
    "出院诊断": "Discharge Diagnosis",
    "入院情况": "Admission Condition",
    "诊疗经过": "Treatment Course",
    "出院情况": "Discharge Condition",
    "出院医嘱": "Discharge Orders",
    "死亡记录": "Death Record",
    "死亡时间": "Death Time",
    "死亡原因": "Cause of Death",
    "死亡诊断": "Death Diagnosis"
}


def compute_subcontent_metrics(golds: List[Dict], predicts: List[Dict], similarity_threshold=0.8):
    results = []
    for gold, pred in zip(golds, predicts):
        matched_golds = [0] * len(gold["sub_contents"])
        matched_preds = [0] * len(pred["sub_contents"])

        gold_subs = []
        pred_subs = []

        for sub in gold["sub_contents"]:
            gold_subs.append((name_maps[sub["sub_meta"]["sec_name"]], sub["sub_content"]))

        for sub in pred["sub_contents"]:
            pred_subs.append((name_maps[sub["sub_meta"]["sec_name"]], sub["sub_content"]))
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

            if best_j != -1 and best_sim >= similarity_threshold:
                matched_golds[i] = 1
                matched_preds[best_j] = 1
                tp += 1

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


def compute_fields_metrics(golds: List[Dict], predicts: List[Dict], similarity_threshold=0.8):
    results = []
    for gold, pred in zip(golds, predicts):
        gold_fields = []
        pred_fields = []

        for sub in gold["sub_contents"]:
            for k, v in sub['fields'].items():
                if k not in {"患者基本信息", "Basic Patient Information"}:
                    gold_fields.append((name_maps[k], v))

        for sub in pred["sub_contents"]:
            for k, v in sub['fields'].items():
                if k not in {"患者基本信息", "Basic Patient Information"}:
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
            if best_j != -1 and best_sim >= similarity_threshold:
                matched_g[i] = 1
                matched_p[best_j] = 1
                tp += 1

        fp = len(pred_fields) - sum(matched_p)
        fn = len(gold_fields) - sum(matched_g)
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        print(f1)
        print(pred_fields)
        print(gold_fields)
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

    model_name = "qwen-agent-14b-grpo-experience"
    model_name = "qwen-extract-7b-sft"
    model_name = "llama-extract-8b"

    # model_name = "qwen-agent-14b-experience"
    # model_name = "qwen-agent-7b"
    # model_name = "qwen-extract-7b-sft"
    # model_name = "qwen-extract-14b"

    # model_name = "qwen-agent-7b"
    # model_name = "qwen-agent-14b-experience"
    # for file_ctype in ["test", "jiwai", "mimic"]:
    for file_ctype in ["mimic"]:
        print(">>>>>>>>>>>>>>>{}<<<<<<<<<<<<<<".format(file_ctype))
        if file_ctype in ["test", "jiwai"]:
            gold_file = "./train_test_datas/base/v1/{}_data.json".format(file_ctype)
        else:
            gold_file = "./datas/note_sample_50_已标注_加噪.json"

        hos_gold_datas, hos_predict_datas = rebuild_datas(
            read_jsonl(gold_file),
            # read_jsonl("./predicts/{}/{}_results_wo_base.json".format(model_name, file_ctype))
            read_jsonl("./predicts/{}/mimic_results_0.json".format(model_name, file_ctype))
        )
        print("加权F1 vs 宏平均F1差异测试 - 使用test_data数据")
        for hos_name, gold_datas in hos_gold_datas.items():
            # 按照意愿类型进行分组
            print("\n" + "=" * 50, hos_name)
            # if hos_name == "wh":
            predict_datas = hos_predict_datas.get(hos_name, [])
            datas = [(gold, predict) for gold, predict in zip(gold_datas, predict_datas)]
            subcontent_metrics = compute_average(compute_subcontent_metrics(gold_datas, predict_datas, similarity_threshold=0.4))
            fields_metrics = compute_average(compute_fields_metrics(gold_datas, predict_datas, similarity_threshold=0.4))

            print("Sub-content metrics:", subcontent_metrics)
            print("Fields metrics:", fields_metrics)
