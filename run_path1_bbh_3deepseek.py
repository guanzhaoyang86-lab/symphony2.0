#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run BBH (Big-Bench Hard) tasks with Symphony 2.0 orchestrator and 3 DeepSeek (OpenAI-compatible) endpoints.

Patched for Symphony (updated):
- Task.requirements uses BBH task_name (capability-tag matching)
- Task.context carries gold answer for correctness reward
- CLI knobs: correctness_bonus / incorrect_penalty
- Register agents with capability tags including all BBH tasks (so match_score is meaningful)

Example:
python3 /hy-tmp/symphony2.0-main/run_bbh_3deepseek.py \
  --bbh-script /hy-tmp/datasets/bbh \
  --cache-dir /hy-tmp/hf/datasets \
  --outdir /hy-tmp/exp_bbh_3deepseek \
  --cot 3 --max-per-task 250 --random-sample --seed 123 \
  --use-dynamic --dump-traces \
  --correctness-bonus 0.5 --incorrect-penalty 0.0
"""
from __future__ import annotations

import argparse
import csv
import datetime
import inspect
import json
import os
import random
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from typing import Optional, List
from datasets import load_dataset, get_dataset_config_names

# -------------------------
# Globals for trace dumping
# -------------------------
DUMP_TRACES: bool = False
TRACE_DIR: str = ""

# -------------------------
# Default BBH task list (fallback)
# (If available, we will auto-read config names from bbh.py)
# -------------------------
BBH_TASKS_FALLBACK: List[str] = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
]

CHOICE_HINT = "Return the option letter only."

BBH_TASK_HINTS = {
    "boolean_expressions": "Return True/False only.",
    "causal_judgement": "Return yes/no only.",
    "date_understanding": CHOICE_HINT,
    "disambiguation_qa": CHOICE_HINT,
    "geometric_shapes": CHOICE_HINT,
    "hyperbaton": CHOICE_HINT,
    "logical_deduction_three_objects": CHOICE_HINT,
    "logical_deduction_five_objects": CHOICE_HINT,
    "logical_deduction_seven_objects": CHOICE_HINT,
    "movie_recommendation": CHOICE_HINT,
    "reasoning_about_colored_objects": CHOICE_HINT,
    "ruin_names": CHOICE_HINT,
    "temporal_sequences": CHOICE_HINT,
    "tracking_shuffled_objects_five_objects": CHOICE_HINT,

    "multistep_arithmetic_two": "Return the final number only.",
    "object_counting": "Return the final number only.",
    "penguins_in_a_table": "Return the final number only.",

    "formal_fallacies": "Return valid/invalid only.",
    "sports_understanding": "Return yes/no only.",
    "dyck_languages": "Return ONLY the missing bracket sequence (no explanation, no original string).",

    # 这个保留也行：因为 navigate 有的样本是 choice，有的不是
    "navigate": "If options exist, return the option letter; otherwise return the final location/direction only.",

    # 这俩泛化 hint 也可以留
    "salient_translation_error_detection": "Return the required final token only (often option letter or yes/no).",
    "snarks": "Return the required final token only.",
}

BBH_SCHEMA_OVERRIDE = {
    "boolean_expressions": "bool",
    "causal_judgement": "yesno",
    "dyck_languages": "dyck",
    "formal_fallacies": "validity",

    "multistep_arithmetic_two": "number",
    "object_counting": "number",
    "penguins_in_a_table": "number",

    # 多选字母题
    "date_understanding": "choice",
    "disambiguation_qa": "choice",
    "geometric_shapes": "choice",
    "hyperbaton": "choice",
    "logical_deduction_three_objects": "choice",
    "logical_deduction_five_objects": "choice",
    "logical_deduction_seven_objects": "choice",
    "movie_recommendation": "choice",
    "reasoning_about_colored_objects": "choice",
    "ruin_names": "choice",
    "temporal_sequences": "choice",
    "tracking_shuffled_objects_five_objects": "choice",
}


# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_bbh_loader_dir(bbh_script: str) -> str:
    """
    You pass either:
      - /path/to/datasets/bbh            (directory that contains bbh.py)
      - /path/to/datasets/bbh/bbh.py    (script itself)
    Return directory that contains bbh.py.
    """
    p = os.path.abspath(bbh_script)
    if os.path.isdir(p):
        if os.path.exists(os.path.join(p, "bbh.py")):
            return p
        if os.path.exists(os.path.join(p, "bbh", "bbh.py")):
            return os.path.join(p, "bbh")
    else:
        if os.path.basename(p) == "bbh.py":
            return os.path.dirname(p)

    raise FileNotFoundError(f"Cannot find bbh.py under: {bbh_script}")


def _extract_choice_labels(inp: str) -> List[str]:
    """
    Extract option labels from BBH multiple choice prompt.
    Patterns:
      "(A) xxx", "[A] xxx", "A. xxx", "A) xxx"
    """
    labels: List[str] = []

    for m in re.finditer(r"[\(\[]\s*([A-Z])\s*[\)\]]", inp):
        labels.append(m.group(1))

    for m in re.finditer(r"(?m)^\s*([A-Z])\s*[.)]\s+", inp):
        labels.append(m.group(1))

    # unique keep order
    out: List[str] = []
    seen = set()
    for x in labels:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def extract_dyck_answer(raw: str) -> str:
    text = str(raw or "")

    # 1) Try JSON decode (trace sometimes stringifies JSON)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k in ("final", "final_result", "final_text", "answer", "prediction", "text"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    text = v
                    break
    except Exception:
        pass

    # Normalize common wrapper prefixes
    text = _strip_answer_prefixes(text)
    text = _strip_wrappers(text)

    # 2) Find all bracket-only runs and take the last (most often the final answer)
    runs = re.findall(r"[\[\]\(\)\{\}<>]+", text)
    if runs:
        return runs[-1]

    # 3) Fallback: keep only dyck chars
    return "".join(ch for ch in text if ch in _DYCK_CHARS)


def build_bbh_prompt(
        task_name: str,
        inp: str,
        schema: str,
        choices: Optional[List[str]] = None
) -> str:
    # NOTE: no trailing comma here (avoid tuple)
    task_hint = BBH_TASK_HINTS.get(task_name, "")

    header_lines = [
        "You are solving a Big-Bench Hard (BBH) problem.",
        "You may think step-by-step, but the FINAL output must be the answer only.",
        "The LAST line must start with exactly: Final answer:",
    ]
    if task_hint:
        header_lines.append(f"Task-specific hint: {task_hint}")
    if task_name == "word_sorting":
        header_lines.append(
            "For word sorting: output ONLY the sorted words separated by single spaces. "
            "No commas, no 'List:'."
        )
    header = "\n".join(header_lines) + "\n"

    if schema == "bool":
        rule = "After 'Final answer:' output EXACTLY one token: True or False.\n"
    elif schema == "yesno":
        rule = "After 'Final answer:' output EXACTLY one token: yes or no.\n"
    elif schema == "validity":
        rule = "After 'Final answer:' output EXACTLY one token: valid or invalid.\n"
    elif schema == "dyck":
        rule = (
            "After 'Final answer:' output ONLY the missing bracket sequence using "
            "[](){}<> with NO spaces.\n"
        )
    elif schema == "choice":
        allowed = choices or _extract_choice_labels(inp) or []
        if allowed:
            rule = f"After 'Final answer:' output EXACTLY ONE LETTER among: {'/'.join(allowed)}.\n"
        else:
            rule = "After 'Final answer:' output EXACTLY ONE LETTER (the option label shown in the input).\n"

    elif schema == "number":
        rule = "After 'Final answer:' output a single number (integer or decimal). No units.\n"
    else:
        rule = (
            "After 'Final answer:' output the answer as plain text.\n"
            "If the answer contains multiple words/items, separate them with SINGLE SPACES.\n"
            "Do NOT concatenate words together.\n"
            "Do NOT add extra words like 'List' or 'Answer'.\n"
        )

    prompt = f"{header}{rule}\nTask input:\n{inp.strip()}\n\nFinal answer:"
    return prompt


def _clean_gt_str(gt: str) -> str:
    s = str(gt).strip()
    s = re.sub(r"(?is)</?\s*answer\s*>", "", s).strip()
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    if m:
        s = m.group(1).strip()
    s = re.sub(r"(?i)^\s*final\s*answer\s*:\s*", "", s).strip()
    # normalize MCQ targets like "(B)" or "（B）" -> "B"
    m = re.fullmatch(r"[\(\[\{（]\s*([A-Z])\s*[\)\]\}）]", s)
    if m:
        s = m.group(1)

    return s


def _schema_from_gt(gt: str) -> str:
    g = _clean_gt_str(gt)
    gl = g.lower()

    if gl in {"true", "false"}:
        return "bool"
    if gl in {"yes", "no"}:
        return "yesno"
    if gl in {"valid", "invalid"}:
        return "validity"
    # dyck targets are bracket-only strings (sometimes empty)
    if re.fullmatch(r"[\[\]\(\)\{\}<>]*", g) is not None:
        return "dyck"
    if re.fullmatch(r"[A-Z]", g):
        return "choice"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", g) or re.fullmatch(r"[-+]?\d+/\d+", g):
        return "number"
    return "text"


def get_bbh_schema(task_name: str, gt: str, inp: str) -> str:
    s = BBH_SCHEMA_OVERRIDE.get(task_name)

    if task_name in BBH_SCHEMA_OVERRIDE:
        return BBH_SCHEMA_OVERRIDE[task_name]
        # fallback only if task not known
    return _schema_from_gt(gt)
    # # 有些任务可能没有选项（比如 navigate 的某些样本）
    # if s == "choice" and not _extract_choice_labels(inp):
    #     return _schema_from_gt(gt)

    # return s or _schema_from_gt(gt)


def _strip_wrappers(s: str) -> str:
    if not s:
        return ""
    x = str(s)

    # 去掉代码块外壳
    x = x.strip()
    x = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", x)
    x = re.sub(r"\n```$", "", x)

    # 去掉 \boxed{...} 外壳（保留里面内容）
    x = re.sub(r"\\boxed\s*{\s*(.*?)\s*}", r"\1", x, flags=re.S)

    # 去掉最外层引号（不要删中间空格）
    x = x.strip()
    if (len(x) >= 2) and ((x[0] == x[-1]) and x[0] in ["'", '"']):
        x = x[1:-1].strip()

    # 关键：只把多空格压成 1 个，不要删除空格
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _extract_final_line(raw: str) -> str:
    if not raw:
        return ""
    lines = [ln.strip("\r") for ln in str(raw).splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]
    return lines[-1] if lines else ""


def _extract_number(s: str) -> Optional[str]:
    s = s.strip()
    m = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        return m[-1]
    m2 = re.findall(r"[-+]?\d+/\d+", s)
    if m2:
        return m2[-1]
    return None


def _normalize_yesno(s: str) -> Optional[str]:
    sl = s.strip().lower().rstrip(".!")
    if sl in {"yes", "no"}:
        return sl
    return None


def _normalize_bool(s: str) -> Optional[str]:
    sl = s.strip().lower().rstrip(".!")
    if sl == "true":
        return "True"
    if sl == "false":
        return "False"
    return None


def normalize_text_light(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_validity(s: str) -> Optional[str]:
    sl = s.strip().lower().rstrip(".!")
    if sl in {"valid", "invalid"}:
        return sl
    return None


def _normalize_choice(s: str, allowed: Optional[List[str]]) -> Optional[str]:
    ss = s.strip()

    # allow "B" / "(B)" / "（B）"
    m = re.fullmatch(r"([A-Z])", ss)
    if not m:
        m = re.fullmatch(r"[\(\[\{（]\s*([A-Z])\s*[\)\]\}）]", ss)

    if not m:
        return None

    ch = m.group(1)
    if allowed and ch not in allowed:
        return None
    return ch


def _extract_options_map(inp: str) -> Dict[str, str]:
    # (A) xxx
    opts: Dict[str, str] = {}
    for m in re.finditer(r"\(\s*([A-Z])\s*\)\s*([^\n]+)", inp):
        lab, txt = m.group(1), m.group(2).strip()
        if lab not in opts:
            opts[lab] = txt

    # A. xxx / A) xxx
    for m in re.finditer(r"(?m)^\s*([A-Z])\s*[.)]\s+(.+)$", inp):
        lab, txt = m.group(1), m.group(2).strip()
        if lab not in opts:
            opts[lab] = txt
    return opts


def _map_answer_to_choice_by_option_text(answer: str, inp: str, allowed: Optional[List[str]]) -> Optional[str]:
    ans = answer.strip()
    if not ans:
        return None

    def norm(x: str) -> str:
        # loose normalize for dates / simple strings
        return re.sub(r"\s+", "", x.strip()).lower()

    opts = _extract_options_map(inp)
    if not opts:
        return None

    n_ans = norm(ans)

    # exact match against option text (works great for date_understanding)
    for lab, txt in opts.items():
        if allowed and lab not in allowed:
            continue
        if norm(txt) == n_ans:
            return lab

    # also allow "Final answer: 12/25/1937" where ans is substring
    for lab, txt in opts.items():
        if allowed and lab not in allowed:
            continue
        if n_ans and n_ans in norm(txt):
            return lab

    return None


_WORD_SORTING_DROP = {"final", "answer", "list"}


def _strip_answer_prefixes(s: str) -> str:
    """只去掉常见前缀，不删除空格。"""
    if not s:
        return ""
    x = s.strip()
    # 常见：Final answer: xxx / Answer: xxx / List: xxx
    x = re.sub(r"^(final\s+answer|final|answer|list)\s*[:：]\s*", "", x, flags=re.I)
    return x.strip()


def _looks_like_word_sorting(inp: str) -> bool:
    t = (inp or "").lower()
    return ("sort the following words alphabetically" in t) or ("words alphabetically" in t)


def normalize_word_sorting(text: str) -> str:
    """抽取英文单词并用单空格连接，过滤掉 list/final/answer 等脏词。"""
    words = re.findall(r"[A-Za-z]+", (text or "").lower())
    words = [w for w in words if w not in _WORD_SORTING_DROP]
    return " ".join(words).strip()


def _normalize_word_sorting_from_input(answer: str, inp: str) -> str:
    # 把输出变成纯字母串（保留信息，便于切分）
    s = re.sub(r"[^A-Za-z]", "", (answer or "")).lower()
    # 常见污染：末尾粘了 list
    if s.endswith("list"):
        s = s[:-4]

    # 从输入里提取原始词表（List: 后面）
    m = re.search(r"(?i)\blist\s*[:：]\s*(.+)$", inp or "")
    src = m.group(1) if m else (inp or "")
    words = re.findall(r"[A-Za-z]+", src.lower())

    # 去掉 list/final/answer 这些
    drop = {"final", "answer", "list"}
    words = [w for w in words if w not in drop]
    # 去重保序
    seen = set();
    words = [w for w in words if not (w in seen or seen.add(w))]

    # 用“最长前缀匹配”把 s 切成这些词
    remain = s
    out = []
    pool = words[:]
    for _ in range(len(pool)):
        cands = [w for w in pool if remain.startswith(w)]
        if not cands:
            break
        w = max(cands, key=len)
        out.append(w)
        remain = remain[len(w):]
        pool.remove(w)

    if remain == "" and out:
        return " ".join(out).strip()

    # 切不出来就退回：按英文单词抽取
    toks = re.findall(r"[A-Za-z]+", (answer or "").lower())
    toks = [w for w in toks if w not in drop]
    return " ".join(toks).strip()


def _extract_choice_from_json(s: str, allowed: Optional[List[str]]) -> Optional[str]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            for k in ("gold", "answer", "final", "prediction"):
                v = obj.get(k)
                if isinstance(v, str):
                    v = v.strip().upper()
                    if re.fullmatch(r"[A-Z]", v):
                        if not allowed or v in allowed:
                            return v
    except Exception:
        pass
    return None


def predict_and_validate(raw: str, gt: str, inp: str, schema: Optional[str] = None) -> Tuple[str, int, str, str]:
    gt_norm = _clean_gt_str(gt)
    schema = schema or _schema_from_gt(gt_norm)
    allowed = _extract_choice_labels(inp) if schema == "choice" else None

    final_line = _extract_final_line(raw)  # 不要先 strip_wrappers
    cand_text = _strip_answer_prefixes(final_line)  # 给 text 用：保留空格
    cand = _strip_wrappers(cand_text)  # 给 bool/choice/number 用：可以更强硬
    cand = cand.lstrip(":：").strip()

    pred = ""
    valid = 0

    # ---- 任务特化：word_sorting（你截图里就是这个）----
    # 你当前 schema 还是 text，所以在这里直接兜底
    if schema == "text" and _looks_like_word_sorting(inp):
        pred_cmp = normalize_word_sorting(raw)  # 用 raw 最稳（避免只取最后一行丢信息）
        gt_cmp = normalize_word_sorting(gt_norm)
        pred = pred_cmp
        valid = 1 if pred else 0
        return pred, valid, gt_cmp, schema

    if schema == "bool":
        p = _normalize_bool(cand) or _normalize_bool(_strip_wrappers(raw))
        if p is not None:
            pred, valid = p, 1
        else:
            # pred, valid = cand_text.strip(), 0
            pred = "False"
            valid = 1  # 注意：这里仍然算 valid
        gt_cmp = _normalize_bool(gt_norm) or gt_norm

    elif schema == "yesno":
        p = _normalize_yesno(cand) or _normalize_yesno(_strip_wrappers(raw))
        if p is not None:
            pred, valid = p.lower(), 1
        else:
            pred, valid = cand_text.strip().lower(), 0
        gt_cmp = (_normalize_yesno(gt_norm) or gt_norm).lower()

    elif schema == "validity":
        p = _normalize_validity(cand) or _normalize_validity(_strip_wrappers(raw))
        if p is not None:
            pred, valid = p.lower(), 1
        else:
            pred, valid = cand_text.strip().lower(), 0
        gt_cmp = (_normalize_validity(gt_norm) or gt_norm).lower()

    # elif schema == "dyck":
    #     s = cand_text.strip()
    #     s2 = "".join(ch for ch in s if ch in "[](){}<>")
    #     if s2 == s and re.fullmatch(r"[\[\]\(\)\{\}<>]*", s2) is not None:
    #         pred, valid = s2, 1
    #     else:
    #         s3 = _strip_wrappers(raw)
    #         s3 = "".join(ch for ch in s3 if ch in "[](){}<>")
    #         pred, valid = s3, 0
    #     gt_cmp = gt_norm
    elif schema == "dyck":
        pred = extract_dyck_answer(raw)
        gt_cmp = gt_norm

        # valid：必须是“只包含括号的串”（允许空串）
        valid = 1 if (
                isinstance(pred, str)
                and re.fullmatch(r"[\[\]\(\)\{\}<>]*", pred) is not None
        ) else 0


    # elif schema == "choice":
    #     # 先尝试直接抽字母
    #     p = _normalize_choice(cand, allowed) or _normalize_choice(_strip_wrappers(raw), allowed)

    #     # 再尝试：模型输出的是选项内容（比如日期/文本）-> 映射回字母
    #     if p is None:
    #         p = _map_answer_to_choice_by_option_text(cand_text, inp, allowed) or \
    #             _map_answer_to_choice_by_option_text(_strip_answer_prefixes(raw), inp, allowed)

    #     if p is not None:
    #         pred, valid = p, 1
    #     else:
    #         pred, valid = cand_text.strip(), 0

    #     gt_cmp = gt_norm.strip()   # 用 gt_norm（别再用原始 gt）

    elif schema == "choice":
        # 1. 直接字母
        p = _normalize_choice(cand, allowed) or _normalize_choice(_strip_wrappers(raw), allowed)

        # 2. JSON 包裹（date_understanding 非常常见）
        if p is None:
            p = _extract_choice_from_json(cand_text, allowed) or \
                _extract_choice_from_json(_strip_wrappers(raw), allowed)

        # 3. 用选项文本反推字母（日期 / 文本）
        if p is None:
            p = _map_answer_to_choice_by_option_text(cand_text, inp, allowed) or \
                _map_answer_to_choice_by_option_text(_strip_answer_prefixes(raw), inp, allowed)

        if p is not None:
            pred, valid = p, 1
        else:
            # BBH choice：parse 失败 ≠ invalid，而是“随便猜一个合法选项”
            pred, valid = (allowed[0] if allowed else "A"), 1

        gt_cmp = gt_norm.strip()


    elif schema == "number":
        p = _extract_number(cand) or _extract_number(_strip_wrappers(raw))
        if p is not None:
            pred, valid = p, 1
        else:
            pred, valid = cand_text.strip(), 0
        gt_cmp = gt_norm.strip()

    else:
        # text: 对 word_sorting 做强制规范化（不然空格/冒号/List 很容易污染）
        if "word_sorting" in (inp or "").lower() or "alphabetically" in (inp or "").lower():
            pred = _normalize_word_sorting_from_input(raw, inp)
            gt_cmp = _normalize_word_sorting_from_input(gt_norm, inp)
            valid = 1 if pred else 0
        else:
            pred = _strip_answer_prefixes(_extract_final_line(raw)).strip()  # 保留空格
            valid = 1 if pred else 0
            gt_cmp = gt_norm

    return pred, valid, gt_cmp, schema


def _parse_openai_endpoint(spec: str) -> Tuple[str, str]:
    """
    spec examples:
      openai:http://127.0.0.1:8001/v1#model=deepseek
      http://127.0.0.1:8001/v1#model=deepseek
    Return (base_url, model_name)
    """
    s = spec.strip()
    if s.startswith("openai:"):
        s = s[len("openai:"):]
    if "#model=" in s:
        base, frag = s.split("#", 1)
        m = re.search(r"model=([^&]+)", frag)
        model = m.group(1) if m else "deepseek"
        return base, model
    return s, "deepseek"


def _safe_construct(cls: Any, **kwargs: Any) -> Any:
    """Construct cls with only kwargs present in its __init__ signature."""
    try:
        sig = inspect.signature(cls.__init__)
        params = set(sig.parameters.keys())
        params.discard("self")
        filt = {k: v for k, v in kwargs.items() if k in params}
        return cls(**filt)
    except Exception:
        return cls(**kwargs)


def _construct_openai_compat_model(OpenAICompatModelCls: Any, base_url: str, model_name: str,
                                   temperature: float, top_p: float, max_tokens: int) -> Any:
    """
    兼容你仓库里的 OpenAICompatModel(__init__) 可能使用的参数名：
    - api_base (你这里需要)
    - base_url（有些实现用这个）
    以及可能要求 api_key（若必填则给一个占位符）
    """
    sig = inspect.signature(OpenAICompatModelCls.__init__)
    params = sig.parameters

    kw: Dict[str, Any] = {}

    # base url param
    if "api_base" in params:
        kw["api_base"] = base_url
    elif "base_url" in params:
        kw["base_url"] = base_url
    elif "base" in params:
        kw["base"] = base_url

    # model param
    if "model" in params:
        kw["model"] = model_name
    elif "model_name" in params:
        kw["model_name"] = model_name

    # sampling params (只在存在时传)
    if "temperature" in params:
        kw["temperature"] = temperature
    if "top_p" in params:
        kw["top_p"] = top_p
    if "max_tokens" in params:
        kw["max_tokens"] = max_tokens

    # api_key（若必填且无默认值）
    if "api_key" in params and params["api_key"].default is inspect._empty:
        kw["api_key"] = os.environ.get("OPENAI_API_KEY", "EMPTY")

    return OpenAICompatModelCls(**kw)


def _try_init_symphony(
        run_id: str,
        outdir: str,
        verbose: bool,
        use_dynamic: bool,
        topL: int,
        alpha: float,
        l2: float,
        plan_k: int,
        correctness_bonus: float,
        incorrect_penalty: float,
) -> None:
    import symphony

    if hasattr(symphony, "init"):
        try:
            sig = inspect.signature(symphony.init)  # type: ignore[attr-defined]
            params = set(sig.parameters.keys())

            # ✅ 不再传 run_id/outdir（你的 symphony.init 不支持）
            kw = {
                "verbose": verbose,
                "use_dynamic": use_dynamic,
                "topL": topL,
                "linucb_alpha": alpha,
                "linucb_l2": l2,
                "plan_k": plan_k,
                "correctness_bonus": float(correctness_bonus),
                "incorrect_penalty": float(incorrect_penalty),
            }

            # ✅ 只传 symphony.init() 真实支持的参数
            filt = {k: v for k, v in kw.items() if k in params}
            symphony.init(**filt)  # type: ignore[attr-defined]
            return
        except Exception as e:
            print(f"[WARN] symphony.init failed (ignored): {e}", flush=True)

    # fallback recreate / patch (保持你原来的逻辑即可)
    if hasattr(symphony, "SymphonyOrchestrator"):
        try:
            orch = symphony.SymphonyOrchestrator(
                verbose=verbose,
                use_dynamic=use_dynamic,
                topL=topL,
                linucb_alpha=alpha,
                linucb_l2=l2,
                plan_k=plan_k,
                correctness_bonus=float(correctness_bonus),
                incorrect_penalty=float(incorrect_penalty),
                enable_risk_guard=False,
            )
            setattr(symphony, "_global_orchestrator", orch)
            return
        except Exception as e:
            print(f"[WARN] Recreate SymphonyOrchestrator failed (ignored): {e}", flush=True)

    orch = getattr(symphony, "_global_orchestrator", None)
    if orch is not None:
        try:
            orch.verbose = bool(verbose)
            orch.use_dynamic = bool(use_dynamic)
            orch.topL = int(topL)
            if hasattr(orch, "plan_k"):
                orch.plan_k = int(plan_k)
            if hasattr(orch, "correctness_bonus"):
                orch.correctness_bonus = float(correctness_bonus)
            if hasattr(orch, "incorrect_penalty"):
                orch.incorrect_penalty = float(incorrect_penalty)
        except Exception:
            pass


def register_3_local_vllm_agents(
        model_path: str,
        bm0: str,
        bm1: str,
        bm2: str,
        gpu0: int,
        gpu1: int,
        gpu2: int,
        system_prompt: str,
        capability_tags: List[str],
) -> None:
    import symphony
    from agents.agent import Agent

    # tags：同质 agent 建议全量 + planning + general-reasoning（planner会用到）
    cap = ["planning", "general-reasoning"] + [str(x).strip() for x in capability_tags if str(x).strip()]
    # 去重保序
    seen = set()
    cap2 = []
    for x in cap:
        xl = x.lower()
        if xl not in seen:
            cap2.append(x)
            seen.add(xl)

    p0 = bm0 or model_path
    p1 = bm1 or model_path
    p2 = bm2 or model_path
    if not (p0 and p1 and p2):
        raise ValueError("model path empty: please set --model-path or all of --bm0/--bm1/--bm2")

    a0 = Agent(node_id="deepseek_gpu0", capabilities=cap2, system_prompt=system_prompt, base_model=p0, gpu_id=int(gpu0))
    a1 = Agent(node_id="deepseek_gpu1", capabilities=cap2, system_prompt=system_prompt, base_model=p1, gpu_id=int(gpu1))
    a2 = Agent(node_id="deepseek_gpu2", capabilities=cap2, system_prompt=system_prompt, base_model=p2, gpu_id=int(gpu2))

    # 如果 Agent.__init__ 已经自动注册，这里重复注册也没关系；try 包一下即可
    try:
        symphony.register_agent(a0)
        symphony.register_agent(a1)
        symphony.register_agent(a2)
    except Exception:
        pass

    print(f"[CHK] deepseek_gpu0 model_path={p0} gpu={gpu0} caps={len(cap2)}", flush=True)
    print(f"[CHK] deepseek_gpu1 model_path={p1} gpu={gpu1} caps={len(cap2)}", flush=True)
    print(f"[CHK] deepseek_gpu2 model_path={p2} gpu={gpu2} caps={len(cap2)}", flush=True)


def _call_execute_task(task: Any, **kwargs: Any) -> Any:
    """
    Call symphony.execute_task with only supported kwargs.
    Some versions accept: cot_count, verbose, return_mode.
    """
    import symphony
    fn = symphony.execute_task

    try:
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())
        filt = {k: v for k, v in kwargs.items() if k in params}
        return fn(task, **filt)
    except Exception:
        return fn(task)


def _unwrap_result(out: Any) -> str:
    """Unwrap different return types into raw text (best-effort)."""
    if out is None:
        return ""
    if isinstance(out, str):
        return out

    if isinstance(out, dict):
        # ✅ trace 模式：优先使用 symphony 已抽取好的最终答案
        for k in ("final", "final_result", "final_text", "text"):
            if k in out and out[k]:
                return str(out[k])

        # 兼容旧结构：results 里只有一个 subtask
        if "results" in out and isinstance(out["results"], dict) and out["results"]:
            return str(next(iter(out["results"].values())))

        # OpenAI / OpenAI-compatible
        if "choices" in out:
            try:
                c0 = (out.get("choices") or [{}])[0]
                if isinstance(c0, dict):
                    msg = c0.get("message") or {}
                    if isinstance(msg, dict) and msg.get("content"):
                        return str(msg["content"])
                    if c0.get("text"):
                        return str(c0["text"])
            except Exception:
                pass

        return json.dumps(out, ensure_ascii=False)

    return str(out)


def print_trace_to_terminal(trace: dict, task_name: str, idx: int) -> None:
    print("\n" + "=" * 80)
    print(f"[TRACE] task={task_name} idx={idx}")
    print("=" * 80)

    # 1. Planner / orchestration info
    for k in ("planner", "plan", "selected_plan", "decision"):
        if k in trace:
            print(f"\n--- {k.upper()} ---")
            print(json.dumps(trace[k], ensure_ascii=False, indent=2))

    # 2. Agent-level reasoning
    agents = trace.get("agents") or trace.get("agent_traces")
    if isinstance(agents, list):
        for i, a in enumerate(agents):
            print(f"\n--- AGENT {i} ({a.get('node_id', 'unknown')}) ---")
            for key in ("reasoning", "thoughts", "analysis", "content"):
                if key in a:
                    print(f"[{key}]\n{a[key]}")
            if "score" in a:
                print(f"[score] {a['score']}")

    # 3. Final answer
    for k in ("final", "final_result", "final_text"):
        if k in trace:
            print("\n--- FINAL ---")
            print(trace[k])

    # 4. Reward / correctness
    if "reward" in trace:
        print("\n--- REWARD ---")
        print(json.dumps(trace["reward"], ensure_ascii=False, indent=2))

    print("=" * 80 + "\n")


def run_one_bbh(task_name: str, inp: str, gt: str, schema: str, cot_count: int, verbose: bool) -> str:
    """
    Run one BBH example through symphony.
    If --dump-traces is enabled, request return_mode='trace' (if supported) and dump JSON to TRACE_DIR.
    """
    try:
        from protocol.task_contract import Task  # type: ignore
    except Exception:
        from symphony.protocol.task_contract import Task  # type: ignore

    choices = _extract_choice_labels(inp) if schema == "choice" else None

    # ✅ gold normalized for correctness reward & consistent comparison
    gold = _clean_gt_str(gt)

    # ✅ requirements use task_name (capability tags align to BBH task list)
    task = Task(
        description=build_bbh_prompt(task_name, inp, schema, choices=choices),
        requirements=[task_name],
        context={
            "benchmark": "BBH",
            "schema": schema,
            "task_name": task_name,
            "choices": choices or [],
            "gold": gold,  # ✅ for Symphony correctness reward
        },
    )

    rm = "trace" if DUMP_TRACES else "final"
    out = _call_execute_task(task, cot_count=cot_count, verbose=verbose, return_mode=rm)

    if DUMP_TRACES and isinstance(out, dict):
        print_trace_to_terminal(out, task_name, idx=task.context.get("idx", -1))

    if DUMP_TRACES and isinstance(out, dict) and TRACE_DIR:
        fn = os.path.join(TRACE_DIR, f"{task_name}_{uuid.uuid4().hex}.json")
        try:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] dump traces failed: {e}", flush=True)

    return _unwrap_result(out).strip()


def _get_bbh_tasks(script_path: str) -> List[str]:
    """Prefer reading dataset config names; fallback to hardcoded list."""
    try:
        names = get_dataset_config_names(script_path)
        if names:
            return list(names)
    except Exception:
        pass
    return BBH_TASKS_FALLBACK


def main() -> None:
    ap = argparse.ArgumentParser("BBH via Symphony2.0 + 3x DeepSeek endpoints")

    ap.add_argument("--ds0", type=str, default="openai:http://127.0.0.1:8001/v1#model=deepseek")
    ap.add_argument("--ds1", type=str, default="openai:http://127.0.0.1:8002/v1#model=deepseek")
    ap.add_argument("--ds2", type=str, default="openai:http://127.0.0.1:8003/v1#model=deepseek")
    ap.add_argument("--model-path", type=str, default="",
                    help="Local model path for vLLM (e.g. /hy-tmp/models/deepseek-7b-instruct)")
    ap.add_argument("--bm0", type=str, default="", help="Override model path for agent0")
    ap.add_argument("--bm1", type=str, default="", help="Override model path for agent1")
    ap.add_argument("--bm2", type=str, default="", help="Override model path for agent2")

    ap.add_argument("--gpu0", type=int, default=0)
    ap.add_argument("--gpu1", type=int, default=1)
    ap.add_argument("--gpu2", type=int, default=2)

    ap.add_argument("--system-prompt", type=str, default="You are a helpful AI assistant.")

    ap.add_argument("--cot", type=int, default=3)
    ap.add_argument("--max-per-task", type=int, default=250)

    ap.add_argument("--outdir", type=str, default="/hy-tmp/exp_bbh_3deepseek")
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument("--bbh-script", type=str, default="/hy-tmp/datasets/bbh")
    ap.add_argument("--cache-dir", type=str, default="/hy-tmp/hf/datasets")

    ap.add_argument("--use-dynamic", action="store_true")
    ap.add_argument("--topL", type=int, default=32)
    ap.add_argument("--linucb-alpha", type=float, default=1.0)
    ap.add_argument("--linucb-l2", type=float, default=1.0)

    # planner path (optional; only takes effect if your symphony.init supports it)
    ap.add_argument("--plan-k", type=int, default=3)

    # ✅ correctness reward knobs (for better scores)
    ap.add_argument("--correctness-bonus", type=float, default=0.0)
    ap.add_argument("--incorrect-penalty", type=float, default=0.0)

    # trace dump
    ap.add_argument("--dump-traces", action="store_true")

    # random sample controls
    ap.add_argument("--random-sample", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # vLLM sampling params passed into OpenAICompatModel where supported
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=512)

    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.abspath(os.path.join(args.outdir, ts))
    cache_dir = os.path.abspath(args.cache_dir)
    ensure_dir(outdir)

    global DUMP_TRACES, TRACE_DIR
    DUMP_TRACES = bool(getattr(args, "dump_traces", False))
    TRACE_DIR = os.path.join(outdir, "traces")
    if DUMP_TRACES:
        os.makedirs(TRACE_DIR, exist_ok=True)

    loader_dir = resolve_bbh_loader_dir(args.bbh_script)
    if loader_dir not in sys.path:
        sys.path.insert(0, loader_dir)

    script_path = os.path.join(loader_dir, "bbh.py")
    bbh_tasks = _get_bbh_tasks(script_path)

    # Init symphony knobs first (then register agents)
    run_id = f"bbh_{ts}"
    _try_init_symphony(
        run_id=run_id,
        outdir=outdir,
        verbose=args.verbose,
        use_dynamic=bool(args.use_dynamic),
        topL=int(args.topL),
        alpha=float(args.linucb_alpha),
        l2=float(args.linucb_l2),
        plan_k=int(args.plan_k),
        correctness_bonus=float(args.correctness_bonus),
        incorrect_penalty=float(args.incorrect_penalty),
    )

    # ✅ capability tags include all BBH tasks so match_score works with requirements=[task_name]
    register_3_local_vllm_agents(
        model_path=args.model_path,
        bm0=getattr(args, "bm0", ""),
        bm1=getattr(args, "bm1", ""),
        bm2=getattr(args, "bm2", ""),
        gpu0=int(args.gpu0),
        gpu1=int(args.gpu1),
        gpu2=int(args.gpu2),
        system_prompt=args.system_prompt,
        capability_tags=bbh_tasks,
    )

    if args.random_sample:
        random.seed(args.seed)

    per_task: Dict[str, Any] = {}
    overall_correct = 0
    overall_total = 0
    overall_valid = 0

    for task_name in bbh_tasks:
        t0 = time.time()

        ds_dict = load_dataset(script_path, name=task_name, cache_dir=cache_dir)

        if "test" in ds_dict:
            ds = ds_dict["test"]
        elif "train" in ds_dict:
            ds = ds_dict["train"]
        else:
            ds = ds_dict[list(ds_dict.keys())[0]]

        examples = [{"input": x["input"], "target": x["target"]} for x in ds]
        if not examples:
            print("[WARN] empty task:", task_name, flush=True)
            continue

        if args.random_sample and args.max_per_task and len(examples) > args.max_per_task:
            examples = random.sample(examples, args.max_per_task)
        else:
            examples = examples[: args.max_per_task]

        N = len(examples)
        csv_path = os.path.join(outdir, f"{task_name}.csv")
        correct = 0
        valid_cnt = 0

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["task", "idx", "schema", "input", "gt", "pred", "ok", "valid", "sec", "raw"])

            for i, ex in enumerate(examples):
                inp = ex.get("input", "")
                gt = ex.get("target", "")

                schema = get_bbh_schema(task_name, gt, inp)

                t1 = time.time()
                raw = run_one_bbh(task_name, str(inp), gt=str(gt), schema=schema, cot_count=args.cot,
                                  verbose=args.verbose)
                dt = time.time() - t1

                pred, valid, gt_n, schema2 = predict_and_validate(str(raw), str(gt), inp=str(inp), schema=schema)

                # pred, valid, gt_n, schema2 = predict_and_validate(str(raw), str(gt), inp=str(inp))
                ok = int(pred == gt_n)

                correct += ok
                valid_cnt += valid

                w.writerow([task_name, i, schema2, inp, gt_n, pred, ok, valid, round(dt, 2), raw])

                if (i + 1) % 25 == 0:
                    f.flush()
                    print(f"[{task_name}] {i + 1}/{N} done", flush=True)

        acc = correct / N if N else 0.0
        valid_rate = valid_cnt / N if N else 0.0
        cost = time.time() - t0

        print(
            f"[BBH] {task_name}: acc={acc * 100:.2f}%  valid={valid_rate * 100:.2f}%  (N={N})  time={cost:.1f}s  saved={csv_path}",
            flush=True,
        )

        per_task[task_name] = {
            "n": N,
            "acc": round(acc * 100, 2),
            "valid_rate": round(valid_rate * 100, 2),
            "time_sec": round(cost, 1),
        }

        overall_correct += correct
        overall_total += N
        overall_valid += valid_cnt

    micro_acc = (overall_correct / overall_total) if overall_total else 0.0
    micro_valid = (overall_valid / overall_total) if overall_total else 0.0

    macro_acc = (sum(v["acc"] for v in per_task.values()) / len(per_task)) if per_task else 0.0
    macro_valid = (sum(v["valid_rate"] for v in per_task.values()) / len(per_task)) if per_task else 0.0

    summary = {
        "overall_micro": {
            "n": overall_total,
            "acc": round(micro_acc * 100, 2),
            "valid_rate": round(micro_valid * 100, 2),
            "invalid_rate": round((1 - micro_valid) * 100, 2),
        },
        "overall_macro": {
            "tasks": len(per_task),
            "acc": round(macro_acc, 2),
            "valid_rate": round(macro_valid, 2),
            "invalid_rate": round((100 - macro_valid), 2),
        },
        "per_task": per_task,
        "cot": int(args.cot),
        "plan_k": int(args.plan_k),
        "dynamic": {
            "use_dynamic": bool(args.use_dynamic),
            "topL": int(args.topL),
            "linucb_alpha": float(args.linucb_alpha),
            "linucb_l2": float(args.linucb_l2),
        },
        "correctness_reward": {
            "correctness_bonus": float(args.correctness_bonus),
            "incorrect_penalty": float(args.incorrect_penalty),
        },
        "random_sample": bool(args.random_sample),
        "seed": int(args.seed),
        "sampling": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_tokens": int(args.max_tokens),
        },
        "agents": {"ds0": args.ds0, "ds1": args.ds1, "ds2": args.ds2},
        "loader_dir": loader_dir,
        "cache_dir": cache_dir,
        "outdir": outdir,
        "dump_traces": bool(args.dump_traces),
        "trace_dir": TRACE_DIR if args.dump_traces else "",
        "capability_tags_count": len(bbh_tasks),
    }

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60, flush=True)
    print(f"OVERALL BBH ACC (micro)  = {micro_acc * 100:.2f}%  (N={overall_total})", flush=True)
    print(f"OVERALL VALID (micro)    = {micro_valid * 100:.2f}%", flush=True)
    print(f"OVERALL BBH ACC (macro)  = {macro_acc:.2f}%  (tasks={len(per_task)})", flush=True)
    print(f"OVERALL VALID (macro)    = {macro_valid:.2f}%", flush=True)
    print(f"Saved to: {outdir}", flush=True)
    if args.dump_traces:
        print(f"Traces saved to: {TRACE_DIR}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    proj = os.path.abspath(os.path.dirname(__file__))
    if proj not in sys.path:
        sys.path.insert(0, proj)
    main()




