"""Generate synthetic demo data for the AI green decision experiment.

This script creates a synthetic dataset that matches the structure of the
real experiment data, so the analysis pipeline can be run end-to-end
without access to the real (private) survey responses.

Usage:
    python data/generate_demo_data.py

Output:
    data/demo_dataset.xlsx  — synthetic data with same column structure
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 20260902
N_PER_GROUP = {0: 31, 1: 56, 2: 64, 3: 51}  # match real sample sizes
TOTAL = sum(N_PER_GROUP.values())

OUT = Path(__file__).resolve().parent / "demo_dataset.xlsx"


def _rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


def _likert(rng: np.random.Generator, n: int, mean: float, sd: float = 1.2,
             low: int = 1, high: int = 7) -> np.ndarray:
    """Generate roughly normal 1-7 Likert data clipped to range."""
    vals = rng.normal(mean, sd, n)
    return np.clip(np.round(vals), low, high).astype(int)


def _choice(rng: np.random.Generator, n: int, options: list[str],
            probs: list[float]) -> np.ndarray:
    return rng.choice(options, size=n, p=probs)


def generate() -> pd.DataFrame:
    rng = _rng()
    rows = []
    gid = 0

    group_labels = {0: "A 对照组", 1: "B 结果型建议",
                    2: "C 过程型建议", 3: "D 范例型建议"}

    for g, n in N_PER_GROUP.items():
        for i in range(n):
            gid += 1
            baseline = _likert(rng, 1, 5.0 + rng.normal(0, 0.3), 1.0)[0]
            rows.append({
                "group_code": g,
                "answer_id": f"demo_{gid:04d}",
                "duration_s": int(np.clip(rng.normal(180, 60), 60, 500)),
                "gender": rng.choice(["男", "女"], p=[0.4, 0.6]),
                "grade": rng.choice(["大一", "大二", "大三", "大四", "研究生"],
                                    p=[0.15, 0.25, 0.30, 0.20, 0.10]),
                "major": rng.choice(["经管类", "理工类", "人文社科类", "其他"],
                                    p=[0.40, 0.25, 0.25, 0.10]),
                "ai_frequency": rng.choice(["几乎每天", "每周几次", "每月几次", "很少/从不"],
                                           p=[0.15, 0.35, 0.30, 0.20]),
                "dining_frequency": rng.choice(["每天", "每周4-5次", "每周2-3次", "每周1次及以下"],
                                               p=[0.30, 0.35, 0.25, 0.10]),
                "green_baseline_1": int(baseline),
                "green_baseline_2": int(np.clip(baseline + rng.normal(0, 0.8), 1, 7)),
                "assigned_condition": group_labels[g],
                # Behavioural DVs — small shift for treatment groups
                "canteen_choice": _canteen_choice(rng, g, baseline),
                "recycling_choice": _recycling_choice(rng, g, baseline),
                "study_space_choice": _study_choice(rng, g, baseline),
                # Manipulation checks — small differences (intentional: weak manipulation)
                "manip_ai_identity": int(_likert(rng, 1, 5.2 + 0.1 * g, 1.1)[0]),
                "manip_clear_conclusion": int(_likert(rng, 1, 5.0 + 0.15 * g, 1.2)[0]),
                "manip_explanation": int(_likert(rng, 1, 4.8 + 0.2 * g, 1.3)[0]),
                "manip_recall": rng.choice(["记得", "不记得"], p=[0.7, 0.3]),
                # Trust — slightly higher for explanation groups
                "trust_1": int(_likert(rng, 1, 4.8 + 0.1 * g, 1.2)[0]),
                "trust_2": int(_likert(rng, 1, 4.6 + 0.1 * g, 1.1)[0]),
                "trust_3": int(_likert(rng, 1, 4.7 + 0.08 * g, 1.2)[0]),
                # Cognitive load — slightly lower for explanation groups
                "load_1": int(_likert(rng, 1, 3.5 - 0.05 * g, 1.2)[0]),
                "load_2": int(_likert(rng, 1, 3.3 - 0.05 * g, 1.1)[0]),
                "load_3": int(_likert(rng, 1, 3.4 - 0.04 * g, 1.2)[0]),
                # Feasibility
                "feasibility_1": int(_likert(rng, 1, 4.8 + 0.1 * g, 1.1)[0]),
                "feasibility_2": int(_likert(rng, 1, 4.9 + 0.08 * g, 1.2)[0]),
                "social_norm": int(_likert(rng, 1, 4.5, 1.3)[0]),
                # Green intention
                "intention_1": int(_likert(rng, 1, 5.0 + 0.15 * g, 1.1)[0]),
                "intention_2": int(_likert(rng, 1, 4.8 + 0.12 * g, 1.2)[0]),
                "intention_3": int(_likert(rng, 1, 4.9 + 0.1 * g, 1.1)[0]),
                "attention_check": rng.choice(["比较同意", "非常同意", "一般"],
                                              p=[0.75, 0.15, 0.10]),
                "scenario_fit": rng.choice(["非常贴近", "比较贴近", "一般", "不太贴近"],
                                           p=[0.20, 0.45, 0.25, 0.10]),
                "open_comment": rng.choice(["", "实验设计合理", "希望有更多情境",
                                            "建议表述可更清晰"],
                                           p=[0.70, 0.10, 0.10, 0.10]),
            })

    df = pd.DataFrame(rows)
    # Shuffle rows so groups aren't clustered
    df = df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
    return df


def _canteen_choice(rng: np.random.Generator, group: int, baseline: float) -> str:
    # Ordinal 0/1/2: baseline shifts probability of greener choice
    base_p = [0.25, 0.45, 0.30]
    # Small group effect (weak, consistent with real null result)
    shift = 0.02 * group + 0.02 * (baseline - 4) / 3
    probs = np.array([base_p[0] - shift, base_p[1], base_p[2] + shift])
    probs = np.clip(probs, 0.05, 0.9)
    probs = probs / probs.sum()
    return rng.choice([
        "选常规份套餐，并领取一次性餐具",
        "选常规份套餐，不领取一次性餐具",
        "选小份菜／按需取餐，不领取一次性餐具",
    ], p=probs)


def _recycling_choice(rng: np.random.Generator, group: int, baseline: float) -> str:
    p_yes = 0.55 + 0.03 * group + 0.05 * (baseline - 4) / 3
    p_yes = float(np.clip(p_yes, 0.2, 0.85))
    return rng.choice(["带去可回收物投放点", "留在座位上"],
                      p=[p_yes, 1 - p_yes])


def _study_choice(rng: np.random.Generator, group: int, baseline: float) -> str:
    p_green = 0.40 + 0.02 * group + 0.04 * (baseline - 4) / 3
    p_green = float(np.clip(p_green, 0.15, 0.75))
    return rng.choice(["A. 已开启设备的开放自习区", "B. 未开启设备的节能自习区"],
                      p=[1 - p_green, p_green])


def main() -> None:
    df = generate()
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="分析数据", index=False)
    print(f"Demo dataset written: {OUT}")
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"  Group sizes: {dict(df['group_code'].value_counts().sort_index())}")


if __name__ == "__main__":
    main()
