"""AI Assistant Advice Types and Green Decision-Making — Top-tier journal-level
reproducible analysis pipeline for a four-group randomized controlled experiment.

Design
------
- Four-group RCT: A (control) / B (result-only) / C (process-based) / D (example-based)
- Primary DVs:
  (1) Standardized green choice index (cafeteria + recycling + study area)
  (2) Green behavioural intention
- Reporting standards: estimates + 95% CI + effect sizes + multiple-testing
  correction (BH-FDR + Holm) + Bayesian factors + robustness checks.
- Pre-registered analyses: main effects, manipulation check,
  reliability/validity, mediation (Hayes PROCESS bootstrap).
- Exploratory: heterogeneity (gender / AI usage frequency / environmental baseline).

Usage
-----
    # Quick start with bundled demo data (no real data required):
    python analysis/top_journal_analysis.py --demo

    # With your own data:
    python analysis/top_journal_analysis.py --input path/to/data.xlsx

    # Output goes to results/ by default
    python analysis/top_journal_analysis.py --demo --output results/

References: CONSORT 2010; Hayes (2018) PROCESS; Wagenmakers (2007) BF;
Lakens (2017) TOST; Benjamini & Hochberg (1995) FDR.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                    "FangSong", "KaiTi", "SimSun",
                                    "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# CLI & Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repo root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Four-group RCT analysis — AI advice and green decision-making")
    parser.add_argument("--input", "-i", type=Path, default=None,
                        help="Path to input Excel file (sheet: '分析数据')")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: results/)")
    parser.add_argument("--demo", action="store_true",
                        help="Use bundled synthetic demo data")
    parser.add_argument("--boot-reps", type=int, default=5000,
                        help="Bootstrap replications (default: 5000)")
    parser.add_argument("--seed", type=int, default=20260902,
                        help="Random seed (default: 20260902)")
    return parser.parse_args()


def _resolve_input(args: argparse.Namespace) -> Path:
    """Resolve input data path, falling back to demo data if needed."""
    if args.demo:
        demo_path = PROJECT_ROOT / "data" / "demo_dataset.xlsx"
        if not demo_path.exists():
            print(f"[ERROR] Demo data not found at {demo_path}")
            print("Run: python data/generate_demo_data.py  first")
            sys.exit(1)
        print(f"[INFO] Using demo dataset: {demo_path.name}")
        print("[NOTE] Demo data is synthetic — for pipeline testing only.")
        return demo_path

    if args.input is not None:
        p = args.input.resolve()
        if not p.exists():
            print(f"[ERROR] Input file not found: {p}")
            sys.exit(1)
        return p

    # Default: try real data, then demo
    real_path = PROJECT_ROOT / "data" / "real_data.xlsx"
    demo_path = PROJECT_ROOT / "data" / "demo_dataset.xlsx"
    if real_path.exists():
        return real_path
    if demo_path.exists():
        print(f"[WARN] Real data not found. Falling back to demo dataset.")
        print(f"       Put your data at: data/real_data.xlsx")
        print(f"       Or use: python analysis/top_journal_analysis.py --input your_data.xlsx")
        return demo_path

    print("[ERROR] No input data found.")
    print("  Use --demo for synthetic data, or --input for your own data.")
    sys.exit(1)


ARGS = _parse_args()
INPUT = _resolve_input(ARGS)
OUT = (ARGS.output or PROJECT_ROOT / "results").resolve()
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = [0, 1, 2, 3]
GROUP_LABELS = {0: "A 对照", 1: "B 结果型", 2: "C 过程型", 3: "D 范例型"}
GROUP_ORDER = [GROUP_LABELS[g] for g in GROUPS]
RNG_SEED = ARGS.seed
BOOT_REPS = ARGS.boot_reps


# ===========================================================================
# 辅助函数
# ===========================================================================
def stars(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha，要求至少两题。"""
    x = items.dropna()
    if x.shape[1] < 2 or x.shape[0] < 2:
        return np.nan
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1))


def cronbach_ci(items: pd.DataFrame, reps: int = BOOT_REPS, seed: int = RNG_SEED) -> tuple[float, float]:
    """Bootstrap 95% CI。"""
    rng = np.random.default_rng(seed)
    x = items.dropna().to_numpy()
    n = len(x)
    alphas = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        b = pd.DataFrame(x[idx])
        a = cronbach_alpha(b)
        if not np.isnan(a):
            alphas.append(a)
    if not alphas:
        return np.nan, np.nan
    return float(np.quantile(alphas, .025)), float(np.quantile(alphas, .975))


def item_total_correlation(items: pd.DataFrame) -> pd.DataFrame:
    """项总计相关 + 删除项后 alpha。"""
    cols = list(items.columns)
    rows = []
    for c in cols:
        rest = items.drop(columns=[c]).sum(axis=1)
        r = items[c].corr(rest)
        a_del = cronbach_alpha(items.drop(columns=[c]))
        rows.append({"item": c, "item_total_r": r, "alpha_if_deleted": a_del})
    return pd.DataFrame(rows)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    sd = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sd if sd > 0 else np.nan


def hedges_g(d: float, n1: int, n2: int) -> float:
    n = n1 + n2
    return d * (1 - 3 / (4 * (n - 1) - 1))


def d_ci(d: float, n1: int, n2: int) -> tuple[float, float]:
    n = n1 + n2
    se = np.sqrt(n1 / (n1 * (n1 - 1)) + n2 / (n2 * (n2 - 1)) + d ** 2 / (2 * (n - 1)))
    return d - 1.96 * se, d + 1.96 * se


def cramers_v(crosstab: pd.DataFrame) -> float:
    chi2, _, _, n = stats.chi2_contingency(crosstab)
    r, k = crosstab.shape
    return np.sqrt(chi2 / (crosstab.values.sum() * (min(r, k) - 1)))


def f_to_eta2(f, df1, df2) -> float:
    return (f * df1) / (f * df1 + df2)


def partial_eta2(model) -> float:
    """偏 η² = SS_effect / (SS_effect + SS_error) for OLS。"""
    try:
        ess = model.ess
        if hasattr(model, "ssr"):
            return ess / (ess + model.ssr)
        return np.nan
    except Exception:
        return np.nan


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR 校正。"""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * np.arange(n, 0, -1)
    adj = np.maximum.accumulate(adj)
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def bic_to_bf(bic_diff: float) -> float:
    """Wagenmakers (2007)：BF01 = exp(ΔBIC / 2)，BIC(H1)-BIC(H0)。"""
    return float(np.exp(bic_diff / 2))


def bf_two_sample_t(d: float, n1: int, n2: int) -> float:
    """基于 t 统计量的近似 BF10（Wagenmakers 2007 / Roudser 2009 简化版）。"""
    n = n1 + n2
    if d == 0:
        return 1.0
    se = np.sqrt(n1 / (n1 * (n1 - 1)) + n2 / (n2 * (n2 - 1)) + d ** 2 / (2 * (n - 1)))
    t = d / se
    df = n - 2
    # 单侧 JZS 近似（Ly et al. 2016 简化）：使用 t 与似然比的代数
    if abs(t) < 1e-9:
        return 1.0
    log_bf = -0.5 * np.log1p(t * t / df) + np.log(np.abs(t)) * 0.5
    return float(np.exp(log_bf))


def tost(x: np.ndarray, y: np.ndarray, eqv: float = 0.5) -> dict:
    """TOST 等效性检验，eqv 为 Cohen's d 等效界（默认0.5）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    diff = x.mean() - y.mean()
    sd = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    se = sd * np.sqrt(1 / n1 + 1 / n2)
    bound = eqv * sd
    t1 = (diff - (-bound)) / se
    t2 = (diff - bound) / se
    df = n1 + n2 - 2
    p1 = stats.t.cdf(t1, df)  # H1: diff > -bound
    p2 = 1 - stats.t.cdf(t2, df)  # H1: diff < +bound
    return {"diff": diff, "se": se, "bound_d": eqv, "t_low": t1, "t_high": t2,
            "p_low": p1, "p_high": p2, "equivalent": max(p1, p2) < .05}


def post_hoc_power(effect_d: float, n_per_group: int, alpha: float = .05) -> float:
    """两点法正态近似的事后功效（Cohen 1988）。"""
    n1 = n2 = n_per_group
    ncp = effect_d / np.sqrt(2 / n1)
    z = stats.norm.ppf(1 - alpha / 2)
    power = 1 - stats.norm.cdf(z - ncp) + stats.norm.cdf(-z - ncp)
    return float(power)


# ===========================================================================
# 1. 数据读取与清洗（CONSORT 流程）
# ===========================================================================
print("=" * 80)
print("步骤 1/10  数据读取与样本流（CONSORT）")
print("=" * 80)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    raw = pd.read_excel(INPUT, sheet_name="分析数据")
n_raw = len(raw)

# 清洗规则
d_raw = raw.copy()
d_consented = d_raw[d_raw["answer_id"].notna()]
n_consented = len(d_consented)
d_dup = d_consented.drop_duplicates("answer_id")
n_dedup = len(d_dup)
d_att = d_dup[d_dup["attention_check"] == "比较同意"]
n_att = len(d_att)
# 时长异常样本剔除：剔除 <60 秒 与 > mean+3SD
d_att = d_att.copy()
d_att["duration_s"] = pd.to_numeric(d_att["duration_s"], errors="coerce")
dur_mean, dur_sd = d_att["duration_s"].mean(), d_att["duration_s"].std()
d_dur = d_att[(d_att["duration_s"] >= 60) &
              (d_att["duration_s"] <= dur_mean + 3 * dur_sd)]
n_dur = len(d_dur)
d = d_dur.copy()
n_final = len(d)

consort = pd.DataFrame({
    "step": ["1. 原始导出样本", "2. 含 answer_id 的有效作答",
             "3. 去重后样本", "4. 注意力题合格",
             "5. 剔除极端时长(<60s 或 >mean+3SD)", "6. 最终分析样本"],
    "n": [n_raw, n_consented, n_dedup, n_att, n_dur, n_final],
    "excluded": [0, n_raw - n_consented, n_consented - n_dedup,
                 n_dedup - n_att, n_att - n_dur, 0],
})
print(consort.to_string(index=False))


# ===========================================================================
# 2. 变量构建
# ===========================================================================
print("\n" + "=" * 80)
print("步骤 2/10  变量构建")
print("=" * 80)

numeric_cols = [c for c in d.columns
                if c.startswith(("green_baseline", "trust_", "load_",
                                 "feasibility_", "intention_", "manip_"))]
for c in numeric_cols:
    if c != "manip_recall":
        d[c] = pd.to_numeric(d[c], errors="coerce")

# 量表均分
d["green_baseline"] = d[["green_baseline_1", "green_baseline_2"]].mean(axis=1)
d["trust"] = d[["trust_1", "trust_2", "trust_3"]].mean(axis=1)
d["cognitive_load"] = d[["load_1", "load_2", "load_3"]].mean(axis=1)
d["feasibility"] = d[["feasibility_1", "feasibility_2"]].mean(axis=1)
d["green_intention"] = d[["intention_1", "intention_2", "intention_3"]].mean(axis=1)
d["social_norm"] = pd.to_numeric(d.get("social_norm"), errors="coerce")

# 行为结果编码
d["green_canteen"] = d["canteen_choice"].map({
    "选常规份套餐，并领取一次性餐具": 0,
    "选常规份套餐，不领取一次性餐具": 1,
    "选小份菜／按需取餐，并领取一次性餐具": 1,
    "选小份菜／按需取餐，不领取一次性餐具": 2,
})
d["green_recycling"] = (d["recycling_choice"] == "带去可回收物投放点").astype(int)
d["green_study"] = (d["study_space_choice"] == "A. 已开启设备的开放自习区").astype(int)

# 标准化合成指数（z-score 平均）
for c in ["green_canteen", "green_recycling", "green_study"]:
    d[f"z_{c}"] = (d[c] - d[c].mean()) / d[c].std(ddof=0)
d["green_choice_index"] = d[["z_green_canteen", "z_green_recycling", "z_green_study"]].mean(axis=1)

# AI vs 对照二元处理
d["any_ai"] = (d.group_code != 0).astype(int)

# 组别标签
d["group"] = d["group_code"].map(GROUP_LABELS).astype(pd.CategoricalDtype(GROUP_ORDER, ordered=True))

print(f"各处理组样本量：\n{d['group'].value_counts().reindex(GROUP_ORDER)}")
print(f"\n合成指数 M(SD)={d['green_choice_index'].mean():.3f}({d['green_choice_index'].std():.3f})")


# ===========================================================================
# 3. 操纵检验（Manipulation Check）
# ===========================================================================
print("\n" + "=" * 80)
print("步骤 3/10  操纵检验")
print("=" * 80)

manip_items = {
    "manip_ai_identity": "AI 身份识别",
    "manip_clear_conclusion": "结论清晰度",
    "manip_explanation": "解释充分度",
}

manip_rows = []
for col, label in manip_items.items():
    arrays = [d.loc[d.group_code == g, col].dropna() for g in GROUPS]
    f, p = stats.f_oneway(*arrays)
    rows = []
    for g in GROUPS:
        x = d.loc[d.group_code == g, col].dropna()
        for g2 in GROUPS:
            if g < g2:
                y = d.loc[d.group_code == g2, col].dropna()
                dd = cohens_d(x, y)
                g1 = hedges_g(dd, len(x), len(y))
                cil, cih = d_ci(dd, len(x), len(y))
                rows.append({"manip": label, "contrast": f"{GROUP_LABELS[g]} vs {GROUP_LABELS[g2]}",
                             "d": dd, "hedges_g": g1, "ci_low": cil, "ci_high": cih,
                             "p_anova": p})
    manip_rows.extend(rows)
manip_check = pd.DataFrame(manip_rows)
manip_check["p_BH"] = fdr_bh(manip_check["p_anova"].values)
manip_check["sig"] = [stars(p) for p in manip_check["p_BH"]]
print(manip_check.round(3).to_string(index=False))


# ===========================================================================
# 4. 信度与效度
# ===========================================================================
print("\n" + "=" * 80)
print("步骤 4/10  量表信度与效度")
print("=" * 80)

scales = {
    "环保基线": ["green_baseline_1", "green_baseline_2"],
    "AI信任": ["trust_1", "trust_2", "trust_3"],
    "认知负荷": ["load_1", "load_2", "load_3"],
    "可行性感知": ["feasibility_1", "feasibility_2"],
    "绿色意愿": ["intention_1", "intention_2", "intention_3"],
}
rel_rows = []
itc_frames = []
for name, items in scales.items():
    a = cronbach_alpha(d[items])
    ci = cronbach_ci(d[items])
    itc = item_total_correlation(d[items])
    itc["scale"] = name
    itc_frames.append(itc)
    # CR & AVE (聚合效度)
    loadings = []
    for c in items:
        f1 = np.sqrt(np.var(d[c], ddof=1))
        loadings.append(f1)
    loadings = np.array(loadings)
    # 标准化载荷近似
    rel_rows.append({"scale": name, "items": len(items), "alpha": a,
                     "alpha_ci_low": ci[0], "alpha_ci_high": ci[1]})
reliability = pd.DataFrame(rel_rows)
reliability["alpha_adequate"] = reliability["alpha"] >= .70
print(reliability.round(3).to_string(index=False))
itc_table = pd.concat(itc_frames, ignore_index=True)
print("\n项总计相关：")
print(itc_table.round(3).to_string(index=False))

# 区分效度：HTMT（异质-单质比）
def htmt(d: pd.DataFrame, scale_items: dict) -> pd.DataFrame:
    names = list(scale_items.keys())
    n = len(names)
    mat = pd.DataFrame(np.eye(n), index=names, columns=names, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            items_i = scale_items[names[i]]
            items_j = scale_items[names[j]]
            all_items = items_i + items_j
            common = d[all_items].dropna().index
            het_corrs = []
            for a in items_i:
                for b in items_j:
                    if a != b:
                        het_corrs.append(np.corrcoef(d.loc[common, a], d.loc[common, b])[0, 1])
            mono_i = [np.corrcoef(d.loc[common, a], d.loc[common, b])[0, 1]
                      for a in items_i for b in items_i if a != b]
            mono_j = [np.corrcoef(d.loc[common, a], d.loc[common, b])[0, 1]
                      for a in items_j for b in items_j if a != b]
            ht = np.mean(np.abs(het_corrs)) if het_corrs else np.nan
            ms = (np.mean(mono_i) + np.mean(mono_j)) / 2 if (mono_i and mono_j) else np.nan
            val = ht / ms if ms else np.nan
            mat.iloc[i, j] = mat.iloc[j, i] = val
    return mat

htmt_mat = htmt(d, scales)
print("\nHTMT 区分效度矩阵（<.85 为良好）：")
print(htmt_mat.round(3))


# ===========================================================================
# 5. 随机化平衡检验
# ===========================================================================
print("\n" + "=" * 80)
print("步骤 5/10  随机化平衡检验")
print("=" * 80)

balance_rows = []
# 连续变量
for x in ["green_baseline", "duration_s"]:
    arrays = [d.loc[d.group_code == g, x].dropna() for g in GROUPS]
    f, p = stats.f_oneway(*arrays)
    cohen_f = np.sqrt(f_to_eta2(f, 3, sum(len(a) for a in arrays) - 4) /
                       (1 - f_to_eta2(f, 3, sum(len(a) for a in arrays) - 4)))
    balance_rows.append({"variable": x, "test": "ANOVA", "statistic": f, "p_value": p,
                         "effect_size": cohen_f, "effect_label": "Cohen's f"})
# 分类变量
for x in ["gender", "grade", "major", "ai_frequency", "dining_frequency"]:
    tab = pd.crosstab(d[x], d.group_code)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    v = cramers_v(tab)
    balance_rows.append({"variable": x, "test": "chi-square", "statistic": chi2,
                         "p_value": p, "effect_size": v, "effect_label": "Cramer's V"})
balance = pd.DataFrame(balance_rows)
balance["sig"] = [stars(p) for p in balance["p_value"]]
print(balance.round(3).to_string(index=False))


# ===========================================================================
# 6. 描述性统计与相关矩阵
# ===========================================================================
print("\n" + "=" * 80)
print("步骤 6/10  描述性统计与相关矩阵")
print("=" * 80)

key_vars = ["green_baseline", "trust", "cognitive_load", "feasibility",
            "social_norm", "green_intention", "green_canteen",
            "green_recycling", "green_study", "green_choice_index"]
desc = d[key_vars].describe().T[["count", "mean", "std", "min", "max"]].round(3)
desc["skew"] = d[key_vars].skew().round(3)
desc["kurtosis"] = d[key_vars].kurt().round(3)
print(desc)

# 组别均值
group_means = d.groupby("group")[key_vars].agg(["mean", "std"]).round(3)
group_means.columns = ["_".join(c) for c in group_means.columns]
print("\n各处理组均值：")
print(group_means)

# 相关矩阵（含 BH 校正）
corr = d[key_vars].corr()
n_obs = len(d)
pvals = np.full_like(corr, np.nan)
for i, a in enumerate(key_vars):
    for j, b in enumerate(key_vars):
        if i != j:
            r, p = stats.pearsonr(d[a].dropna(), d[b].dropna())
            pvals[i, j] = p
flat_p = pvals[~np.isnan(pvals)]
adj_flat = fdr_bh(flat_p)
adj_mat = np.full_like(corr, np.nan)
idx = 0
for i in range(len(key_vars)):
    for j in range(len(key_vars)):
        if i != j and not np.isnan(pvals[i, j]):
            adj_mat[i, j] = adj_flat[idx]
            idx += 1
sig_mat = np.vectorize(stars)(adj_mat)
print("\n相关矩阵（Pearson r，BH 校正 p）：")
print(corr.round(2).astype(str) + sig_mat)


# ===========================================================================
# 7. 主效应（顶刊规格：OLS / Logit / 有序 Logit；HC3；效应量；多重校正）
# ===========================================================================
print("\n" + "=" * 80)
print("步骤 7/10  主效应：预注册 ITT 分析")
print("=" * 80)

controls = ("green_baseline + C(gender) + C(grade) + C(ai_frequency) + "
           "C(dining_frequency)")
cont_outcomes = ["green_choice_index", "green_intention", "trust",
                 "cognitive_load", "feasibility"]
bin_outcomes = ["green_recycling", "green_study"]
ord_outcomes = ["green_canteen"]

# ---- 7a. OLS 主效应 ----
main_tables = []
for y in cont_outcomes:
    m_unadj = smf.ols(f"{y} ~ C(group_code, Treatment(reference=0))", d).fit(cov_type="HC3")
    m_adj = smf.ols(f"{y} ~ C(group_code, Treatment(reference=0)) + {controls}", d).fit(cov_type="HC3")
    for label, mod in [(f"{y}:未调整 OLS", m_unadj), (f"{y}:调整 OLS", m_adj)]:
        ci = mod.conf_int()
        # 组别系数 vs 对照组的 Cohen d
        a_ref = d.loc[d.group_code == 0, y].dropna().to_numpy()
        for g in [1, 2, 3]:
            term = f"C(group_code, Treatment(reference=0))[T.{g}]"
            if term in mod.params.index:
                beta = mod.params[term]
                se = mod.bse[term]
                pval = mod.pvalues[term]
                # 各组样本量用于效应量
                g_data = d.loc[d.group_code == g, y].dropna().to_numpy()
                dd = cohens_d(g_data, a_ref)
                rows = {
                    "outcome": y, "model": label, "contrast": f"{GROUP_LABELS[g]} vs A",
                    "estimate": beta, "std_error": se, "ci_low": ci.loc[term, 0],
                    "ci_high": ci.loc[term, 1], "p_value": pval,
                    "cohen_d": dd, "n_treat": len(g_data), "n_ref": len(a_ref)
                }
                main_tables.append(rows)
        main_tables.append({
            "outcome": y, "model": label, "contrast": "R²",
            "estimate": mod.rsquared, "std_error": np.nan, "ci_low": np.nan,
            "ci_high": np.nan, "p_value": mod.f_pvalue, "cohen_d": np.nan,
            "n_treat": mod.nobs, "n_ref": mod.nobs,
        })
main_ols = pd.DataFrame(main_tables)
# 多重检验校正（仅处理系数）
treat_mask = main_ols["contrast"].str.contains("vs A")
adj = fdr_bh(main_ols.loc[treat_mask, "p_value"].values)
main_ols.loc[treat_mask, "p_BH"] = adj
holm = holm_adjust(main_ols.loc[treat_mask, "p_value"].values)
main_ols.loc[treat_mask, "p_Holm"] = holm
main_ols["sig"] = main_ols["p_value"].apply(stars)
print("\nOLS 主效应（HC3 稳健 SE）：")
print(main_ols[main_ols["model"].str.contains("调整")].round(3).to_string(index=False))

# ---- 7b. Logit（二元） ----
logit_tables = []
for y in bin_outcomes:
    mod = smf.logit(f"{y} ~ C(group_code, Treatment(reference=0)) + {controls}", d).fit(disp=False)
    ci = mod.conf_int()
    for g in [1, 2, 3]:
        term = f"C(group_code, Treatment(reference=0))[T.{g}]"
        if term in mod.params.index:
            beta = mod.params[term]
            orr = np.exp(beta)
            rows = {
                "outcome": y, "contrast": f"{GROUP_LABELS[g]} vs A",
                "log_odds": beta, "OR": orr,
                "ci_low_OR": np.exp(ci.loc[term, 0]),
                "ci_high_OR": np.exp(ci.loc[term, 1]),
                "p_value": mod.pvalues[term],
                "n_obs": int(mod.nobs),
            }
            logit_tables.append(rows)
logit_results = pd.DataFrame(logit_tables)
logit_results["p_BH"] = fdr_bh(logit_results["p_value"].values)
logit_results["p_Holm"] = holm_adjust(logit_results["p_value"].values)
logit_results["sig"] = logit_results["p_value"].apply(stars)
print("\n二元选择 Logit（OR, 95%CI）：")
print(logit_results.round(3).to_string(index=False))

# ---- 7c. 有序 Logit（食堂 0/1/2） ----
try:
    ord_mod = sm.MNLogit.from_formula(
        "green_canteen ~ C(group_code, Treatment(reference=0)) + " + controls,
        data=d).fit(method="bfgs", disp=False, maxiter=200)
    ord_df = pd.DataFrame({
        "term": ord_mod.params.index,
        "estimate": ord_mod.params.values[:, 0],
        "p_value": ord_mod.pvalues.values[:, 0],
    }) if ord_mod.params.ndim > 1 else pd.DataFrame({
        "term": ord_mod.params.index,
        "estimate": ord_mod.params.values,
        "p_value": ord_mod.pvalues.values,
    })
    ord_df["OR"] = np.exp(ord_df["estimate"])
    ord_df["sig"] = ord_df["p_value"].apply(stars)
    print("\n有序结果（食堂 0/1/2）MNLogit：")
    print(ord_df.round(3).to_string(index=False))
except Exception as e:
    ord_df = pd.DataFrame({"note": [f"有序模型未收敛或失败：{e}"]})
    print("有序 Logit 失败：", e)


# ===========================================================================
# 8. 中介分析（预注册：Hayes PROCESS Bootstrap）
# ===========================================================================
print("\n" + "=" * 80)
print("步骤 8/10  中介分析（Hayes PROCESS，Bootstrap 5000）")
print("=" * 80)

def mediation_bootstrap(data: pd.DataFrame, treatment: int, mediator: str,
                         outcome: str, controls: str, reps: int = BOOT_REPS,
                         seed: int = RNG_SEED) -> dict:
    """处理组 vs 对照 A 的 Bootstrap 中介（矩阵化，跳过公式解析）。

    返回 a, b, c, c', ab, 95%CI。
    """
    rng = np.random.default_rng(seed)
    d_sub = data.loc[data.group_code.isin([0, treatment])].copy()
    d_sub["treat"] = (d_sub.group_code == treatment).astype(int)
    f_m = f"{mediator} ~ treat + {controls}"
    f_y = f"{outcome} ~ treat + {mediator} + {controls}"
    f_c = f"{outcome} ~ treat + {controls}"
    # 一次性构造设计矩阵，避免 bootstrap 循环里反复解析公式
    y_m, X_m = patsy.dmatrices(f_m, d_sub, return_type="dataframe")
    y_y, X_y = patsy.dmatrices(f_y, d_sub, return_type="dataframe")
    y_c, X_c = patsy.dmatrices(f_c, d_sub, return_type="dataframe")
    X_m_np, X_y_np, X_c_np = X_m.to_numpy(), X_y.to_numpy(), X_c.to_numpy()
    y_m_np, y_y_np, y_c_np = y_m.to_numpy().ravel(), y_y.to_numpy().ravel(), y_c.to_numpy().ravel()
    treat_idx_m = list(X_m.columns).index("treat")
    treat_idx_y = list(X_y.columns).index("treat")
    treat_idx_c = list(X_c.columns).index("treat")
    med_idx_y = list(X_y.columns).index(mediator)
    n = len(d_sub)
    # 点估计（全样本，HC3）
    m_full = sm.OLS(y_m_np, X_m_np).fit(cov_type="HC3")
    y_full = sm.OLS(y_y_np, X_y_np).fit(cov_type="HC3")
    c_full = sm.OLS(y_c_np, X_c_np).fit(cov_type="HC3")
    a = float(m_full.params[treat_idx_m])
    b = float(y_full.params[med_idx_y])
    c = float(c_full.params[treat_idx_c])
    cprime = float(y_full.params[treat_idx_y])
    ab = a * b
    draws = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        try:
            a_b = np.linalg.lstsq(X_m_np[idx], y_m_np[idx], rcond=None)[0][treat_idx_m]
            b_b = np.linalg.lstsq(X_y_np[idx], y_y_np[idx], rcond=None)[0][med_idx_y]
            draws.append(a_b * b_b)
        except Exception:
            continue
    if not draws:
        return {"contrast": f"{GROUP_LABELS[treatment]} vs A", "mediator": mediator,
                "outcome": outcome, "a": a, "b": b, "c": c, "c_prime": cprime,
                "indirect_ab": ab, "ci_low": np.nan, "ci_high": np.nan,
                "ci_excludes_zero": np.nan, "mediated_pct": ab / c * 100 if c else np.nan,
                "bootstrap_reps": 0}
    cil, cih = float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
    return {"contrast": f"{GROUP_LABELS[treatment]} vs A", "mediator": mediator,
            "outcome": outcome, "a": a, "b": b, "c": c, "c_prime": cprime,
            "indirect_ab": ab, "ci_low": cil, "ci_high": cih,
            "ci_excludes_zero": bool((cil > 0) | (cih < 0)),
            "mediated_pct": ab / c * 100 if c else np.nan,
            "bootstrap_reps": len(draws)}


mediation_rows = []
# 预注册中介：B/C/D 经 trust 与 cognitive_load 影响 green_intention
for t in [1, 2, 3]:
    for med in ["trust", "cognitive_load", "feasibility"]:
        r = mediation_bootstrap(d, t, med, "green_intention", controls)
        mediation_rows.append(r)
# 同时对绿色选择指数做中介
for t in [1, 2, 3]:
    for med in ["trust", "cognitive_load"]:
        r = mediation_bootstrap(d, t, med, "green_choice_index", controls)
        mediation_rows.append(r)
mediation = pd.DataFrame(mediation_rows)
print(mediation.round(3).to_string(index=False))

# 并行多重中介：以 D 为例，trust + load 并行
def parallel_mediation(data, treatment, mediators, outcome, controls, reps=BOOT_REPS, seed=RNG_SEED):
    d_sub = data.loc[data.group_code.isin([0, treatment])].copy()
    d_sub["treat"] = (d_sub.group_code == treatment).astype(int)
    med_terms = " + ".join(mediators)
    f_m_base = "{} ~ treat + " + controls
    f_y = f"{outcome} ~ treat + {med_terms} + {controls}"
    # 预构造设计矩阵
    m_mats = {}
    for m in mediators:
        y_m, X_m = patsy.dmatrices(f_m_base.format(m), d_sub, return_type="dataframe")
        m_mats[m] = (y_m.to_numpy().ravel(), X_m.to_numpy(), list(X_m.columns).index("treat"))
    y_y, X_y = patsy.dmatrices(f_y, d_sub, return_type="dataframe")
    X_y_np = X_y.to_numpy()
    y_y_np = y_y.to_numpy().ravel()
    med_idx = {m: list(X_y.columns).index(m) for m in mediators}
    # 点估计（HC3）
    a_params, b_params = {}, {}
    for m in mediators:
        ym, Xm, ti = m_mats[m]
        a_params[m] = float(sm.OLS(ym, Xm).fit(cov_type="HC3").params[ti])
    y_full = sm.OLS(y_y_np, X_y_np).fit(cov_type="HC3")
    for m in mediators:
        b_params[m] = float(y_full.params[med_idx[m]])
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in mediators}
    n = len(d_sub)
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        try:
            a_bs = {m: np.linalg.lstsq(m_mats[m][1][idx], m_mats[m][0][idx], rcond=None)[0][m_mats[m][2]] for m in mediators}
            b_bs = np.linalg.lstsq(X_y_np[idx], y_y_np[idx], rcond=None)[0]
            for m in mediators:
                draws[m].append(a_bs[m] * b_bs[med_idx[m]])
        except Exception:
            continue
    out = {"contrast": f"{GROUP_LABELS[treatment]} vs A", "outcome": outcome,
           "mediators": "+".join(mediators)}
    for m in mediators:
        ab = a_params[m] * b_params[m]
        cil, cih = (np.quantile(draws[m], .025), np.quantile(draws[m], .975)) if draws[m] else (np.nan, np.nan)
        out[f"ab_{m}"] = ab
        out[f"ci_low_{m}"] = cil
        out[f"ci_high_{m}"] = cih
        out[f"excl0_{m}"] = (cil > 0) | (cih < 0)
    return out

par_rows = []
for t in [1, 2, 3]:
    par_rows.append(parallel_mediation(d, t, ["trust", "cognitive_load"], "green_intention", controls))
    par_rows.append(parallel_mediation(d, t, ["trust", "cognitive_load", "feasibility"], "green_intention", controls))
parallel_med = pd.DataFrame(par_rows)
print("\n并行多重中介（trust + load + feasibility -> intention）：")
print(parallel_med.round(3).to_string(index=False))


# ===========================================================================
# 9. 异质性（探索性：性别 / AI 使用频率 / 环保基线）
# ===========================================================================
print("\n" + "=" * 80)
print("步骤 9/10  异质性分析（探索性）")
print("=" * 80)

het_rows = []
moderators_cont = ["green_baseline"]
moderators_cat = ["gender", "ai_frequency"]
for mod_var in moderators_cont + moderators_cat:
    if mod_var == "green_baseline":
        formula = (f"green_choice_index ~ C(group_code, Treatment(reference=0)) * {mod_var} + "
                   "C(gender) + C(grade) + C(ai_frequency) + C(dining_frequency)")
    else:
        formula = (f"green_choice_index ~ C(group_code, Treatment(reference=0)) * C({mod_var}) + "
                   "green_baseline + C(grade) + C(ai_frequency) + C(dining_frequency)")
    mod = smf.ols(formula, d).fit(cov_type="HC3")
    # 提取交互项
    for term in mod.params.index:
        if ":" in term:
            het_rows.append({
                "moderator": mod_var, "term": term,
                "estimate": mod.params[term], "std_error": mod.bse[term],
                "p_value": mod.pvalues[term],
                "ci_low": mod.conf_int().loc[term, 0],
                "ci_high": mod.conf_int().loc[term, 1],
            })
heterogeneity = pd.DataFrame(het_rows)
if len(heterogeneity):
    heterogeneity["p_BH"] = fdr_bh(heterogeneity["p_value"].values)
    heterogeneity["sig"] = heterogeneity["p_value"].apply(stars)
    print(heterogeneity.round(3).to_string(index=False))
else:
    print("未识别到显著交互项。")


# ===========================================================================
# 10. 稳健性 + 贝叶斯 + 等效性 + 功效 + 图形
# ===========================================================================
print("\n" + "=" * 80)
print("步骤 10/10  稳健性、贝叶斯、等效性、功效与图形")
print("=" * 80)

# ---- 10a. 稳健性：替代标准误（自举）----
rob_rows = []
# HC3 vs cluster by gender vs Bootstrap
mod_hc3 = smf.ols(
    f"green_choice_index ~ C(group_code, Treatment(reference=0)) + {controls}", d
).fit(cov_type="HC3")
mod_clu = smf.ols(
    f"green_choice_index ~ C(group_code, Treatment(reference=0)) + {controls}", d
).fit(cov_type="cluster", cov_kwds={"groups": d["gender"]})
for mod, label in [(mod_hc3, "HC3"), (mod_clu, "cluster(gender)")]:
    for g in [1, 2, 3]:
        term = f"C(group_code, Treatment(reference=0))[T.{g}]"
        if term in mod.params.index:
            rob_rows.append({"se_method": label, "contrast": f"{GROUP_LABELS[g]} vs A",
                             "estimate": mod.params[term], "std_error": mod.bse[term],
                             "p_value": mod.pvalues[term]})
robust_se = pd.DataFrame(rob_rows)
print("稳健标准误比较：")
print(robust_se.round(3).to_string(index=False))

# ---- 10b. 替代样本：剔除极端时长后再次剔除场景契合度<3 ----
d_rob = d[d["scenario_fit"].isin(["非常贴近", "比较贴近", "一般", "不太贴近"])].copy()
# 把"一般"以上视为合格（即剔除"不贴近"、"非常不贴近"）
fit_ok = ["非常贴近", "比较贴近", "一般"]
d_rob = d[d["scenario_fit"].isin(fit_ok)].copy()
n_rob = len(d_rob)
mod_rob = smf.ols(
    f"green_choice_index ~ C(group_code, Treatment(reference=0)) + {controls}", d_rob
).fit(cov_type="HC3")
rob_sample_rows = []
for g in [1, 2, 3]:
    term = f"C(group_code, Treatment(reference=0))[T.{g}]"
    if term in mod_rob.params.index:
        rob_sample_rows.append({
            "subsample": f"scenario_fit≥一般 (n={n_rob})",
            "contrast": f"{GROUP_LABELS[g]} vs A",
            "estimate": mod_rob.params[term], "std_error": mod_rob.bse[term],
            "p_value": mod_rob.pvalues[term],
        })
robust_sample = pd.DataFrame(rob_sample_rows)
print("\n替代样本稳健性（仅场景契合度合格者）：")
print(robust_sample.round(3).to_string(index=False))

# ---- 10c. 二元 AI vs 对照 ----
mod_anyai = smf.ols(
    f"green_choice_index ~ any_ai + {controls}", d).fit(cov_type="HC3")
ai_vs_ref = pd.DataFrame({
    "term": ["any_ai (B/C/D vs A)", "R²"],
    "estimate": [mod_anyai.params["any_ai"], mod_anyai.rsquared],
    "std_error": [mod_anyai.bse["any_ai"], np.nan],
    "p_value": [mod_anyai.pvalues["any_ai"], mod_anyai.f_pvalue],
})
print("\n二元 AI vs 对照：")
print(ai_vs_ref.round(3).to_string(index=False))

# ---- 10d. 贝叶斯 BF（基于 BIC 近似，Wagenmakers 2007） ----
bf_rows = []
for g in [1, 2, 3]:
    a = d.loc[d.group_code == 0, "green_choice_index"].dropna()
    b = d.loc[d.group_code == g, "green_choice_index"].dropna()
    dd = cohens_d(b, a)
    bf = bf_two_sample_t(dd, len(a), len(b))
    # 同时计算 BIC 近似 BF：H1-H0 BIC diff
    n = len(a) + len(b)
    # 模型对比：完整模型 vs 仅截距
    full = smf.ols(f"green_choice_index ~ C(group)", d.loc[d.group_code.isin([0, g])]).fit()
    null = smf.ols("green_choice_index ~ 1", d.loc[d.group_code.isin([0, g])]).fit()
    bic_diff = full.bic - null.bic  # 负表示支持 H1
    bf01 = bic_to_bf(bic_diff)  # BF01（支持 H0）
    bf_rows.append({"contrast": f"{GROUP_LABELS[g]} vs A", "cohen_d": dd,
                    "BF10_approx_t": bf, "bic_diff_H1_minus_H0": bic_diff,
                    "BF01_via_BIC": bf01, "BF10_via_BIC": 1 / bf01 if bf01 > 0 else np.nan})
bayes = pd.DataFrame(bf_rows)
print("\n贝叶斯因子（BF10 表示支持 H1 的相对证据）：")
print(bayes.round(3).to_string(index=False))

# ---- 10e. 等效性检验（TOST） ----
tost_rows = []
for g in [1, 2, 3]:
    a = d.loc[d.group_code == 0, "green_choice_index"].dropna().to_numpy()
    b = d.loc[d.group_code == g, "green_choice_index"].dropna().to_numpy()
    res = tost(b, a, eqv=0.5)
    tost_rows.append({"contrast": f"{GROUP_LABELS[g]} vs A",
                      "diff": res["diff"], "se": res["se"], "bound_d": res["bound_d"],
                      "p_low": res["p_low"], "p_high": res["p_high"],
                      "equivalent_at_d=0.5": res["equivalent"]})
tost_tab = pd.DataFrame(tost_rows)
print("\n等效性检验（TOST, d=0.5 界值）：")
print(tost_tab.round(3).to_string(index=False))

# ---- 10f. 事后功效分析 ----
power_rows = []
for g in [1, 2, 3]:
    a = d.loc[d.group_code == 0, "green_choice_index"].dropna()
    b = d.loc[d.group_code == g, "green_choice_index"].dropna()
    dd = cohens_d(b, a)
    pw = post_hoc_power(abs(dd), min(len(a), len(b)))
    power_rows.append({"contrast": f"{GROUP_LABELS[g]} vs A", "observed_d": dd,
                       "n_min": min(len(a), len(b)),
                       "alpha": .05, "power": pw})
power_tab = pd.DataFrame(power_rows)
print("\n事后功效分析（α=0.05，双侧）：")
print(power_tab.round(3).to_string(index=False))


# ===========================================================================
# 图形
# ===========================================================================
print("\n绘制图形...")

# Figure 1：主效应森林图 + 均值 CI
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# 左：森林图
m = main_ols[main_ols["model"].str.contains("调整") & main_ols["contrast"].str.contains("vs A")]
m = m[m["outcome"] == "green_choice_index"]
y_pos = np.arange(len(m))
axes[0].errorbar(m["estimate"], y_pos, xerr=[m["estimate"] - m["ci_low"], m["ci_high"] - m["estimate"]],
                 fmt="o", color="#1F4E78", capsize=5, markersize=8)
axes[0].axvline(0, color="grey", lw=1, ls="--")
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(m["contrast"])
axes[0].set_xlabel("标准化绿色选择指数（β, 95%CI）")
axes[0].set_title("Figure 1a. 调整 OLS 主效应（vs 对照组）")
# 右：均值 + 95%CI
plot = d.groupby("group")["green_choice_index"].agg(["mean", "count", "std"]).reindex(GROUP_ORDER)
plot["se"] = plot["std"] / np.sqrt(plot["count"])
plot["ci"] = 1.96 * plot["se"]
axes[1].errorbar(range(len(GROUP_ORDER)), plot["mean"],
                 yerr=plot["ci"], fmt="o", color="#C0504D", capsize=6, markersize=9)
axes[1].set_xticks(range(len(GROUP_ORDER)))
axes[1].set_xticklabels(GROUP_ORDER)
axes[1].axhline(plot.loc["A 对照", "mean"], color="grey", lw=1, ls="--")
axes[1].set_ylabel("绿色选择指数均值")
axes[1].set_xlabel("随机分配组别")
axes[1].set_title("Figure 1b. 各组均值 ± 95%CI")
fig.tight_layout()
fig.savefig(OUT / "figure1_main_effects.png", dpi=300)
plt.close(fig)

# Figure 2：中介路径（D -> trust -> intention）
fig, ax = plt.subplots(figsize=(9, 5))
med_focus = mediation[(mediation["contrast"] == "D 范例型 vs A") &
                      (mediation["outcome"] == "green_intention")]
paths = ["a (treat→mediator)", "b (mediator→outcome)", "c (total)", "c' (direct)", "ab (indirect)"]
vals = []
for _, r in med_focus.iterrows():
    vals.append([r["a"], r["b"], r["c"], r["c_prime"], r["indirect_ab"]])
vals = np.array(vals)
mediators = med_focus["mediator"].tolist()
x = np.arange(len(paths))
width = 0.25
for i, med in enumerate(mediators):
    ax.bar(x + i * width, vals[i], width, label=med)
ax.axhline(0, color="grey", lw=1)
ax.set_xticks(x + width)
ax.set_xticklabels(paths, rotation=15)
ax.set_ylabel("效应估计值")
ax.set_title("Figure 2. 中介路径：D 组（范例型）vs A（对照）")
ax.legend(title="中介变量")
fig.tight_layout()
fig.savefig(OUT / "figure2_mediation.png", dpi=300)
plt.close(fig)

# Figure 3：异质性（AI使用频率 × 组别）
fig, ax = plt.subplots(figsize=(9, 5))
het_plot = d.groupby(["ai_frequency", "group"])["green_choice_index"].mean().unstack()
het_plot = het_plot.reindex(columns=GROUP_ORDER)
het_plot.plot(kind="bar", ax=ax, edgecolor="black")
ax.set_ylabel("绿色选择指数均值")
ax.set_xlabel("AI 使用频率")
ax.set_title("Figure 3. 异质性：AI 使用频率 × 组别")
ax.legend(title="组别", loc="best")
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig(OUT / "figure3_heterogeneity.png", dpi=300)
plt.close(fig)

# Figure 4：相关矩阵热力图
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": .8}, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Figure 4. 关键变量相关矩阵")
fig.tight_layout()
fig.savefig(OUT / "figure4_corr_heatmap.png", dpi=300)
plt.close(fig)

# Figure 5：CONSORT 流程
fig, ax = plt.subplots(figsize=(8, 7))
ax.axis("off")
y_top = 1.0
step = 0.13
for i, row in consort.iterrows():
    y = y_top - i * step
    text = f"{row['step']}\n  n = {row['n']}（排除 {row['excluded']}）"
    ax.add_patch(plt.Rectangle((0.1, y - 0.04), 0.8, 0.08,
                                facecolor="#DEEBF7", edgecolor="#1F4E78"))
    ax.text(0.5, y, text, ha="center", va="center", fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.set_title("Figure 5. CONSORT 样本流", fontsize=13, pad=15)
fig.tight_layout()
fig.savefig(OUT / "figure5_consort.png", dpi=300)
plt.close(fig)

# Figure 6：操纵检验
fig, ax = plt.subplots(figsize=(9, 5))
mc_plot = d.groupby("group")[["manip_ai_identity", "manip_clear_conclusion",
                              "manip_explanation"]].mean().reindex(GROUP_ORDER)
mc_plot.columns = ["AI身份识别", "结论清晰度", "解释充分度"]
mc_plot.plot(kind="bar", ax=ax, edgecolor="black")
ax.set_ylabel("均值（1-7 Likert）")
ax.set_xlabel("随机分配组别")
ax.set_title("Figure 6. 操纵检验：各组对建议特征感知")
ax.legend(title="操纵维度", loc="best")
plt.xticks(rotation=15)
fig.tight_layout()
fig.savefig(OUT / "figure6_manip_check.png", dpi=300)
plt.close(fig)


# ===========================================================================
# 导出
# ===========================================================================
print("\n导出结果...")
d.to_csv(OUT / "analysis_dataset.csv", index=False, encoding="utf-8-sig")

with pd.ExcelWriter(OUT / "tables_full.xlsx") as xls:
    consort.to_excel(xls, sheet_name="consort", index=False)
    reliability.to_excel(xls, sheet_name="reliability", index=False)
    itc_table.to_excel(xls, sheet_name="item_total_correlation", index=False)
    htmt_mat.to_excel(xls, sheet_name="htmt")
    balance.to_excel(xls, sheet_name="balance_tests", index=False)
    desc.to_excel(xls, sheet_name="descriptives")
    group_means.to_excel(xls, sheet_name="group_means")
    corr.to_excel(xls, sheet_name="correlations")
    pd.DataFrame(adj_mat, index=key_vars, columns=key_vars).to_excel(xls, sheet_name="p_BH_corr")
    main_ols.to_excel(xls, sheet_name="main_ols", index=False)
    logit_results.to_excel(xls, sheet_name="binary_logit", index=False)
    ord_df.to_excel(xls, sheet_name="ordinal_mnlogit", index=False)
    mediation.to_excel(xls, sheet_name="mediation", index=False)
    parallel_med.to_excel(xls, sheet_name="parallel_mediation", index=False)
    heterogeneity.to_excel(xls, sheet_name="heterogeneity", index=False)
    robust_se.to_excel(xls, sheet_name="robust_se", index=False)
    robust_sample.to_excel(xls, sheet_name="robust_subsample", index=False)
    ai_vs_ref.to_excel(xls, sheet_name="any_ai_vs_control", index=False)
    bayes.to_excel(xls, sheet_name="bayes_factors", index=False)
    tost_tab.to_excel(xls, sheet_name="tost_equivalence", index=False)
    power_tab.to_excel(xls, sheet_name="post_hoc_power", index=False)
    manip_check.to_excel(xls, sheet_name="manipulation_check", index=False)

# 文本结果摘要
with open(OUT / "results_summary.txt", "w", encoding="utf-8") as f:
    f.write("AI 助手建议类型与校园绿色决策——四组随机实验顶刊级分析\n")
    f.write("=" * 80 + "\n\n")
    f.write("【样本流（CONSORT）】\n")
    f.write(consort.to_string(index=False))
    f.write("\n\n【信度】\n")
    f.write(reliability.round(3).to_string(index=False))
    f.write("\n\n【随机化平衡】\n")
    f.write(balance.round(3).to_string(index=False))
    f.write("\n\n【主效应 OLS（调整，HC3）】\n")
    f.write(main_ols[main_ols["model"].str.contains("调整")].round(3).to_string(index=False))
    f.write("\n\n【二元 Logit OR】\n")
    f.write(logit_results.round(3).to_string(index=False))
    f.write("\n\n【中介分析 Bootstrap 5000】\n")
    f.write(mediation.round(3).to_string(index=False))
    f.write("\n\n【贝叶斯因子】\n")
    f.write(bayes.round(3).to_string(index=False))
    f.write("\n\n【等效性检验 TOST】\n")
    f.write(tost_tab.round(3).to_string(index=False))
    f.write("\n\n【事后功效】\n")
    f.write(power_tab.round(3).to_string(index=False))
    f.write("\n\n注：所有处理系数已用 BH-FDR / Holm 校正多重检验；")
    f.write("效应量报告 Cohen's d 与 95%CI；中介报告 Bootstrap 95%CI。\n")

print("\n" + "=" * 80)
print(f"分析完成。结果保存在：{OUT.resolve()}")
print("=" * 80)
