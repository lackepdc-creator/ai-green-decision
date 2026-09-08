"""AI助手建议类型与校园绿色决策：四组随机实验的可复现分析。

输入：outputs/randomized_assignment_recode/AI助手绿色决策_随机实验分析编码版.xlsx
输出：analysis_results/ 下的样本、量表、主效应、稳健性、中介与图形结果。

安装依赖：py -m pip install pandas numpy scipy statsmodels openpyxl matplotlib
运行：py randomized_experiment_analysis.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

INPUT = Path("outputs/randomized_assignment_recode/AI助手绿色决策_随机实验分析编码版.xlsx")
OUT = Path("analysis_results")
OUT.mkdir(exist_ok=True)
GROUPS = [0, 1, 2, 3]
GROUP_LABELS = {0: "A 对照", 1: "B 结果型", 2: "C 过程型", 3: "D 范例型"}

def alpha(x):
    """Cronbach's alpha。"""
    x = x.dropna()
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1))

def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

def model_table(model, model_name):
    ci = model.conf_int()
    return pd.DataFrame({
        "model": model_name, "term": model.params.index, "estimate": model.params.values,
        "std_error": model.bse.values, "ci_low": ci.iloc[:, 0].values,
        "ci_high": ci.iloc[:, 1].values, "p_value": model.pvalues.values,
        "significance": [stars(p) for p in model.pvalues.values], "n": int(model.nobs),
    })

def bootstrap_indirect(data, treatment, mediator, outcome, controls, reps=5000, seed=20260902):
    """二元处理（C组 vs A组）的非参数 Bootstrap 中介效应。"""
    rng = np.random.default_rng(seed)
    d = data.loc[data.group_code.isin([0, treatment])].copy()
    d["treat"] = (d.group_code == treatment).astype(int)
    formula_m = f"{mediator} ~ treat + {controls}"
    formula_y = f"{outcome} ~ treat + {mediator} + {controls}"
    draws = []
    for _ in range(reps):
        b = d.iloc[rng.integers(0, len(d), len(d))]
        try:
            a = smf.ols(formula_m, b).fit().params["treat"]
            b_path = smf.ols(formula_y, b).fit().params[mediator]
            draws.append(a * b_path)
        except Exception:
            pass
    point = smf.ols(formula_m, d).fit().params["treat"] * smf.ols(formula_y, d).fit().params[mediator]
    return {"contrast": f"{GROUP_LABELS[treatment]} vs A 对照", "mediator": mediator,
            "outcome": outcome, "indirect_effect": point, "ci_low": np.quantile(draws, .025),
            "ci_high": np.quantile(draws, .975), "bootstrap_reps": len(draws)}

# 1. 读取已编码数据并执行事先定义的质量规则
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    d = pd.read_excel(INPUT, sheet_name="分析数据")
d = d.loc[(d["attention_check"] == "比较同意") & d["answer_id"].notna()].copy()
d = d.drop_duplicates("answer_id")
d["group_code"] = pd.to_numeric(d["group_code"], errors="coerce")

# 2. 变量构建
numeric_cols = [c for c in d.columns if c.startswith(("green_baseline", "trust_", "load_", "feasibility_", "intention_", "manip_"))]
for c in numeric_cols:
    if c != "manip_recall":
        d[c] = pd.to_numeric(d[c], errors="coerce")
d["green_baseline"] = d[["green_baseline_1", "green_baseline_2"]].mean(axis=1)
d["trust"] = d[["trust_1", "trust_2", "trust_3"]].mean(axis=1)
d["cognitive_load"] = d[["load_1", "load_2", "load_3"]].mean(axis=1)
d["feasibility"] = d[["feasibility_1", "feasibility_2"]].mean(axis=1)
d["green_intention"] = d[["intention_1", "intention_2", "intention_3"]].mean(axis=1)

d["green_canteen"] = d["canteen_choice"].map({
    "选常规份套餐，并领取一次性餐具": 0,
    "选常规份套餐，不领取一次性餐具": 1,
    "选小份菜／按需取餐，并领取一次性餐具": 1,
    "选小份菜／按需取餐，不领取一次性餐具": 2,
})
d["green_recycling"] = (d["recycling_choice"] == "带去可回收物投放点").astype(int)
d["green_study"] = (d["study_space_choice"] == "A. 已开启设备的开放自习区").astype(int)
for c in ["green_canteen", "green_recycling", "green_study"]:
    d[f"z_{c}"] = (d[c] - d[c].mean()) / d[c].std(ddof=0)
d["green_choice_index"] = d[["z_green_canteen", "z_green_recycling", "z_green_study"]].mean(axis=1)
d["any_ai"] = (d.group_code != 0).astype(int)

# 3. 样本流、信度、随机化平衡与描述性统计
flow = pd.DataFrame({"stage": ["原始编码样本", "注意力题合格且去重后样本"], "n": [len(pd.read_excel(INPUT, sheet_name="分析数据")), len(d)]})
reliability = pd.DataFrame({"construct": ["环保基线", "AI信任", "认知负荷", "可行性感知", "绿色行为意愿"],
                            "items": [2, 3, 3, 2, 3],
                            "cronbach_alpha": [alpha(d[["green_baseline_1", "green_baseline_2"]]), alpha(d[["trust_1", "trust_2", "trust_3"]]), alpha(d[["load_1", "load_2", "load_3"]]), alpha(d[["feasibility_1", "feasibility_2"]]), alpha(d[["intention_1", "intention_2", "intention_3"]])]})
group_counts = d.groupby("group_code").size().rename("n").reset_index()
group_counts["group"] = group_counts.group_code.map(GROUP_LABELS)

balance_rows = []
for x in ["green_baseline", "duration_s"]:
    arrays = [d.loc[d.group_code == g, x].dropna() for g in GROUPS]
    f, p = stats.f_oneway(*arrays)
    balance_rows.append({"variable": x, "test": "ANOVA", "statistic": f, "p_value": p})
for x in ["gender", "grade", "major", "ai_frequency", "dining_frequency"]:
    tab = pd.crosstab(d[x], d.group_code)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    balance_rows.append({"variable": x, "test": "chi-square", "statistic": chi2, "p_value": p})
balance = pd.DataFrame(balance_rows)

outcomes = ["green_choice_index", "green_canteen", "green_recycling", "green_study", "green_intention", "trust", "cognitive_load", "feasibility"]
means = d.groupby("group_code")[outcomes].agg(["mean", "std", "count"]).round(3)
means.columns = ["_".join(c) for c in means.columns]
means = means.reset_index()
means["group"] = means.group_code.map(GROUP_LABELS)

# 4. 顶刊常见的主规格：ITT、处理组虚拟变量、HC3稳健标准误；协变量仅提高精度。
controls = "green_baseline + C(gender) + C(grade) + C(ai_frequency) + C(dining_frequency)"
main_models = []
for y in ["green_choice_index", "green_intention", "trust", "cognitive_load", "feasibility"]:
    m1 = smf.ols(f"{y} ~ C(group_code, Treatment(reference=0))", d).fit(cov_type="HC3")
    m2 = smf.ols(f"{y} ~ C(group_code, Treatment(reference=0)) + {controls}", d).fit(cov_type="HC3")
    main_models += [model_table(m1, f"{y}: unadjusted OLS"), model_table(m2, f"{y}: adjusted OLS")]
main_results = pd.concat(main_models, ignore_index=True)

# 二元选择使用 Logit；D1 是有序选择，保留 OLS 得分作为简洁的主规格并用有序模型稳健性检查。
logit_results = []
for y in ["green_recycling", "green_study"]:
    m = smf.logit(f"{y} ~ C(group_code, Treatment(reference=0)) + {controls}", d).fit(disp=False)
    tab = model_table(m, f"{y}: adjusted Logit")
    tab["odds_ratio"] = np.exp(tab["estimate"])
    logit_results.append(tab)
logit_results = pd.concat(logit_results, ignore_index=True)

# 5. 预先限定的机制：过程型与范例型相对于对照组，经由信任/负荷影响绿色意愿。
mediation = pd.DataFrame([
    bootstrap_indirect(d, treatment, mediator, "green_intention", controls)
    for treatment in [2, 3] for mediator in ["trust", "cognitive_load"]
])
mediation["ci_excludes_zero"] = (mediation.ci_low > 0) | (mediation.ci_high < 0)

# 6. 预先限定异质性：AI使用频率与环保基线。交互项结果必须标为探索性。
heterogeneity = []
for moderator in ["green_baseline", "ai_frequency"]:
    model = smf.ols(f"green_choice_index ~ C(group_code, Treatment(reference=0))*{moderator} + {controls.replace(moderator + ' + ', '')}", d).fit(cov_type="HC3")
    heterogeneity.append(model_table(model, f"exploratory moderation: {moderator}"))
heterogeneity = pd.concat(heterogeneity, ignore_index=True)

# 7. 图形：主结果均值和95%置信区间
plot = d.groupby("group_code")["green_choice_index"].agg(["mean", "count", "std"]).reindex(GROUPS)
plot["se"] = plot["std"] / np.sqrt(plot["count"])
plot["ci"] = 1.96 * plot["se"]
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.errorbar([GROUP_LABELS[g] for g in GROUPS], plot["mean"], yerr=plot["ci"], fmt="o", color="#1F4E78", capsize=4)
ax.axhline(0, color="#808080", lw=.8)
ax.set_ylabel("标准化绿色选择指数")
ax.set_xlabel("随机分配组别")
ax.set_title("Figure 1. Green choice index by assigned condition")
fig.tight_layout(); fig.savefig(OUT / "figure1_green_choice_index.png", dpi=300); plt.close(fig)

# 8. 导出。论文结果应报告估计值、95%CI、效应量和多重检验校正；不要只报告p值。
d.to_csv(OUT / "analysis_dataset.csv", index=False, encoding="utf-8-sig")
with pd.ExcelWriter(OUT / "tables_for_paper.xlsx") as xls:
    flow.to_excel(xls, "sample_flow", index=False); group_counts.to_excel(xls, "group_counts", index=False)
    reliability.to_excel(xls, "reliability", index=False); balance.to_excel(xls, "balance_tests", index=False)
    means.to_excel(xls, "outcome_means", index=False); main_results.to_excel(xls, "main_ols", index=False)
    logit_results.to_excel(xls, "binary_logit", index=False); mediation.to_excel(xls, "mediation", index=False)
    heterogeneity.to_excel(xls, "heterogeneity", index=False)
print("完成：", OUT.resolve())
