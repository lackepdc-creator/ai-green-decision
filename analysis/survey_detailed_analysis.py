"""问卷数据详细分析：信度、效度、相关、操纵检验、回归、中介、调节、贝叶斯。
输出到 ./survey_analysis/
"""
from __future__ import annotations
import warnings
warnings.simplefilter("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(r'c:\Users\ASUS\Desktop\成功计划')
SRC = ROOT / 'top_journal_results'
OUT = ROOT / 'survey_analysis'
OUT.mkdir(exist_ok=True)

SEED = 20260902
BOOT = 5000
rng = np.random.default_rng(SEED)

d = pd.read_csv(SRC / 'analysis_dataset.csv', encoding='utf-8-sig')

GROUPS = ['A 对照', 'B 结果型', 'C 过程型', 'D 范例型']
G = {g: d[d['group'] == g] for g in GROUPS}

results = {}  # 收集所有表

def stars(p):
    return '***' if p < .01 else '**' if p < .05 else '*' if p < .10 else ''

# ===========================================================================
# 1. 样本人口统计
# ===========================================================================
print('[1] 人口统计...')
demo_rows = []
for var, label in [('gender','性别'), ('grade','年级'), ('major','专业'),
                   ('ai_frequency','AI使用频率'), ('dining_frequency','食堂就餐频率'),
                   ('scenario_fit','场景契合度')]:
    vc = d[var].value_counts()
    for level, n in vc.items():
        demo_rows.append({'变量': label, '类别': str(level), 'n': n,
                          '%': f'{n/len(d)*100:.1f}'})
demo = pd.DataFrame(demo_rows)
results['demographics'] = demo

# 各组样本量
gn = d.groupby('group').size().reindex(GROUPS)
results['group_n'] = pd.DataFrame({'组别': GROUPS, 'n': gn.values,
                                   '%': (gn.values/len(d)*100).round(1)})

# ===========================================================================
# 2. 信度分析
# ===========================================================================
print('[2] 信度分析...')
SCALES = {
    '环保基线': ['green_baseline_1','green_baseline_2'],
    'AI信任': ['trust_1','trust_2','trust_3'],
    '认知负荷': ['load_1','load_2','load_3'],
    '可行性感知': ['feasibility_1','feasibility_2'],
    '绿色行为意愿': ['intention_1','intention_2','intention_3'],
}
# 认知负荷反向题（load 越高=负荷越大，在信任/意愿语境下反转使"低负荷"为高）
# 按 top_journal_analysis.py 的口径：load 直接用原始值做量表均值，信度按同向计算

def cronbach_alpha(x):
    x = np.asarray(x, float)
    k = x.shape[1]
    if k < 2: return np.nan
    item_var = x.var(ddof=1, axis=0).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    return k/(k-1)*(1 - item_var/total_var)

def alpha_boot(items, reps=BOOT):
    x = items.dropna().to_numpy()
    n = len(x)
    a = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        v = cronbach_alpha(x[idx])
        if not np.isnan(v): a.append(v)
    return np.quantile(a, .025), np.quantile(a, .975)

rel_rows, itc_rows = [], []
for name, cols in SCALES.items():
    items = d[cols].dropna()
    alpha = cronbach_alpha(items)
    lo, hi = alpha_boot(items)
    rel_rows.append({'量表': name, '题数': len(cols), 'n': len(items),
                     'Cronbach α': f'{alpha:.3f}',
                     '95% CI': f'[{lo:.3f}, {hi:.3f}]',
                     '判定': '良好' if alpha>=0.7 else ('可接受' if alpha>=0.6 else '偏低')})
    # 项总计相关 + 删除后 alpha
    for c in cols:
        rest = items.drop(columns=[c]).sum(axis=1)
        r_it = items[c].corr(rest)
        a_del = cronbach_alpha(items.drop(columns=[c])) if len(cols)>2 else np.nan
        itc_rows.append({'量表': name, '题项': c,
                         '题均值得': f'{items[c].mean():.2f}',
                         '题标准差': f'{items[c].std():.2f}',
                         '项总计相关r': f'{r_it:.3f}',
                         '删除后α': f'{a_del:.3f}' if not np.isnan(a_del) else '—'})
results['reliability'] = pd.DataFrame(rel_rows)
results['item_total'] = pd.DataFrame(itc_rows)

# ===========================================================================
# 3. 效度分析
# ===========================================================================
print('[3] 效度分析（EFA/KMO/Bartlett/AVE/CR/HTMT）...')

# --- 手动实现 KMO / Bartlett / EFA（避免 factor_analyzer 与新版 sklearn 不兼容） ---
def kmo_test(X):
    """KMO 取样适当性度量。"""
    corr = np.corrcoef(X, rowvar=False)
    n_var = corr.shape[0]
    # 偏相关 = -inv(corr) 归一化
    inv = np.linalg.pinv(corr)
    d = np.sqrt(np.diag(inv))
    partial = -inv / np.outer(d, d)
    np.fill_diagonal(partial, 0)
    r2 = corr**2; p2 = partial**2
    np.fill_diagonal(r2, 0)
    kmo_total = r2.sum() / (r2.sum() + p2.sum())
    # 各变量 KMO（MSA）
    kmo_per = []
    for i in range(n_var):
        r_sum = r2[i].sum() - r2[i,i]
        p_sum = p2[i].sum()
        kmo_per.append(r_sum/(r_sum+p_sum))
    return kmo_total, np.array(kmo_per)

def bartlett_test(X):
    """Bartlett 球形检验。"""
    n, p = X.shape
    corr = np.corrcoef(X, rowvar=False)
    det = np.linalg.det(corr)
    det = max(det, 1e-12)
    chi2 = -(n - 1 - (2*p+5)/6) * np.log(det)
    dof = p*(p-1)/2
    pval = stats.chi2.sf(chi2, dof)
    return chi2, dof, pval

def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    """Kaiser varimax 旋转。"""
    p, k = Phi.shape
    R = np.eye(k)
    d_old = 0
    for _ in range(q):
        Lam = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lam**3 - (gamma/p) * Lam @ np.diag(np.diag(Lam.T@Lam))))
        R = u @ vt
        d_new = s.sum()
        if d_old != 0 and d_new/d_old < 1 + tol:
            break
        d_old = d_new
    return Phi @ R

# 构念测量题（潜变量指标）：5个量表的全部题项
latent_items = (SCALES['环保基线'] + SCALES['AI信任'] + SCALES['认知负荷']
                + SCALES['可行性感知'] + SCALES['绿色行为意愿'])
X = d[latent_items].dropna().to_numpy(float)

# KMO & Bartlett
kmo_model, kmo_per_var = kmo_test(X)
chi2, dof_bart, p_bart = bartlett_test(X)

# EFA：主成分法 + varimax 旋转，提取5因子
corr_mat = np.corrcoef(X, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(corr_mat)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
n_factors = 5
loadings_unrot = eigvecs[:, :n_factors] * np.sqrt(np.maximum(eigvals[:n_factors], 0))
loadings_rot = varimax(loadings_unrot)
loadings = pd.DataFrame(loadings_rot, index=latent_items,
                        columns=[f'因子{i+1}' for i in range(n_factors)])
# 方差解释（旋转后）
var_exp = (loadings_rot**2).sum(axis=0) / X.shape[1]
cum = np.cumsum(var_exp)
efa_var = pd.DataFrame({'因子': [f'因子{i+1}' for i in range(n_factors)],
                        '特征值': np.round(eigvals[:n_factors],3),
                        '旋转后方差解释%': np.round(var_exp*100,2),
                        '累计%': np.round(cum*100,2)})
# Harman 单因子：未旋转第一主成分方差
harman_var = eigvals[0] / len(latent_items) * 100

# AVE & CR（用各构念题项在其理论因子上的载荷；先把每个题项归到理论构念）
construct_items = {
    '环保基线': SCALES['环保基线'], 'AI信任': SCALES['AI信任'],
    '认知负荷': SCALES['认知负荷'], '可行性': SCALES['可行性感知'],
    '绿色意愿': SCALES['绿色行为意愿']}
# 用主成分载荷（每个构念单因子 PCA）来算 AVE/CR，更稳健
ave_cr_rows = []
construct_scores = {}
for cname, cols in construct_items.items():
    xi = d[cols].dropna()
    # 标准化载荷：第一主成分
    z = (xi - xi.mean())/xi.std()
    u, s, vt = np.linalg.svd(z.cov(), hermitian=True)
    lam = vt[0]  # 第一主成分载荷（协方差阵特征向量）
    lam_std = lam / np.abs(lam).max()  # 缩放
    # 用相关阵的 PCA 载荷
    corr = xi.corr().to_numpy()
    vals, vecs = np.linalg.eigh(corr)
    l = vecs[:, -1]
    load = l * np.sqrt(vals[-1])
    load = np.abs(load)
    ave = np.mean(load**2)
    cr = (load.sum()**2) / (load.sum()**2 + np.sum(1-load**2))
    construct_scores[cname] = xi.mean(axis=1)
    ave_cr_rows.append({'构念': cname, '题数': len(cols),
                        '标准化载荷范围': f'{load.min():.3f}–{load.max():.3f}',
                        'AVE': f'{ave:.3f}',
                        'CR': f'{cr:.3f}',
                        '聚合效度': '达标' if ave>=0.5 and cr>=0.7 else ('CR达标AVE偏低' if cr>=0.7 else '未达标')})
results['ave_cr'] = pd.DataFrame(ave_cr_rows)

# 构念相关
con_df = pd.DataFrame(construct_scores)
con_corr = con_df.corr()
results['construct_corr'] = con_corr.round(3)

# HTMT（用题项相关：异构念题项间均值相关 / 同构念题项内均值相关）
htmt = pd.DataFrame(index=construct_items.keys(), columns=construct_items.keys(), dtype=float)
for a in construct_items:
    for b in construct_items:
        if a == b:
            htmt.loc[a,b] = np.nan; continue
        ca, cb = construct_items[a], construct_items[b]
        # 跨构念相关均值
        cross = []
        for ia in ca:
            for ib in cb:
                cross.append(abs(d[ia].corr(d[ib])))
        # 同构念相关均值
        def mono(cols):
            rr = []
            for i in range(len(cols)):
                for j in range(i+1,len(cols)):
                    rr.append(abs(d[cols[i]].corr(d[cols[j]])))
            return np.mean(rr) if rr else np.nan
        ma, mb = mono(ca), mono(cb)
        htmt.loc[a,b] = np.mean(cross)/np.sqrt(ma*mb)
results['htmt'] = htmt.astype(float).round(3)

# Harman 单因子检验（harman_var 已在上面按未旋转第一主成分计算）

results['validity_summary'] = pd.DataFrame([
    {'指标': 'KMO（取样适当性）', '数值': f'{kmo_model:.3f}', '判定': '良好(>0.8)' if kmo_model>0.8 else ('可接受(>0.6)' if kmo_model>0.6 else '差')},
    {'指标': 'Bartlett球形 χ²(df)', '数值': f'{chi2:.1f}({int(dof_bart)})', '判定': 'p<.001' if p_bart<.001 else f'p={p_bart:.3f}'},
    {'指标': 'Harman单因子方差解释%', '数值': f'{harman_var:.2f}%', '判定': '<40%无严重共同方法偏差' if harman_var<40 else '存在共同方法偏差风险'},
    {'指标': 'EFA五因子累计方差%', '数值': f'{cum[-1]*100:.2f}%', '判定': '>60%良好' if cum[-1]>0.6 else '一般'},
])

# 题项在5因子上的载荷（标记最高载荷）
load_out = loadings.round(3)
load_out['归属因子'] = loadings.abs().idxmax(axis=1)
results['efa_loadings'] = load_out.reset_index().rename(columns={'index':'题项'})
results['efa_variance'] = efa_var

# ===========================================================================
# 4. 描述统计（量表/行为）
# ===========================================================================
print('[4] 描述统计...')
vars_desc = {
    '环保基线': d['green_baseline'], 'AI信任': d['trust'],
    '认知负荷': d['cognitive_load'], '可行性感知': d['feasibility'],
    '社会规范': d['social_norm'], '绿色行为意愿': d['green_intention'],
    '绿色选择指数': d['green_choice_index'], '食堂选择(0-2)': d['green_canteen'],
    '回收选择(0/1)': d['green_recycling'], '自习选择(0/1)': d['green_study'],
}
desc_rows = []
for name, v in vars_desc.items():
    desc_rows.append({'变量': name, 'n': v.notna().sum(),
                      '均值': f'{v.mean():.3f}', '标准差': f'{v.std():.3f}',
                      '最小值': f'{v.min():.2f}', '最大值': f'{v.max():.2f}',
                      '偏度': f'{v.skew():.3f}', '峰度': f'{v.kurtosis():.3f}'})
results['descriptives'] = pd.DataFrame(desc_rows)

# 各组量表均值
grp_mean = d.groupby('group')[['green_baseline','trust','cognitive_load','feasibility',
                              'green_intention','green_choice_index']].mean().reindex(GROUPS)
grp_mean.index.name = '组别'
results['group_means_scale'] = grp_mean.round(3).reset_index()

# ===========================================================================
# 5. 操纵检验（ANOVA + 两两 Hedges g）
# ===========================================================================
print('[5] 操纵检验...')
manip_vars = {'AI身份识别':'manip_ai_identity','结论清晰度':'manip_clear_conclusion','解释充分度':'manip_explanation'}
manip_anova_rows, manip_pair_rows = [], []
for label, col in manip_vars.items():
    groups_vals = [G[g][col].values for g in GROUPS]
    f, p = stats.f_oneway(*groups_vals)
    manip_anova_rows.append({'操纵题项': label, 'F': f'{f:.3f}', 'p': f'{p:.3f}{stars(p)}'})
    # 两两 hedges g
    for i in range(4):
        for j in range(i+1,4):
            a, b = groups_vals[i], groups_vals[j]
            na, nb = len(a), len(b)
            sp = np.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2))
            g_h = (a.mean()-b.mean())/sp * (1 - 3/(4*(na+nb)-9))
            # Welch t
            t, pt = stats.ttest_ind(a, b, equal_var=False)
            manip_pair_rows.append({'操纵题项': label, '对比': f'{GROUPS[i]} vs {GROUPS[j]}',
                                    'Hedges g': f'{g_h:.3f}', 'p': f'{pt:.3f}{stars(pt)}'})
results['manip_anova'] = pd.DataFrame(manip_anova_rows)
results['manip_pairwise'] = pd.DataFrame(manip_pair_rows)

# ===========================================================================
# 6. 相关分析
# ===========================================================================
print('[6] 相关分析...')
corr_vars = {'环保基线':'green_baseline','AI信任':'trust','认知负荷':'cognitive_load',
             '可行性':'feasibility','社会规范':'social_norm','绿色意愿':'green_intention',
             '食堂':'green_canteen','回收':'green_recycling','自习':'green_study',
             '绿色指数':'green_choice_index'}
cm = d[list(corr_vars.values())].corr()
cm.index = list(corr_vars.keys()); cm.columns = list(corr_vars.keys())
# p 值
pm = cm.copy().astype(object)
for i,a in enumerate(corr_vars.values()):
    for j,b in enumerate(corr_vars.values()):
        if i==j: pm.iloc[i,j]='—'
        elif i<j:
            r,p = stats.pearsonr(d[a],d[b])
            pm.iloc[i,j]=f'{r:.3f}{stars(p)}'
        else:
            pm.iloc[i,j]=''
results['correlation'] = pm.reset_index().rename(columns={'index':'变量'})

# ===========================================================================
# 7. 回归分析
# ===========================================================================
print('[7] 回归分析...')
# 7a. 主效应 OLS（HC3），因变量=green_choice_index
reg_rows = []
d['treatB'] = (d['group_code']==1).astype(int)
d['treatC'] = (d['group_code']==2).astype(int)
d['treatD'] = (d['group_code']==3).astype(int)
covariates = ['green_baseline','C(gender)','C(grade)','C(major)','C(ai_frequency)','C(dining_frequency)']

def ols_fit(formula, data, label, model_type):
    m = smf.ols(formula, data=data).fit(cov_type='HC3')
    return m

# 未调整
m0 = smf.ols('green_choice_index ~ treatB + treatC + treatD', data=d).fit(cov_type='HC3')
# 调整
f_adj = 'green_choice_index ~ treatB + treatC + treatD + ' + ' + '.join(covariates)
m1 = smf.ols(f_adj, data=d).fit(cov_type='HC3')

for tname, tlab in [('treatB','B 结果型 vs A'),('treatC','C 过程型 vs A'),('treatD','D 范例型 vs A')]:
    for mdl, tag in [(m0,'未调整'),(m1,'调整后')]:
        b = mdl.params[tname]; se = mdl.bse[tname]; p = mdl.pvalues[tname]
        ci = mdl.conf_int().loc[tname]
        # cohen d (偏)
        d_eff = b/np.sqrt(mdl.mse_resid) if hasattr(mdl,'mse_resid') else np.nan
        reg_rows.append({'模型': tag, '对比': tlab, 'β': f'{b:.3f}', 'SE': f'{se:.3f}',
                         '95%CI': f'[{ci[0]:.3f}, {ci[1]:.3f}]', 't': f'{mdl.tvalues[tname]:.3f}',
                         'p': f'{p:.3f}{stars(p)}', 'd': f'{d_eff:.2f}'})
    reg_rows.append({'模型':'','对比':'','β':'','SE':'','95%CI':'','t':'','p':'','d':''})
# 模型统计
for mdl, tag in [(m0,'未调整'),(m1,'调整后')]:
    reg_rows.append({'模型': tag, '对比': f'R²={mdl.rsquared:.3f}, 调整R²={mdl.rsquared_adj:.3f}, F={mdl.fvalue:.2f}, n={int(mdl.nobs)}',
                     'β':'','SE':'','95%CI':'','t':'','p':'','d':''})
results['ols_main'] = pd.DataFrame(reg_rows)

# 7b. 意愿回归（分层）
print('  分层回归（意愿）...')
hier_rows = []
h1 = smf.ols('green_intention ~ C(gender)+C(grade)', data=d).fit(cov_type='HC3')
h2 = smf.ols('green_intention ~ C(gender)+C(grade)+green_baseline+trust+cognitive_load+feasibility+social_norm', data=d).fit(cov_type='HC3')
h3 = smf.ols('green_intention ~ C(gender)+C(grade)+green_baseline+trust+cognitive_load+feasibility+social_norm+treatB+treatC+treatD', data=d).fit(cov_type='HC3')
for mdl, tag in [(h1,'M1 人口'),(h2,'M2 +心理变量'),(h3,'M3 +处理')]:
    hier_rows.append({'模型': tag, 'R²': f'{mdl.rsquared:.3f}', '调整R²': f'{mdl.rsquared_adj:.3f}',
                      'F': f'{mdl.fvalue:.2f}', 'p': f'{mdl.f_pvalue:.3f}{stars(mdl.f_pvalue)}'})
# ΔR²
hier_rows.append({'模型':'ΔR²(M2-M1)', 'R²': f'{h2.rsquared-h1.rsquared:.3f}', '调整R²':'', 'F':'', 'p':''})
hier_rows.append({'模型':'ΔR²(M3-M2)', 'R²': f'{h3.rsquared-h2.rsquared:.3f}', '调整R²':'', 'F':'', 'p':''})
results['hierarchical'] = pd.DataFrame(hier_rows)

# M3 系数
m3_rows = []
m1_params = ['green_baseline','trust','cognitive_load','feasibility','social_norm','treatB','treatC','treatD']
for v in m1_params:
    if v in h3.params.index:
        b=h3.params[v]; se=h3.bse[v]; p=h3.pvalues[v]; ci=h3.conf_int().loc[v]
        m3_rows.append({'预测变量': v, 'β': f'{b:.3f}', 'SE': f'{se:.3f}',
                        '95%CI': f'[{ci[0]:.3f},{ci[1]:.3f}]', 't': f'{h3.tvalues[v]:.3f}',
                        'p': f'{p:.3f}{stars(p)}'})
results['hier_coef'] = pd.DataFrame(m3_rows)

# 7c. Logit 二元结果 —— 直接采用原始分析结果（tables_full.xlsx），确保与论文一致
print('  Logit（读取原始分析结果）...')
binlog = pd.read_excel(SRC / 'tables_full.xlsx', sheet_name='binary_logit')
logit_rows = []
for _, r_ in binlog.iterrows():
    olab = '回收' if r_['outcome'] == 'green_recycling' else '自习'
    glab = r_['contrast'].replace(' vs A', ' vs A')
    logit_rows.append({'因变量': olab, '对比': glab,
                       'OR': f"{r_['OR']:.2f}",
                       '95%CI': f"[{r_['ci_low_OR']:.2f},{r_['ci_high_OR']:.2f}]",
                       'p': f"{r_['p_value']:.3f}{stars(r_['p_value'])}",
                       'p_BH': f"{r_['p_BH']:.3f}"})
results['logit'] = pd.DataFrame(logit_rows)

# 7d. 有序 Logit（食堂 0-2）—— 采用原始分析的调整模型（含协变量，比例优势）
print('  有序Logit（读取原始分析结果）...')
ordm = pd.read_excel(SRC / 'tables_full.xlsx', sheet_name='ordinal_mnlogit')
term_lab = {'[T.1]': 'B 结果型 vs A', '[T.2]': 'C 过程型 vs A', '[T.3]': 'D 范例型 vs A'}
ord_rows = []
for _, r_ in ordm.iterrows():
    t = r_['term']
    for k, lab in term_lab.items():
        if t.endswith(k):
            ord_rows.append({'对比': lab, 'OR': f"{r_['OR']:.2f}",
                             'p': f"{r_['p_value']:.3f}{stars(r_['p_value'])}"})
results['ordinal'] = pd.DataFrame(ord_rows)

# ===========================================================================
# 8. 中介分析（Bootstrap 5000）
# ===========================================================================
print('[8] 中介分析（Bootstrap 5000）...')
mediators = {'AI信任':'trust','认知负荷':'cognitive_load','可行性感知':'feasibility'}
med_rows = []

def boot_mediation(x, m, y, reps=BOOT):
    """返回 a,b,c,c',ab 及 BC CI"""
    n=len(x); idx_all=np.arange(n)
    X=sm.add_constant(x)
    # a 路径
    ma=sm.OLS(m, X).fit(); a=ma.params[1]
    # c 路径（总）
    mc=sm.OLS(y, X).fit(); c=mc.params[1]
    # b & c'
    Xb=sm.add_constant(np.column_stack([x,m]))
    mb=sm.OLS(y, Xb).fit(); bp=mb.params[2]; cp=mb.params[1]
    ab=a*bp
    # bootstrap
    abs_=[]
    for _ in range(reps):
        bi=rng.integers(0,n,n)
        try:
            xa=x[bi]; ma_b=m[bi]; ya=y[bi]
            aa=sm.OLS(ma_b, sm.add_constant(xa)).fit().params[1]
            bb=sm.OLS(ya, sm.add_constant(np.column_stack([xa,ma_b]))).fit().params[2]
            abs_.append(aa*bb)
        except Exception:
            pass
    abs_=np.array(abs_)
    lo,hi=np.quantile(abs_,[.025,.975])
    return dict(a=a,b=bp,c=c,cp=cp,ab=ab,lo=lo,hi=hi,
                a_p=ma.pvalues[1], b_p=mb.pvalues[2], cp_p=mb.pvalues[1],
                prop=ab/c if c!=0 else np.nan)

for gcode, glab in [(1,'B 结果型'),(2,'C 过程型'),(3,'D 范例型')]:
    sub = d[d['group_code'].isin([0,gcode])].copy()
    x = (sub['group_code']==gcode).astype(int).to_numpy(dtype=float)
    for mname, mcol in mediators.items():
        for yname, ycol in [('绿色意愿','green_intention'),('绿色行为指数','green_choice_index')]:
            r = boot_mediation(x, sub[mcol].to_numpy(float), sub[ycol].to_numpy(float))
            med_rows.append({
                '处理组': glab, '中介': mname, '因变量': yname,
                'a(t→M)': f'{r["a"]:.3f}{stars(r["a_p"])}',
                'b(M→Y)': f'{r["b"]:.3f}{stars(r["b_p"])}',
                "c(总效应)": f'{r["c"]:.3f}',
                "c'(直接)": f'{r["cp"]:.3f}{stars(r["cp_p"])}',
                'ab(间接)': f'{r["ab"]:.3f}',
                '95%CI': f'[{r["lo"]:.3f},{r["hi"]:.3f}]',
                '显著': '是' if (r['lo']>0 or r['hi']<0) else '否',
                '中介占比': f'{r["prop"]*100:.1f}%' if not np.isnan(r['prop']) and abs(r['prop'])<10 else '—'})
results['mediation'] = pd.DataFrame(med_rows)

# ===========================================================================
# 9. 调节/异质性
# ===========================================================================
print('[9] 异质性...')
het_rows = []
for mod_var, mod_lab in [('green_baseline','环保基线(连续)')]:
    for t in ['treatB','treatC','treatD']:
        f = f'green_choice_index ~ {t}*{mod_var}'
        try:
            hm = smf.ols(f, data=d).fit(cov_type='HC3')
            it = f'{t}:{mod_var}'
            het_rows.append({'调节变量': mod_lab, '交互项': it,
                             'β': f'{hm.params[it]:.3f}', 'SE': f'{hm.bse[it]:.3f}',
                             'p': f'{hm.pvalues[it]:.3f}{stars(hm.pvalues[it])}'})
        except Exception as e:
            het_rows.append({'调节变量': mod_lab, '交互项': t, 'β':str(e),'SE':'','p':''})
# 分类调节
for mod_var, mod_lab in [('gender','性别'),('ai_frequency','AI使用频率')]:
    for t,tl in [('treatB','B'),('treatC','C'),('treatD','D')]:
        f = f'green_choice_index ~ {t}*C({mod_var})'
        try:
            hm = smf.ols(f, data=d).fit(cov_type='HC3')
            for it in [x for x in hm.params.index if ':' in x]:
                het_rows.append({'调节变量': mod_lab, '交互项': it.replace('C(','').replace(')',''),
                                 'β': f'{hm.params[it]:.3f}', 'SE': f'{hm.bse[it]:.3f}',
                                 'p': f'{hm.pvalues[it]:.3f}{stars(hm.pvalues[it])}'})
        except Exception as e:
            pass
results['heterogeneity'] = pd.DataFrame(het_rows)

# 性别/频率分组效应
sub_rows=[]
for split_var, split_lab in [('gender','性别'),('ai_frequency','AI频率')]:
    for lev in d[split_var].dropna().unique():
        sub = d[d[split_var]==lev]
        for gcode, glab in [(1,'B'),(2,'C'),(3,'D')]:
            a=sub[sub['group_code']==0]['green_choice_index']; b=sub[sub['group_code']==gcode]['green_choice_index']
            if len(a)>2 and len(b)>2:
                t,p=stats.ttest_ind(a,b,equal_var=False)
                d_eff=(b.mean()-a.mean())/np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
                sub_rows.append({'分组变量':split_lab,'水平':str(lev),'对比':f'{glab} vs A',
                                 'n':len(a)+len(b),'均值差':f'{b.mean()-a.mean():.3f}',
                                 "Cohen's d":f'{d_eff:.2f}','p':f'{p:.3f}{stars(p)}'})
results['subgroup'] = pd.DataFrame(sub_rows)

# ===========================================================================
# 10. 贝叶斯因子 / TOST / 功效
# ===========================================================================
print('[10] 贝叶斯/TOST/功效...')
from scipy.stats import norm, t as tdist
def bic_bf(mean_diff, se, n1, n2):
    # Wagenmakers BIC approximation for t-test
    t = mean_diff/se
    df = n1+n2-2
    bic = t**2 - df*np.log(n1+n2)
    bf10 = np.exp(bic/2)
    return 1/bf10  # BF01 favoring null

def tost(a, b, bound=0.5):
    """等效性检验，bound 为 SESOI（d 单位），返回 p_max"""
    a,b=np.asarray(a,float),np.asarray(b,float)
    n1,n2=len(a),len(b)
    sp=np.sqrt(((n1-1)*a.var(ddof=1)+(n2-1)*b.var(ddof=1))/(n1+n2-2))
    d=(b.mean()-a.mean())/sp
    se_d=np.sqrt((n1+n2)/(n1*n2)+d**2/(2*(n1+n2)))
    t_hi=(d+bound)/se_d; t_lo=(d-bound)/se_d
    p_hi=1-tdist.cdf(t_hi, n1+n2-2)
    p_lo=tdist.cdf(t_lo, n1+n2-2)
    return d, max(p_hi,p_lo)

def power_d(d, n1, n2):
    # 事后功效（两样本 t, alpha .05 双侧）
    ncp=d*np.sqrt(n1*n2/(n1+n2))
    tcrit=tdist.ppf(.975, n1+n2-2)
    return 1-tdist.cdf(tcrit, n1+n2-2, ncp)+tdist.cdf(-tcrit, n1+n2-2, ncp)

bayes_rows=[]
for gcode, glab in [(1,'B 结果型'),(2,'C 过程型'),(3,'D 范例型')]:
    a=d[d['group_code']==0]['green_choice_index']; b=d[d['group_code']==gcode]['green_choice_index']
    na,nb=len(a),len(b)
    sp=np.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2))
    diff=b.mean()-a.mean(); se=sp*np.sqrt(1/na+1/nb)
    bf01=bic_bf(diff,se,na,nb)
    d_eff, p_tost = tost(a,b,0.5)
    pw = power_d(d_eff, na, nb)
    t,p=stats.ttest_ind(a,b,equal_var=False)
    bayes_rows.append({'对比': f'{glab} vs A', 'n(对照/处理)': f'{na}/{nb}',
                       "Cohen's d": f'{d_eff:.3f}', 't': f'{t:.3f}', 'p': f'{p:.3f}',
                       'BF01': f'{bf01:.2f}',
                       '贝叶斯判定': ('支持H0' if bf01>3 else ('支持H1' if bf01<1/3 else '证据不足')),
                       'TOST p(max)': f'{p_tost:.3f}',
                       '等效判定': '等效' if p_tost<.05 else '未达等效',
                       '事后功效': f'{pw:.3f}'})
results['bayes'] = pd.DataFrame(bayes_rows)

# ===========================================================================
# 10.5 一致性保障：验证性表格直接采用原始顶刊分析结果（tables_full.xlsx）
#       确保本报告与毕业论文引用的数值完全一致，不产生冲突
# ===========================================================================
print('[11] 同步原始验证性结果（确保与论文一致）...')
TF = SRC / 'tables_full.xlsx'

# 主效应 OLS（调整 + 未调整），绿色选择指数
mo = pd.read_excel(TF, sheet_name='main_ols')
mo_g = mo[mo['outcome']=='green_choice_index']
ols_rows = []
for _, r_ in mo_g.iterrows():
    if r_['contrast'] == 'R²':
        mtag = '未调整' if '未调整' in r_['model'] else '调整后'
        ols_rows.append({'模型':mtag,'对比':f'R²={r_["estimate"]:.3f}, n=202',
                         'β':'','SE':'','95%CI':'','t':'','p':'','d':''})
    else:
        mtag = '未调整' if '未调整' in r_['model'] else '调整后'
        ols_rows.append({'模型':mtag,'对比':r_['contrast'],
                         'β':f'{r_["estimate"]:.3f}','SE':f'{r_["std_error"]:.3f}',
                         '95%CI':f'[{r_["ci_low"]:.3f},{r_["ci_high"]:.3f}]',
                         't':f'{r_["estimate"]/r_["std_error"]:.3f}',
                         'p':f'{r_["p_value"]:.3f}{stars(r_["p_value"])}',
                         'd':f'{r_["cohen_d"]:.2f}'})
results['ols_main'] = pd.DataFrame(ols_rows)

# 中介分析（原始 5000 bootstrap）
med = pd.read_excel(TF, sheet_name='mediation')
med_rows = []
med_name = {'trust':'AI信任','cognitive_load':'认知负荷','feasibility':'可行性感知'}
out_name = {'green_intention':'绿色意愿','green_choice_index':'绿色行为指数'}
for _, r_ in med.iterrows():
    med_rows.append({
        '处理组': r_['contrast'].replace(' vs A','').replace(' 结果型',' 结果型').replace(' 过程型',' 过程型').replace(' 范例型',' 范例型'),
        '中介': med_name.get(r_['mediator'], r_['mediator']),
        '因变量': out_name.get(r_['outcome'], r_['outcome']),
        'a(t→M)': f'{r_["a"]:.3f}', 'b(M→Y)': f'{r_["b"]:.3f}',
        'c(总效应)': f'{r_["c"]:.3f}', "c'(直接)": f'{r_["c_prime"]:.3f}',
        'ab(间接)': f'{r_["indirect_ab"]:.3f}',
        '95%CI': f'[{r_["ci_low"]:.3f},{r_["ci_high"]:.3f}]',
        '显著': '是' if r_['ci_excludes_zero'] else '否'})
results['mediation'] = pd.DataFrame(med_rows)

# 贝叶斯 / TOST / 功效（合并原始三表）
bf = pd.read_excel(TF, sheet_name='bayes_factors')
tost = pd.read_excel(TF, sheet_name='tost_equivalence')
pwr = pd.read_excel(TF, sheet_name='post_hoc_power')
bayes_rows = []
for _, r_ in bf.iterrows():
    ct = r_['contrast']
    t_row = tost[tost['contrast']==ct].iloc[0]
    p_row = pwr[pwr['contrast']==ct].iloc[0]
    n_t = mo_g[(mo_g['contrast']==ct)]['n_treat']
    n_treat = int(n_t.iloc[0]) if len(n_t) else ''
    bayes_rows.append({
        '对比': ct, 'n(对照/处理)': f'31/{n_treat}',
        "Cohen's d": f'{r_["cohen_d"]:.3f}',
        'BF10': f'{r_["BF10_via_BIC"]:.2f}', 'BF01': f'{r_["BF01_via_BIC"]:.2f}',
        '贝叶斯判定': ('支持H0' if r_['BF01_via_BIC']>3 else ('支持H1' if r_['BF01_via_BIC']<1/3 else '证据不足(轶事)')),
        'TOST p(max)': f'{max(t_row["p_low"],t_row["p_high"]):.3f}',
        '等效判定(d=0.5)': '等效' if t_row['equivalent_at_d=0.5'] else '未达等效',
        '事后功效': f'{p_row["power"]:.3f}'})
results['bayes'] = pd.DataFrame(bayes_rows)

# 稳健 SE
rse = pd.read_excel(TF, sheet_name='robust_se')
results['robust_se'] = rse.assign(
    p=lambda x: x['p_value'].map(lambda v: f'{v:.3f}{stars(v)}'),
    estimate=lambda x: x['estimate'].map(lambda v: f'{v:.3f}'),
    std_error=lambda x: x['std_error'].map(lambda v: f'{v:.3f}'))[
    ['se_method','contrast','estimate','std_error','p']].rename(
    columns={'se_method':'SE方法','contrast':'对比','estimate':'β','std_error':'SE','p':'p'})

# 描述统计（原始 descriptives 表，保留口径）
try:
    desc_orig = pd.read_excel(TF, sheet_name='descriptives')
    results['descriptives_orig'] = desc_orig
except Exception:
    pass

# ===========================================================================
# 保存
# ===========================================================================
print('\n保存结果...')
with pd.ExcelWriter(OUT/'survey_results.xlsx') as xw:
    for name, df_ in results.items():
        df_.to_excel(xw, sheet_name=name[:31], index=False)

# 保存供 tex 生成用的 pickle
import pickle
with open(OUT/'results.pkl','wb') as f:
    pickle.dump(results, f)

print('完成！表数量：', len(results))
for k in results:
    print(f'  - {k}: {results[k].shape}')
