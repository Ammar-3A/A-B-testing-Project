"""
main.py
تحليل تجربة A/B: أثر لوحة الاستهلاك اللحظي على استهلاك الكهرباء اليومي.

التشغيل:
    pip install -r requirements.txt
    python main.py
(إذا لم يوجد data/dataset.csv يتم توليده تلقائيًا عبر generate_data.py)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

import generate_data
from generate_data import assign_group

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "dataset.csv"
CHARTS = BASE / "outputs" / "charts"
RESULTS = BASE / "outputs" / "results"
CHARTS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ----------------------- معاملات التصميم (محددة مسبقًا) -----------------------
ALPHA = 0.05
POWER = 0.80
MDE_REL = 0.05                 # أصغر أثر نريد اكتشافه: انخفاض 5% نسبيًا
PRACTICAL_MIN_REL = 0.02       # أقل انخفاض يستحق الإطلاق تجاريًا: 2%
SRM_ALPHA = 0.001
SMD_LIMIT = 0.1               # التوازن جيد إذا كان SMD أقل من 0.1


# ----------------------------- دوال مساعدة -----------------------------
def welch(treat, ctrl, alpha=ALPHA):
    """فرق المتوسطات (B - A) مع فترة ثقة واختبار Welch."""
    treat, ctrl = np.asarray(treat, float), np.asarray(ctrl, float)
    nt, nc = len(treat), len(ctrl)
    vt, vc = treat.var(ddof=1), ctrl.var(ddof=1)
    diff = treat.mean() - ctrl.mean()
    se = math.sqrt(vt / nt + vc / nc)
    df = (vt / nt + vc / nc) ** 2 / ((vt / nt) ** 2 / (nt - 1) + (vc / nc) ** 2 / (nc - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    p = 2 * (1 - stats.t.cdf(abs(diff / se), df))
    return {"mean_B": treat.mean(), "mean_A": ctrl.mean(), "diff": diff, "se": se,
            "ci_low": diff - tcrit * se, "ci_high": diff + tcrit * se, "p_value": p,
            "n_B": nt, "n_A": nc}


def cuped_adjust(y, x):
    """تعديل CUPED باستخدام الاستهلاك قبل التجربة كمتغير مساعد."""
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), theta


def required_n_per_group(mean, sd, mde_rel, alpha=ALPHA, power=POWER):
    delta = mde_rel * mean
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return math.ceil(2 * (z ** 2) * (sd ** 2) / (delta ** 2))


def smd(a, b):
    """الفرق المعياري للمتوسطات لفحص توازن المتغيرات."""
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (b.mean() - a.mean()) / pooled


# ------------------------------- 1) تحميل البيانات -------------------------------
if not DATA.exists():
    generate_data.generate()
raw = pd.read_csv(DATA)
print(f"Loaded {len(raw)} rows")

# ------------------------------- 2) فحوص الجودة -------------------------------
quality = []
n_dups = int(raw.duplicated(subset="user_id").sum())
df = raw.drop_duplicates(subset="user_id").copy()
quality.append(("duplicate_user_rows_removed", n_dups))

miss_post = df["post_avg_kwh"].isna()
quality.append(("missing_post_avg_kwh", int(miss_post.sum())))
miss_by_group = df.groupby("group")["post_avg_kwh"].apply(lambda s: s.isna().mean())
quality.append(("missing_rate_A", float(miss_by_group["A"])))
quality.append(("missing_rate_B", float(miss_by_group["B"])))
df_an = df.dropna(subset=["post_avg_kwh", "pre_avg_kwh"]).copy()   # تحليل الحالات المكتملة
df_an["treat"] = (df_an["group"] == "B").astype(int)

# SRM: هل نسبة التوزيع الفعلية = 50/50؟
counts = df["group"].value_counts().reindex(["A", "B"])
srm_chi2, srm_p = stats.chisquare(counts.values, f_exp=[len(df) / 2] * 2)
quality += [("n_A", int(counts["A"])), ("n_B", int(counts["B"])),
            ("srm_chi2", float(srm_chi2)), ("srm_p_value", float(srm_p)),
            ("srm_pass", bool(srm_p > SRM_ALPHA))]

# هل التوزيع الثابت بالـ hash قابل لإعادة الحساب؟ (المستخدم نفسه دائمًا في النسخة نفسها)
quality.append(("assignment_matches_hash", bool((df["user_id"].map(assign_group) == df["group"]).all())))

# توازن المتغيرات قبل المعالجة
A, B = df_an[df_an.treat == 0], df_an[df_an.treat == 1]
for col in ["pre_avg_kwh", "area_m2", "household_size"]:
    s = float(smd(A[col], B[col]))
    quality.append((f"SMD_{col}", s))
    quality.append((f"SMD_{col}_below_{SMD_LIMIT}", bool(abs(s) < SMD_LIMIT)))
# فحص على طريقة A/A: لا يجب أن يظهر فرق في استهلاك فترة ما قبل التجربة
aa = welch(B["pre_avg_kwh"], A["pre_avg_kwh"])
quality.append(("pre_period_AA_check_p_value", float(aa["p_value"])))
chi_region = stats.chi2_contingency(pd.crosstab(df_an["region"], df_an["treat"]))[1]
quality.append(("region_balance_p_value", float(chi_region)))

# معدل الاستجابة لاستبيان الرضا بين المجموعتين (استجابة تفاضلية؟)
resp = df.groupby("group")["satisfaction"].apply(lambda s: s.notna().mean())
quality += [("survey_response_rate_A", float(resp["A"])),
            ("survey_response_rate_B", float(resp["B"]))]
pd.DataFrame(quality, columns=["check", "value"]).to_csv(RESULTS / "quality_checks.csv", index=False)

# ------------------------------- 3) حجم العينة والمدة -------------------------------
base_mean, base_sd = df_an["pre_avg_kwh"].mean(), df_an["pre_avg_kwh"].std()
rho = np.corrcoef(A["pre_avg_kwh"], A["post_avg_kwh"])[0, 1]
n_req = required_n_per_group(base_mean, base_sd, MDE_REL)
n_req_cuped = required_n_per_group(base_mean, base_sd * math.sqrt(1 - rho ** 2), MDE_REL)
daily_enroll = generate_data.DAILY_ENROLLMENT
raw_days = 2 * n_req / daily_enroll
enroll_days = math.ceil(raw_days / 7) * 7        # نقرب إلى أسابيع كاملة (سلوك المستخدم يختلف بين أيام الأسبوع)
duration_days = enroll_days + generate_data.OBSERVATION_DAYS - 1
n_actual = int(min(len(A), len(B)))
mde_achieved = math.sqrt(2 * (stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)) ** 2 * base_sd ** 2 / n_actual)
sample_size = {
    "alpha": ALPHA, "power": POWER, "mde_relative": MDE_REL,
    "baseline_mean_kwh_day": base_mean, "baseline_sd_kwh_day": base_sd,
    "mde_absolute_kwh_day": MDE_REL * base_mean,
    "required_n_per_group": n_req, "required_total": 2 * n_req,
    "required_n_per_group_with_cuped": n_req_cuped, "cuped_correlation_rho": rho,
    "expected_daily_enrollment": daily_enroll, "enrollment_days_raw": raw_days,
    "enrollment_days_rounded_to_full_weeks": enroll_days,
    "mde_achieved_kwh_day": mde_achieved, "mde_achieved_relative": mde_achieved / base_mean,
    "observation_days_per_user": generate_data.OBSERVATION_DAYS,
    "total_duration_days": duration_days,
    "actual_n_per_group": n_actual,
}
pd.DataFrame([sample_size]).T.rename(columns={0: "value"}).to_csv(RESULTS / "sample_size.csv")

# ------------------------------- 4) التحليل الرئيسي -------------------------------
primary = welch(B["post_avg_kwh"], A["post_avg_kwh"])
primary["rel_change_pct"] = 100 * primary["diff"] / primary["mean_A"]
primary["rel_ci_low_pct"] = 100 * primary["ci_low"] / primary["mean_A"]
primary["rel_ci_high_pct"] = 100 * primary["ci_high"] / primary["mean_A"]
primary["cohens_d"] = primary["diff"] / math.sqrt((A.post_avg_kwh.var() + B.post_avg_kwh.var()) / 2)

# تحليل CUPED (لزيادة الدقة)
y_adj, theta = cuped_adjust(df_an["post_avg_kwh"].values, df_an["pre_avg_kwh"].values)
df_an["post_cuped"] = y_adj
cu = welch(df_an.loc[df_an.treat == 1, "post_cuped"], df_an.loc[df_an.treat == 0, "post_cuped"])
cu["rel_change_pct"] = 100 * cu["diff"] / primary["mean_A"]
cu["theta"] = theta
cu["variance_reduction_pct"] = 100 * (1 - df_an["post_cuped"].var() / df_an["post_avg_kwh"].var())

# انحدار OLS بمتغيرات مساعدة (فحص صلابة) مع أخطاء قياسية متينة HC3
ols = smf.ols("post_avg_kwh ~ treat + pre_avg_kwh + C(region) + area_m2 + household_size",
              data=df_an).fit(cov_type="HC3")
ols_ci = ols.conf_int().loc["treat"]
reg = {"diff": ols.params["treat"], "ci_low": ols_ci[0], "ci_high": ols_ci[1],
       "p_value": ols.pvalues["treat"]}

pd.DataFrame({"unadjusted_welch (PRIMARY)": primary, "CUPED": cu,
              "OLS_covariates_HC3": reg}).to_csv(RESULTS / "primary_results.csv")

# ------------------------------- 5) المقاييس الثانوية (تصحيح Holm) -------------------------------
sec_rows = []
for col in ["app_opens_per_week", "dashboard_interactions"]:
    r = welch(B[col], A[col]); r["metric"] = col; sec_rows.append(r)
sec = pd.DataFrame(sec_rows).set_index("metric")
sec["p_holm"] = multipletests(sec["p_value"], method="holm")[1]
sec.to_csv(RESULTS / "secondary_metrics.csv")

# ------------------------------- 6) مقاييس الحماية -------------------------------
sat_B = df.loc[df.group == "B", "satisfaction"].dropna()
sat_A = df.loc[df.group == "A", "satisfaction"].dropna()
sat = welch(sat_B, sat_A)
sat["pass"] = bool(not sat["ci_high"] < 0)       # سليم ما لم يكن هناك تدهور دال (فترة الثقة كلها تحت الصفر)

cB, cA = df.loc[df.group == "B", "complaints"] > 0, df.loc[df.group == "A", "complaints"] > 0
pB, pA, nB, nA = cB.mean(), cA.mean(), len(cB), len(cA)
d_c = pB - pA
se_c = math.sqrt(pB * (1 - pB) / nB + pA * (1 - pA) / nA)
zc = stats.norm.ppf(0.975)
pool = (cB.sum() + cA.sum()) / (nB + nA)
z_stat = d_c / math.sqrt(pool * (1 - pool) * (1 / nA + 1 / nB))
comp = {"rate_B": pB, "rate_A": pA, "diff": d_c, "ci_low": d_c - zc * se_c,
        "ci_high": d_c + zc * se_c, "p_value": 2 * (1 - stats.norm.cdf(abs(z_stat))),
        "pass": bool(not d_c - zc * se_c > 0)}  # سليم ما لم تزد الشكاوى زيادة دالة
pd.DataFrame({"satisfaction": sat, "complaint_rate": comp}).to_csv(RESULTS / "guardrails.csv")

# ------------------------------- 7) تحليل الشرائح (استكشافي) -------------------------------
df_an["tier"] = pd.qcut(df_an["pre_avg_kwh"], 3, labels=["Low", "Medium", "High"])
seg_rows = []
for seg_col in ["tier", "region"]:
    for level, g in df_an.groupby(seg_col, observed=True):
        gb, ga = g[g.treat == 1], g[g.treat == 0]
        r = welch(gb["post_avg_kwh"], ga["post_avg_kwh"])
        r.update({"segment": f"{seg_col}={level}",
                  "rel_change_pct": 100 * r["diff"] / r["mean_A"]})
        seg_rows.append(r)
seg = pd.DataFrame(seg_rows).set_index("segment")
seg.to_csv(RESULTS / "segments_exploratory.csv")

# ------------------------------- 7ب) أثر الحداثة: الأثر في كل أسبوع -------------------------------
wk_rows = []
for w in range(1, 5):
    col = f"wk{w}_avg_kwh"
    r = welch(B[col], A[col]); r["week"] = w
    r["rel_change_pct"] = 100 * r["diff"] / r["mean_A"]
    wk_rows.append(r)
novelty = pd.DataFrame(wk_rows).set_index("week")
novelty.to_csv(RESULTS / "weekly_effect_novelty.csv")

# ------------------------------- 8) قاعدة القرار (حسب فترة الثقة وأصغر أثر يهمنا) -------------------------------
benefit_lo, benefit_hi = -primary["rel_ci_high_pct"], -primary["rel_ci_low_pct"]   # نسبة الانخفاض %
min_pct = 100 * PRACTICAL_MIN_REL
guardrails_ok = sat["pass"] and comp["pass"]
if benefit_hi < 0:
    decision = "DO NOT LAUNCH: harm, consumption increased"
elif not guardrails_ok:
    decision = "DO NOT LAUNCH NOW: a guardrail metric worsened; fix and re-verify"
elif benefit_lo > 0:
    decision = "LAUNCH: clear win, reduction is significant and guardrails are healthy"
elif benefit_hi < min_pct:
    decision = "DO NOT LAUNCH: an effect of the size that matters is ruled out"
else:
    decision = "INCONCLUSIVE: extend the experiment or strengthen the design"

summary = {
    "decision": decision,
    "primary": primary, "cuped": cu, "ols": reg,
    "sample_size": sample_size, "guardrail_satisfaction": sat,
    "guardrail_complaints": comp,
    "reduction_ci_pct": [benefit_lo, benefit_hi], "min_effect_pct": min_pct,
    "kwh_saved_per_household_per_year": -primary["diff"] * 365,
}
with open(RESULTS / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else bool(o))

# ------------------------------- 9) الرسوم -------------------------------
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
COL = {"A": "#7f8c8d", "B": "#1f77b4"}

# (1) توزيع الاستهلاك
fig, ax = plt.subplots(figsize=(7, 4))
for g, lab in [("A", "A - Control"), ("B", "B - Dashboard")]:
    ax.hist(df_an.loc[df_an.group == g, "post_avg_kwh"], bins=45, alpha=0.55,
            color=COL[g], label=lab, density=True)
    ax.axvline(df_an.loc[df_an.group == g, "post_avg_kwh"].mean(), color=COL[g], ls="--")
ax.set_xlabel("Average daily consumption (kWh/day)"); ax.set_ylabel("Density")
ax.set_title("Distribution of daily consumption by group"); ax.legend()
fig.tight_layout(); fig.savefig(CHARTS / "01_distribution.png"); plt.close(fig)

# (2) الأثر المقدر بالطرق الثلاث
fig, ax = plt.subplots(figsize=(7, 3.5))
items = [("Welch (primary)", primary), ("CUPED", cu), ("OLS + covariates", reg)]
for i, (lab, r) in enumerate(items):
    ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["ci_low"]], [r["ci_high"] - r["diff"]]],
                fmt="o", color="#1f77b4", capsize=5)
ax.axvline(0, color="k", lw=1)
ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in items])
ax.set_xlabel("Effect on daily consumption, B - A (kWh/day), 95% CI")
ax.set_title("Estimated treatment effect"); fig.tight_layout()
fig.savefig(CHARTS / "02_effect_estimates.png"); plt.close(fig)

# (3) منحنى القوة الإحصائية: العينة مقابل MDE
fig, ax = plt.subplots(figsize=(7, 4))
mdes = np.linspace(0.02, 0.10, 40)
ax.plot(100 * mdes, [required_n_per_group(base_mean, base_sd, m) for m in mdes], label="Without CUPED")
ax.plot(100 * mdes, [required_n_per_group(base_mean, base_sd * math.sqrt(1 - rho ** 2), m) for m in mdes],
        label="With CUPED")
ax.axhline(sample_size["actual_n_per_group"], color="red", ls=":", label="Actual n per group")
ax.set_yscale("log"); ax.set_xlabel("Minimum detectable effect (% reduction)")
ax.set_ylabel("Required n per group (log)"); ax.set_title("Sample size vs MDE (alpha=0.05, power=0.8)")
ax.legend(); fig.tight_layout(); fig.savefig(CHARTS / "03_sample_size_vs_mde.png"); plt.close(fig)

# (4) الشرائح
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, (name, r) in enumerate(seg.iterrows()):
    ax.errorbar(r["rel_change_pct"], i,
                xerr=[[r["rel_change_pct"] - 100 * r["ci_low"] / r["mean_A"]],
                      [100 * r["ci_high"] / r["mean_A"] - r["rel_change_pct"]]],
                fmt="s", color="#2c3e50", capsize=4)
ax.axvline(0, color="k", lw=1)
ax.set_yticks(range(len(seg))); ax.set_yticklabels(seg.index)
ax.set_xlabel("Relative change in consumption (%), 95% CI")
ax.set_title("Exploratory segment effects"); fig.tight_layout()
fig.savefig(CHARTS / "04_segments.png"); plt.close(fig)

# (5) مقاييس الحماية
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].errorbar(sat["diff"], 0, xerr=[[sat["diff"] - sat["ci_low"]], [sat["ci_high"] - sat["diff"]]],
                 fmt="o", capsize=5)
axes[0].axvline(0, color="k", lw=1); axes[0].set_yticks([])
axes[0].set_title("Satisfaction diff (B-A)")
axes[1].errorbar(100 * comp["diff"], 0, xerr=[[100 * (comp["diff"] - comp["ci_low"])],
                                               [100 * (comp["ci_high"] - comp["diff"])]],
                 fmt="o", capsize=5, color="#c0392b")
axes[1].axvline(0, color="k", lw=1); axes[1].set_yticks([])
axes[1].set_title("Complaint-rate diff (pp, B-A)")
fig.tight_layout(); fig.savefig(CHARTS / "05_guardrails.png"); plt.close(fig)

# (6) الأثر الأسبوعي
fig, ax = plt.subplots(figsize=(7, 4))
for w, r in novelty.iterrows():
    ax.errorbar(w, r["rel_change_pct"],
                yerr=[[r["rel_change_pct"] - 100 * r["ci_low"] / r["mean_A"]],
                      [100 * r["ci_high"] / r["mean_A"] - r["rel_change_pct"]]],
                fmt="o", color="#1f77b4", capsize=5)
ax.axhline(0, color="k", lw=1); ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("Week after enrollment"); ax.set_ylabel("Relative change in consumption (%)")
ax.set_title("Weekly effect (novelty check), 95% CI"); fig.tight_layout()
fig.savefig(CHARTS / "06_weekly_effect.png"); plt.close(fig)

# ------------------------------- 10) ملخص الطباعة -------------------------------
print("\n=== QUALITY ===")
print(pd.DataFrame(quality, columns=["check", "value"]).to_string(index=False))
print("\n=== SAMPLE SIZE ===")
for k, v in sample_size.items():
    print(f"{k}: {v}")
print("\n=== PRIMARY (B - A, kWh/day) ===")
print(f"A={primary['mean_A']:.2f}  B={primary['mean_B']:.2f}  diff={primary['diff']:.2f} "
      f"[{primary['ci_low']:.2f}, {primary['ci_high']:.2f}]  p={primary['p_value']:.2g}  "
      f"rel={primary['rel_change_pct']:.2f}%")
print(f"CUPED diff={cu['diff']:.2f} [{cu['ci_low']:.2f}, {cu['ci_high']:.2f}] p={cu['p_value']:.2g} "
      f"variance reduction={cu['variance_reduction_pct']:.1f}%")
print(f"OLS   diff={reg['diff']:.2f} [{reg['ci_low']:.2f}, {reg['ci_high']:.2f}] p={reg['p_value']:.2g}")
print("\n=== SECONDARY ===")
print(sec[["mean_A", "mean_B", "diff", "ci_low", "ci_high", "p_value", "p_holm"]].round(4).to_string())
print("\n=== GUARDRAILS ===")
print(f"Satisfaction diff={sat['diff']:.3f} [{sat['ci_low']:.3f}, {sat['ci_high']:.3f}] pass={sat['pass']}")
print(f"Complaint rate A={comp['rate_A']:.3%} B={comp['rate_B']:.3%} diff[{comp['ci_low']:.3%}, "
      f"{comp['ci_high']:.3%}] pass={comp['pass']}")
print("\n=== WEEKLY EFFECT (novelty check) ===")
print(novelty[["rel_change_pct", "diff", "ci_low", "ci_high", "p_value"]].round(3).to_string())
print("\n=== SEGMENTS ===")
print(seg[["rel_change_pct", "p_value", "n_A", "n_B"]].round(3).to_string())
print(f"\nDECISION: {decision}")
