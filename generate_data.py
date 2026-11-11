"""
generate_data.py
توليد بيانات محاكاة لتجربة A/B حول لوحة الاستهلاك اللحظي للكهرباء.

تنبيه: البيانات محاكاة بالكامل (لا توجد بيانات مشتركين حقيقية).
الافتراضات الرئيسية موثقة في README.md وفي التعليقات أدناه.
"""
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 3500            # إجمالي المنازل المشاركة في التجربة
DAILY_ENROLLMENT = 250    # عدد المنازل التي تدخل التجربة يوميًا
OBSERVATION_DAYS = 28     # مدة متابعة كل منزل بعد الدخول
EXPERIMENT = "electricity_dashboard_v1"
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "dataset.csv"


def assign_group(user_id: str, experiment: str = EXPERIMENT, pct_b: int = 50) -> str:
    """توزيع ثابت بالـ hash: رقم 0-99 من (user_id + اسم التجربة)، أقل من النسبة يعني A وغير ذلك B."""
    h = int(hashlib.md5(f"{user_id}_{experiment}".encode()).hexdigest(), 16) % 100
    return "A" if h < 100 - pct_b else "B"


def generate(path: Path = DATA_PATH, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = N_USERS

    # ---- خصائص المنزل (قبل المعالجة) ----
    regions = ["Central", "Western", "Eastern", "Southern", "Northern"]
    region = rng.choice(regions, size=n, p=[0.35, 0.25, 0.2, 0.1, 0.1])
    region_effect = {"Central": 0.05, "Western": 0.0, "Eastern": 0.08,
                     "Southern": -0.08, "Northern": -0.03}
    area = np.clip(rng.normal(250, 80, n), 80, 600)              # م2
    household_size = np.clip(rng.poisson(4, n) + 1, 1, 12)

    # الاستهلاك الأساسي اليومي (كيلوواط ساعة/يوم) ~ متوسط 45
    log_base = (np.log(36)
                + 0.004 * (area - 250)
                + 0.05 * (household_size - 5)
                + np.array([region_effect[r] for r in region])
                + rng.normal(0, 0.25, n))
    base = np.exp(log_base)

    # ---- التوزيع الثابت بالـ hash (50/50 على مستوى المنزل) ----
    user_ids = [f"H{str(i).zfill(5)}" for i in range(1, n + 1)]
    group = np.array([assign_group(u) for u in user_ids])
    treat = (group == "B").astype(int)

    # ---- المقياس الرئيسي ----
    pre_avg = base * rng.lognormal(0, 0.15, n)                   # متوسط 28 يومًا قبل التجربة
    # الأثر الحقيقي (مفترض): انخفاض ~4.5% في المتوسط، أكبر عند المستهلكين الأعلى
    effect_pct = 0.025 + 0.04 * (base > np.median(base)) + rng.normal(0, 0.02, n)
    # أثر أكبر في الأسبوع الأول ثم يتراجع قليلًا (أثر الحداثة)، متوسط المعاملات = 1
    week_mult = [1.25, 1.10, 0.90, 0.75]
    weekly = [base * rng.lognormal(0, 0.25, n) * (1 - treat * effect_pct * m) for m in week_mult]
    post_avg = np.mean(weekly, axis=0)

    # ---- المقاييس الثانوية ----
    app_opens_week = rng.poisson(np.where(treat == 1, 5.5, 3.0))
    dashboard_interactions = rng.poisson(np.where(treat == 1, 4.5, 1.2))

    # ---- مقاييس الحماية ----
    responded = rng.random(n) < 0.45                             # استبيان الرضا اختياري
    satisfaction = np.clip(np.round(rng.normal(3.9 + 0.03 * treat, 0.9, n)), 1, 5)
    satisfaction = np.where(responded, satisfaction, np.nan)
    complaints = rng.poisson(np.where(treat == 1, 0.062, 0.060))

    # ---- تاريخ الدخول للتجربة ----
    enroll_day = np.sort(rng.integers(0, int(np.ceil(n / DAILY_ENROLLMENT)), n))
    enroll_date = pd.Timestamp("2026-01-04") + pd.to_timedelta(enroll_day, unit="D")

    df = pd.DataFrame({
        "user_id": user_ids,
        "group": group,
        "enroll_date": enroll_date.strftime("%Y-%m-%d"),
        "region": region,
        "area_m2": area.round(1),
        "household_size": household_size,
        "pre_avg_kwh": pre_avg.round(2),
        "wk1_avg_kwh": weekly[0].round(2),
        "wk2_avg_kwh": weekly[1].round(2),
        "wk3_avg_kwh": weekly[2].round(2),
        "wk4_avg_kwh": weekly[3].round(2),
        "post_avg_kwh": post_avg.round(2),
        "app_opens_per_week": app_opens_week,
        "dashboard_interactions": dashboard_interactions,
        "satisfaction": satisfaction,
        "complaints": complaints,
    })

    # ---- عيوب بيانات واقعية متعمدة لاختبار فحوص الجودة ----
    dup = df.sample(10, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)                 # 10 سجلات مكررة
    miss_idx = rng.choice(len(df), size=12, replace=False)
    df.loc[miss_idx, "post_avg_kwh"] = np.nan                    # قراءات عداد مفقودة

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out = generate()
    print(f"Saved {len(out)} rows to {DATA_PATH}")
