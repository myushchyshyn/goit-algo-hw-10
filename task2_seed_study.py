"""Дослідження розкиду оцінок Монте-Карло залежно від зерна генератора.

Для кожного n запускаємо оцінювачі з різними зернами і порівнюємо:
  - емпіричний розкид оцінок між запусками (std по зернах);
  - очікуваний розкид, тобто середній SE, який повертає сам метод;
  - частку запусків, де |I − 8/3| > 2·SE (для нормального розподілу ≈ 4.6%).

Запуск:
    python3 task2_seed_study.py
"""

import numpy as np

from task2_monte_carlo import ANALYTIC, A, B, f, monte_carlo_integral

SEEDS = range(500)
SAMPLE_SIZES = [1_000, 100_000, 1_000_000]
F_MAX = 4.0


def run_study(n):
    rows = {"mean_value": [], "hit_or_miss": []}
    ses = {"mean_value": [], "hit_or_miss": []}
    for seed in SEEDS:
        r = monte_carlo_integral(f, A, B, n, f_max=F_MAX, seed=seed)
        for key in rows:
            rows[key].append(r[key])
            ses[key].append(r[f"{key}_se"])
    summary = {}
    for key in rows:
        est = np.array(rows[key])
        se = np.array(ses[key])
        z = np.abs(est - ANALYTIC) / se
        summary[key] = {
            "mean": est.mean(),
            "bias": est.mean() - ANALYTIC,
            "emp_std": est.std(ddof=1),
            "mean_se": se.mean(),
            "min": est.min(),
            "max": est.max(),
            "share_gt2se": np.mean(z > 2),
        }
    return summary


if __name__ == "__main__":
    print(f"Зерна: {len(SEEDS)} шт., точне значення 8/3 = {ANALYTIC:.6f}\n")
    header = (
        f"{'n':>9} | {'метод':<12} | {'середнє':>10} | {'зсув':>10} | {'емп. std':>10} | "
        f"{'середній SE':>11} | {'min..max':>23} | {'|z|>2':>6}"
    )
    print(header)
    print("-" * len(header))
    for n in SAMPLE_SIZES:
        summary = run_study(n)
        for key, label in [("mean_value", "середнє"), ("hit_or_miss", "влучення")]:
            s = summary[key]
            print(
                f"{n:>9} | {label:<12} | {s['mean']:>10.6f} | {s['bias']:>+10.6f} | "
                f"{s['emp_std']:>10.6f} | {s['mean_se']:>11.6f} | "
                f"{s['min']:>10.6f}..{s['max']:<10.6f} | {s['share_gt2se']*100:>5.1f}%"
            )
