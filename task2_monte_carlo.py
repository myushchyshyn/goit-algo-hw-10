"""Завдання 2. Обчислення визначеного інтеграла методом Монте-Карло.

Функція: f(x) = x^2 на відрізку [a, b] = [0, 2].
Аналітичне значення: ∫_0^2 x^2 dx = 2^3 / 3 = 8/3 ≈ 2.6666666667

Запуск (у віртуальному середовищі з numpy, scipy, matplotlib):
    python3 task2_monte_carlo.py
Графік зберігається у task2_plot.png.
"""

import matplotlib

matplotlib.use("Agg")  # збереження у файл без графічного дисплея

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import integrate  # noqa: E402

A = 0.0
B = 2.0
ANALYTIC = B**3 / 3 - A**3 / 3  # 8/3 для f(x) = x^2


def f(x):
    return x**2


def monte_carlo_integral(func, a, b, n, f_max, seed=None):
    """Оцінка інтеграла двома способами та їх стандартні похибки (SE).

    1) Середнє значення (mean value):
           I ≈ (b - a) * mean(f(x_i)),  x_i ~ U[a, b]
    2) Влучення (hit-or-miss) — площа під графіком напряму:
           точки (x_i, y_i) рівномірно в прямокутнику [a, b] × [0, f_max];
           I ≈ (b - a) * f_max * (частка точок під графіком)

    95% довірчий інтервал ≈ ±1.96 * SE. f має бути невід'ємною на [a, b],
    f_max — верхня межа f на [a, b].
    """
    if n <= 0:
        raise ValueError("n має бути додатним")
    rng = np.random.default_rng(seed)

    x = rng.uniform(a, b, n)
    fx = func(x)
    mv_estimate = (b - a) * fx.mean()
    mv_se = (b - a) * fx.std(ddof=0) / np.sqrt(n)

    y = rng.uniform(0.0, f_max, n)
    p = np.mean(y <= func(x))
    box_area = (b - a) * f_max
    hm_estimate = box_area * p
    hm_se = box_area * np.sqrt(p * (1 - p) / n)

    return {
        "n": n,
        "mean_value": mv_estimate,
        "mean_value_se": mv_se,
        "hit_or_miss": hm_estimate,
        "hit_or_miss_se": hm_se,
    }


def save_plot(path="task2_plot.png"):
    """Графік з умови завдання: f(x) і сіра зона під кривою на [a, b]."""
    x = np.linspace(-0.5, 2.5, 400)
    y = f(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, "r", linewidth=2)

    ix = np.linspace(A, B)
    iy = f(ix)
    ax.fill_between(ix, iy, color="gray", alpha=0.3)

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, max(y) + 0.1])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.axvline(x=A, color="gray", linestyle="--")
    ax.axvline(x=B, color="gray", linestyle="--")
    ax.set_title(f"Графік інтегрування f(x) = x^2 від {A:g} до {B:g}")
    ax.grid()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(f"Аналітичне значення ∫_{A:g}^{B:g} x^2 dx = {ANALYTIC:.12f}")

    quad_value, quad_error = integrate.quad(f, A, B)
    print(f"scipy.integrate.quad:          {quad_value:.12f}  (оцінка помилки {quad_error:.2e})\n")

    header = (
        f"{'n':>9} | {'середнє, I':>12} | {'|похибка|':>10} | {'SE':>9} | "
        f"{'влучення, I':>12} | {'|похибка|':>10} | {'SE':>9}"
    )
    print(header)
    print("-" * len(header))
    for n in [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        r = monte_carlo_integral(f, A, B, n, f_max=4.0, seed=42)
        print(
            f"{n:>9} | {r['mean_value']:>12.6f} | {abs(r['mean_value'] - ANALYTIC):>10.6f} | "
            f"{r['mean_value_se']:>9.6f} | {r['hit_or_miss']:>12.6f} | "
            f"{abs(r['hit_or_miss'] - ANALYTIC):>10.6f} | {r['hit_or_miss_se']:>9.6f}"
        )

    final = monte_carlo_integral(f, A, B, 10_000_000, f_max=4.0, seed=42)
    z_mv = abs(final["mean_value"] - ANALYTIC) / final["mean_value_se"]
    z_hm = abs(final["hit_or_miss"] - ANALYTIC) / final["hit_or_miss_se"]
    print(f"\nПеревірка (n = 10 000 000): |I_mv - 8/3| / SE = {z_mv:.2f}, |I_hm - 8/3| / SE = {z_hm:.2f}")
    print(f"Графік збережено: {save_plot()}")
