"""Завдання 2. Обчислення визначеного інтеграла методом Монте-Карло.

Функція: f(x) = x^2 на відрізку [a, b] = [0, 2].
Аналітичне значення: ∫_0^2 x^2 dx = 2^3 / 3 = 8/3 ≈ 2.6666666667

Запуск:
    python3 task2_monte_carlo.py
Модуль використовує лише стандартну бібліотеку. Якщо встановлено SciPy,
додатково виводиться результат scipy.integrate.quad.
"""

import math
import random

A = 0.0
B = 2.0
ANALYTIC = B**3 / 3 - A**3 / 3  # 8/3 для f(x) = x^2


def f(x):
    return x**2


def monte_carlo_integral(func, a, b, n, f_max=None, seed=None):
    """Оцінка інтеграла двома способами та оцінка похибки.

    1) Метод середнього значення (mean value):
           I ≈ (b - a) * mean(f(x_i)),  x_i ~ U[a, b]
    2) Метод «влучення» (hit-or-miss) — пряма оцінка площі під графіком:
           точки (x_i, y_i) рівномірно в прямокутнику [a, b] × [0, f_max];
           I ≈ площа прямокутника * (частка точок, що лежать під графіком).

    Похибка — стандартна помилка (SE) оцінки. 95% довірчий інтервал ≈ ±1.96 * SE.
    f має бути невід'ємною на [a, b]. f_max, якщо не задано, знаходиться
    на сітці з 10001 точки (точний для монотонних функцій, як x^2 на [0, 2]).
    """
    if n <= 0:
        raise ValueError("n має бути додатним")
    rng = random.Random(seed)

    if f_max is None:
        steps = 10_000
        f_max = max(func(a + (b - a) * i / steps) for i in range(steps + 1))

    box_area = (b - a) * f_max

    sum_f = 0.0
    sum_f2 = 0.0
    hits = 0
    for _ in range(n):
        x = rng.uniform(a, b)
        fx = func(x)
        sum_f += fx
        sum_f2 += fx * fx
        if rng.uniform(0.0, f_max) <= fx:
            hits += 1

    mean = sum_f / n
    variance = max(sum_f2 / n - mean * mean, 0.0)
    mv_estimate = (b - a) * mean
    mv_se = (b - a) * math.sqrt(variance / n)

    p = hits / n
    hm_estimate = box_area * p
    hm_se = box_area * math.sqrt(p * (1 - p) / n)

    return {
        "n": n,
        "mean_value": mv_estimate,
        "mean_value_se": mv_se,
        "hit_or_miss": hm_estimate,
        "hit_or_miss_se": hm_se,
    }


def scipy_quad(func, a, b):
    """Повертає (результат, оцінка помилки) від scipy.integrate.quad або None, якщо SciPy відсутній."""
    try:
        from scipy import integrate
    except ImportError:
        return None
    return integrate.quad(func, a, b)


if __name__ == "__main__":
    print(f"Аналітичне значення ∫_{A:g}^{B:g} x^2 dx = {ANALYTIC:.10f}\n")

    quad = scipy_quad(f, A, B)
    if quad is not None:
        print(f"scipy.integrate.quad:       {quad[0]:.10f}  (помилка {quad[1]:.2e})\n")
    else:
        print("SciPy не встановлено — порівнюємо лише з аналітичним значенням.\n")

    header = (
        f"{'n':>9} | {'середнє, I':>12} | {'|похибка|':>10} | {'SE':>9} | "
        f"{'влучення, I':>12} | {'|похибка|':>10} | {'SE':>9}"
    )
    print(header)
    print("-" * len(header))
    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        r = monte_carlo_integral(f, A, B, n, f_max=4.0, seed=42)
        print(
            f"{n:>9} | {r['mean_value']:>12.6f} | {abs(r['mean_value'] - ANALYTIC):>10.6f} | "
            f"{r['mean_value_se']:>9.6f} | {r['hit_or_miss']:>12.6f} | "
            f"{abs(r['hit_or_miss'] - ANALYTIC):>10.6f} | {r['hit_or_miss_se']:>9.6f}"
        )

    # Перевірка, що оцінка лежить у межах 3 SE від точного значення для великого n
    final = monte_carlo_integral(f, A, B, 1_000_000, f_max=4.0, seed=42)
    print(
        f"\nПеревірка (n = 1 000 000): |I_mv - 8/3| / SE = "
        f"{abs(final['mean_value'] - ANALYTIC) / final['mean_value_se']:.2f}, "
        f"|I_hm - 8/3| / SE = {abs(final['hit_or_miss'] - ANALYTIC) / final['hit_or_miss_se']:.2f}"
    )
