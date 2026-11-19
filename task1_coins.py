"""Завдання 1. Видача решти монетами: жадібний алгоритм та динамічне програмування.

Запуск:
    python3 task1_coins.py
"""

import time

COINS = [50, 25, 10, 5, 2, 1]


def find_coins_greedy(amount, coins=COINS):
    """Жадібний алгоритм: беремо найбільший номінал, поки сума не менша за нього.

    Повертає словник {номінал: кількість}, лише для номіналів, що використані.
    Для канонічної системи номіналів (як у COINS) результат оптимальний.
    """
    if amount < 0:
        raise ValueError("Сума не може бути від'ємною")

    result = {}
    remaining = amount
    for coin in sorted(coins, reverse=True):
        if remaining == 0:
            break
        count, remaining = divmod(remaining, coin)
        if count:
            result[coin] = count
    if remaining:
        raise ValueError(f"Неможливо видати суму {amount} заданими номіналами")
    return result


def find_min_coins(amount, coins=COINS):
    """Динамічне програмування: мінімальна кількість монет для суми amount.

    min_coins[v]  — мінімальна кількість монет для суми v;
    last_coin[v]  — останній номінал, який дає цей мінімум (для відновлення відповіді).
    Повертає словник {номінал: кількість}, відсортований за номіналом.
    """
    if amount < 0:
        raise ValueError("Сума не може бути від'ємною")

    inf = float("inf")
    min_coins = [0] + [inf] * amount
    last_coin = [0] * (amount + 1)

    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value and min_coins[value - coin] + 1 < min_coins[value]:
                min_coins[value] = min_coins[value - coin] + 1
                last_coin[value] = coin

    if amount and min_coins[amount] == inf:
        raise ValueError(f"Неможливо видати суму {amount} заданими номіналами")

    result = {}
    value = amount
    while value > 0:
        coin = last_coin[value]
        result[coin] = result.get(coin, 0) + 1
        value -= coin
    return dict(sorted(result.items()))


def _benchmark(func, amount, repeats=3):
    """Найкращий час (с) з кількох запусків."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        func(amount)
        best = min(best, time.perf_counter() - start)
    return best


if __name__ == "__main__":
    print("Приклад для суми 113:")
    print("  жадібний:            ", find_coins_greedy(113))
    print("  динамічне програмув.:", find_min_coins(113))

    print("\nКонтрпример для НЕканонічних номіналів [1, 3, 4], сума 6:")
    print("  жадібний:            ", find_coins_greedy(6, [1, 3, 4]))
    print("  динамічне програмув.:", find_min_coins(6, [1, 3, 4]))

    print("\nЧас виконання (номінали за замовчуванням):")
    print(f"{'сума':>10} | {'жадібний, с':>12} | {'DP, с':>12}")
    for amount in [10**3, 10**4, 10**5, 10**6]:
        t_greedy = _benchmark(find_coins_greedy, amount)
        t_dp = _benchmark(find_min_coins, amount, repeats=1)
        print(f"{amount:>10} | {t_greedy:>12.8f} | {t_dp:>12.6f}")
