import unittest

from task1_coins import COINS, find_coins_greedy, find_min_coins


def total(coins_dict):
    return sum(coin * count for coin, count in coins_dict.items())


def count_coins(coins_dict):
    return sum(coins_dict.values())


class TestCoins(unittest.TestCase):
    def test_greedy_example_113(self):
        self.assertEqual(find_coins_greedy(113), {50: 2, 10: 1, 2: 1, 1: 1})

    def test_dp_example_113(self):
        self.assertEqual(find_min_coins(113), {1: 1, 2: 1, 10: 1, 50: 2})

    def test_zero(self):
        self.assertEqual(find_coins_greedy(0), {})
        self.assertEqual(find_min_coins(0), {})

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            find_coins_greedy(-1)
        with self.assertRaises(ValueError):
            find_min_coins(-1)

    def test_greedy_equals_dp_for_canonical_coins(self):
        # Для канонічних номіналів жадібний алгоритм оптимальний.
        for amount in range(0, 1001):
            g = find_coins_greedy(amount)
            d = find_min_coins(amount)
            self.assertEqual(total(g), amount)
            self.assertEqual(total(d), amount)
            self.assertEqual(count_coins(g), count_coins(d), msg=f"сума {amount}")

    def test_non_canonical_greedy_can_be_worse(self):
        # Жадібний дає 4+1+1 (3 монети), DP знаходить 3+3 (2 монети).
        self.assertEqual(count_coins(find_coins_greedy(6, [1, 3, 4])), 3)
        self.assertEqual(count_coins(find_min_coins(6, [1, 3, 4])), 2)

    def test_impossible_amount_raises(self):
        with self.assertRaises(ValueError):
            find_coins_greedy(3, [5, 2])
        with self.assertRaises(ValueError):
            find_min_coins(3, [5, 2])

    def test_default_coins(self):
        self.assertEqual(COINS, [50, 25, 10, 5, 2, 1])


if __name__ == "__main__":
    unittest.main()
