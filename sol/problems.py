"""Problem pools for the signs-of-life gauntlet.

Hard pool: classic LC-hard / CF-style algorithmic problems, execution-checked.
Easy pool: trivial functions, used for the length-matched easy-coding control.
Every problem has a reference solution; run this file to self-test them.
"""
from dataclasses import dataclass, field
from typing import Any, Callable
import random


@dataclass
class Problem:
    pid: str
    difficulty: str  # "hard" | "easy"
    func: str
    prompt: str
    tests: list  # list of (args_tuple, expected)
    ref: Callable = None


PROBLEMS: list[Problem] = []


def P(pid, difficulty, func, prompt, tests, ref):
    PROBLEMS.append(Problem(pid, difficulty, func, prompt, tests, ref))


# ---------------------------------------------------------------- hard pool

def _ref_distinct_subseq(s):
    MOD = 10**9 + 7
    dp = 1
    last = {}
    for ch in s:
        new = (2 * dp) % MOD
        if ch in last:
            new = (new - last[ch]) % MOD
        last[ch] = dp
        dp = new
    return (dp - 1) % MOD

P("distinct_subseq", "hard", "count_distinct_subsequences",
  "Write a Python function `count_distinct_subsequences(s: str) -> int` that returns the "
  "number of distinct non-empty subsequences of `s`, modulo 10**9 + 7. "
  "`s` consists of lowercase letters, len(s) up to 10**5, so the solution must be O(n).",
  [(("abc",), 7), (("aba",), 6), (("aaa",), 3), (("",), 0),
   (("abcdefghijklmnopqrstuvwxyz" * 100,), _ref_distinct_subseq("abcdefghijklmnopqrstuvwxyz" * 100)),
   (("abab" * 500,), _ref_distinct_subseq("abab" * 500))],
  _ref_distinct_subseq)


def _ref_min_window(s, t):
    from collections import Counter
    if not t or not s:
        return ""
    need = Counter(t)
    missing = len(t)
    left = start = end = 0
    for right, ch in enumerate(s, 1):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        if missing == 0:
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if end == 0 or right - left < end - start:
                start, end = left, right
            need[s[left]] += 1
            missing += 1
            left += 1
    return s[start:end]

P("min_window", "hard", "min_window",
  "Write a Python function `min_window(s: str, t: str) -> str` that returns the minimum "
  "window substring of `s` containing every character of `t` (with multiplicity). "
  "Return \"\" if no such window exists. Must run in O(len(s) + len(t)).",
  [(("ADOBECODEBANC", "ABC"), "BANC"), (("a", "a"), "a"), (("a", "aa"), ""),
   (("aaflslflsldkalskaaa", "aaa"), "aaa"),
   (("ab" * 2000 + "c", "abc"), _ref_min_window("ab" * 2000 + "c", "abc"))],
  _ref_min_window)


def _ref_count_range_sums(nums, lower, upper):
    import bisect
    pre = [0]
    for x in nums:
        pre.append(pre[-1] + x)

    def sort_count(lo, hi):
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        cnt = sort_count(lo, mid) + sort_count(mid, hi)
        j = k = mid
        for left_i in range(lo, mid):
            while j < hi and pre[j] - pre[left_i] < lower:
                j += 1
            while k < hi and pre[k] - pre[left_i] <= upper:
                k += 1
            cnt += k - j
        pre[lo:hi] = sorted(pre[lo:hi])
        return cnt

    return sort_count(0, len(pre))

_rng = random.Random(7)
_big = [_rng.randint(-1000, 1000) for _ in range(3000)]
P("count_range_sums", "hard", "count_range_sums",
  "Write a Python function `count_range_sums(nums: list[int], lower: int, upper: int) -> int` "
  "that returns the number of contiguous subarrays whose sum lies in [lower, upper] inclusive. "
  "len(nums) up to 10**5 with values in [-10**9, 10**9]; O(n^2) is too slow, aim for O(n log n).",
  [(([-2, 5, -1], -2, 2), 3), (([0], 0, 0), 1), (([1, -1, 1, -1], 0, 0), 4),
   ((_big, -50, 50), _ref_count_range_sums(list(_big), -50, 50))],
  _ref_count_range_sums)


def _ref_decode_star(s):
    MOD = 10**9 + 7
    if not s:
        return 0
    prev2, prev1 = 1, 0
    prev_ch = ''
    prev1 = 9 if s[0] == '*' else (1 if s[0] != '0' else 0)
    prev_ch = s[0]
    for i in range(1, len(s)):
        ch = s[i]
        cur = 0
        if ch == '*':
            cur += 9 * prev1
        elif ch != '0':
            cur += prev1
        a, b = prev_ch, ch
        if a == '*':
            if b == '*':
                cur += 15 * prev2
            elif b <= '6':
                cur += 2 * prev2
            else:
                cur += prev2
        elif a == '1':
            cur += (9 if b == '*' else 1) * prev2
        elif a == '2':
            if b == '*':
                cur += 6 * prev2
            elif b <= '6':
                cur += prev2
        prev2, prev1 = prev1, cur % MOD
        prev_ch = ch
    return prev1

P("decode_star", "hard", "num_decodings",
  "A message of digits was encoded with A=1..Z=26. The string may also contain '*', which "
  "represents any digit from '1' to '9'. Write `num_decodings(s: str) -> int` returning the "
  "number of ways to decode `s`, modulo 10**9 + 7. len(s) up to 10**5.",
  [(("*",), 9), (("1*",), 18), (("2*",), 15), (("**",), 96), (("0",), 0), (("*0",), 2),
   (("*1*0" * 200,), _ref_decode_star("*1*0" * 200))],
  _ref_decode_star)


def _ref_shortest_subarray(nums, k):
    from collections import deque
    pre = [0]
    for x in nums:
        pre.append(pre[-1] + x)
    dq = deque()
    ans = len(nums) + 1
    for i, p in enumerate(pre):
        while dq and p - pre[dq[0]] >= k:
            ans = min(ans, i - dq.popleft())
        while dq and pre[dq[-1]] >= p:
            dq.pop()
        dq.append(i)
    return ans if ans <= len(nums) else -1

_big2 = [_rng.randint(-30, 100) for _ in range(5000)]
P("shortest_subarray_k", "hard", "shortest_subarray",
  "Write a Python function `shortest_subarray(nums: list[int], k: int) -> int` returning the "
  "length of the shortest non-empty contiguous subarray with sum >= k, or -1 if none exists. "
  "`nums` may contain negative numbers. len(nums) up to 10**5; O(n^2) is too slow.",
  [(([1], 1), 1), (([1, 2], 4), -1), (([2, -1, 2], 3), 3), (([84, -37, 32, 40, 95], 167), 3),
   ((_big2, 5000), _ref_shortest_subarray(list(_big2), 5000))],
  _ref_shortest_subarray)


def _ref_count_smaller(nums):
    res = [0] * len(nums)
    order = list(range(len(nums)))

    def merge(idx):
        n = len(idx)
        if n <= 1:
            return idx
        left = merge(idx[:n // 2])
        right = merge(idx[n // 2:])
        merged = []
        j = 0
        for i in left:
            while j < len(right) and nums[right[j]] < nums[i]:
                merged.append(right[j])
                j += 1
            res[i] += j
            merged.append(i)
        merged.extend(right[j:])
        return merged

    merge(order)
    return res

_big3 = [_rng.randint(-10**4, 10**4) for _ in range(4000)]
P("count_smaller", "hard", "count_smaller",
  "Write a Python function `count_smaller(nums: list[int]) -> list[int]` where output[i] is "
  "the number of elements to the right of nums[i] that are strictly smaller than nums[i]. "
  "len(nums) up to 10**5; O(n^2) is too slow, aim for O(n log n).",
  [(([5, 2, 6, 1],), [2, 1, 1, 0]), (([-1],), [0]), (([-1, -1],), [0, 0]),
   ((_big3,), _ref_count_smaller(list(_big3)))],
  _ref_count_smaller)


def _ref_super_egg(k, n):
    moves = 0
    dp = [0] * (k + 1)
    while dp[k] < n:
        moves += 1
        for i in range(k, 0, -1):
            dp[i] = dp[i] + dp[i - 1] + 1
    return moves

P("super_egg_drop", "hard", "super_egg_drop",
  "You have `k` identical eggs and a building with `n` floors. There is a critical floor F "
  "(0 <= F <= n) such that eggs dropped from above F break and from F or below do not. "
  "Write `super_egg_drop(k: int, n: int) -> int` returning the minimum number of drops that "
  "guarantees identifying F in the worst case. k up to 100, n up to 10**4; must be efficient.",
  [((1, 2), 2), ((2, 6), 3), ((3, 14), 4), ((2, 100), 14), ((100, 10000), 14),
   ((4, 5000), _ref_super_egg(4, 5000))],
  _ref_super_egg)


def _ref_min_cut(s):
    n = len(s)
    if n == 0:
        return 0
    pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i < 2 or pal[i + 1][j - 1]):
                pal[i][j] = True
    cut = [0] * n
    for j in range(n):
        if pal[0][j]:
            cut[j] = 0
        else:
            cut[j] = min(cut[i] + 1 for i in range(j) if pal[i + 1][j])
    return cut[-1]

P("palindrome_min_cut", "hard", "min_cut",
  "Write a Python function `min_cut(s: str) -> int` returning the minimum number of cuts "
  "needed to partition `s` so that every part is a palindrome. len(s) up to 2000.",
  [(("aab",), 1), (("a",), 0), (("ab",), 1), (("abcba",), 0),
   (("abccbaxyzzyx",), 1), (("ab" * 300,), _ref_min_cut("ab" * 300))],
  _ref_min_cut)


def _ref_kth_mult_table(m, n, k):
    lo, hi = 1, m * n
    while lo < hi:
        mid = (lo + hi) // 2
        cnt = sum(min(mid // i, n) for i in range(1, m + 1))
        if cnt >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo

P("kth_mult_table", "hard", "find_kth_number",
  "Write a Python function `find_kth_number(m: int, n: int, k: int) -> int` returning the "
  "k-th smallest number in the m x n multiplication table (table[i][j] = i*j, 1-indexed). "
  "m, n up to 3*10**4, so you cannot materialize the table.",
  [((3, 3, 5), 3), ((2, 3, 6), 6), ((1, 1, 1), 1), ((9895, 28405, 100787757), 31666344),
   ((30000, 30000, 450000000), _ref_kth_mult_table(30000, 30000, 450000000))],
  _ref_kth_mult_table)


def _ref_russian_doll(envelopes):
    import bisect
    envelopes = sorted(envelopes, key=lambda e: (e[0], -e[1]))
    tails = []
    for _, h in envelopes:
        i = bisect.bisect_left(tails, h)
        if i == len(tails):
            tails.append(h)
        else:
            tails[i] = h
    return len(tails)

_envs = [[_rng.randint(1, 500), _rng.randint(1, 500)] for _ in range(4000)]
P("russian_doll", "hard", "max_envelopes",
  "Write a Python function `max_envelopes(envelopes: list[list[int]]) -> int`. Each envelope "
  "is [w, h]; envelope A fits inside B iff A's width and height are both strictly smaller. "
  "Return the maximum number of envelopes you can nest. Up to 10**5 envelopes; O(n^2) too slow.",
  [(([[5, 4], [6, 4], [6, 7], [2, 3]],), 3), (([[1, 1], [1, 1]],), 1),
   (([[1, 2]],), 1), ((_envs,), _ref_russian_doll([list(e) for e in _envs]))],
  _ref_russian_doll)


# ---------------------------------------------------------------- easy pool

P("reverse_words", "easy", "reverse_words",
  "Write a Python function `reverse_words(s: str) -> str` that returns the words of `s` in "
  "reverse order, joined by single spaces (words are separated by whitespace).",
  [(("hello world",), "world hello"), (("a b c",), "c b a"), (("one",), "one")],
  lambda s: " ".join(s.split()[::-1]))

P("sum_evens", "easy", "sum_evens",
  "Write a Python function `sum_evens(nums: list[int]) -> int` returning the sum of the even "
  "numbers in the list.",
  [(([1, 2, 3, 4],), 6), (([],), 0), (([-2, 5],), -2)],
  lambda nums: sum(x for x in nums if x % 2 == 0))

P("is_palindrome", "easy", "is_palindrome",
  "Write a Python function `is_palindrome(s: str) -> bool` that returns True iff `s` reads the "
  "same forwards and backwards, ignoring case (do not ignore spaces or punctuation).",
  [(("Level",), True), (("abc",), False), (("",), True)],
  lambda s: s.lower() == s.lower()[::-1])

P("count_vowels", "easy", "count_vowels",
  "Write a Python function `count_vowels(s: str) -> int` counting the vowels (aeiou, either "
  "case) in `s`.",
  [(("hello",), 2), (("XYZ",), 0), (("AeIoU",), 5)],
  lambda s: sum(c in "aeiouAEIOU" for c in s))

P("fizzbuzz_str", "easy", "fizzbuzz",
  "Write a Python function `fizzbuzz(n: int) -> list[str]` returning the classic FizzBuzz list "
  "for 1..n: multiples of 3 -> 'Fizz', of 5 -> 'Buzz', of both -> 'FizzBuzz', else the number "
  "as a string.",
  [((5,), ["1", "2", "Fizz", "4", "Buzz"]), ((1,), ["1"]),
   ((15,), ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"])],
  lambda n: [("FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i)) for i in range(1, n + 1)])

P("second_largest", "easy", "second_largest",
  "Write a Python function `second_largest(nums: list[int]) -> int` returning the second "
  "largest distinct value in the list (the list always has at least two distinct values).",
  [(([1, 3, 2],), 2), (([5, 5, 4],), 4), (([-1, -2],), -2)],
  lambda nums: sorted(set(nums))[-2])

P("factorial", "easy", "factorial",
  "Write a Python function `factorial(n: int) -> int` returning n! for n >= 0.",
  [((0,), 1), ((5,), 120), ((10,), 3628800)],
  lambda n: 1 if n == 0 else n * __import__("math").factorial(n - 1))

P("merge_dicts", "easy", "merge_sum",
  "Write a Python function `merge_sum(a: dict, b: dict) -> dict` returning a new dict whose "
  "keys are the union of a and b, with values summed where a key appears in both.",
  [(({"x": 1}, {"x": 2, "y": 3}), {"x": 3, "y": 3}), (({}, {}), {}), (({"a": 1}, {}), {"a": 1})],
  lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)})

P("running_max", "easy", "running_max",
  "Write a Python function `running_max(nums: list[int]) -> list[int]` where output[i] is the "
  "maximum of nums[0..i].",
  [(([1, 3, 2, 5],), [1, 3, 3, 5]), (([],), []), (([4],), [4])],
  lambda nums: [max(nums[:i + 1]) for i in range(len(nums))])

P("caesar", "easy", "caesar",
  "Write a Python function `caesar(s: str, k: int) -> str` shifting each lowercase letter of "
  "`s` forward by k positions in the alphabet, wrapping around; other characters unchanged.",
  [(("abc", 1), "bcd"), (("xyz", 3), "abc"), (("a b!", 2), "c d!")],
  lambda s, k: "".join(chr((ord(c) - 97 + k) % 26 + 97) if c.islower() else c for c in s))

P("flatten_once", "easy", "flatten_once",
  "Write a Python function `flatten_once(lst: list) -> list` flattening exactly one level of "
  "nesting: each element that is itself a list gets its items spliced in; other elements are "
  "kept as-is.",
  [(([[1, 2], [3]],), [1, 2, 3]), (([1, [2, 3], 4],), [1, 2, 3, 4]), (([],), [])],
  lambda lst: [y for x in lst for y in (x if isinstance(x, list) else [x])])

P("char_freq_top", "easy", "most_common_char",
  "Write a Python function `most_common_char(s: str) -> str` returning the most frequent "
  "character in the non-empty string `s`; break ties by earliest first occurrence in `s`.",
  [(("aabbb",), "b"), (("abab",), "a"), (("z",), "z")],
  lambda s: max(sorted(set(s), key=s.index), key=s.count))

P("digits_sum", "easy", "digit_sum",
  "Write a Python function `digit_sum(n: int) -> int` returning the sum of the decimal digits "
  "of the non-negative integer n.",
  [((0,), 0), ((1234,), 10), ((999,), 27)],
  lambda n: sum(int(d) for d in str(n)))

P("unique_sorted", "easy", "unique_sorted",
  "Write a Python function `unique_sorted(nums: list[int]) -> list[int]` returning the "
  "distinct values of nums in ascending order.",
  [(([3, 1, 3, 2],), [1, 2, 3]), (([],), []), (([5, 5],), [5])],
  lambda nums: sorted(set(nums)))

P("title_case", "easy", "title_case",
  "Write a Python function `title_case(s: str) -> str` that capitalizes the first letter of "
  "each whitespace-separated word and lowercases the rest, joining with single spaces.",
  [(("hello WORLD",), "Hello World"), (("a",), "A"), (("foo  bar",), "Foo Bar")],
  lambda s: " ".join(w.capitalize() for w in s.split()))

P("clamp_list", "easy", "clamp_list",
  "Write a Python function `clamp_list(nums: list[int], lo: int, hi: int) -> list[int]` "
  "clamping each value into the inclusive range [lo, hi].",
  [(([1, 10, -5], 0, 5), [1, 5, 0]), (([], 0, 1), []), (([3], 3, 3), [3])],
  lambda nums, lo, hi: [min(max(x, lo), hi) for x in nums])


HARD = [p for p in PROBLEMS if p.difficulty == "hard"]
EASY = [p for p in PROBLEMS if p.difficulty == "easy"]


def self_test():
    for p in PROBLEMS:
        for args, expected in p.tests:
            import copy
            got = p.ref(*copy.deepcopy(list(args)))
            assert got == expected, f"{p.pid}{args if len(repr(args)) < 60 else ''}: ref gave {got!r}, expected {expected!r}"
    print(f"self-test OK: {len(HARD)} hard, {len(EASY)} easy problems")


if __name__ == "__main__":
    self_test()
