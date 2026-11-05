"""The CQ test's problems (coding qualification), written for Aegist, 2026-11-05.

Each problem: id, section, prompt (what the brain is asked), and either
  - "tests": Python asserts run against the brain's code (code problems), or
  - "gold": the exact expected output (code-reading problems: "what does this print?").
"ref" is a reference solution: `aegist cq --selftest` checks every test passes on it, so a wrong test
can't silently mark right answers wrong.
"""
from __future__ import annotations

PROBLEMS: list[dict] = []


def code(pid, section, prompt, tests, ref):
    PROBLEMS.append({"id": pid, "section": section, "prompt": prompt.strip(), "tests": tests.strip(), "ref": ref.strip()})


def read(pid, prompt, gold):
    PROBLEMS.append({"id": pid, "section": "reading", "prompt": prompt.strip(), "gold": gold})


# ---------------------------------------------------------------- basics
code("rev_words", "basics", """
Write a Python function `reverse_words(s: str) -> str` that reverses the order of the words in s.
Words are separated by any amount of whitespace; the result has single spaces and no leading or trailing space.
""", """
assert reverse_words("hello world") == "world hello"
assert reverse_words("  a   b  c ") == "c b a"
assert reverse_words("") == ""
assert reverse_words("one") == "one"
""", """
def reverse_words(s):
    return " ".join(reversed(s.split()))
""")

code("vowels", "basics", """
Write a Python function `count_vowels(s: str) -> dict` that returns how many times each vowel (a, e, i, o, u)
appears in s, ignoring case. Every vowel is a key, even with count 0.
""", """
assert count_vowels("Education") == {"a": 1, "e": 1, "i": 1, "o": 1, "u": 1}
assert count_vowels("xyz") == {"a": 0, "e": 0, "i": 0, "o": 0, "u": 0}
assert count_vowels("AAaa ee") == {"a": 4, "e": 2, "i": 0, "o": 0, "u": 0}
""", """
def count_vowels(s):
    s = s.lower()
    return {v: s.count(v) for v in "aeiou"}
""")

code("chunks", "basics", """
Write a Python function `chunk(xs: list, n: int) -> list` that splits xs into consecutive lists of length n;
the last one may be shorter. Raise ValueError if n < 1.
""", """
assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
assert chunk([], 3) == []
assert chunk([1, 2, 3], 3) == [[1, 2, 3]]
try:
    chunk([1], 0)
    assert False
except ValueError:
    pass
""", """
def chunk(xs, n):
    if n < 1:
        raise ValueError("n must be at least 1")
    return [xs[i:i + n] for i in range(0, len(xs), n)]
""")

code("palin", "basics", """
Write a Python function `is_palindrome(s: str) -> bool` that says whether s reads the same backwards,
considering only letters and digits and ignoring case.
""", """
assert is_palindrome("A man, a plan, a canal: Panama")
assert not is_palindrome("race a car")
assert is_palindrome("")
assert is_palindrome("No 'x' in Nixon")
assert not is_palindrome("ab")
""", """
def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]
""")

code("flatten", "basics", """
Write a Python function `flatten(x) -> list` that flattens arbitrarily nested lists (and tuples) into one flat
list, keeping order. Strings are not split.
""", """
assert flatten([1, [2, [3, [4]], 5]]) == [1, 2, 3, 4, 5]
assert flatten([]) == []
assert flatten([["ab", ("c", [])], "d"]) == ["ab", "c", "d"]
""", """
def flatten(x):
    out = []
    for v in x:
        if isinstance(v, (list, tuple)):
            out.extend(flatten(v))
        else:
            out.append(v)
    return out
""")

code("fizz", "basics", """
Write a Python function `fizzbuzz(n: int) -> list` returning the strings for 1..n: "Fizz" for multiples of 3,
"Buzz" for multiples of 5, "FizzBuzz" for both, otherwise the number as a string.
""", """
assert fizzbuzz(15)[-1] == "FizzBuzz"
assert fizzbuzz(5) == ["1", "2", "Fizz", "4", "Buzz"]
assert fizzbuzz(0) == []
""", """
def fizzbuzz(n):
    return ["FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i) for i in range(1, n + 1)]
""")

code("caesar", "basics", """
Write a Python function `caesar(s: str, k: int) -> str` that shifts every letter k places along the alphabet,
wrapping around and keeping case; other characters are unchanged. k may be negative or larger than 26.
""", """
assert caesar("abc xyz", 3) == "def abc"
assert caesar("Hello, World!", 13) == "Uryyb, Jbeyq!"
assert caesar("def", -3) == "abc"
assert caesar("a", 53) == "b"
""", """
def caesar(s, k):
    out = []
    for c in s:
        if c.isascii() and c.isalpha():
            base = ord("A") if c.isupper() else ord("a")
            out.append(chr((ord(c) - base + k) % 26 + base))
        else:
            out.append(c)
    return "".join(out)
""")

code("second", "basics", """
Write a Python function `second_largest(xs: list[int])` that returns the second largest DISTINCT value in xs,
or None if there isn't one.
""", """
assert second_largest([3, 1, 4, 4, 2]) == 3
assert second_largest([5, 5]) is None
assert second_largest([]) is None
assert second_largest([-1, -2]) == -2
""", """
def second_largest(xs):
    s = sorted(set(xs))
    return s[-2] if len(s) >= 2 else None
""")

# ---------------------------------------------------------------- algorithms
code("bsearch", "algorithms", """
Write a Python function `first_at_least(xs: list[int], t: int) -> int` for a sorted list xs: return the index
of the first element >= t, or len(xs) if none. It must run in O(log n) (don't use the bisect module).
""", """
assert first_at_least([1, 3, 3, 5], 3) == 1
assert first_at_least([1, 3, 3, 5], 4) == 3
assert first_at_least([1, 3, 3, 5], 9) == 4
assert first_at_least([], 1) == 0
assert first_at_least([2, 2, 2], 1) == 0
import random
r = random.Random(1)
for _ in range(200):
    xs = sorted(r.randint(0, 20) for _ in range(r.randint(0, 15)))
    t = r.randint(-1, 21)
    assert first_at_least(xs, t) == next((i for i, v in enumerate(xs) if v >= t), len(xs))
""", """
def first_at_least(xs, t):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo
""")

code("intervals", "algorithms", """
Write a Python function `merge_intervals(iv: list[list[int]]) -> list[list[int]]` that merges overlapping
closed intervals [start, end] (touching ones like [1,2],[2,3] merge too) and returns them sorted by start.
""", """
assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
assert merge_intervals([]) == []
assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]
assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
""", """
def merge_intervals(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
""")

code("edit", "algorithms", """
Write a Python function `edit_distance(a: str, b: str) -> int` returning the Levenshtein distance
(insertions, deletions and substitutions each cost 1).
""", """
assert edit_distance("kitten", "sitting") == 3
assert edit_distance("", "abc") == 3
assert edit_distance("flaw", "lawn") == 2
assert edit_distance("same", "same") == 0
""", """
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
""")

code("lis", "algorithms", """
Write a Python function `lis_length(xs: list[int]) -> int` returning the length of the longest strictly
increasing subsequence. It must handle 10,000 numbers in well under a second.
""", """
assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert lis_length([]) == 0
assert lis_length([7, 7, 7]) == 1
import random, time
r = random.Random(2)
xs = [r.randint(0, 10**6) for _ in range(10000)]
t = time.perf_counter()
assert lis_length(list(range(10000))) == 10000
assert lis_length(xs) > 100
assert time.perf_counter() - t < 1.0, "too slow (needs O(n log n))"
""", """
from bisect import bisect_left
def lis_length(xs):
    tails = []
    for x in xs:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
""")

code("topo", "algorithms", """
Write a Python function `build_order(tasks: list[str], deps: list[tuple[str, str]]) -> list[str]` where
(a, b) in deps means a must come before b. Return an order containing every task that respects all deps;
when several tasks are ready, take them in alphabetical order. Raise ValueError if there is a cycle.
""", """
assert build_order(["c", "b", "a"], [("a", "b"), ("b", "c")]) == ["a", "b", "c"]
assert build_order(["x", "y", "z"], []) == ["x", "y", "z"]
assert build_order(["a", "b", "c", "d"], [("d", "a"), ("c", "a")]) == ["b", "c", "d", "a"]
try:
    build_order(["a", "b"], [("a", "b"), ("b", "a")])
    assert False
except ValueError:
    pass
""", """
import heapq
def build_order(tasks, deps):
    indeg = {t: 0 for t in tasks}
    nxt = {t: [] for t in tasks}
    for a, b in deps:
        nxt[a].append(b)
        indeg[b] += 1
    ready = [t for t in tasks if indeg[t] == 0]
    heapq.heapify(ready)
    out = []
    while ready:
        t = heapq.heappop(ready)
        out.append(t)
        for b in nxt[t]:
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(ready, b)
    if len(out) != len(tasks):
        raise ValueError("cycle")
    return out
""")

code("dijkstra", "algorithms", """
Write a Python function `shortest(n: int, edges: list[tuple[int, int, int]], s: int) -> list` for a directed
graph with nodes 0..n-1 and edges (u, v, w) with w >= 0: return the shortest distance from s to every node,
with None for nodes that can't be reached.
""", """
assert shortest(4, [(0, 1, 1), (1, 2, 2), (0, 2, 5)], 0) == [0, 1, 3, None]
assert shortest(1, [], 0) == [0]
assert shortest(3, [(0, 1, 0), (1, 2, 0)], 0) == [0, 0, 0]
assert shortest(3, [(1, 0, 1)], 0) == [0, None, None]
""", """
import heapq
def shortest(n, edges, s):
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [None] * n
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if dist[u] is not None:
            continue
        dist[u] = d
        for v, w in adj[u]:
            if dist[v] is None:
                heapq.heappush(pq, (d + w, v))
    return dist
""")

code("coins", "algorithms", """
Write a Python function `min_coins(coins: list[int], amount: int) -> int` returning the fewest coins that add
up to amount (each coin can be used any number of times), or -1 if it can't be done.
""", """
assert min_coins([1, 2, 5], 11) == 3
assert min_coins([2], 3) == -1
assert min_coins([1], 0) == 0
assert min_coins([1, 3, 4], 6) == 2
assert min_coins([186, 419, 83, 408], 6249) == 20
""", """
def min_coins(coins, amount):
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] < INF else -1
""")

code("islands", "algorithms", """
Write a Python function `count_islands(grid: list[str]) -> int` where grid rows are strings of "#" (land) and
"." (water). Count groups of land connected up/down/left/right. Grids can be 300x300: avoid deep recursion.
""", """
assert count_islands(["##..", "#..#", "...#"]) == 2
assert count_islands([]) == 0
assert count_islands(["#.#", ".#.", "#.#"]) == 5
assert count_islands(["#" * 300] * 300) == 1
""", """
def count_islands(grid):
    seen = set()
    n = 0
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == "#" and (r, c) not in seen:
                n += 1
                stack = [(r, c)]
                seen.add((r, c))
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < len(grid) and 0 <= nx < len(grid[ny]) and grid[ny][nx] == "#" and (ny, nx) not in seen:
                            seen.add((ny, nx))
                            stack.append((ny, nx))
    return n
""")

# ---------------------------------------------------------------- data structures
code("lru", "structures", """
Write a Python class `LRUCache` with `__init__(self, capacity: int)`, `get(self, key)` returning the value or -1,
and `put(self, key, value)`. When full, putting a new key evicts the least recently used key (get and put both
count as use).
""", """
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
c.put(4, 4)
assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4
c.put(3, 30)
c.put(5, 5)
assert c.get(4) == -1 and c.get(3) == 30
""", """
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.cap, self.d = capacity, OrderedDict()
    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)
        return self.d[key]
    def put(self, key, value):
        self.d[key] = value
        self.d.move_to_end(key)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
""")

code("minstack", "structures", """
Write a Python class `MinStack` with `push(x)`, `pop()` (returns the value), `top()` and `get_min()`, all O(1).
""", """
s = MinStack()
s.push(3); s.push(1); s.push(2); s.push(1)
assert s.get_min() == 1
assert s.pop() == 1
assert s.get_min() == 1
s.pop(); s.pop()
assert s.get_min() == 3 and s.top() == 3
""", """
class MinStack:
    def __init__(self):
        self.s = []
    def push(self, x):
        self.s.append((x, min(x, self.s[-1][1]) if self.s else x))
    def pop(self):
        return self.s.pop()[0]
    def top(self):
        return self.s[-1][0]
    def get_min(self):
        return self.s[-1][1]
""")

code("trie", "structures", """
Write a Python class `Trie` with `insert(word)`, `contains(word) -> bool` and `count_prefix(prefix) -> int`
(how many inserted words start with prefix; inserting the same word twice counts once).
""", """
t = Trie()
for w in ["apple", "app", "apply", "bat", "app"]:
    t.insert(w)
assert t.contains("app") and not t.contains("ap")
assert t.count_prefix("app") == 3
assert t.count_prefix("") == 4
assert t.count_prefix("c") == 0
""", """
class Trie:
    def __init__(self):
        self.root = {}
        self.words = set()
    def insert(self, word):
        if word in self.words:
            return
        self.words.add(word)
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {"#n": 0})
            node["#n"] += 1
        node["#end"] = True
    def contains(self, word):
        return word in self.words
    def count_prefix(self, prefix):
        if not prefix:
            return len(self.words)
        node = self.root
        for ch in prefix:
            if ch not in node:
                return 0
            node = node[ch]
        return node["#n"]
""")

code("uf", "structures", """
Write a Python class `DisjointSet` with `__init__(self, n)`, `union(a, b) -> bool` (False if already joined),
`same(a, b) -> bool` and `groups() -> int` (current number of separate groups).
""", """
d = DisjointSet(5)
assert d.groups() == 5
assert d.union(0, 1) and d.union(3, 4)
assert not d.union(1, 0)
assert d.same(0, 1) and not d.same(1, 3)
assert d.groups() == 3
d.union(1, 4)
assert d.same(0, 3) and d.groups() == 2
""", """
class DisjointSet:
    def __init__(self, n):
        self.p = list(range(n))
        self.g = n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        self.g -= 1
        return True
    def same(self, a, b):
        return self.find(a) == self.find(b)
    def groups(self):
        return self.g
""")

code("topk", "structures", """
Write a Python function `top_k_words(text: str, k: int) -> list[str]` returning the k most frequent words
(lowercase, words are runs of letters a-z), most frequent first, ties broken alphabetically.
""", """
assert top_k_words("the cat and the hat and the bat", 2) == ["the", "and"]
assert top_k_words("b a c b a", 3) == ["a", "b", "c"]
assert top_k_words("Hello, hello! HELLO world", 1) == ["hello"]
assert top_k_words("", 2) == []
""", """
import re
from collections import Counter
def top_k_words(text, k):
    c = Counter(re.findall(r"[a-z]+", text.lower()))
    return [w for w, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
""")

# ---------------------------------------------------------------- bug fixing
code("fix_avg", "bugfix", """
This function should return the average of a list, or 0.0 for an empty list, but it has a bug. Return the fixed
function (same name).

def average(xs):
    total = 0
    for x in xs:
        total += x
    return total / len(xs) if xs else 0
""", """
assert average([1, 2, 3, 4]) == 2.5
assert average([]) == 0.0 and isinstance(average([]), float)
assert average([5]) == 5
""", """
def average(xs):
    return sum(xs) / len(xs) if xs else 0.0
""")

code("fix_default", "bugfix", """
This function should add an item to a list and return the list, starting a NEW empty list each call when none
is given. It has a bug: calls share one list. Return the fixed function (same name).

def add_item(item, items=[]):
    items.append(item)
    return items
""", """
assert add_item(1) == [1]
assert add_item(2) == [2]
a = [9]
assert add_item(3, a) == [9, 3] and a == [9, 3]
""", """
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
""")

code("fix_range", "bugfix", """
This function should return the sum of the integers from 1 to n inclusive, but it is off by one. Fix it
(same name).

def sum_to(n):
    return sum(range(1, n))
""", """
assert sum_to(1) == 1
assert sum_to(10) == 55
assert sum_to(0) == 0
""", """
def sum_to(n):
    return sum(range(1, n + 1))
""")

code("fix_mutate", "bugfix", """
This function should return the list without its duplicates, keeping the first occurrence of each value in
order. It skips elements because it changes the list while looping. Fix it (same name); don't change the input.

def dedupe(xs):
    seen = set()
    for x in xs:
        if x in seen:
            xs.remove(x)
        seen.add(x)
    return xs
""", """
a = [1, 1, 1, 2, 2, 3, 1]
assert dedupe(a) == [1, 2, 3]
assert a == [1, 1, 1, 2, 2, 3, 1]
assert dedupe([]) == []
assert dedupe(["b", "a", "b"]) == ["b", "a"]
""", """
def dedupe(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
""")

code("fix_bsearch", "bugfix", """
This binary search should return the index of t in the sorted list xs, or -1. It can loop forever and misses
some elements. Fix it (same name).

def find(xs, t):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] == t:
            return mid
        if xs[mid] < t:
            lo = mid
        else:
            hi = mid - 1
    return -1
""", """
xs = [1, 3, 5, 7, 9, 11]
for i, v in enumerate(xs):
    assert find(xs, v) == i
for v in (0, 2, 12):
    assert find(xs, v) == -1
assert find([], 1) == -1
assert find([4], 4) == 0
""", """
def find(xs, t):
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == t:
            return mid
        if xs[mid] < t:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""")

code("fix_closure", "bugfix", """
This should return a list of functions where the i-th one multiplies its argument by i, but they all use the
last i. Fix it (same name).

def multipliers(n):
    return [lambda x: x * i for i in range(n)]
""", """
fs = multipliers(4)
assert [f(10) for f in fs] == [0, 10, 20, 30]
assert multipliers(0) == []
""", """
def multipliers(n):
    return [lambda x, i=i: x * i for i in range(n)]
""")

code("fix_compare", "bugfix", """
This should say whether a word is in a list, ignoring case and surrounding spaces, but it returns wrong
answers. Fix it (same name).

def has_word(words, w):
    for x in words:
        if x.lower is w.lower:
            return True
    return False
""", """
assert has_word(["Apple", " banana "], "banana")
assert has_word(["Apple"], "APPLE ")
assert not has_word(["Apple"], "pear")
assert not has_word([], "a")
""", """
def has_word(words, w):
    w = w.strip().lower()
    return any(x.strip().lower() == w for x in words)
""")

code("fix_dict", "bugfix", """
This should count how many times each letter appears in a string, but it crashes. Fix it (same name).

def letter_counts(s):
    counts = {}
    for ch in s:
        counts[ch] += 1
    return counts
""", """
assert letter_counts("banana") == {"b": 1, "a": 3, "n": 2}
assert letter_counts("") == {}
""", """
def letter_counts(s):
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    return counts
""")

# ---------------------------------------------------------------- robustness (careful specs)
code("roman", "careful", """
Write a Python function `to_roman(n: int) -> str` for 1 <= n <= 3999 (standard subtractive form, e.g. 4 = "IV",
1994 = "MCMXCIV"), and `from_roman(s: str) -> int` that reverses it.
""", """
assert to_roman(4) == "IV" and to_roman(1994) == "MCMXCIV" and to_roman(3999) == "MMMCMXCIX"
for n in range(1, 4000):
    assert from_roman(to_roman(n)) == n
""", """
VALS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
def to_roman(n):
    out = []
    for v, s in VALS:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)
def from_roman(s):
    m = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = m[ch]
        total += -v if i + 1 < len(s) and m[s[i + 1]] > v else v
    return total
""")

code("parse_dur", "careful", """
Write a Python function `seconds(text: str) -> int` that parses durations like "1h30m", "45s", "2h 5s",
"3m10s" (any of h, m, s, in that order, each at most once, optional spaces between parts) into seconds.
Raise ValueError for anything else, including an empty string, "5", "1m1h" or "1x".
""", """
assert seconds("1h30m") == 5400
assert seconds("45s") == 45
assert seconds("2h 5s") == 7205
assert seconds("3m10s") == 190
for bad in ["", "5", "1m1h", "1x", "h", "1h1h", "-1s"]:
    try:
        seconds(bad)
        assert False, bad
    except ValueError:
        pass
""", """
import re
def seconds(text):
    m = re.fullmatch(r"\\s*(?:(\\d+)h)?\\s*(?:(\\d+)m)?\\s*(?:(\\d+)s)?\\s*", text)
    if not m or not any(m.groups()):
        raise ValueError(text)
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s
""")

code("csvline", "careful", """
Write a Python function `split_csv(line: str) -> list[str]` that splits one CSV line on commas, where a field
in double quotes may contain commas, and "" inside quotes means one quote character. Don't use the csv module.
""", '''
assert split_csv('a,b,c') == ["a", "b", "c"]
assert split_csv('"x, y",z') == ["x, y", "z"]
assert split_csv('"she said ""hi""",2') == ['she said "hi"', "2"]
assert split_csv('') == [""]
assert split_csv('a,,b,') == ["a", "", "b", ""]
''', """
def split_csv(line):
    out, cur, q, i = [], [], False, 0
    while i < len(line):
        c = line[i]
        if q:
            if c == '"' and i + 1 < len(line) and line[i + 1] == '"':
                cur.append('"'); i += 1
            elif c == '"':
                q = False
            else:
                cur.append(c)
        elif c == '"':
            q = True
        elif c == ",":
            out.append("".join(cur)); cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur))
    return out
""")

code("money", "careful", """
Write a Python function `split_bill(total_cents: int, people: int) -> list[int]` that splits a bill in whole
cents as evenly as possible: the amounts differ by at most 1 cent, add up exactly to total_cents, and the
larger amounts come first.
""", """
assert split_bill(1000, 3) == [334, 333, 333]
assert split_bill(10, 4) == [3, 3, 2, 2]
assert split_bill(0, 2) == [0, 0]
assert sum(split_bill(99999, 7)) == 99999
""", """
def split_bill(total_cents, people):
    q, r = divmod(total_cents, people)
    return [q + 1] * r + [q] * (people - r)
""")

code("matrix", "careful", """
Write a Python function `rotate(m: list[list[int]]) -> list[list[int]]` that returns the matrix rotated 90
degrees clockwise. It may be rectangular (not square). Don't change the input.
""", """
a = [[1, 2, 3], [4, 5, 6]]
assert rotate(a) == [[4, 1], [5, 2], [6, 3]]
assert a == [[1, 2, 3], [4, 5, 6]]
assert rotate([[1]]) == [[1]]
assert rotate([]) == []
""", """
def rotate(m):
    return [list(r) for r in zip(*m[::-1])]
""")

# ---------------------------------------------------------------- code reading (exact output)
read("r_slice", """What does this Python code print? Answer with only the exact output.

xs = [0, 1, 2, 3, 4, 5]
print(xs[1:5:2], xs[::-2])""", "[1, 3] [5, 3, 1]")

read("r_scope", """What does this Python code print? Answer with only the exact output.

x = 1
def f():
    x = 2
    def g():
        nonlocal x
        x += 10
        return x
    return g() + x
print(f(), x)""", "24 1")

read("r_dict", """What does this Python code print? Answer with only the exact output.

d = {}
for w in "to be or not to be".split():
    d[w] = d.get(w, 0) + 1
print(sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:2])""", "[('be', 2), ('to', 2)]")

read("r_gen", """What does this Python code print? Answer with only the exact output.

def gen():
    for i in range(3):
        yield i * i
g = gen()
next(g)
print(list(g), list(g))""", "[1, 4] []")

read("r_int", """What does this Python code print? Answer with only the exact output.

print(7 // 2, -7 // 2, 7 % -3, round(2.5), round(3.5))""", "3 -4 -2 2 4")

read("r_alias", """What does this Python code print? Answer with only the exact output.

a = [[0] * 2] * 2
a[0][0] = 5
b = [x[:] for x in a]
b[1][1] = 7
print(a, b)""", "[[5, 0], [5, 0]] [[5, 0], [5, 7]]")
