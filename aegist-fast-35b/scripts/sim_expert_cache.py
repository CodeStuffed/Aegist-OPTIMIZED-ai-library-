"""Replay a routing trace through expert-cache policies and print each one's hit rate.

    AEGIST_ROUTE_TRACE=trace.txt aegist ask "..."        # record (one line per layer step)
    python -I scripts/sim_expert_cache.py trace.txt --slots 826 1650 --windows 1 4 16 64

Window 1 is plain LRU; a wider window evicts the least used of that many least recently used
slots, as the engine's Pool does (counts halve every 65536 uses). "best" is Belady's optimum
(evict the expert needed furthest in the future), the most any policy could get.
"""
import argparse
from collections import OrderedDict, defaultdict

AGE = 1 << 16


def read(path):
    steps = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if p:
                layer = int(p[0])
                steps.append([(layer, int(e)) for e in p[1:]])
    return steps


def windowed(steps, slots, window):
    lru = OrderedDict()  # key -> None, oldest first
    uses = defaultdict(int)
    since, hits, total = 0, 0, 0
    for step in steps:
        for k in step:
            uses[k] = min(uses[k] + 1, 65535)
        since += len(step)
        if since >= AGE:
            since = 0
            for k in uses:
                uses[k] //= 2
        busy = set(step)
        for k in step:
            total += 1
            if k in lru:
                hits += 1
                lru.move_to_end(k)
                continue
            if len(lru) >= slots:
                cands = []
                for old in lru:
                    if old not in busy:
                        cands.append(old)
                        if len(cands) >= window:
                            break
                victim = min(cands, key=lambda c: uses[c])
                del lru[victim]
            lru[k] = None
    return hits / max(total, 1)


def belady(steps, slots):
    import heapq
    flat = [(i, k) for i, s in enumerate(steps) for k in s]
    nxt = [0] * len(flat)
    last = {}
    for j in range(len(flat) - 1, -1, -1):
        k = flat[j][1]
        nxt[j] = last.get(k, 1 << 60)
        last[k] = j
    cache, heap, hits = {}, [], 0
    for j, (_, k) in enumerate(flat):
        if k in cache:
            hits += 1
        elif len(cache) >= slots:
            while True:
                negn, v = heapq.heappop(heap)
                if cache.get(v) == -negn:
                    del cache[v]
                    break
        cache[k] = nxt[j]
        heapq.heappush(heap, (-nxt[j], k))
    return hits / max(len(flat), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--slots", type=int, nargs="+", default=[826, 1650])
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 4, 16, 64])
    a = ap.parse_args()
    steps = read(a.trace)
    n = sum(len(s) for s in steps)
    print(f"{len(steps)} layer steps, {n} expert uses, {len({k for s in steps for k in s})} distinct experts")
    for slots in a.slots:
        row = [f"window {w}: {100 * windowed(steps, slots, w):.1f}%" for w in a.windows]
        row.append(f"best: {100 * belady(steps, slots):.1f}%")
        print(f"{slots} slots: " + ", ".join(row))


if __name__ == "__main__":
    main()
