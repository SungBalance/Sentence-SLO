"""Analyze the conditions under which unit-deadline violations occur
for progress_serve in the cmp run."""
import json
import os
import statistics as st

ROOT = "exp/run_sslo/output_cmp_ps/progress_serve/run_1"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
    return xs[i]


def summ(xs):
    if not xs:
        return "n=0"
    return (f"n={len(xs)} mean={st.mean(xs):.1f} p50={pct(xs,50):.1f} "
            f"p90={pct(xs,90):.1f} max={max(xs):.1f}")


for rate in (8, 16, 24):
    cdir = f"{ROOT}/rate_{rate}"
    chunks = load(f"{cdir}/chunks.jsonl")
    reqs = {r["request_id"]: r for r in load(f"{cdir}/requests.jsonl")}
    sched = [r for r in load(f"{cdir}/scheduler_stats.jsonl")
             if r.get("kind") == "step"]
    sched.sort(key=lambda r: r["ts"])
    sched_ts = [r["ts"] for r in sched]

    viol = [c for c in chunks if c.get("unit_deadline_missed")]
    ok = [c for c in chunks if not c.get("unit_deadline_missed")]
    print(f"\n===== rate {rate}: {len(viol)}/{len(chunks)} units missed "
          f"({100*len(viol)/max(1,len(chunks)):.3f}%) =====")
    if not viol:
        continue

    # 1) Which chunk position misses? (unit_index)
    print("  unit_index   viol:", summ([c["unit_index"] for c in viol]),
          "| ok:", summ([c["unit_index"] for c in ok]))
    print("  num_token    viol:", summ([c["num_token"] for c in viol]),
          "| ok:", summ([c["num_token"] for c in ok]))
    print("  word_count   viol:", summ([c["word_count"] for c in viol]),
          "| ok:", summ([c["word_count"] for c in ok]))
    print("  miss_s       viol:", summ([c["unit_deadline_miss_s"]
                                        for c in viol]))

    # 2) System state at the violating chunk's generation time.
    import bisect
    run_at, pend_at, ev_at, kv_at = [], [], [], []
    for c in viol:
        t = c["text_generation_end_time"]
        i = min(len(sched) - 1, bisect.bisect_left(sched_ts, t))
        s = sched[i]
        run_at.append(s.get("running", 0))
        pend_at.append(s.get("pending", 0))
        ev_at.append(s.get("e_viol", 0.0))
        kv = s.get("kv_blocks_used", 0)
        kvt = s.get("kv_blocks_total", 1)
        kv_at.append(100 * kv / kvt)
    # overall step distribution for reference
    print("  --- system state AT violation vs overall steps ---")
    print("  running  at-viol:", summ(run_at),
          "| all-steps:", summ([s.get("running", 0) for s in sched]))
    print("  pending  at-viol:", summ(pend_at),
          "| all-steps:", summ([s.get("pending", 0) for s in sched]))
    print("  e_viol   at-viol:", summ(ev_at),
          "| all-steps:", summ([s.get("e_viol", 0.0) for s in sched]))
    print("  KV%      at-viol:", summ(kv_at))

    # 3) Per-request: do violators concentrate in high-stall requests?
    vreq = {c["request_id"] for c in viol}
    stall_v = [reqs[r]["request_max_stall_s"] for r in vreq if r in reqs]
    stall_all = [r["request_max_stall_s"] for r in reqs.values()]
    print(f"  requests with >=1 missed unit: {len(vreq)}/{len(reqs)}")
    print("  req_max_stall  violators:", summ(stall_v),
          "| all-reqs:", summ(stall_all))
    # fraction of all missed units that are the request's FIRST chunk
    first = sum(1 for c in viol if c["unit_index"] == 0)
    print(f"  missed units that are unit_index==0 (first chunk): "
          f"{first}/{len(viol)}")
