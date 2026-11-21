"""Generate every figure and summary table of the report from ``results/``.

    python -m analysis.figures          # writes report/figures/*.png and report/tables.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TABLES: dict = {}

# validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 200,
})


def _log2(ax, axis="x"):
    """Log2 axis labelled with plain numbers (1, 2, 4, ...)."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log", base=2)
    a.set_major_locator(LogLocator(base=2))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _load(p: Path):
    return json.loads(p.read_text())


def _save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- E0 / E1
PREC_ORDER = ["fp32", "bf16", "fp16", "int8", "int4", "int4_noemb"]
PREC_LABEL = {"fp32": "FP32", "bf16": "BF16", "fp16": "FP16", "int8": "INT8 (W8)", "int4": "INT4 (W4, g64)",
              "int4_noemb": "INT4, FP16 emb/head"}


def profiles(model="qwen0.5b"):
    out = {}
    for k in PREC_ORDER:
        p = RES / "profile" / f"{model}_{k}.json"
        if p.exists():
            out[k] = _load(p)
    return out


def fig_gqa():
    a, b = RES / "profile" / "qwen0.5b_fp16.json", RES / "profile" / "qwen0.5b_fp16_hfgqa.json"
    if not (a.exists() and b.exists()):
        return
    fast, slow = _load(a), _load(b)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    rows = []
    for ax, T in zip(axes, [512, 2048]):
        for d, lab, col in [(slow, "HF repeat_kv", C[1]), (fast, "GQA-aware (ours)", C[0])]:
            pts = sorted((r["B"], r["s"] * 1000) for r in d["decode"] if r["T"] == T)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, label=lab)
        _log2(ax)
        ax.set_xlabel("batch size")
        ax.set_title(f"context {T} tokens")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("decode step (ms)")
    axes[0].legend()
    _save(fig, "e0_gqa_ablation")
    for T in [128, 512, 1024, 2048]:
        for B in [1, 8, 16, 32]:
            f = [r["s"] for r in fast["decode"] if r["T"] == T and r["B"] == B]
            s = [r["s"] for r in slow["decode"] if r["T"] == T and r["B"] == B]
            if f and s:
                rows.append({"T": T, "B": B, "hf_ms": s[0] * 1e3, "ours_ms": f[0] * 1e3, "speedup": s[0] / f[0]})
    TABLES["gqa"] = rows


def fig_precision():
    for model in ["qwen0.5b", "qwen1.5b"]:
        pr = profiles(model)
        if not pr:
            continue
        keys = [k for k in PREC_ORDER if k in pr]
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
        # (a) weight memory
        mem = [pr[k]["info"]["weight_bytes_tensors"] / 2**20 for k in keys]
        axes[0].barh(range(len(keys)), mem, color=C[0], height=0.6)
        axes[0].set_yticks(range(len(keys)), [PREC_LABEL[k] for k in keys])
        axes[0].invert_yaxis()
        for i, m in enumerate(mem):
            axes[0].text(m, i, f" {m:.0f}", va="center", fontsize=8, color=INK)
        axes[0].set_xlabel("weight memory (MiB)")
        axes[0].set_title("(a) resident weights")
        axes[0].grid(axis="y", visible=False)
        axes[0].set_xlim(0, max(mem) * 1.25)
        # (b) decode throughput vs batch, T=512
        for i, k in enumerate(keys):
            pts = sorted((r["B"], r["B"] / r["s"]) for r in pr[k]["decode"] if r["T"] == 512)
            axes[1].plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=C[i], label=PREC_LABEL[k])
        _log2(axes[1])
        axes[1].set_xlabel("batch size (context 512)")
        axes[1].set_ylabel("decode tokens / s")
        axes[1].set_title("(b) decode throughput")
        axes[1].legend(fontsize=7)
        # (c) prefill latency
        for i, k in enumerate(keys):
            pts = sorted((r["L"], r["s"] * 1000) for r in pr[k]["prefill"])
            axes[2].plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=C[i], label=PREC_LABEL[k])
        _log2(axes[2])
        axes[2].set_yscale("log")
        axes[2].set_xlabel("prompt tokens")
        axes[2].set_ylabel("prefill latency (ms)")
        axes[2].set_title("(c) prefill (time to first token)")
        _save(fig, f"e1_precision_{model}")
        rows = []
        for k in keys:
            d = {r["B"]: r["s"] for r in pr[k]["decode"] if r["T"] == 512}
            pre = {r["L"]: r["s"] for r in pr[k]["prefill"]}
            rows.append({"precision": k, "weight_mib": pr[k]["info"]["weight_bytes_tensors"] / 2**20,
                         "load_s": pr[k]["info"]["load_seconds"],
                         "kv_bytes_per_token": pr[k]["info"]["kv_bytes_per_token"],
                         "prefill_512_ms": pre.get(512, np.nan) * 1e3,
                         "tok_s_b1": 1 / d[1] if 1 in d else None,
                         "tok_s_b32": 32 / d[32] if 32 in d else None,
                         "tok_s_bmax": max(B / s for B, s in d.items())})
        TABLES[f"precision_{model}"] = rows


def table_quality():
    rows = []
    for p in sorted((RES / "quality").glob("*.json")) if (RES / "quality").exists() else []:
        d = _load(p)
        rows.append({"run": p.stem, "precision": d["precision"], "model": d["model"],
                     "ppl": d["ppl_wikitext2"], "arc_acc": d["arc_easy"]["acc"],
                     "arc_acc_norm": d["arc_easy"]["acc_norm"],
                     "exact_match": d.get("agreement_vs_ref", {}).get("exact_match"),
                     "prefix_frac": d.get("agreement_vs_ref", {}).get("mean_prefix_frac"),
                     "weight_mib": d["info"]["weight_bytes_tensors"] / 2**20})
    if rows:
        TABLES["quality"] = rows


def fig_tradeoff():
    q = {r["run"]: r for r in TABLES.get("quality", [])}
    pr = profiles("qwen0.5b")
    if not q or not pr:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    label = dict(PREC_LABEL, fp16="FP16 = BF16")
    offset = {"fp32": (-8, 8), "int8": (-14, 8), "int4_noemb": (6, 4), "int4": (6, 4), "fp16": (-4, 8)}
    for i, k in enumerate([k for k in PREC_ORDER if k in pr]):
        run = f"qwen0.5b_{k}"
        if run not in q or k == "bf16":  # BF16 coincides with FP16 in both quality and speed
            continue
        d = {r["B"]: r["s"] for r in pr[k]["decode"] if r["T"] == 512}
        mem = pr[k]["info"]["weight_bytes_tensors"] / 2**20
        for ax, x in [(axes[0], mem), (axes[1], 32 / d[32])]:
            ax.scatter([x], [q[run]["ppl"]], s=60, color=C[i], edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(label[k], (x, q[run]["ppl"]), xytext=offset.get(k, (5, 4)), textcoords="offset points",
                        fontsize=7, color=INK)
    axes[0].set_xlabel("weight memory (MiB)  - lower is better")
    axes[1].set_xlabel("decode tok/s at batch 32  - higher is better")
    axes[0].set_ylabel("WikiText-2 perplexity  - lower is better")
    axes[0].set_title("quality vs memory")
    axes[1].set_title("quality vs throughput")
    _save(fig, "e2_tradeoff")


# --------------------------------------------------------------------------- E3 batching
BATCH_STYLE = {
    "serial": ("serial (B=1)", C[7]),
    "static_b8_w50": ("static B=8, 50 ms window", C[3]),
    "static_b32_w50": ("static B=32, 50 ms window", C[1]),
    "continuous_b8": ("continuous B=8", C[2]),
    "continuous_b32": ("continuous B=32", C[0]),
    "continuous_b64": ("continuous B=64", C[6]),
}


def fig_batching():
    d = RES / "batching"
    if not d.exists():
        return
    data = defaultdict(dict)
    for p in d.glob("*_c*.json"):
        j = _load(p)
        data[j["meta"]["config"]][j["meta"]["concurrency"]] = j["summary"]
    if not data:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    metrics = [("throughput_tok_s", "output tokens / s", "(a) throughput"),
               ("ttft_p95", "P95 time to first token (s)", "(b) P95 TTFT"),
               ("e2e_p95", "P95 end-to-end latency (s)", "(c) P95 latency")]
    rows = []
    for name, (lab, col) in BATCH_STYLE.items():
        if name not in data:
            continue
        cs = sorted(data[name])
        for ax, (m, yl, t) in zip(axes, metrics):
            ax.plot(cs, [data[name][c][m] for c in cs], "-o", color=col, label=lab)
            _log2(ax)
            ax.set_xlabel("concurrent clients")
            ax.set_ylabel(yl)
            ax.set_title(t)
        for c in cs:
            s = data[name][c]
            rows.append({"config": name, "concurrency": c, **{k: s.get(k) for k in
                         ["throughput_tok_s", "throughput_req_s", "ttft_p50", "ttft_p95", "e2e_p50", "e2e_p95",
                          "tpot_p50", "active_mean", "kv_padding_overhead", "mem_peak_bytes", "busy_last"]}})
    axes[2].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7)
    _save(fig, "e3_batching")
    TABLES["batching"] = rows

    od = d / "open"
    if od.exists():
        opn = defaultdict(dict)
        for p in od.glob("*.json"):
            j = _load(p)
            opn[j["meta"]["config"]][j["meta"]["rate"]] = j["summary"]
        names = [("continuous_b32", "continuous", C[0]), ("static_b32_w0", "static, 0 ms", C[3]),
                 ("static_b32_w25", "static, 25 ms", C[1]), ("static_b32_w100", "static, 100 ms", C[4]),
                 ("static_b32_w400", "static, 400 ms", C[7])]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
        orows = []
        for name, lab, col in names:
            if name not in opn:
                continue
            rs = sorted(opn[name])
            axes[0].plot(rs, [opn[name][r]["ttft_p50"] for r in rs], "-o", color=col, label=lab)
            axes[1].plot(rs, [opn[name][r]["e2e_p95"] for r in rs], "-o", color=col, label=lab)
            for r in rs:
                orows.append({"config": name, "rate": r, **{k: opn[name][r].get(k) for k in
                              ["ttft_p50", "ttft_p95", "e2e_p50", "e2e_p95", "throughput_tok_s"]}})
        axes[0].set_ylabel("median TTFT (s)")
        axes[1].set_ylabel("P95 end-to-end (s)")
        for ax in axes:
            ax.set_xlabel("arrival rate (req/s, Poisson)")
        axes[0].set_title("(a) time to first token")
        axes[1].set_title("(b) tail latency")
        axes[0].legend(fontsize=7)
        _save(fig, "e3_open_loop")
        TABLES["batching_open"] = orows


# --------------------------------------------------------------------------- E4 scaling
def fig_scaling():
    d = RES / "scaling"
    if not d.exists():
        return
    data = defaultdict(dict)
    for p in d.glob("*.json"):
        j = _load(p)
        data[j["meta"]["replicas"]][j["meta"]["concurrency"]] = j["summary"]
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    rows = []
    for i, k in enumerate(sorted(data)):
        cs = sorted(data[k])
        axes[0].plot(cs, [data[k][c]["throughput_tok_s"] for c in cs], "-o", color=C[i], label=f"{k} worker(s)")
        axes[1].plot(cs, [data[k][c]["e2e_p95"] for c in cs], "-o", color=C[i], label=f"{k} worker(s)")
        for c in cs:
            rows.append({"replicas": k, "concurrency": c, **{m: data[k][c].get(m) for m in
                         ["throughput_tok_s", "e2e_p50", "e2e_p95", "ttft_p95", "mem_peak_bytes"]}})
    for ax in axes:
        _log2(ax)
        ax.set_xlabel("concurrent clients")
    axes[0].set_ylabel("output tokens / s")
    axes[1].set_ylabel("P95 end-to-end (s)")
    axes[0].set_title("(a) throughput")
    axes[1].set_title("(b) tail latency")
    axes[0].legend()
    _save(fig, "e4_scaling")
    TABLES["scaling"] = rows


# --------------------------------------------------------------------------- E5 routing (live)
POL = [("random", "random", C[3]), ("round_robin", "round-robin", C[1]),
       ("least_outstanding", "least-outstanding", C[2]), ("gpu_aware", "GPU-aware", C[0])]


def fig_routing_live():
    d = RES / "routing"
    if not d.exists():
        return
    runs = defaultdict(list)
    for p in d.glob("*.json"):
        j = _load(p)
        m = j["meta"]
        runs[(m["cluster"], m["workload"], m["policy"])].append(j["summary"])
    if not runs:
        return
    rows = []
    for cluster in ["hetero", "homo"]:
        wls = [w for w in ["poisson", "bursty", "heavy_tail"] if any(k[:2] == (cluster, w) for k in runs)]
        if not wls:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
        x = np.arange(len(wls))
        wbar = 0.2
        for j, (pol, lab, col) in enumerate(POL):
            for ax, m in zip(axes, ["e2e_p50", "e2e_p95"]):
                vals = [np.mean([s[m] for s in runs.get((cluster, w, pol), [])]) if runs.get((cluster, w, pol))
                        else np.nan for w in wls]
                ax.bar(x + (j - 1.5) * wbar, vals, width=wbar * 0.92, color=col, label=lab)
        for (cl, w, pol), ss in runs.items():
            if cl == cluster:
                rows.append({"cluster": cl, "workload": w, "policy": pol,
                             **{m: float(np.mean([s[m] for s in ss])) for m in
                                ["e2e_p50", "e2e_p95", "ttft_p50", "ttft_p95", "throughput_tok_s"]},
                             "per_worker": ss[0].get("per_worker")})
        for ax, t in zip(axes, ["median end-to-end (s)", "P95 end-to-end (s)"]):
            ax.set_xticks(x, wls)
            ax.set_ylabel(t)
            ax.grid(axis="x", visible=False)
        axes[0].legend(fontsize=7)
        _save(fig, f"e5_routing_{cluster}")
    TABLES["routing_live"] = rows


# --------------------------------------------------------------------------- E6 fault
def fig_fault():
    p = RES / "fault" / "kill_worker.json"
    if not p.exists():
        return
    j = _load(p)
    res = j["results"]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    workers = sorted({r["worker"] for r in res if r["worker"]})
    for i, w in enumerate(workers):
        pts = [(r["t_send"], r["e2e"]) for r in res if r["ok"] and r["worker"] == w]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=10, color=C[i], label=w)
    bad = [r["t_send"] for r in res if not r["ok"]]
    if bad:
        ax.scatter(bad, [0] * len(bad), marker="x", s=30, color=C[7], label=f"failed ({len(bad)})")
    ax.axvline(j["meta"]["kill_at"], color=INK2, linestyle="--", linewidth=1)
    ax.text(j["meta"]["kill_at"], ax.get_ylim()[1] * 0.92, " worker W2 killed", fontsize=8, color=INK2)
    ax.set_xlabel("request send time (s)")
    ax.set_ylabel("end-to-end latency (s)")
    ax.legend(fontsize=7, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.25))
    _save(fig, "e6_fault")
    TABLES["fault"] = {**{k: j["summary"].get(k) for k in ["n", "n_ok", "n_err", "router_retries", "per_worker"]},
                       "errors": sorted({r["error"][:60] for r in res if r.get("error")})}


# --------------------------------------------------------------------------- E7 simulation
def fig_sim():
    d = RES / "sim"
    if not d.exists():
        return
    if (d / "s1_validate.json").exists():
        TABLES["sim_validation"] = _load(d / "s1_validate.json")
    for name, fname in [("hetero", "s2_hetero.json"), ("homo", "s3_homo.json")]:
        if not (d / fname).exists():
            continue
        rows = _load(d / fname)
        wls = sorted({r["workload"] for r in rows}, key=["poisson", "bursty", "heavy_tail"].index)
        fig, axes = plt.subplots(1, len(wls), figsize=(3.3 * len(wls), 2.7), sharey=False)
        axes = np.atleast_1d(axes)
        agg = []
        for ax, w in zip(axes, wls):
            for pol, lab, col in POL:
                rates = sorted({r["rate"] for r in rows if r["workload"] == w})
                med, lo, hi = [], [], []
                for rt in rates:
                    v = [r["e2e_p95"] for r in rows if r["workload"] == w and r["policy"] == pol and r["rate"] == rt]
                    med.append(np.mean(v))
                    lo.append(np.min(v))
                    hi.append(np.max(v))
                    agg.append({"cluster": name, "workload": w, "policy": pol, "rate": rt, "e2e_p95_mean": np.mean(v),
                                "e2e_p95_min": np.min(v), "e2e_p95_max": np.max(v),
                                "e2e_p50_mean": np.mean([r["e2e_p50"] for r in rows if r["workload"] == w
                                                         and r["policy"] == pol and r["rate"] == rt]),
                                "ttft_p95_mean": np.mean([r["ttft_p95"] for r in rows if r["workload"] == w
                                                          and r["policy"] == pol and r["rate"] == rt])})
                ax.plot(rates, med, "-o", color=col, label=lab)
                ax.fill_between(rates, lo, hi, color=col, alpha=0.15, linewidth=0)
            ax.set_yscale("log")
            ax.set_xlabel("offered load (req/s)")
            ax.set_title(w)
        axes[0].set_ylabel("P95 end-to-end (s)")
        axes[0].legend(fontsize=7)
        _save(fig, f"e7_sim_{name}")
        TABLES[f"sim_{name}"] = agg
    if (d / "s4_stale.json").exists():
        rows = _load(d / "s4_stale.json")
        fig, ax = plt.subplots(figsize=(4.6, 2.7))
        labs = {"gpu_aware": ("GPU-aware, local counters + polled KV", C[0]),
                "gpu_aware_polled": ("GPU-aware, polled counters only", C[6]),
                "least_outstanding_polled": ("least-outstanding, polled only", C[2])}
        agg = []
        for pol, (lab, col) in labs.items():
            polls = sorted({r["poll"] for r in rows if r["policy"] == pol})
            v = [np.mean([r["e2e_p95"] for r in rows if r["policy"] == pol and r["poll"] == p]) for p in polls]
            xs = [max(p, 0.01) for p in polls]
            ax.plot(xs, v, "-o", color=col, label=lab)
            agg += [{"policy": pol, "poll": p, "e2e_p95": x} for p, x in zip(polls, v)]
        ax.set_xscale("log")
        ax.set_xlabel("telemetry poll interval (s)  [0 plotted at 0.01]")
        ax.set_ylabel("P95 end-to-end (s)")
        ax.legend(fontsize=7)
        _save(fig, "e7_sim_staleness")
        TABLES["sim_stale"] = agg


def main():
    fig_gqa()
    fig_precision()
    table_quality()
    fig_tradeoff()
    fig_batching()
    fig_scaling()
    fig_routing_live()
    fig_fault()
    fig_sim()
    (ROOT / "report" / "tables.json").write_text(json.dumps(TABLES, indent=1, default=float))
    print("tables:", list(TABLES))


if __name__ == "__main__":
    main()
