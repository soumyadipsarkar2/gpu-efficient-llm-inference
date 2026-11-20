from llmserve.policies import GpuAwarePolicy, ReqInfo, WorkerView, make_policy
from llmserve.sim.simulator import Profile, simulate, synthetic


def views():
    return [WorkerView("fast", inflight=4, inflight_tokens=400, step_a=0.03, step_c=0.001),
            WorkerView("slow", inflight=0, inflight_tokens=0, step_a=0.15, step_c=0.001),
            WorkerView("full", inflight=0, kv_free_tokens=10, kv_budget_tokens=100, step_a=0.03, step_c=0.001)]


def test_round_robin_cycles_and_skips_unhealthy():
    v = views()
    v[1].healthy = False
    p = make_policy("round_robin")
    assert [p.choose(ReqInfo(10, 10), v) for _ in range(4)] == [0, 2, 0, 2]


def test_least_outstanding():
    assert make_policy("least_outstanding").choose(ReqInfo(10, 10), views()) in (1, 2)


def test_gpu_aware_prefers_fast_worker_and_respects_budget():
    p = GpuAwarePolicy()
    r = ReqInfo(prompt_tokens=100, max_tokens=100)
    est = [p.estimate(r, w) for w in views()]
    assert est[2] == float("inf")  # can never fit a 200-token reservation in a 100-token budget
    assert est[0] < est[1]  # fast worker with a small batch beats an idle worker 5x slower
    assert p.choose(r, views()) == 0


def test_simulator_conserves_requests_and_gpu_aware_beats_random_on_hetero():
    profs = [Profile("fast", 0.03, 0.0005, 1e-7, 0.01, 2e-4), Profile("slow", 0.12, 0.002, 4e-7, 0.04, 8e-4)]
    reqs = synthetic(300, "balanced", "poisson", rate=6.0, seed=1)
    out = {}
    for pol in ["random", "gpu_aware"]:
        res = simulate(profs, reqs, pol)
        s = res.summary()
        assert s["n_done"] == 300
        out[pol] = s["e2e_p95"]
    assert out["gpu_aware"] < out["random"]
