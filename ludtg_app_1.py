"""
╔══════════════════════════════════════════════════════════════╗
║     LOOMIS-UNIVERSAL DIGITAL TWIN GENERATOR  (L-UDTG)       ║
║     Version 1.0  |  Lead Architect: Factory Intelligence    ║
╚══════════════════════════════════════════════════════════════╝

Phases:
  1 – Guided Wizard (input + validation)
  2 – High-Fidelity SimPy Engine (DES + Monte Carlo)
  3 – Versioned JSON Data Bridge
  4 – Performance Dashboard (Graphviz + Plotly)
  5 – Optimization Layer (Decision Support)

Install dependencies:
    pip install streamlit simpy plotly graphviz
"""

# ─────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────
import random
import json
import math
import copy
import io
import streamlit as st
import simpy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
#  GLOBAL SIMULATION CONSTANTS
# ─────────────────────────────────────────────────────────────
SIM_TIME    = 480   # minutes per run
NUM_RUNS    = 5
WARMUP_TIME = 50    # minutes — data before this is discarded

# ─────────────────────────────────────────────────────────────
#  PHASE 3 — JSON SCHEMA & DATA BRIDGE
# ─────────────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0"

def build_json_payload(arrival: dict, stages: list) -> dict:
    """Serialize current configuration to versioned JSON."""
    return {
        "version": SCHEMA_VERSION,
        "simulation": {
            "sim_time": SIM_TIME,
            "num_runs": NUM_RUNS,
            "warmup_time": WARMUP_TIME,
        },
        "arrival": arrival,
        "stages": stages,
    }

def validate_json_payload(payload: dict) -> tuple[bool, str]:
    """Schema consistency check before loading a saved config."""
    if not isinstance(payload, dict):
        return False, "Payload must be a JSON object."
    if payload.get("version") != SCHEMA_VERSION:
        return False, f"Version mismatch. Expected {SCHEMA_VERSION}, got {payload.get('version')}."
    if "simulation" not in payload:
        return False, "Missing 'simulation' key."
    if "arrival" not in payload:
        return False, "Missing 'arrival' key."
    if "stages" not in payload or not isinstance(payload["stages"], list):
        return False, "Missing or invalid 'stages' key."
    return True, "OK"

def safe_parse_json(raw: str) -> tuple[dict | None, str]:
    """Safe JSON parsing with error reporting."""
    try:
        payload = json.loads(raw)
        ok, msg = validate_json_payload(payload)
        if not ok:
            return None, msg
        return payload, "OK"
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

# ─────────────────────────────────────────────────────────────
#  PHASE 1 — VALIDATION / GUARDRAIL LAYER
# ─────────────────────────────────────────────────────────────
def validate_triangular(label: str, mn: float, mode: float, mx: float) -> list[str]:
    """Return list of error strings; empty list means valid."""
    errors = []
    if any(v is None for v in [mn, mode, mx]):
        errors.append(f"[{label}] One or more values are missing.")
        return errors
    if any(v <= 0 for v in [mn, mode, mx]):
        errors.append(f"[{label}] All values must be > 0.")
    if not (mn <= mode <= mx):
        errors.append(f"[{label}] Must satisfy Min ≤ Mode ≤ Max. Got {mn} / {mode} / {mx}.")
    return errors

def validate_all_inputs(arrival: dict, stages: list) -> list[str]:
    """Full guardrail sweep. Returns list of error strings."""
    errors = []

    # Arrival
    errors += validate_triangular(
        "Arrival Inter-Arrival Time",
        arrival.get("min"), arrival.get("mode"), arrival.get("max")
    )

    for i, s in enumerate(stages):
        label_prefix = f"Stage {i+1} ({s.get('name', '?')})"

        # Process time
        errors += validate_triangular(
            f"{label_prefix} Process Time",
            s.get("proc_min"), s.get("proc_mode"), s.get("proc_max")
        )

        # Capacity
        cap = s.get("capacity", 0)
        if not isinstance(cap, int) or cap < 1:
            errors.append(f"[{label_prefix}] Capacity must be an integer ≥ 1.")

        # Reliability
        if s.get("enable_failures", False):
            errors += validate_triangular(
                f"{label_prefix} TBF",
                s.get("tbf_min"), s.get("tbf_mode"), s.get("tbf_max")
            )
            errors += validate_triangular(
                f"{label_prefix} Repair Time",
                s.get("repair_min"), s.get("repair_mode"), s.get("repair_max")
            )

        # Routing
        if not s.get("route_all", True):
            pct_next   = s.get("route_pct_next",   0.0)
            pct_rework = s.get("route_pct_rework",  0.0)
            pct_scrap  = s.get("route_pct_scrap",   0.0)
            total = pct_next + pct_rework + pct_scrap
            if abs(total - 100.0) > 0.01:
                errors.append(
                    f"[{label_prefix}] Routing percentages must sum to 100%. Got {total:.1f}%."
                )

    return errors

# ─────────────────────────────────────────────────────────────
#  PHASE 2 — HIGH-FIDELITY SIMPY ENGINE
# ─────────────────────────────────────────────────────────────

class Part:
    """Individual unit flowing through the shop floor."""
    def __init__(self, part_id: int, env_time: float):
        self.id              = part_id
        self.entry_time      = env_time
        self.wait_time       = 0.0
        self.processing_time = 0.0
        self.completed       = False
        self.scrapped        = False


class StageMonitor:
    """Accumulates per-stage metrics during a single run."""
    def __init__(self, name: str):
        self.name             = name
        self.queue_log        = []   # (time, queue_length)
        self.wait_times       = []
        self.proc_times       = []
        self.busy_time        = 0.0
        self.parts_completed  = 0
        self.utilization      = 0.0  # filled in post-run


class FactoryEngine:
    """
    Discrete-Event Simulation engine.

    Supports:
    - N stages defined at runtime
    - Triangular stochastic times
    - FIFO queues (SimPy default)
    - Machine failures (TBF + Repair)
    - Warm-up period exclusion
    - Per-stage metrics
    """

    def __init__(self, arrival: dict, stages: list,
                 sim_time: int = SIM_TIME, warmup: int = WARMUP_TIME):
        self.arrival    = arrival
        self.stages_cfg = stages
        self.sim_time   = sim_time
        self.warmup     = warmup

        self.env        = simpy.Environment()
        self.resources  = {}        # stage_name -> simpy.Resource
        self.monitors   = {}        # stage_name -> StageMonitor
        self.parts_done = []        # completed Part objects (post warmup)
        self._part_counter = 0

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def tri(mn, mode, mx) -> float:
        return random.triangular(mn, mx, mode)

    # ── build resources & monitors ─────────────────────────────

    def _setup(self):
        for s in self.stages_cfg:
            name = s["name"]
            self.resources[name] = simpy.Resource(self.env, capacity=s["capacity"])
            self.monitors[name]  = StageMonitor(name)

    # ── failure process ────────────────────────────────────────

    def _failure_process(self, stage_cfg):
        """Preempts the resource to simulate breakdowns."""
        resource = self.resources[stage_cfg["name"]]
        while True:
            # time until next failure
            tbf = self.tri(stage_cfg["tbf_min"],
                           stage_cfg["tbf_mode"],
                           stage_cfg["tbf_max"])
            yield self.env.timeout(tbf)

            # "break" the machine by consuming all capacity slots
            reqs = [resource.request() for _ in range(resource.capacity)]
            for req in reqs:
                yield req

            # repair time
            repair = self.tri(stage_cfg["repair_min"],
                              stage_cfg["repair_mode"],
                              stage_cfg["repair_max"])
            yield self.env.timeout(repair)

            # release all slots
            for req in reqs:
                resource.release(req)

    # ── part process ───────────────────────────────────────────

    def _part_process(self, part: Part, stage_index: int):
        """Recursive process — part walks through stages from stage_index onward."""
        if stage_index >= len(self.stages_cfg):
            part.completed = True
            if self.env.now >= self.warmup:
                self.parts_done.append(part)
            return

        stage_cfg = self.stages_cfg[stage_index]
        name      = stage_cfg["name"]
        monitor   = self.monitors[name]
        resource  = self.resources[name]

        # Queue behavior
        queue_behavior = stage_cfg.get("queue_behavior", "wait")

        if queue_behavior == "skip" and len(resource.queue) >= resource.capacity:
            # Skip this stage entirely
            yield from self._part_process(part, stage_index + 1)
            return

        # Queue entry
        arrive_queue = self.env.now
        monitor.queue_log.append((self.env.now, len(resource.queue)))

        with resource.request() as req:
            yield req  # FIFO wait

            wait = self.env.now - arrive_queue
            if self.env.now >= self.warmup:
                part.wait_time += wait
                monitor.wait_times.append(wait)

            monitor.queue_log.append((self.env.now, len(resource.queue)))

            # Process
            proc = self.tri(stage_cfg["proc_min"],
                            stage_cfg["proc_mode"],
                            stage_cfg["proc_max"])

            # Handle rework by repeating the stage
            if queue_behavior == "rework":
                rework_prob = stage_cfg.get("rework_prob", 0.1)
                iterations  = 1
                while random.random() < rework_prob and iterations < 5:
                    iterations += 1
                proc *= iterations

            start_proc = self.env.now
            yield self.env.timeout(proc)

            if self.env.now >= self.warmup:
                part.processing_time += proc
                monitor.proc_times.append(proc)
                monitor.busy_time    += proc
                monitor.parts_completed += 1

        # Routing
        if stage_cfg.get("route_all", True):
            yield from self._part_process(part, stage_index + 1)
        else:
            r = random.uniform(0, 100)
            pct_next   = stage_cfg.get("route_pct_next",  100.0)
            pct_rework = stage_cfg.get("route_pct_rework",  0.0)
            # pct_scrap  = remainder
            if r < pct_next:
                yield from self._part_process(part, stage_index + 1)
            elif r < pct_next + pct_rework:
                # Rework = repeat same stage
                yield from self._part_process(part, stage_index)
            else:
                part.scrapped = True  # scrap

    # ── arrival generator ──────────────────────────────────────

    def _arrivals(self):
        part_id = 0
        while True:
            iat = self.tri(self.arrival["min"],
                           self.arrival["mode"],
                           self.arrival["max"])
            yield self.env.timeout(iat)
            part = Part(part_id, self.env.now)
            part_id += 1
            self.env.process(self._part_process(part, 0))

    # ── run ────────────────────────────────────────────────────

    def run(self) -> dict:
        self._setup()

        # Start failure processes
        for s in self.stages_cfg:
            if s.get("enable_failures", False):
                self.env.process(self._failure_process(s))

        # Start arrivals
        self.env.process(self._arrivals())

        # Execute
        self.env.run(until=self.sim_time)

        # Compute utilization
        active_time = self.sim_time - self.warmup
        for s in self.stages_cfg:
            mon = self.monitors[s["name"]]
            denom = active_time * s["capacity"]
            mon.utilization = (mon.busy_time / denom * 100) if denom > 0 else 0.0

        return self._collect_metrics()

    # ── metric collection ──────────────────────────────────────

    def _collect_metrics(self) -> dict:
        throughput = len([p for p in self.parts_done if p.completed])

        avg_wait = (
            sum(p.wait_time for p in self.parts_done) / len(self.parts_done)
            if self.parts_done else 0.0
        )
        avg_proc = (
            sum(p.processing_time for p in self.parts_done) / len(self.parts_done)
            if self.parts_done else 0.0
        )

        stages_out = {}
        for name, mon in self.monitors.items():
            avg_q = 0.0
            if len(mon.queue_log) > 1:
                weighted = sum(
                    mon.queue_log[i][1] * (mon.queue_log[i+1][0] - mon.queue_log[i][0])
                    for i in range(len(mon.queue_log)-1)
                )
                span = mon.queue_log[-1][0] - mon.queue_log[0][0]
                avg_q = weighted / span if span > 0 else 0.0

            stages_out[name] = {
                "utilization":      mon.utilization,
                "avg_wait":         sum(mon.wait_times) / len(mon.wait_times) if mon.wait_times else 0.0,
                "avg_proc":         sum(mon.proc_times) / len(mon.proc_times) if mon.proc_times else 0.0,
                "avg_queue_length": avg_q,
                "parts_completed":  mon.parts_completed,
                "queue_log":        mon.queue_log,
            }

        return {
            "throughput": throughput,
            "avg_wait":   avg_wait,
            "avg_proc":   avg_proc,
            "stages":     stages_out,
        }


# ─────────────────────────────────────────────────────────────
#  MONTE CARLO RUNNER
# ─────────────────────────────────────────────────────────────

def run_monte_carlo(arrival: dict, stages: list, num_runs: int = NUM_RUNS) -> dict:
    """Execute NUM_RUNS independent simulation runs and aggregate statistics."""
    all_runs = []
    for i in range(num_runs):
        engine = FactoryEngine(arrival, stages)
        result = engine.run()
        all_runs.append(result)

    stage_names = [s["name"] for s in stages]

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def std(vals):
        if len(vals) < 2:
            return 0.0
        m = mean(vals)
        return math.sqrt(sum((v - m)**2 for v in vals) / (len(vals) - 1))

    throughputs = [r["throughput"] for r in all_runs]
    avg_waits   = [r["avg_wait"]   for r in all_runs]

    aggregate = {
        "num_runs":  num_runs,
        "throughput_mean": mean(throughputs),
        "throughput_std":  std(throughputs),
        "avg_wait_mean":   mean(avg_waits),
        "avg_wait_std":    std(avg_waits),
        "stages": {},
        "raw_runs": all_runs,
    }

    for name in stage_names:
        utils  = [r["stages"][name]["utilization"]      for r in all_runs if name in r["stages"]]
        waits  = [r["stages"][name]["avg_wait"]          for r in all_runs if name in r["stages"]]
        procs  = [r["stages"][name]["avg_proc"]          for r in all_runs if name in r["stages"]]
        queues = [r["stages"][name]["avg_queue_length"]  for r in all_runs if name in r["stages"]]

        aggregate["stages"][name] = {
            "util_mean":  mean(utils),
            "util_std":   std(utils),
            "wait_mean":  mean(waits),
            "wait_std":   std(waits),
            "proc_mean":  mean(procs),
            "proc_std":   std(procs),
            "queue_mean": mean(queues),
            "queue_std":  std(queues),
        }

    return aggregate


# ─────────────────────────────────────────────────────────────
#  PHASE 5 — OPTIMIZATION LAYER
# ─────────────────────────────────────────────────────────────

def run_optimization(arrival: dict, stages: list, baseline: dict) -> dict:
    """
    For each stage, add 1 resource and re-run NUM_RUNS simulations.
    Returns per-stage improvement comparison.
    """
    optimizations = {}
    for i, stage in enumerate(stages):
        modified = copy.deepcopy(stages)
        modified[i]["capacity"] += 1
        result = run_monte_carlo(arrival, modified)
        tp_gain_pct = (
            (result["throughput_mean"] - baseline["throughput_mean"])
            / baseline["throughput_mean"] * 100
            if baseline["throughput_mean"] > 0 else 0.0
        )
        wait_reduction_pct = (
            (baseline["avg_wait_mean"] - result["avg_wait_mean"])
            / baseline["avg_wait_mean"] * 100
            if baseline["avg_wait_mean"] > 0 else 0.0
        )
        optimizations[stage["name"]] = {
            "new_throughput_mean":  result["throughput_mean"],
            "new_throughput_std":   result["throughput_std"],
            "tp_gain_pct":          tp_gain_pct,
            "wait_reduction_pct":   wait_reduction_pct,
            "new_util_mean":        result["stages"].get(stage["name"], {}).get("util_mean", 0),
        }
    return optimizations


# ─────────────────────────────────────────────────────────────
#  PHASE 4 — GRAPHVIZ FLOW DIAGRAM
# ─────────────────────────────────────────────────────────────

def build_flow_diagram(stages: list, agg: dict) -> "graphviz.Digraph | None":
    if not GRAPHVIZ_AVAILABLE:
        return None

    dot = graphviz.Digraph(
        name="DigitalTwin",
        graph_attr={
            "rankdir": "LR",
            "bgcolor": "#0f172a",
            "fontname": "Courier New",
            "splines": "ortho",
        },
        node_attr={
            "fontname": "Courier New",
            "fontcolor": "#e2e8f0",
            "style": "filled",
            "shape": "box",
            "margin": "0.2,0.1",
        },
        edge_attr={
            "color": "#475569",
            "fontcolor": "#94a3b8",
            "fontname": "Courier New",
            "fontsize": "10",
        },
    )

    def util_color(u):
        if u < 70:   return "#16a34a"
        if u < 85:   return "#d97706"
        return "#dc2626"

    # Source node
    dot.node("SOURCE", "▶ SOURCE", fillcolor="#1e40af", shape="ellipse")

    prev = "SOURCE"
    for s in stages:
        name = s["name"]
        util = agg["stages"].get(name, {}).get("util_mean", 0)
        q    = agg["stages"].get(name, {}).get("queue_mean", 0)

        # Queue node
        q_id = f"Q_{name}"
        dot.node(q_id, f"QUEUE\n{name}\nAvg: {q:.1f}", fillcolor="#334155", shape="cylinder")

        # Process node
        p_id = f"P_{name}"
        label = f"{name}\nUtil: {util:.1f}%\nCap: {s['capacity']}"
        dot.node(p_id, label, fillcolor=util_color(util))

        dot.edge(prev,  q_id)
        dot.edge(q_id,  p_id)

        # Routing edges
        if not s.get("route_all", True):
            pct_next   = s.get("route_pct_next",   0)
            pct_rework = s.get("route_pct_rework",  0)
            pct_scrap  = s.get("route_pct_scrap",   0)
            prev = p_id
            # routing labels added on the outgoing edge
            dot.edge(p_id, "NEXT_PLACEHOLDER",
                     label=f"→Next {pct_next:.0f}%")
        else:
            prev = p_id

    # Sink node
    dot.node("SINK", "⬛ SINK", fillcolor="#7c3aed", shape="ellipse")
    dot.edge(prev, "SINK")

    return dot


# ─────────────────────────────────────────────────────────────
#  PHASE 4 — PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────

CHART_THEME = {
    "paper_bgcolor": "#0f172a",
    "plot_bgcolor":  "#1e293b",
    "font_color":    "#e2e8f0",
    "grid_color":    "#334155",
}

def chart_wait_vs_proc(agg: dict) -> go.Figure:
    names      = list(agg["stages"].keys())
    wait_means = [agg["stages"][n]["wait_mean"] for n in names]
    wait_stds  = [agg["stages"][n]["wait_std"]  for n in names]
    proc_means = [agg["stages"][n]["proc_mean"] for n in names]
    proc_stds  = [agg["stages"][n]["proc_std"]  for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Wait Time",
        x=names, y=wait_means,
        error_y=dict(type="data", array=wait_stds, visible=True),
        marker_color="#3b82f6",
    ))
    fig.add_trace(go.Bar(
        name="Avg Process Time",
        x=names, y=proc_means,
        error_y=dict(type="data", array=proc_stds, visible=True),
        marker_color="#10b981",
    ))

    fig.update_layout(
        title="Wait Time vs Processing Time per Stage (Mean ± Std Dev)",
        barmode="group",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(color=CHART_THEME["font_color"]),
        yaxis_title="Time (minutes)",
        legend_bgcolor="#1e293b",
    )
    fig.update_xaxes(gridcolor=CHART_THEME["grid_color"])
    fig.update_yaxes(gridcolor=CHART_THEME["grid_color"])
    return fig


def chart_utilization(agg: dict) -> go.Figure:
    names = list(agg["stages"].keys())
    utils = [agg["stages"][n]["util_mean"] for n in names]
    stds  = [agg["stages"][n]["util_std"]  for n in names]
    colors = [
        "#16a34a" if u < 70 else "#d97706" if u < 85 else "#dc2626"
        for u in utils
    ]

    fig = go.Figure(go.Bar(
        x=names, y=utils,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=colors,
    ))
    fig.add_hline(y=70, line_dash="dot", line_color="#d97706",
                  annotation_text="70% threshold", annotation_font_color="#d97706")
    fig.add_hline(y=85, line_dash="dot", line_color="#dc2626",
                  annotation_text="85% threshold", annotation_font_color="#dc2626")
    fig.update_layout(
        title="Resource Utilization per Stage (Mean ± Std Dev)",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(color=CHART_THEME["font_color"]),
        yaxis_title="Utilization (%)",
        yaxis_range=[0, 110],
    )
    fig.update_xaxes(gridcolor=CHART_THEME["grid_color"])
    fig.update_yaxes(gridcolor=CHART_THEME["grid_color"])
    return fig


def chart_queue_over_time(agg: dict) -> go.Figure:
    """Plot average queue length trend from the last run."""
    last_run = agg["raw_runs"][-1]
    fig = go.Figure()

    for name, sdata in last_run["stages"].items():
        qlog = sdata.get("queue_log", [])
        if not qlog:
            continue
        times = [q[0] for q in qlog]
        qlen  = [q[1] for q in qlog]
        fig.add_trace(go.Scatter(
            x=times, y=qlen,
            mode="lines",
            name=name,
            line=dict(width=1.5),
        ))

    fig.add_vline(x=WARMUP_TIME, line_dash="dash", line_color="#94a3b8",
                  annotation_text="Warm-Up End", annotation_font_color="#94a3b8")
    fig.update_layout(
        title="Queue Length Over Time (Last Run)",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(color=CHART_THEME["font_color"]),
        xaxis_title="Simulation Time (min)",
        yaxis_title="Queue Length",
        legend_bgcolor="#1e293b",
    )
    fig.update_xaxes(gridcolor=CHART_THEME["grid_color"])
    fig.update_yaxes(gridcolor=CHART_THEME["grid_color"])
    return fig


def chart_optimization(baseline: dict, optimizations: dict) -> go.Figure:
    names  = list(optimizations.keys())
    gains  = [optimizations[n]["tp_gain_pct"]       for n in names]
    w_reds = [optimizations[n]["wait_reduction_pct"] for n in names]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Throughput Gain (%)", "Wait Reduction (%)"))
    fig.add_trace(go.Bar(x=names, y=gains,
                         marker_color="#6366f1", name="TP Gain %"), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=w_reds,
                         marker_color="#06b6d4", name="Wait Reduction %"), row=1, col=2)

    fig.update_layout(
        title="Effect of Adding +1 Resource per Stage",
        showlegend=False,
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=dict(color=CHART_THEME["font_color"]),
    )
    for axis in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{axis: dict(gridcolor=CHART_THEME["grid_color"])})
    return fig


# ─────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────────

def set_page_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        background-color: #0a0f1e;
        color: #e2e8f0;
    }
    .stApp { background-color: #0a0f1e; }

    h1, h2, h3 { font-family: 'Share Tech Mono', monospace; color: #38bdf8; }
    h4, h5     { font-family: 'Rajdhani', sans-serif; color: #94a3b8; font-weight: 600; }

    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #7c3aed);
        color: #f8fafc;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #9333ea);
        box-shadow: 0 0 12px #7c3aed88;
    }

    .stExpander > div:first-child {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 4px;
    }

    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.8rem;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        font-family: 'Share Tech Mono', monospace;
    }

    .bottleneck-box {
        background: #450a0a;
        border: 2px solid #dc2626;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-family: 'Share Tech Mono', monospace;
        color: #fca5a5;
    }
    .ok-box {
        background: #052e16;
        border: 2px solid #16a34a;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-family: 'Share Tech Mono', monospace;
        color: #86efac;
    }
    .warn-box {
        background: #1c1001;
        border: 2px solid #d97706;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-family: 'Share Tech Mono', monospace;
        color: #fcd34d;
    }
    .divider {
        border: none;
        border-top: 1px solid #334155;
        margin: 1.2rem 0;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)


def header_banner():
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid #3730a3;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    ">
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#6366f1; letter-spacing:0.15em;">
            LOOMIS-UNIVERSAL ● INDUSTRIAL INTELLIGENCE DIVISION
        </div>
        <h1 style="margin:0.3rem 0 0 0; font-size:2rem; color:#e0f2fe;">
            Digital Twin Generator
        </h1>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#475569; margin-top:0.3rem;">
            L-UDTG v1.0 ● DES ● Monte Carlo ● Stochastic ● FIFO ● Triangular
        </div>
    </div>
    """, unsafe_allow_html=True)


def tri_inputs(label: str, key_prefix: str, defaults=(1.0, 5.0, 10.0)):
    cols = st.columns(3)
    mn   = cols[0].number_input(f"Min",  min_value=0.01, value=float(defaults[0]),
                                 key=f"{key_prefix}_min",  step=0.5, format="%.2f")
    mode = cols[1].number_input(f"Mode", min_value=0.01, value=float(defaults[1]),
                                 key=f"{key_prefix}_mode", step=0.5, format="%.2f")
    mx   = cols[2].number_input(f"Max",  min_value=0.01, value=float(defaults[2]),
                                 key=f"{key_prefix}_max",  step=0.5, format="%.2f")
    return mn, mode, mx


def metric_card(label: str, mean_val: float, std_val: float, unit: str = ""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-sub">{label}</div>
        <div class="metric-val">{mean_val:.2f}{unit}</div>
        <div class="metric-sub">± {std_val:.2f} std dev</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "num_stages":   2,
        "sim_results":  None,
        "opt_results":  None,
        "errors":       [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="L-UDTG | Digital Twin Generator",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    set_page_style()
    init_state()
    header_banner()

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Simulation Control")
        sim_time_val  = st.number_input("Sim Duration (min)",  value=SIM_TIME,    min_value=60,  step=60)
        num_runs_val  = st.number_input("Monte Carlo Runs",    value=NUM_RUNS,    min_value=1,   max_value=30, step=1)
        warmup_val    = st.number_input("Warm-Up Period (min)",value=WARMUP_TIME, min_value=0,   step=10)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 📦 Stage Count")
        n_stages = st.number_input("Number of Stages", min_value=1, max_value=10,
                                   value=st.session_state["num_stages"], step=1)
        st.session_state["num_stages"] = int(n_stages)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 💾 Config Export / Import")
        export_btn = st.button("📥 Export Config JSON")
        uploaded   = st.file_uploader("Import Config JSON", type=["json"])

    # ── TAB LAYOUT ───────────────────────────────────────────
    tab_wizard, tab_results, tab_opt, tab_json = st.tabs([
        "🔧 Wizard (Input)",
        "📊 Dashboard (Results)",
        "🚀 Optimization",
        "📋 JSON Bridge",
    ])

    # ══════════════════════════════════════════════════════════
    #  TAB 1 — WIZARD
    # ══════════════════════════════════════════════════════════
    with tab_wizard:
        st.subheader("Phase 1 — System Configuration Wizard")

        # ── SECTION 1: ARRIVAL ──────────────────────────────
        with st.expander("▶  SECTION 1 — Arrival System", expanded=True):
            st.markdown("**Inter-Arrival Time (minutes between units)**")
            st.caption("Fastest day → Normal day → Slowest day")
            arr_min, arr_mode, arr_max = tri_inputs("Arrival", "arr", (1.0, 3.0, 8.0))

        # ── SECTION 2: STAGES ───────────────────────────────
        stages_input = []
        for i in range(st.session_state["num_stages"]):
            with st.expander(f"▶  STAGE {i+1}", expanded=(i == 0)):
                c1, c2 = st.columns([2, 1])
                sname = c1.text_input("Stage Name", value=f"Stage_{i+1}", key=f"sname_{i}")
                cap   = c2.number_input("Capacity (machines/workers)", min_value=1,
                                        value=1, step=1, key=f"cap_{i}")

                st.markdown("**Process Time (min)** — Triangular Distribution")
                p_min, p_mode, p_max = tri_inputs("Process", f"proc_{i}", (3.0, 7.0, 15.0))

                st.markdown("**Queue Behavior when resource is busy:**")
                qb = st.radio("Behavior", ["Wait (FIFO)", "Skip", "Rework"],
                              key=f"qb_{i}", horizontal=True)
                rework_prob = 0.0
                if qb == "Rework":
                    rework_prob = st.slider("Rework probability", 0.0, 1.0, 0.1,
                                            key=f"rp_{i}")
                queue_map = {"Wait (FIFO)": "wait", "Skip": "skip", "Rework": "rework"}

                st.markdown("**Machine Reliability:**")
                enable_fail = st.checkbox("Enable Failure Modeling", key=f"fail_{i}")
                tbf_min = tbf_mode = tbf_max = 999.0
                rep_min = rep_mode = rep_max = 1.0
                if enable_fail:
                    st.caption("Time Between Failures (min)")
                    tbf_min, tbf_mode, tbf_max = tri_inputs("TBF", f"tbf_{i}", (60.0, 120.0, 240.0))
                    st.caption("Repair Time (min)")
                    rep_min, rep_mode, rep_max = tri_inputs("Repair", f"rep_{i}", (5.0, 15.0, 30.0))

                st.markdown("**Routing Logic:**")
                route_all = st.radio("Route all units to next stage?",
                                     ["Yes", "No — define split"],
                                     key=f"ra_{i}", horizontal=True) == "Yes"
                pct_next = pct_rework = pct_scrap = 0.0
                if not route_all:
                    c1r, c2r, c3r = st.columns(3)
                    pct_next   = c1r.number_input("% Next Stage",  0.0, 100.0, 80.0, key=f"pn_{i}")
                    pct_rework = c2r.number_input("% Rework",      0.0, 100.0, 10.0, key=f"pr_{i}")
                    pct_scrap  = c3r.number_input("% Scrap",       0.0, 100.0, 10.0, key=f"ps_{i}")

                stages_input.append({
                    "name":          sname,
                    "capacity":      int(cap),
                    "proc_min":      p_min,
                    "proc_mode":     p_mode,
                    "proc_max":      p_max,
                    "queue_behavior":queue_map[qb],
                    "rework_prob":   rework_prob,
                    "enable_failures": enable_fail,
                    "tbf_min":       tbf_min,
                    "tbf_mode":      tbf_mode,
                    "tbf_max":       tbf_max,
                    "repair_min":    rep_min,
                    "repair_mode":   rep_mode,
                    "repair_max":    rep_max,
                    "route_all":     route_all,
                    "route_pct_next":   pct_next,
                    "route_pct_rework": pct_rework,
                    "route_pct_scrap":  pct_scrap,
                })

        arrival_cfg = {
            "min": arr_min, "mode": arr_mode, "max": arr_max
        }

        # ── SECTION 3: GUARDRAIL VALIDATION ─────────────────
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.subheader("Guardrail Validation")
        errors = validate_all_inputs(arrival_cfg, stages_input)
        st.session_state["errors"] = errors

        if errors:
            st.markdown(
                "<div class='bottleneck-box'><b>⛔ VALIDATION FAILED — Simulation Blocked</b><br>"
                + "<br>".join(f"• {e}" for e in errors)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='ok-box'>✅ All inputs valid — ready to simulate</div>",
                        unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── RUN BUTTON ───────────────────────────────────────
        col_run, col_opt = st.columns(2)
        run_clicked = col_run.button("▶  RUN SIMULATION", disabled=bool(errors), use_container_width=True)
        opt_clicked = col_opt.button("🚀  RUN + OPTIMIZE", disabled=bool(errors), use_container_width=True)

        if run_clicked or opt_clicked:
            with st.spinner("Running Monte Carlo simulation..."):
                agg = run_monte_carlo(arrival_cfg, stages_input, int(num_runs_val))
                st.session_state["sim_results"] = {
                    "agg": agg,
                    "arrival": arrival_cfg,
                    "stages": stages_input,
                }

            if opt_clicked:
                with st.spinner("Running optimization layer..."):
                    opt = run_optimization(arrival_cfg, stages_input, agg)
                    st.session_state["opt_results"] = opt

            st.success("Simulation complete! Switch to the Dashboard tab.")

        # Export
        if export_btn and stages_input:
            payload = build_json_payload(arrival_cfg, stages_input)
            json_str = json.dumps(payload, indent=2)
            st.sidebar.download_button(
                "⬇ Download JSON",
                data=json_str,
                file_name="ludtg_config.json",
                mime="application/json",
            )

        # Import
        if uploaded:
            raw = uploaded.read().decode("utf-8")
            parsed, msg = safe_parse_json(raw)
            if parsed:
                st.sidebar.success(f"Config loaded: {len(parsed['stages'])} stages")
            else:
                st.sidebar.error(f"Import failed: {msg}")

    # ══════════════════════════════════════════════════════════
    #  TAB 2 — DASHBOARD
    # ══════════════════════════════════════════════════════════
    with tab_results:
        if st.session_state["sim_results"] is None:
            st.info("Run the simulation from the Wizard tab to see results here.")
        else:
            res    = st.session_state["sim_results"]
            agg    = res["agg"]
            stages = res["stages"]

            st.subheader("Phase 4 — Performance Dashboard")
            st.caption(f"Results aggregated over {agg['num_runs']} Monte Carlo runs")

            # ── KPI Row ──────────────────────────────────────
            kpi_cols = st.columns(4)
            with kpi_cols[0]: metric_card("Throughput (parts)", agg["throughput_mean"], agg["throughput_std"])
            with kpi_cols[1]: metric_card("Avg Wait Time", agg["avg_wait_mean"], agg["avg_wait_std"], " min")
            # Find bottleneck
            bn_name = max(agg["stages"], key=lambda n: agg["stages"][n]["util_mean"])
            bn_util = agg["stages"][bn_name]["util_mean"]
            bn_queue= agg["stages"][bn_name]["queue_mean"]
            with kpi_cols[2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-sub">Bottleneck Stage</div>
                    <div class="metric-val" style="font-size:1.2rem;">{bn_name}</div>
                    <div class="metric-sub">{bn_util:.1f}% util | Q={bn_queue:.1f}</div>
                </div>""", unsafe_allow_html=True)
            with kpi_cols[3]:
                metric_card("Num Stages", float(len(stages)), 0.0)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # ── Bottleneck Alert ─────────────────────────────
            if bn_util > 85 and bn_queue > 0.5:
                st.markdown(
                    f"<div class='bottleneck-box'>🔴 BOTTLENECK DETECTED: <b>{bn_name}</b> "
                    f"(Utilization: {bn_util:.1f}%, Avg Queue: {bn_queue:.1f} units). "
                    "This stage is constraining system throughput.</div>",
                    unsafe_allow_html=True,
                )
            elif bn_util > 70:
                st.markdown(
                    f"<div class='warn-box'>⚠️ WATCH: <b>{bn_name}</b> "
                    f"at {bn_util:.1f}% utilization — approaching bottleneck territory.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='ok-box'>✅ System appears balanced — no critical bottleneck detected.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # ── Flow Diagram ─────────────────────────────────
            st.subheader("Shop Floor Flow Diagram")
            if GRAPHVIZ_AVAILABLE:
                dot = build_flow_diagram(stages, agg)
                if dot:
                    st.graphviz_chart(dot.source, use_container_width=True)
            else:
                st.warning("graphviz Python package not installed. "
                           "Install with: pip install graphviz")

            # ── Charts ───────────────────────────────────────
            st.plotly_chart(chart_wait_vs_proc(agg),   use_container_width=True)
            st.plotly_chart(chart_utilization(agg),    use_container_width=True)
            st.plotly_chart(chart_queue_over_time(agg), use_container_width=True)

            # ── Per-Stage Table ───────────────────────────────
            st.subheader("Per-Stage Metrics Summary")
            tbl_data = []
            for name, s in agg["stages"].items():
                tbl_data.append({
                    "Stage":           name,
                    "Util Mean %":     f"{s['util_mean']:.1f}",
                    "Util Std %":      f"{s['util_std']:.1f}",
                    "Avg Wait (min)":  f"{s['wait_mean']:.2f}",
                    "Avg Proc (min)":  f"{s['proc_mean']:.2f}",
                    "Avg Queue Len":   f"{s['queue_mean']:.2f}",
                })
            st.table(tbl_data)

    # ══════════════════════════════════════════════════════════
    #  TAB 3 — OPTIMIZATION
    # ══════════════════════════════════════════════════════════
    with tab_opt:
        if st.session_state["sim_results"] is None:
            st.info("Run the simulation first.")
        else:
            st.subheader("Phase 5 — Optimization Layer (Decision Support)")

            res = st.session_state["sim_results"]
            agg = res["agg"]

            if st.session_state["opt_results"] is None:
                if st.button("▶  Run Optimization Analysis"):
                    with st.spinner("Testing +1 resource per stage..."):
                        opt = run_optimization(res["arrival"], res["stages"], agg)
                        st.session_state["opt_results"] = opt
                    st.rerun()
            else:
                opt = st.session_state["opt_results"]

                # Narrative
                best_stage = max(opt, key=lambda n: opt[n]["tp_gain_pct"])
                best        = opt[best_stage]
                bn_name2    = max(agg["stages"], key=lambda n: agg["stages"][n]["util_mean"])

                st.markdown(
                    f"<div class='bottleneck-box'>"
                    f"🔍 <b>Primary Bottleneck:</b> {bn_name2} "
                    f"(Utilization: {agg['stages'][bn_name2]['util_mean']:.1f}%).<br>"
                    f"✅ <b>Best ROI:</b> Adding 1 machine to <b>{best_stage}</b> increases "
                    f"throughput by <b>{best['tp_gain_pct']:.1f}%</b> and reduces average wait "
                    f"by <b>{best['wait_reduction_pct']:.1f}%</b> "
                    f"(averaged over {NUM_RUNS} runs)."
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                st.plotly_chart(chart_optimization(agg, opt), use_container_width=True)

                # Detail table
                tbl = []
                for name, o in opt.items():
                    tbl.append({
                        "Stage":              name,
                        "TP Gain %":          f"{o['tp_gain_pct']:.1f}",
                        "Wait Reduction %":   f"{o['wait_reduction_pct']:.1f}",
                        "New TP Mean":        f"{o['new_throughput_mean']:.1f}",
                        "New TP Std":         f"{o['new_throughput_std']:.1f}",
                        "New Util %":         f"{o['new_util_mean']:.1f}",
                    })
                st.table(tbl)

    # ══════════════════════════════════════════════════════════
    #  TAB 4 — JSON BRIDGE
    # ══════════════════════════════════════════════════════════
    with tab_json:
        st.subheader("Phase 3 — Versioned JSON Data Bridge")

        if st.session_state["sim_results"]:
            res = st.session_state["sim_results"]
            payload = build_json_payload(res["arrival"], res["stages"])
            json_str = json.dumps(payload, indent=2)

            st.code(json_str, language="json")

            st.download_button(
                "⬇ Download Configuration JSON",
                data=json_str,
                file_name="ludtg_config.json",
                mime="application/json",
            )
        else:
            st.info("Run a simulation to generate the JSON payload.")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Validate / Import Custom JSON**")
        raw_input = st.text_area("Paste JSON configuration here", height=200)
        if st.button("Validate & Parse"):
            if raw_input.strip():
                parsed, msg = safe_parse_json(raw_input)
                if parsed:
                    st.success(f"✅ Valid — {len(parsed['stages'])} stage(s) found.")
                    st.json(parsed)
                else:
                    st.error(f"❌ {msg}")
            else:
                st.warning("Paste a JSON config above first.")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
