"""
Explainability API endpoints:
- Synthetic data seeding for demo
- Attribute space (common attributes for all algorithms)
- Step-by-step full pipeline (Granger → PCMCI → NOTEARS → Evaluate → AI fallback)
- Unknown cause discovery + AI prediction layer
- Audit trail + PDF export
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import io
import numpy as np

router = APIRouter(tags=["Explainability"])


def _sanitize_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_numpy(item) for item in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Pydantic Models ──────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    zone: Optional[str] = None
    max_lag: int = 3
    alpha: float = 0.05
    l1_lambda: float = 0.01
    hours: int = 168

class AuditEntryRequest(BaseModel):
    zone: str
    event_type: str
    event_description: Optional[str] = ""
    event_timestamp: Optional[str] = None
    severity: Optional[str] = "medium"
    detection_method: Optional[str] = "system"
    action_taken: Optional[str] = None
    resolved: Optional[bool] = False
    resolution_description: Optional[str] = None
    outcome: Optional[str] = None
    operator: Optional[str] = "system"
    notes: Optional[str] = None

class AuditActionRequest(BaseModel):
    action: str
    operator: str = "system"

class AuditResolveRequest(BaseModel):
    description: str
    outcome: str = "resolved"

class AuditNoteRequest(BaseModel):
    note: str

class AuditTrailResponse(BaseModel):
    zone: Optional[str]
    entries: List[Dict[str, Any]]
    total: int

class IngestAuditRequest(BaseModel):
    entries: List[Dict[str, Any]]


# ── Seed Demo Data ───────────────────────────────────────────────────────────

# Mapping: event type → which timeseries attribute to spike
EVENT_TO_ATTRIBUTE = {
    "flood": "flooding_level",
    "flooding": "flooding_level",
    "rain": "rainfall",
    "rainfall": "rainfall",
    "traffic_jam": "traffic_congestion",
    "traffic": "traffic_congestion",
    "accident": "accident_count",
    "power_outage": "power_outage",
    "power": "power_outage",
    "fire": "fire_incident",
    "heatwave": "heatwave_index",
    "heat": "heatwave_index",
    "pollution": "air_quality_index",
    "air_quality": "air_quality_index",
    "construction": "construction_activity",
    "crowd": "public_event_crowd",
    "public_event": "public_event_crowd",
    "industrial": "industrial_discharge",
    "drainage": "drainage_load",
    "water": "water_supply_pressure",
    "emergency": "emergency_delay",
}

SEVERITY_MULTIPLIER = {"low": 0.5, "medium": 1.0, "high": 2.0, "critical": 3.0}


@router.post("/inject-event")
async def inject_event_to_timeseries(
    zone: str = Query("Bibwewadi"),
    event_type: str = Query(...),
    severity: str = Query("high"),
):
    """
    Inject a causal event into the timeseries data.
    This writes spike rows so the explainability algorithms pick up real injected events.
    Also triggers cascade effects based on known causal priors.
    """
    from database import db
    from causal_engine.attribute_space import ATTRIBUTE_NAMES, KNOWN_CAUSES

    attr = EVENT_TO_ATTRIBUTE.get(event_type.lower())
    if not attr or attr not in ATTRIBUTE_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown event type '{event_type}'. Valid: {list(EVENT_TO_ATTRIBUTE.keys())}")

    mult = SEVERITY_MULTIPLIER.get(severity.lower(), 1.0)
    now = datetime.now()

    # Fetch recent data to get baseline values
    recent = db.get_timeseries(zone, hours=24)
    baseline = {}
    if recent:
        for a in ATTRIBUTE_NAMES:
            vals = [float(r.get(a, 0)) for r in recent]
            baseline[a] = sum(vals) / len(vals) if vals else 0
    else:
        baseline = {a: 0 for a in ATTRIBUTE_NAMES}

    # Build spike row: baseline + spike on the target attribute and downstream cascade
    rows_added = 0
    for hour_offset in range(3):  # inject a 3-hour spike
        row = dict(baseline)
        row["timestamp"] = (now + timedelta(hours=hour_offset)).isoformat()

        # Primary spike
        spike_val = max(baseline.get(attr, 0) * (1.5 + mult), 5 * mult)
        row[attr] = round(spike_val, 2)

        # Cascade: if the spiked attribute is a known cause of other attributes, bump them too
        for target, causes in KNOWN_CAUSES.items():
            if attr in causes and target != attr:
                cascade_val = baseline.get(target, 0) * (1.0 + 0.3 * mult) + 1.0 * mult
                row[target] = round(max(row.get(target, 0), cascade_val), 2)

        db.store_timeseries(zone, row)
        rows_added += 1

    return {
        "status": "injected",
        "zone": zone,
        "event_type": event_type,
        "attribute_spiked": attr,
        "severity": severity,
        "rows_added": rows_added,
        "message": f"Injected {event_type} event into timeseries for {zone}. "
                   f"Re-run the pipeline to see updated causal graph.",
    }


@router.post("/seed-demo-data")
async def seed_demo_data(zone: str = Query("Bibwewadi"), hours: int = Query(168)):
    """Seed synthetic time-series data for demo/judging. Data has known causal relationships."""
    from simulation.synthetic_seeder import seed_zone
    count = seed_zone(zone=zone, hours=hours)
    return {"status": "seeded", "zone": zone, "rows": count, "hours": hours}


# ── Attribute Space Endpoints ─────────────────────────────────────────────────

@router.get("/attributes")
async def get_attributes():
    """Return the full common attribute space used by all algorithms."""
    from causal_engine.attribute_space import get_attributes_info, ATTRIBUTE_NAMES, KNOWN_CAUSES
    return {
        "attributes": get_attributes_info(),
        "attribute_names": ATTRIBUTE_NAMES,
        "known_causal_priors": KNOWN_CAUSES,
        "total": len(ATTRIBUTE_NAMES),
    }


@router.get("/attributes/timeseries")
async def get_attribute_timeseries(zone: str = Query(...), hours: int = Query(168)):
    """Get time-series data for the common attribute space."""
    from database import db
    data = db.get_timeseries(zone, hours=hours)
    from causal_engine.attribute_space import ATTRIBUTE_NAMES
    return {"zone": zone, "data": data, "count": len(data), "attributes": ATTRIBUTE_NAMES}


# ── Helper: build consensus edges ─────────────────────────────────────────────

def _build_consensus(g_edges, p_edges, n_edges):
    """Compute consensus edges from Granger, PCMCI, NOTEARS edge lists."""
    edge_votes = {}
    for e in g_edges:
        key = f"{e['from']}->{e['to']}"
        edge_votes[key] = edge_votes.get(key, {"from": e["from"], "to": e["to"], "algorithms": [], "scores": []})
        edge_votes[key]["algorithms"].append("granger")
        edge_votes[key]["scores"].append(e.get("f_stat", 0))
    for e in p_edges:
        key = f"{e['from']}->{e['to']}"
        edge_votes[key] = edge_votes.get(key, {"from": e["from"], "to": e["to"], "algorithms": [], "scores": []})
        edge_votes[key]["algorithms"].append("pcmci")
        edge_votes[key]["scores"].append(e.get("mci_value", 0))
    for e in n_edges:
        key = f"{e['from']}->{e['to']}"
        edge_votes[key] = edge_votes.get(key, {"from": e["from"], "to": e["to"], "algorithms": [], "scores": []})
        edge_votes[key]["algorithms"].append("notears")
        edge_votes[key]["scores"].append(e.get("abs_weight", 0))
    consensus = []
    for key, info in edge_votes.items():
        consensus.append({
            "from": info["from"], "to": info["to"],
            "agreement": len(info["algorithms"]), "algorithms": info["algorithms"],
            "avg_score": round(sum(info["scores"]) / len(info["scores"]), 4) if info["scores"] else 0,
        })
    consensus.sort(key=lambda e: e["agreement"], reverse=True)
    return consensus


def _find_unresolved_targets(consensus_edges):
    """Return target variables not explained by any consensus edge with agreement >= 2."""
    from causal_engine.attribute_space import KNOWN_CAUSES
    resolved = set()
    for e in consensus_edges:
        if e.get("agreement", 0) >= 2:
            resolved.add(e["to"])
    return [t for t in KNOWN_CAUSES if t not in resolved]


# ── Step-by-Step Full Pipeline ────────────────────────────────────────────────

@router.post("/full-pipeline")
async def run_full_pipeline(req: PipelineRequest):
    """
    Step-by-step causal graph generation pipeline for judges/demonstration.

    Returns a structured response with 5 steps:
    1. Data loaded from common attribute space
    2. Step 1 — Granger Causality: edges + explanation
    3. Step 2 — PCMCI: edges + explanation
    4. Step 3 — NOTEARS: edges + explanation
    5. Step 4 — Evaluate consensus causal graph
    6. Step 5 — If unresolved targets exist, AI Prediction Layer
       predicts root cause and re-evaluates (shown in Unknown Causes)
    """
    from database import db as database
    from causal_engine.granger import run_granger_full
    from causal_engine.pcmci import run_pcmci
    from causal_engine.notears import run_notears
    from analysis.ai_prediction_layer import predict_root_cause, predict_all_targets
    from causal_engine.attribute_space import KNOWN_CAUSES

    zone = req.zone or "Bibwewadi"
    timeseries = database.get_timeseries(zone, hours=req.hours)

    # If no data in DB, auto-seed synthetic data for demo
    if len(timeseries) < 10:
        from simulation.synthetic_seeder import seed_zone
        seed_zone(zone=zone, hours=req.hours)
        timeseries = database.get_timeseries(zone, hours=req.hours)
        if len(timeseries) < 10:
            raise HTTPException(status_code=400, detail="Could not generate sufficient data")

    now = datetime.now().isoformat()
    data_points = len(timeseries)

    # ── STEP 1: Granger Causality ──
    g_result = run_granger_full(timeseries, max_lag=req.max_lag)
    # Progressive graph: after Granger, all its edges form the initial graph
    graph_after_granger = [
        {"from": e["from"], "to": e["to"], "source": "granger",
         "score": e.get("f_stat", 0), "lag": e.get("lag", 0)}
        for e in g_result["edges"]
    ]
    step1 = {
        "step": 1,
        "algorithm": "Granger Causality",
        "status": "completed",
        "explanation": (
            "Granger Causality tests whether past values of variable X help predict "
            "variable Y beyond Y's own past. Uses VAR (Vector Auto-Regression) with "
            "F-test at lags 1-{}. Edges are directed: X → Y means X Granger-causes Y."
        ).format(req.max_lag),
        "method": "Pairwise F-test on {} variable pairs across {} data points".format(
            len(g_result.get("variables", [])) * (len(g_result.get("variables", [])) - 1),
            g_result["data_points"],
        ),
        "edges": g_result["edges"],
        "edge_count": len(g_result["edges"]),
        "variables": g_result.get("variables", []),
        "data_points": g_result["data_points"],
        "parameters": {"max_lag": req.max_lag, "significance": 0.05},
        "graph_so_far": graph_after_granger,
        "graph_nodes": sorted({e["from"] for e in graph_after_granger} | {e["to"] for e in graph_after_granger}),
    }

    # ── STEP 2: PCMCI ──
    p_result = run_pcmci(timeseries, tau_max=min(req.max_lag, 3), alpha=req.alpha)
    # Progressive graph: merge Granger + PCMCI edges (union, track which algos found each)
    _merged_2 = {}
    for e in g_result["edges"]:
        key = f"{e['from']}->{e['to']}"
        _merged_2[key] = {"from": e["from"], "to": e["to"], "sources": ["granger"], "scores": [e.get("f_stat", 0)]}
    for e in p_result["edges"]:
        key = f"{e['from']}->{e['to']}"
        if key in _merged_2:
            _merged_2[key]["sources"].append("pcmci")
            _merged_2[key]["scores"].append(e.get("mci_value", 0))
        else:
            _merged_2[key] = {"from": e["from"], "to": e["to"], "sources": ["pcmci"], "scores": [e.get("mci_value", 0)]}
    graph_after_pcmci = [{"from": v["from"], "to": v["to"], "sources": v["sources"], "agreement": len(v["sources"])} for v in _merged_2.values()]
    step2 = {
        "step": 2,
        "algorithm": "PCMCI",
        "status": "completed",
        "explanation": (
            "PCMCI (Peter-Clark Momentary Conditional Independence) is a constraint-based "
            "method. Phase 1: PC-stable algorithm removes spurious edges via conditional "
            "independence tests. Phase 2: MCI test orients remaining edges using "
            "time-lag structure. More conservative than Granger — eliminates indirect effects."
        ),
        "method": "PC-stable skeleton ({} links) → MCI orientation → {} significant edges".format(
            p_result.get("skeleton_links", 0),
            len(p_result["edges"]),
        ),
        "edges": p_result["edges"],
        "edge_count": len(p_result["edges"]),
        "variables": p_result.get("variables", []),
        "data_points": p_result["data_points"],
        "parameters": {"tau_max": min(req.max_lag, 3), "alpha": req.alpha},
        "graph_so_far": graph_after_pcmci,
        "graph_nodes": sorted({e["from"] for e in graph_after_pcmci} | {e["to"] for e in graph_after_pcmci}),
        "new_edges_added": len(p_result["edges"]),
        "edges_confirmed": sum(1 for e in graph_after_pcmci if e["agreement"] >= 2),
    }

    # ── STEP 3: NOTEARS ──
    n_result = run_notears(timeseries, l1_lambda=req.l1_lambda)
    # Progressive graph: merge all 3 algorithms
    _merged_3 = {}
    for e in g_result["edges"]:
        key = f"{e['from']}->{e['to']}"
        _merged_3[key] = {"from": e["from"], "to": e["to"], "sources": ["granger"]}
    for e in p_result["edges"]:
        key = f"{e['from']}->{e['to']}"
        if key in _merged_3:
            _merged_3[key]["sources"].append("pcmci")
        else:
            _merged_3[key] = {"from": e["from"], "to": e["to"], "sources": ["pcmci"]}
    for e in n_result["edges"]:
        key = f"{e['from']}->{e['to']}"
        if key in _merged_3:
            _merged_3[key]["sources"].append("notears")
        else:
            _merged_3[key] = {"from": e["from"], "to": e["to"], "sources": ["notears"]}
    graph_after_notears = [{"from": v["from"], "to": v["to"], "sources": v["sources"], "agreement": len(v["sources"])} for v in _merged_3.values()]
    step3 = {
        "step": 3,
        "algorithm": "NOTEARS",
        "status": "completed",
        "explanation": (
            "NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented "
            "Lagrangian) learns a DAG by solving a continuous optimization problem. "
            "Minimizes least-squares loss with L1 sparsity penalty subject to an acyclicity "
            "constraint h(W) = tr(e^{W∘W}) - d = 0. Discovers instantaneous relationships."
        ),
        "method": "Augmented Lagrangian optimization, L1 penalty={}, DAG constraint h={:.6f}".format(
            req.l1_lambda,
            n_result.get("h_value", 0),
        ),
        "edges": n_result["edges"],
        "edge_count": len(n_result["edges"]),
        "variables": n_result.get("variables", []),
        "data_points": n_result["data_points"],
        "parameters": {"l1_lambda": req.l1_lambda, "h_value": n_result.get("h_value", 0)},
        "graph_so_far": graph_after_notears,
        "graph_nodes": sorted({e["from"] for e in graph_after_notears} | {e["to"] for e in graph_after_notears}),
        "new_edges_added": len(n_result["edges"]),
        "edges_confirmed": sum(1 for e in graph_after_notears if e["agreement"] >= 2),
        "edges_by_3": sum(1 for e in graph_after_notears if e["agreement"] >= 3),
    }

    # ── STEP 4: Evaluate Consensus Causal Graph ──
    consensus = _build_consensus(g_result["edges"], p_result["edges"], n_result["edges"])
    strong_edges = [e for e in consensus if e["agreement"] >= 2]
    unresolved = _find_unresolved_targets(consensus)

    step4 = {
        "step": 4,
        "algorithm": "Consensus Evaluation",
        "status": "completed",
        "explanation": (
            "The causal graph is evaluated by combining edges from all 3 algorithms. "
            "Edges found by 2+ algorithms form the consensus graph — these are high-confidence "
            "causal relationships. Targets from the known causal priors that have NO incoming "
            "consensus edge are flagged as 'unresolved' — the algorithms could not find their root cause."
        ),
        "method": "{} consensus edges (2+ agreement) from {} total unique edges".format(
            len(strong_edges), len(consensus),
        ),
        "consensus_edges": consensus,
        "strong_edges": strong_edges,
        "total_targets": len(KNOWN_CAUSES),
        "resolved_targets": [t for t in KNOWN_CAUSES if t not in unresolved],
        "unresolved_targets": unresolved,
        "causal_graph_complete": len(unresolved) == 0,
    }

    # ── STEP 5: AI Prediction Layer (only if unresolved targets) ──
    step5 = None
    ai_iteration_log = []
    if unresolved:
        # The AI prediction layer fills in for targets the algorithms couldn't resolve
        causal_result = {"consensus_edges": consensus}
        ai_predictions = predict_all_targets(timeseries, causal_result)

        # Build iteration log showing AI "re-running" for each unresolved target
        for target in unresolved:
            pred = ai_predictions["predictions"].get(target, {})
            top_causes = pred.get("predicted_causes", [])[:3]
            ai_iteration_log.append({
                "target": target,
                "iteration": 1,
                "ai_method": "Correlation (0.3) + Mutual Information (0.4) + Anomaly Coincidence (0.3)",
                "predicted_root_cause": top_causes[0]["variable"] if top_causes else "unknown",
                "confidence": top_causes[0]["total_score"] if top_causes else 0,
                "top_candidates": top_causes,
                "data_points": pred.get("data_points", 0),
                "variables_analyzed": pred.get("variables_analyzed", 0),
                "method_details": pred.get("method_details", {}),
            })

        # Build AI-augmented causal graph: original consensus + AI-predicted edges
        ai_augmented_edges = list(strong_edges)
        for entry in ai_iteration_log:
            if entry["predicted_root_cause"] != "unknown":
                ai_augmented_edges.append({
                    "from": entry["predicted_root_cause"],
                    "to": entry["target"],
                    "agreement": 0,
                    "algorithms": ["ai_prediction"],
                    "avg_score": round(entry["confidence"], 4),
                    "ai_predicted": True,
                })

        still_unresolved = [e["target"] for e in ai_iteration_log if e["predicted_root_cause"] == "unknown"]

        step5 = {
            "step": 5,
            "algorithm": "AI Prediction Layer",
            "status": "completed",
            "explanation": (
                "All 3 algorithms failed to identify root causes for {} target(s): {}. "
                "The AI Prediction Layer uses an ensemble of Pearson correlation ranking, "
                "mutual information (captures non-linear dependencies), and anomaly "
                "coincidence detection (z-score spike overlap) to predict the most likely "
                "root cause. Weights: Correlation=0.3, MI=0.4, Anomaly=0.3."
            ).format(len(unresolved), ", ".join(unresolved)),
            "method": "Ensemble prediction across {} data points, {} variables".format(
                data_points, len(g_result.get("variables", [])),
            ),
            "unresolved_targets": unresolved,
            "iterations": ai_iteration_log,
            "ai_predictions": ai_predictions,
            "augmented_causal_graph": ai_augmented_edges,
            "still_unresolved": still_unresolved,
            "causal_graph_complete": len(still_unresolved) == 0,
        }

    # ── Build final conclusion ──
    final_graph = step5["augmented_causal_graph"] if step5 else strong_edges
    # Identify key causal chains in the final graph
    _edge_map = {}
    for e in final_graph:
        _edge_map.setdefault(e["to"], []).append(e["from"])
    chains = []
    for target, sources in _edge_map.items():
        for src in sources:
            if src in _edge_map:
                for root in _edge_map[src]:
                    chains.append(f"{root} → {src} → {target}")

    conclusion = {
        "total_edges": len(final_graph),
        "algorithm_edges": len(strong_edges),
        "ai_predicted_edges": len(final_graph) - len(strong_edges) if step5 else 0,
        "resolved_targets": [t for t in KNOWN_CAUSES if t not in unresolved],
        "unresolved_targets": [e["target"] for e in ai_iteration_log if e.get("predicted_root_cause") == "unknown"] if step5 else unresolved,
        "ai_resolved_targets": [e["target"] for e in ai_iteration_log if e.get("predicted_root_cause") != "unknown"] if step5 else [],
        "key_causal_chains": chains[:10],
        "graph_complete": (step5["causal_graph_complete"] if step5 else len(unresolved) == 0),
        "summary": (
            "The causal graph has been fully constructed with {} edges. "
            "{} edges were discovered by algorithm consensus (2+ agreement), "
            "and {} edges were predicted by the AI layer. "
            "Key causal chains: {}. {}"
        ).format(
            len(final_graph),
            len(strong_edges),
            len(final_graph) - len(strong_edges) if step5 else 0,
            "; ".join(chains[:5]) if chains else "none detected",
            "All target variables now have identified root causes." if (step5 and step5["causal_graph_complete"]) or len(unresolved) == 0
            else f"Still unresolved: {', '.join(unresolved)}."
        ),
    }

    return _sanitize_numpy({
        "zone": zone,
        "data_points": data_points,
        "steps": [step1, step2, step3, step4] + ([step5] if step5 else []),
        "total_steps": 5 if step5 else 4,
        "final_causal_graph": final_graph,
        "conclusion": conclusion,
        "ai_fallback_activated": step5 is not None,
        "timestamp": now,
    })


# ── Unknown Cause Discovery + AI Prediction ──────────────────────────────────

@router.get("/unknown-causes/discover")
async def discover_unknown_causes_endpoint(zone: str = Query(...), hours: int = Query(168)):
    """
    Run unknown cause discovery + AI prediction layer.
    First discovers anomalies via residual/correlation/change-point analysis,
    then runs the AI prediction layer for targets the algorithms couldn't resolve.
    The AI prediction results are shown side-by-side with unknown cause findings.
    """
    from analysis.unknown_cause_engine import discover_unknown_causes
    from analysis.ai_prediction_layer import predict_all_targets
    from database import db
    from causal_engine.granger import run_granger_full
    from causal_engine.pcmci import run_pcmci
    from causal_engine.notears import run_notears

    timeseries = db.get_timeseries(zone, hours=hours)

    # Auto-seed if empty
    if len(timeseries) < 10:
        from simulation.synthetic_seeder import seed_zone
        seed_zone(zone=zone, hours=hours)
        timeseries = db.get_timeseries(zone, hours=hours)

    # 1. Standard unknown cause discovery
    uc_result = discover_unknown_causes(zone, timeseries)

    # 2. Run all 3 algorithms to find unresolved targets
    if len(timeseries) >= 10:
        g = run_granger_full(timeseries, max_lag=3)
        p = run_pcmci(timeseries, tau_max=3, alpha=0.05)
        n = run_notears(timeseries, l1_lambda=0.01)
        consensus = _build_consensus(g["edges"], p["edges"], n["edges"])
        unresolved = _find_unresolved_targets(consensus)
    else:
        consensus = []
        unresolved = []

    # 3. AI prediction for unresolved targets
    ai_predictions = None
    ai_iterations = []
    if len(timeseries) >= 10:
        causal_result = {"consensus_edges": consensus}
        ai_result = predict_all_targets(timeseries, causal_result)
        ai_predictions = ai_result

        # Build iteration log for display
        for target in unresolved:
            pred = ai_result["predictions"].get(target, {})
            top = pred.get("predicted_causes", [])[:5]
            ai_iterations.append({
                "target": target,
                "predicted_root_cause": top[0]["variable"] if top else "unknown",
                "confidence": top[0]["total_score"] if top else 0,
                "candidates": top,
                "method_details": pred.get("method_details", {}),
                "status": "ai_resolved" if top else "unresolved",
            })

    # ── Build conclusion summary ──
    conclusion_parts = []
    # Unknown variables from residual analysis
    residual_unknowns = [
        {"variable": k, "unexplained_ratio": v.get("unexplained_ratio", 0),
         "drift": v.get("residual_drift", 0)}
        for k, v in uc_result.get("residual_analysis", {}).items()
        if v.get("anomaly")
    ]
    # AI-predicted placements
    ai_placements = []
    for it in ai_iterations:
        if it.get("predicted_root_cause") and it["predicted_root_cause"] != "unknown":
            ai_placements.append({
                "unknown_target": it["target"],
                "predicted_cause": it["predicted_root_cause"],
                "confidence": it["confidence"],
                "graph_position": f"{it['predicted_root_cause']} → {it['target']}",
            })

    summary_lines = []
    if residual_unknowns:
        summary_lines.append(
            f"{len(residual_unknowns)} variable(s) have significant unexplained variance: "
            + ", ".join(f"{u['variable']} ({u['unexplained_ratio']:.0%} unexplained)" for u in residual_unknowns)
            + "."
        )
    if ai_placements:
        summary_lines.append(
            "The AI prediction layer identified root causes for these unknowns: "
            + "; ".join(
                f"{p['unknown_target']} ← {p['predicted_cause']} ({p['confidence']:.0%} confidence, graph edge: {p['graph_position']})"
                for p in ai_placements
            )
            + "."
        )
    change_pts = uc_result.get("change_points", [])
    if change_pts:
        summary_lines.append(
            f"{len(change_pts)} sudden shift(s) detected: "
            + ", ".join(f"{cp['variable']} ({cp['direction']}, {cp['magnitude']}σ)" for cp in change_pts)
            + "."
        )
    if not summary_lines:
        summary_lines.append("No unknown variables or anomalies detected — all patterns explained by known causes.")

    conclusion = {
        "summary": " ".join(summary_lines),
        "unknown_variables": residual_unknowns,
        "ai_graph_placements": ai_placements,
        "change_points_detected": len(change_pts),
        "total_anomalies": uc_result.get("total_findings", 0),
        "all_resolved": len(unresolved) == 0 or all(it.get("status") == "ai_resolved" for it in ai_iterations),
    }

    return _sanitize_numpy({
        "zone": uc_result["zone"],
        "discoveries": uc_result["discoveries"],
        "residual_analysis": uc_result.get("residual_analysis", {}),
        "correlation_spikes": uc_result.get("correlation_spikes", []),
        "change_points": uc_result.get("change_points", []),
        "total_findings": uc_result.get("total_findings", 0),
        # AI prediction layer results (merged)
        "unresolved_targets": unresolved,
        "ai_predictions": ai_predictions,
        "ai_iterations": ai_iterations,
        "ai_fallback_activated": len(unresolved) > 0,
        "conclusion": conclusion,
        "timestamp": uc_result.get("timestamp", datetime.now().isoformat()),
    })


@router.get("/unknown-causes/history")
async def get_unknown_causes_history(zone: Optional[str] = None, limit: int = 50):
    """Get past unknown cause discoveries."""
    from database import db
    causes = db.get_unknown_causes(zone=zone, limit=limit)
    return {"causes": causes, "count": len(causes)}


# ── Audit Trail Endpoints ────────────────────────────────────────────────────

@router.post("/audit/record")
async def create_audit_entry(req: AuditEntryRequest):
    """Create a new audit trail entry."""
    from analysis.audit_service import record_event_detection
    entry_id = record_event_detection(
        zone=req.zone,
        event_type=req.event_type,
        severity=req.severity,
        description=req.event_description,
        detection_method=req.detection_method,
    )
    if req.action_taken:
        from analysis.audit_service import record_action
        record_action(entry_id, req.action_taken, req.operator)
    if req.resolved:
        from analysis.audit_service import record_resolution
        record_resolution(entry_id, req.resolution_description or "", req.outcome or "resolved")
    return {"id": entry_id, "status": "recorded"}


@router.post("/audit/{entry_id}/action")
async def record_audit_action(entry_id: int, req: AuditActionRequest):
    """Record an action on an audit entry."""
    from analysis.audit_service import record_action
    record_action(entry_id, req.action, req.operator)
    return {"id": entry_id, "status": "action_recorded"}


@router.post("/audit/{entry_id}/resolve")
async def resolve_audit_entry(entry_id: int, req: AuditResolveRequest):
    """Resolve an audit entry."""
    from analysis.audit_service import record_resolution
    record_resolution(entry_id, req.description, req.outcome)
    return {"id": entry_id, "status": "resolved"}


@router.post("/audit/{entry_id}/note")
async def add_audit_note(entry_id: int, req: AuditNoteRequest):
    """Add a note to an audit entry."""
    from analysis.audit_service import add_note
    add_note(entry_id, req.note)
    return {"id": entry_id, "status": "note_added"}


@router.get("/audit/trail", response_model=AuditTrailResponse)
async def get_audit_trail_endpoint(zone: Optional[str] = None, limit: int = 100):
    """Get the audit trail."""
    from analysis.audit_service import get_trail
    entries = get_trail(zone=zone, limit=limit)
    return AuditTrailResponse(zone=zone, entries=entries, total=len(entries))


@router.get("/audit/pdf")
async def download_audit_pdf(zone: Optional[str] = None, limit: int = 100):
    """Download audit trail as PDF."""
    from analysis.audit_service import get_trail, generate_pdf
    entries = get_trail(zone=zone, limit=limit)
    pdf_bytes = generate_pdf(entries, zone=zone)
    filename = f"audit_trail_{zone or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/audit/ingest")
async def ingest_audit_data(req: IngestAuditRequest):
    """Re-ingest historical audit data back into the system."""
    from analysis.audit_service import ingest_audit_pdf_data
    ids = []
    for entry in req.entries:
        try:
            entry_id = ingest_audit_pdf_data(entry)
            ids.append(entry_id)
        except Exception as e:
            ids.append({"error": str(e)})
    return {"ingested": len([i for i in ids if isinstance(i, int)]), "ids": ids}
