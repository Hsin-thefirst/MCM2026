"""Build paper assets from saved evidence only; never load market data or fit policies."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def escape(text):
    return "".join({"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
                    "{": r"\{", "}": r"\}"}.get(c, c) for c in text)


def write_rows(path, rows):
    path.write_text("\n".join(" & ".join(row) + r" \\" for row in rows) + "\n", encoding="utf-8")


def read_saved_severe_path(path, severe):
    """Extract recorded annual wealth; do not replay returns or policy decisions."""
    columns = ["policy", "scenario_id", "simulation_horizon", "cost_rate", "period",
               "real_wealth", "action", "moved_amount", "review_pending"]
    matches = []
    for block in pd.read_csv(path, usecols=columns, chunksize=100_000, float_precision="round_trip"):
        selected = ((block.policy == "plan_a_balanced") &
                    (block.scenario_id == severe["scenario_id"]) &
                    (block.simulation_horizon == 30) & np.isclose(block.cost_rate, .0025))
        matches.append(block.loc[selected])
    frame = pd.concat(matches, ignore_index=True).sort_values("period")
    assert frame.period.tolist() == list(range(1, 31))
    assert np.isfinite(frame.real_wealth).all() and (frame.real_wealth > 0).all()
    assert np.isclose(frame.real_wealth.max(), severe["peak"])
    assert np.isclose(frame.real_wealth.min(), severe["trough"])
    assert np.isclose(frame.real_wealth.iloc[-1], severe["terminal_real_wealth"])
    assert frame.loc[frame.action == "human_review", "period"].tolist() == severe["review_periods"]
    post_initial_trades = frame.loc[frame.period > 1, "moved_amount"].gt(0)
    assert int(post_initial_trades.sum()) == severe["trades"] == 6
    pending = frame.period >= severe["review_periods"][0]
    assert frame.loc[pending, "review_pending"].all()
    assert frame.loc[pending, "moved_amount"].eq(0).all()
    return frame


def save_figure(fig, base, require_matplotlib_panel_alignment=None):
    """Export editable vectors and a print-size preview; allow external geometry QA."""
    if require_matplotlib_panel_alignment is not None:
        qa = base.parent.parent / "figure_qa"
        qa.mkdir(exist_ok=True)
        require_matplotlib_panel_alignment(fig, json_out=qa / (base.name + ".alignment.json"), strict=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def build(require_matplotlib_panel_alignment=None):
    config = read_json(ROOT / "configs/report.json")
    out = ROOT / config["output"]
    generated = out / "generated"
    figures = out / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    sources = {key: ROOT / value for key, value in config["sources"].items()}
    hashes = {key: sha(path) for key, path in sources.items()}
    bundle, lock, release, manifest = [read_json(sources[key]) for key in ("bundle", "lock", "release", "manifest")]
    assert bundle["selected_policy"] == release["selected_policy"] == config["selected_policy"] == "plan_a_balanced"
    assert release["status"] == "ready" and release["customer_verification_passed"]
    assert not release["holdout_reused"] and not release["new_final_evaluation"]
    assert lock["bundle_sha256"] == bundle["bundle_sha256"] == release["bundle_sha256"]
    assert manifest["code_sha256"] == bundle["code_sha256"] == release["code_sha256"]
    assert manifest["scenario_ids"] == list(range(121, 181)) and manifest["phase"] == "final"
    assert release["protected_hashes"][config["sources"]["bundle"]] == hashes["bundle"]
    summary = pd.read_csv(sources["summary"], float_precision="round_trip")
    paired = pd.read_csv(sources["paired"], float_precision="round_trip")
    customers = pd.read_csv(sources["customers"], float_precision="round_trip")
    screen = pd.read_csv(sources["screen"], float_precision="round_trip")
    severe = read_json(sources["severe"])
    severe_path = read_saved_severe_path(sources["ledger"], severe)
    verification = read_json(sources["verification"])
    assert verification["all_checks_passed"] and not verification["market_data_reopened"]
    assert not summary.duplicated(["policy", "horizon", "cost_rate"]).any() and summary.n.eq(60).all()
    expected = {"A": "human_review", "B": "conservative", "C": "conservative", "D": "human_review", "E": "balanced"}
    assert customers.set_index("profile_id").portfolio.to_dict() == expected
    assert {row["profile_id"]: row["portfolio"] for row in bundle["frozen_suitability"]} == expected

    def row(policy="plan_a_balanced", horizon=30, cost=.0025):
        selected = summary[(summary.policy == policy) & (summary.horizon == horizon) & np.isclose(summary.cost_rate, cost)]
        assert len(selected) == 1
        return selected.iloc[0]

    def pct(value):
        return f"{100 * value:.2f}"

    macros = {"PaperTitle": escape(config["title"]), "PaperDate": config["date"],
              "PaperAuthors": escape(", ".join(config["members"])) if config["members"] else "[Team member names to be supplied]",
              "PaperTeam": escape(config["team_name"]) if config["team_name"] else "Team review draft",
              "AITools": escape(", ".join(config["ai_tools"]))}
    fields = {"Growth": "median_growth", "Pten": "p10_growth", "MedianLoss": "median_loss", "Pninety": "p90_loss",
              "Volatility": "median_volatility", "Trades": "mean_trade_rate", "TailLoss": "cvar90_drawdown"}
    for prefix, policy in [("Balanced", "plan_a_balanced"), ("Equal", "equal_annual"), ("Annual", "plan_a_annual"), ("Hold", "plan_a_hold")]:
        r = row(policy)
        for suffix, field in fields.items():
            macros[prefix + suffix] = pct(r[field])
        macros[prefix + "Cost"] = f"{r.mean_real_cost:.4f}"
    a, annual = row(), row("plan_a_annual")
    macros.update(CostSaving=pct(1 - a.mean_real_cost / annual.mean_real_cost),
                  GrowthGain=pct(a.median_growth - row("equal_annual").median_growth),
                  RiskIncrease=pct(a.p90_loss - row("equal_annual").p90_loss),
                  AnnualGrowthGain=pct(a.median_growth - annual.median_growth),
                  HoldRiskReduction=pct(row("plan_a_hold").p90_loss - a.p90_loss),
                  HoldGrowthLoss=pct(row("plan_a_hold").median_growth - a.median_growth),
                  IncludingInitialTrades=pct(a.mean_trade_rate_including_initial),
                  DoubleGrowth=pct(row(cost=.005).median_growth), DoubleTrades=pct(row(cost=.005).mean_trade_rate),
                  DoubleCost=f"{row(cost=.005).mean_real_cost:.4f}",
                  SeverePeak=f"{severe['peak']:.2f}", SevereTerminal=f"{severe['terminal_real_wealth']:.2f}",
                  SevereLoss=pct(severe["max_drawdown"]), SevereGrowth=pct(severe["real_cagr"]),
                  ReviewThreshold=pct(severe["review_threshold"]))
    for metric, prefix in [("median_growth", "GrowthCI"), ("p90_loss", "LossCI")]:
        ci = paired[(paired.policy == "plan_a_balanced") & (paired.horizon == 30) & np.isclose(paired.cost_rate, .0025) & (paired.metric == metric)].iloc[0]
        for name, field in [("Lower", "lower95"), ("Upper", "upper95"), ("FamilyLower", "family_lower"), ("FamilyUpper", "family_upper")]:
            macros[prefix + name] = f"{100 * ci[field]:.3f}"
    (generated / "numbers.tex").write_text("\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in macros.items()) + "\n", encoding="utf-8")

    weights = []
    for h in [5, 6, 15, 25, 30]:
        values = [str(h)]
        for product in ["conservative", "balanced", "growth"]:
            params = bundle["params"]["plan_a_" + product]
            entry = bundle["model"]["plan_a"]["entries"][f"{product}:{h}:{params['risk_ratio']:g}"]
            assert entry["status"] != "infeasible" and abs(sum(entry["weights"]) - 1) < 1e-8
            values.append(" / ".join(pct(w) for w in entry["weights"]))
        weights.append(values)
    write_rows(generated / "weights.tex", weights)
    write_rows(generated / "assignments.tex", [[r.profile_id, str(r.horizon), pct(r.loss_tolerance), r.liquidity.capitalize(),
               "Human review" if r.portfolio == "human_review" else r.portfolio.capitalize()] for r in customers.itertuples()])
    selected = [("A", "conservative"), ("B", "conservative"), ("C", "conservative"), ("D", "conservative"), ("E", "balanced")]
    write_rows(generated / "screen.tex", [[p, product.capitalize(), f"{int(g.breaches)}/120", pct(g.upper_bound), pct(g.stress_upper_bound)]
               for p, product in selected for _, g in screen[(screen.profile == p) & (screen.portfolio == product)].iterrows()])
    labels = [("equal_annual", "Equal-weight annual"), ("plan_a_balanced", "Balanced A"),
              ("plan_a_annual", "Same-anchor annual"), ("plan_a_hold", "Same-anchor hold"),
              ("plan_a_conservative", "Conservative A"), ("plan_a_growth", "Growth A")]
    write_rows(generated / "performance.tex", [[label] + [pct(row(policy)[f]) for f in ["median_growth", "p10_growth", "median_loss", "p90_loss", "mean_trade_rate"]] for policy, label in labels])
    write_rows(generated / "execution.tex", [[label, f"{row(policy).mean_real_cost:.4f}", pct(row(policy).mean_trade_rate), pct(row(policy).review_rate)] for policy, label in labels[:4]])
    customer_rows = []
    for c in customers.itertuples():
        if c.policy == "human_review":
            customer_rows.append([c.profile_id, str(c.horizon), "--", "--", "--", "--", "--"])
            continue
        r = row(c.policy, c.horizon)
        assert abs(r.median_growth - c.final_median_growth) < 1e-12
        customer_rows.append([c.profile_id, str(c.horizon), pct(r.median_growth), pct(r.p10_growth), pct(r.median_loss), pct(r.p90_loss), f"{round(c.final_breach_probability * 60)}/60"])
    write_rows(generated / "customers.tex", customer_rows)
    write_rows(generated / "customer_risk.tex", [[c.profile_id, pct(row(c.policy, c.horizon).median_volatility),
               f"{round(row(c.policy, c.horizon).terminal_loss_probability * 60)}/60", pct(row(c.policy, c.horizon).mean_trade_rate)]
               for c in customers.itertuples() if c.policy != "human_review"])
    costs = summary[(summary.policy == "plan_a_balanced") & (summary.horizon == 30)].sort_values("cost_rate")
    write_rows(generated / "costs.tex", [[f"{100*r.cost_rate:.3f}", pct(r.median_growth), pct(r.p10_growth), pct(r.p90_loss), f"{r.mean_real_cost:.4f}", pct(r.mean_trade_rate)] for r in costs.itertuples()])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 11, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.75, 2.6))
    names = ["equal_annual", "plan_a_conservative", "plan_a_balanced", "plan_a_annual", "plan_a_hold", "plan_a_growth"]
    short = ["Equal weight", "Conservative A", "Balanced A", "Same-anchor annual", "Same-anchor hold", "Growth A"]
    positions = [(34, 3.75), (28.5, 3.08), (41, 4.90), (49.7, 3.45), (56, 4.75), (56, 5.55)]
    for name, label, position in zip(names, short, positions):
        r = row(name)
        primary = name == "plan_a_balanced"
        ax.scatter(100*r.p90_loss, 100*r.median_growth, s=58 if primary else 30,
                   marker="D" if primary else "o", color="#007c83" if primary else "#65758a", zorder=3)
        ax.annotate(label, (100*r.p90_loss, 100*r.median_growth), xytext=position, fontsize=11,
                    color="#007c83" if primary else "#26333e",
                    arrowprops=dict(arrowstyle="-", lw=.6, color="#82909c"))
    ax.set(xlabel="90th-percentile maximum real drawdown (%)", ylabel="Median real CAGR (%)",
           xlim=(26, 70), ylim=(2.35, 5.95), xticks=[30, 40, 50, 60, 70], yticks=[3, 4, 5])
    fig.subplots_adjust(left=.12, right=.985, bottom=.25, top=.96)
    save_figure(fig, figures / "risk_growth", require_matplotlib_panel_alignment)
    pd.DataFrame([dict(policy=name, label=label, n=60, horizon=30, cost_rate=.0025,
                       p90_loss=row(name).p90_loss, median_growth=row(name).median_growth)
                  for name, label in zip(names, short)]).to_csv(generated / "risk_growth.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.75, 2.05))
    years = np.r_[0, severe_path.period.to_numpy()]
    wealth = np.r_[100., severe_path.real_wealth.to_numpy()]
    review_time = severe["review_periods"][0] - 1
    ax.axvspan(review_time, 30, facecolor="#e5edf1", edgecolor="none", zorder=0)
    ax.axhline(100, color="#929ca3", lw=.7, linestyle="--", zorder=1)
    ax.plot(years, wealth, color="#007c83", lw=1.6, marker="o", markersize=2.6, zorder=3)
    ax.axvline(review_time, color="#65758a", lw=.7, linestyle=":", zorder=2)
    ax.annotate(f"Peak {severe['peak']:.2f}", (6, severe["peak"]), xytext=(8, 173), fontsize=11,
                arrowprops=dict(arrowstyle="-", color="#65758a", lw=.6))
    ax.text(23, 142, "Review\npending", fontsize=11, va="top", color="#26333e")
    ax.annotate(f"Final {severe['terminal_real_wealth']:.2f}", (30, severe["terminal_real_wealth"]), xytext=(25, 80), fontsize=11,
                arrowprops=dict(arrowstyle="-", color="#65758a", lw=.6), ha="center")
    ax.set(xlabel="Investment year", ylabel="Real wealth", xlim=(0, 31), ylim=(40, 190),
           xticks=[0, 5, 10, 15, 20, 25, 30], yticks=[50, 100, 150])
    fig.subplots_adjust(left=.12, right=.985, bottom=.29, top=.96)
    save_figure(fig, figures / "severe_path", require_matplotlib_panel_alignment)
    severe_path.to_csv(generated / "severe_path.csv", index=False)
    fig, axes = plt.subplots(1,2,figsize=(6.65,2.6),layout="constrained")
    for ax, field, label in zip(axes, ["median_growth", "mean_trade_rate"], ["Median real CAGR (%)", "Reviews with a trade (%)"]):
        ax.plot(100*costs.cost_rate, 100*costs[field], "o-", color="#007c83", lw=1.8, markersize=4)
        ax.set(xlabel="Transaction cost (%)", ylabel=label);ax.grid(alpha=.18)
    save_figure(fig, figures / "cost_sensitivity", require_matplotlib_panel_alignment)
    assert {key: sha(path) for key, path in sources.items()} == hashes
    evidence = dict(selected_policy=config["selected_policy"], final_scenarios=60, market_data_read=False,
                    policies_reevaluated=False, models_refitted=False, source_hashes={config["sources"][k]: v for k,v in hashes.items()},
                    figures={"risk_growth.pdf": config["sources"]["summary"], "cost_sensitivity.pdf": config["sources"]["summary"],
                             "severe_path.pdf": config["sources"]["ledger"]},
                    generated_files={p.relative_to(out).as_posix(): sha(p) for folder in [generated, figures] for p in sorted(folder.iterdir()) if p.is_file()},
                    unresolved_team_members=not bool(config["members"]), compiler="See validation.json for the verified build engine", template="5writing/templates/en/mcm-latex")
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="saved_evidence_verified", generated_files=len(evidence["generated_files"]),
                         market_data_read=False, unresolved_team_members=evidence["unresolved_team_members"])))


if __name__ == "__main__":
    build()
