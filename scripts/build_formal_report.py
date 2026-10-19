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
    path.write_text("\n".join(" & ".join(row) + r" \\" for row in rows) + "\n", encoding="utf-8", newline="\n")


def build():
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
    (generated / "numbers.tex").write_text("\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in macros.items()) + "\n", encoding="utf-8", newline="\n")

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
    suitability = []
    for profile in customers.itertuples():
        pairs = screen[screen.profile == profile.profile_id]
        assert len(pairs) == 3 and not pairs.duplicated("portfolio").any()
        screened = {row.portfolio: row for row in pairs.itertuples()}
        eligible = [name for name, value in screened.items() if bool(value.eligible)]
        if profile.policy == "human_review":
            assert not eligible, f"{profile.profile_id} has an eligible product but was referred"
        else:
            assert eligible, f"{profile.profile_id} was served without an eligible product"
            best = max(eligible, key=lambda name: screened[name].median_growth)
            assert profile.policy == "plan_a_" + best, f"{profile.profile_id} assignment is not the top eligible product"
        entry = [profile.profile_id, str(profile.horizon), pct(profile.loss_tolerance), profile.liquidity.capitalize()]
        for name in ["conservative", "balanced", "growth"]:
            value = screened[name]
            cell = f"{int(value.breaches)}/120"
            if bool(value.eligible):
                entry += [rf"\textbf{{{cell}}}", rf"\textbf{{{pct(value.upper_bound)}}}"]
            else:
                entry += [cell, pct(value.upper_bound)]
        entry.append("Human review" if profile.policy == "human_review" else profile.portfolio.capitalize())
        suitability.append(entry)
    write_rows(generated / "suitability.tex", suitability)
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
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 11, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.65, 2.7))
    names = ["equal_annual", "plan_a_conservative", "plan_a_balanced", "plan_a_annual", "plan_a_hold", "plan_a_growth"]
    short = ["Equal weight", "Conservative A", "Balanced A", "Same-anchor annual", "Same-anchor hold", "Growth A"]
    offsets = [(9,-14), (8,-14), (-94,15), (8,-18), (8,-12), (-75,9)]
    for name, label, offset in zip(names, short, offsets):
        r = row(name)
        ax.scatter(100*r.p90_loss, 100*r.median_growth, s=42, color="#007c83" if name == "plan_a_balanced" else "#55697f", zorder=3)
        ax.annotate(label, (100*r.p90_loss, 100*r.median_growth), xytext=offset, textcoords="offset points", fontsize=11,
                    arrowprops=dict(arrowstyle="-", lw=.5, color="#82909c"))
    ax.set(xlabel="90th-percentile maximum real drawdown (%)", ylabel="Median real CAGR (%)", xlim=(28,68), ylim=(2.35,5.65))
    ax.grid(alpha=.18);fig.tight_layout()
    fig.savefig(figures / "risk_growth.pdf");fig.savefig(figures / "risk_growth.png", dpi=180);plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(6.65,2.6),layout="constrained")
    for ax, field, label in zip(axes, ["median_growth", "mean_trade_rate"], ["Median real CAGR (%)", "Reviews with a trade (%)"]):
        ax.plot(100*costs.cost_rate, 100*costs[field], "o-", color="#007c83", lw=1.8, markersize=4)
        ax.set(xlabel="Transaction cost (%)", ylabel=label);ax.grid(alpha=.18)
    fig.savefig(figures / "cost_sensitivity.pdf");fig.savefig(figures / "cost_sensitivity.png", dpi=180);plt.close(fig)
    assert {key: sha(path) for key, path in sources.items()} == hashes
    evidence = dict(selected_policy=config["selected_policy"], final_scenarios=60, market_data_read=False,
                    policies_reevaluated=False, models_refitted=False, source_hashes={config["sources"][k]: v for k,v in hashes.items()},
                    figures={"risk_growth.pdf": config["sources"]["summary"], "cost_sensitivity.pdf": config["sources"]["summary"]},
                    generated_files={p.relative_to(out).as_posix(): sha(p) for folder in [generated, figures] for p in sorted(folder.iterdir()) if p.is_file()},
                    unresolved_team_members=not bool(config["members"]), compiler="pdfLaTeX (XeLaTeX-compatible source)", template="5writing/templates/en/mcm-latex")
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status="saved_evidence_verified", generated_files=len(evidence["generated_files"]),
                         market_data_read=False, unresolved_team_members=evidence["unresolved_team_members"])))


if __name__ == "__main__":
    build()
