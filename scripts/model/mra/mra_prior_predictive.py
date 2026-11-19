#!/usr/bin/env python3
"""Prior predictive calibration for the mra mutation-supply model.

Simulates counts from a candidate prior specification *without* fitting anything, then
checks the implied data against the real observed data. This is how you choose prior
scales before ever running PyMC: adjust the spec until the simulated data brackets reality
comfortably without being absurd, then fit.

The generative structure mirrors the fit script but replaces the empirically-centred and
transferred priors with a hierarchical parameterisation anchored on external
order-of-magnitude knowledge. See PRIORS_mra_framework.md for the rationale.

    log bio rate = mu_bio + type/context structure + centred positional spline
    log err rate = mu_err + type structure
    counts       ~ NegBinom(exposure * rate, alpha)

Acceptance criteria (the point of the exercise):

  * The prior predictive must CONTAIN the observed data -- the observed pooled rate and
    total mutation count should sit inside the central 90% of the simulated distribution.
  * The prior predictive must be WIDER than the observed data by a good margin -- the
    simulated 5-95% range should span at least two orders of magnitude, so the prior is
    admitting possibilities the data will have to rule out.
  * The prior must not be absurd -- essentially no draws should imply per-molecule rates
    above ~1e-2, and the simulated zero-fraction should be able to reach the observed 0.90.

A prior that satisfies all three is informative (it excludes the impossible) without being
deterministic (it does not pick the answer).

Usage:
    python mra_prior_predictive.py                      # default candidate spec
    python mra_prior_predictive.py --spec myspec.json   # override any subset of the spec
    python mra_prior_predictive.py --draws 1000 --seed 7
    python mra_prior_predictive.py --dump-spec          # print the default spec as JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "models" / "mra" / "mutation_supply_data.csv"

GA_MUT_TYPE = "G->A"
INDEL_TYPES = ("insertion", "deletion")
GA_POOL_2MERS = ("CC", "CG", "TC", "TT")  # pooled into OTHER, as in the fit script
SPLINE_DF = 30

# ---------------------------------------------------------------------------------------
# Candidate prior specification.
#
# Every entry is (mean, sd) for a Normal or (scale,) for a HalfNormal, on the natural log
# rate scale unless noted. Anchors are deliberately round numbers: they encode
# order-of-magnitude knowledge, not fitted values.
# ---------------------------------------------------------------------------------------
DEFAULT_SPEC: dict = {
    # --- external anchors (order of magnitude, per consensus molecule) -----------------
    # Technical error floor of the assay after UMI consensus calling.
    "mu_err": {"log10_rate": -6.0, "sd": 1.0},
    # Typical biological supply for a single substitution type at a single site.
    # sd 1.5 on the natural-log scale is a 95% prior interval of roughly 5e-8 to 2e-3,
    # i.e. deliberately two-plus orders of magnitude wide in each direction.
    "mu_bio": {"log10_rate": -5.0, "sd": 1.5},

    # --- hierarchical spread of mutation types (no per-type centring) ------------------
    # tau_type = 1.0 means types typically differ by e^1 ~ 2.7x, with 2 sd ~ 7x.
    "tau_type": {"scale": 1.0},
    # Is G->A elevated? Centred on zero: the prior is agnostic, so any G->A dominance
    # in the results is a finding rather than an assumption. sd 2.5 is wide enough that
    # the observed 68% G->A share sits inside the prior predictive 90% interval.
    "delta_ga": {"mean": 0.0, "sd": 2.5},
    "tau_ga_context": {"scale": 1.0},
    # Indels relative to substitutions: centred on ~0.3x, wide.
    "delta_indel": {"mean": np.log(0.3), "sd": 1.0},

    # --- 3' single-base context deviations, nested inside type effects -----------------
    # Deliberately smaller than tau_type: context modulates a type, it does not redefine it.
    "tau_context_delta": {"scale": 0.5},

    # --- error-arm type effects --------------------------------------------------------
    "tau_err_type": {"scale": 0.75},

    # --- positional effect: P-spline, no amplitude floor -------------------------------
    # One identical prior for all six splines, so no mutation class is privileged and any
    # difference between classes is a result rather than an assumption.
    # sigma_wiggle is the sd of nonlinear positional variation in LOG-RATE UNITS:
    # 0.2 means typical positional effects of roughly +/-20%, with the prior still
    # admitting both a flat profile and a genome-wide range near 2 log units.
    "sigma_wiggle": {"scale": 0.20},
    # Linear gradient across the genome, in log units from one end to the other.
    "slope": {"sd": 0.3},

    # --- negative-binomial dispersion --------------------------------------------------
    # Prior on 1/alpha so that alpha may be large (near-Poisson), which is where count
    # data of this sparsity usually lands.
    "recip_alpha": {"scale": 1.0},
}


def load_data(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d.rename(
        columns={
            "Mut_Type": "mut_type",
            "WT_mutation_count": "wt_count",
            "Control_mutation_count": "control_count",
            "WT_coverage": "wt_exp",
            "Control_coverage": "control_exp",
        }
    )
    d["three_prime"] = d["three_prime"].astype(str)
    return d


def ga_context(d: pd.DataFrame) -> pd.Series:
    """3' dinucleotide at G sites, with the four sparsest 2-mers pooled to OTHER."""
    two = d["three_prime"].str.slice(0, 2).str.upper()
    two = two.where(two.str.match(r"^[ACGT]{2}$"), "OTHER")
    return two.where(~two.isin(GA_POOL_2MERS), "OTHER")


def spline_basis(positions: np.ndarray, df: int = SPLINE_DF) -> np.ndarray:
    scaled = (positions - positions.mean()) / positions.std()
    return np.asarray(
        patsy.dmatrix(
            f"bs(x, df={df}, degree=3, include_intercept=False)",
            {"x": scaled},
            return_type="dataframe",
        ),
        dtype=np.float64,
    )


def second_difference_matrix(n: int) -> np.ndarray:
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    return D


def wiggle_basis(B: np.ndarray) -> np.ndarray:
    """P-spline mixed-model reparameterisation of the positional effect.

    The second-difference penalty P = D'D has a two-dimensional null space (constant and
    linear), which is exactly why a naive random-walk prior lets an unpenalised linear
    trend run away. Here the null space is removed and the remaining penalised directions
    are rescaled so that IID unit-normal coefficients produce a positional function with
    unit standard deviation across positions.

    The consequence is a single interpretable knob: with coefficients b ~ Normal(0, s^2 I),
    the nonlinear positional effect has standard deviation s *in log-rate units*. The
    overall level and any linear gradient are modelled separately and explicitly.

    Returns Z with shape (n_pos, n_basis - 2). This transformation is fixed and
    data-independent, so the identical construction can be used in the PyMC model.
    """
    n = B.shape[1]
    # errstate: this BLAS build emits spurious overflow warnings from matmul on small
    # well-conditioned matrices. Nothing here can overflow -- D holds only 1, -2 and 1.
    with np.errstate(all="ignore"):
        D = second_difference_matrix(n)
        eigvals, eigvecs = np.linalg.eigh(D.T @ D)
        keep = eigvals > 1e-8 * eigvals.max()
        Z = B @ (eigvecs[:, keep] / np.sqrt(eigvals[keep]))
        # remove any residual constant/linear leakage, then scale to unit marginal sd
        x = np.linspace(-1.0, 1.0, B.shape[0])
        A = np.column_stack([np.ones_like(x), x])
        Z = Z - A @ np.linalg.lstsq(A, Z, rcond=None)[0]
        Z /= np.sqrt(np.mean(np.sum(Z**2, axis=1)))
    return Z


def positional_effect(
    rng: np.random.Generator,
    Z: np.ndarray,
    n_draws: int,
    sigma_wiggle: np.ndarray,
    slope_sd: float,
) -> np.ndarray:
    """(n_draws, n_pos) positional log-rate deviation: linear gradient + nonlinear wiggle."""
    b = rng.normal(0.0, 1.0, (n_draws, Z.shape[1])) * sigma_wiggle[:, None]
    with np.errstate(all="ignore"):  # see note in wiggle_basis
        wiggle = b @ Z.T
    x = np.linspace(-1.0, 1.0, Z.shape[0])
    gradient = rng.normal(0.0, slope_sd, n_draws)[:, None] * x[None, :]
    vals = wiggle + gradient
    return vals - vals.mean(axis=1, keepdims=True)


def simulate(spec: dict, d: pd.DataFrame, n_draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    mut_types = sorted(d["mut_type"].unique())
    sub_types = [m for m in mut_types if m not in INDEL_TYPES and m != GA_MUT_TYPE]

    is_ga = (d["mut_type"] == GA_MUT_TYPE).to_numpy()
    is_indel = d["mut_type"].isin(INDEL_TYPES).to_numpy()
    is_sub = ~is_ga & ~is_indel

    # positional basis, evaluated once on unique positions then broadcast to rows
    pos = np.sort(d["ref_position"].unique())
    B = spline_basis(pos.astype(float))
    pos_index = pd.Series(np.arange(len(pos)), index=pos).loc[d["ref_position"]].to_numpy()
    n_basis = B.shape[1]

    # --- draw hyperparameters ----------------------------------------------------------
    mu_bio = rng.normal(
        np.log(10.0 ** spec["mu_bio"]["log10_rate"]), spec["mu_bio"]["sd"], n_draws
    )
    mu_err = rng.normal(
        np.log(10.0 ** spec["mu_err"]["log10_rate"]), spec["mu_err"]["sd"], n_draws
    )
    tau_type = np.abs(rng.normal(0.0, spec["tau_type"]["scale"], n_draws))
    delta_ga = rng.normal(spec["delta_ga"]["mean"], spec["delta_ga"]["sd"], n_draws)
    tau_ga = np.abs(rng.normal(0.0, spec["tau_ga_context"]["scale"], n_draws))
    delta_indel = rng.normal(spec["delta_indel"]["mean"], spec["delta_indel"]["sd"], n_draws)
    tau_ctx = np.abs(rng.normal(0.0, spec["tau_context_delta"]["scale"], n_draws))
    tau_err = np.abs(rng.normal(0.0, spec["tau_err_type"]["scale"], n_draws))
    sigma_wiggle = np.abs(rng.normal(0.0, spec["sigma_wiggle"]["scale"], n_draws))
    recip_alpha = np.abs(rng.normal(0.0, spec["recip_alpha"]["scale"], n_draws))
    alpha = 1.0 / np.maximum(recip_alpha, 1e-6)

    # --- type-level effects ------------------------------------------------------------
    sub_eff = mu_bio[:, None] + tau_type[:, None] * rng.normal(0, 1, (n_draws, len(sub_types)))
    sub_col = {t: i for i, t in enumerate(sub_types)}

    ga_ctx = ga_context(d)
    ga_levels = sorted(ga_ctx[is_ga].unique())
    ga_eff = (
        mu_bio[:, None]
        + delta_ga[:, None]
        + tau_ga[:, None] * rng.normal(0, 1, (n_draws, len(ga_levels)))
    )
    ga_col = {c: i for i, c in enumerate(ga_levels)}

    indel_eff = mu_bio[:, None] + delta_indel[:, None] + tau_type[:, None] * rng.normal(
        0, 1, (n_draws, len(INDEL_TYPES))
    )
    indel_col = {t: i for i, t in enumerate(INDEL_TYPES)}

    err_eff = mu_err[:, None] + tau_err[:, None] * rng.normal(0, 1, (n_draws, len(mut_types)))
    err_col = {t: i for i, t in enumerate(mut_types)}

    # --- assemble per-row log rates ----------------------------------------------------
    n_obs = len(d)
    eta_bio = np.empty((n_draws, n_obs))

    row_sub = np.array([sub_col.get(t, -1) for t in d["mut_type"]])
    row_ga = np.array([ga_col.get(c, -1) for c in ga_ctx])
    row_indel = np.array([indel_col.get(t, -1) for t in d["mut_type"]])
    row_err = np.array([err_col[t] for t in d["mut_type"]])

    eta_bio[:, is_sub] = np.take_along_axis(
        sub_eff, np.broadcast_to(row_sub[is_sub], (n_draws, is_sub.sum())), axis=1
    )
    eta_bio[:, is_ga] = np.take_along_axis(
        ga_eff, np.broadcast_to(row_ga[is_ga], (n_draws, is_ga.sum())), axis=1
    )
    eta_bio[:, is_indel] = np.take_along_axis(
        indel_eff, np.broadcast_to(row_indel[is_indel], (n_draws, is_indel.sum())), axis=1
    )

    # context deviation, applied to non-G->A substitutions only
    ctx_dev = tau_ctx[:, None] * rng.normal(0, 1, (n_draws, is_sub.sum()))
    eta_bio[:, is_sub] += ctx_dev

    # positional effect, centred so the intercept owns the overall level
    Z = wiggle_basis(B)
    spline_vals = positional_effect(rng, Z, n_draws, sigma_wiggle, spec["slope"]["sd"])
    eta_bio += spline_vals[:, pos_index]

    eta_err = np.take_along_axis(
        err_eff, np.broadcast_to(row_err, (n_draws, n_obs)), axis=1
    )

    bio_rate = np.exp(eta_bio)
    err_rate = np.exp(eta_err)

    wt_exp = d["wt_exp"].to_numpy(dtype=float)
    mu_wt = wt_exp * (bio_rate + err_rate)

    # NB sampling with mean mu and dispersion alpha: var = mu + mu^2/alpha
    a = alpha[:, None]
    p = a / (a + mu_wt)
    wt_sim = rng.negative_binomial(np.broadcast_to(a, mu_wt.shape), p)

    return {
        "bio_rate": bio_rate,
        "spline_vals": spline_vals,
        "mu_wt": mu_wt,
        "wt_sim": wt_sim,
        "wt_exp": wt_exp,
        "is_ga": is_ga,
        "alpha": alpha,
        "n_draws": n_draws,
    }


def q(x, probs=(0.05, 0.5, 0.95)) -> tuple:
    return tuple(np.quantile(x, probs))


def report(sim: dict, d: pd.DataFrame, spec: dict) -> None:
    wt_sim = sim["wt_sim"]
    wt_exp = sim["wt_exp"]
    obs_total = int(d["wt_count"].sum())
    obs_rate = obs_total / wt_exp.sum()
    obs_zero = float((d["wt_count"] == 0).mean())
    obs_max = int(d["wt_count"].max())
    obs_ga_share = float(
        d.loc[d["mut_type"] == GA_MUT_TYPE, "wt_count"].sum() / d["wt_count"].sum()
    )

    sim_total = wt_sim.sum(axis=1)
    sim_rate = sim_total / wt_exp.sum()
    sim_zero = (wt_sim == 0).mean(axis=1)
    sim_max = wt_sim.max(axis=1)
    ga = sim["is_ga"]
    sim_ga_share = wt_sim[:, ga].sum(axis=1) / np.maximum(sim_total, 1)
    sim_range = sim["spline_vals"].max(axis=1) - sim["spline_vals"].min(axis=1)
    pooled_bio = (sim["bio_rate"] * wt_exp).sum(axis=1) / wt_exp.sum()

    print("\n" + "=" * 78)
    print(f"PRIOR PREDICTIVE CHECK  ({sim['n_draws']} draws)")
    print("=" * 78)

    rows = [
        ("pooled WT rate / molecule", q(sim_rate), obs_rate, "{:.3g}"),
        ("pooled biological rate",    q(pooled_bio), None,    "{:.3g}"),
        ("total WT mutations",        q(sim_total), obs_total, "{:,.0f}"),
        ("fraction zero cells",       q(sim_zero), obs_zero,  "{:.3f}"),
        ("max count in one cell",     q(sim_max), obs_max,    "{:,.0f}"),
        ("G->A share of mutations",   q(sim_ga_share), obs_ga_share, "{:.3f}"),
        ("spline log-range (genome)", q(sim_range), None,     "{:.3f}"),
        ("NB alpha",                  q(sim["alpha"]), None,  "{:.3g}"),
    ]
    print(f"\n{'quantity':28s} {'5%':>12s} {'50%':>12s} {'95%':>12s} {'observed':>12s}")
    print("-" * 78)
    for name, (lo, mid, hi), obs, fmt in rows:
        o = fmt.format(obs) if obs is not None else "-"
        print(f"{name:28s} {fmt.format(lo):>12s} {fmt.format(mid):>12s} "
              f"{fmt.format(hi):>12s} {o:>12s}")

    # ---- acceptance criteria ---------------------------------------------------------
    print("\n" + "-" * 78)
    print("ACCEPTANCE CRITERIA")
    print("-" * 78)

    def check(label: str, ok: bool, detail: str) -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}\n         {detail}")
        return ok

    lo_r, _, hi_r = q(sim_rate)
    c1 = check(
        "prior predictive contains the observed pooled rate",
        lo_r <= obs_rate <= hi_r,
        f"observed {obs_rate:.3g} vs simulated 5-95% [{lo_r:.3g}, {hi_r:.3g}]",
    )

    lo_t, _, hi_t = q(sim_total)
    c2 = check(
        "prior predictive contains the observed mutation total",
        lo_t <= obs_total <= hi_t,
        f"observed {obs_total:,} vs simulated 5-95% [{lo_t:,.0f}, {hi_t:,.0f}]",
    )

    decades = np.log10(max(hi_r, 1e-12)) - np.log10(max(lo_r, 1e-12))
    c3 = check(
        "prior is wide enough to be overridable (>=2 decades on pooled rate)",
        decades >= 2.0,
        f"simulated 5-95% pooled rate spans {decades:.2f} decades",
    )

    frac_absurd = float((pooled_bio > 1e-2).mean())
    c4 = check(
        "prior excludes absurd rates (<1% of draws above 1e-2 per molecule)",
        frac_absurd < 0.01,
        f"{frac_absurd:.3%} of draws imply pooled biological rate > 1e-2",
    )

    c5 = check(
        "prior allows a sparse table (simulated zero-fraction can reach observed)",
        np.quantile(sim_zero, 0.95) >= obs_zero,
        f"observed zero-fraction {obs_zero:.3f} vs simulated 95% {np.quantile(sim_zero, 0.95):.3f}",
    )

    lo_g, _, hi_g = q(sim_ga_share)
    c6 = check(
        "prior is agnostic about G->A dominance (interval contains both 1/14 and observed)",
        lo_g <= 1.0 / 14 and hi_g >= obs_ga_share,
        f"simulated 5-95% G->A share [{lo_g:.3f}, {hi_g:.3f}]; "
        f"uniform expectation {1/14:.3f}, observed {obs_ga_share:.3f}",
    )

    lo_s, med_s, hi_s = q(sim_range)
    c7 = check(
        "positional prior admits both flat and structured profiles",
        lo_s < 0.2 and hi_s > 1.0,
        f"spline log-range 5-95% [{lo_s:.3f}, {hi_s:.3f}] (median {med_s:.3f}); "
        f"want the low end near flat and the high end past 1 log unit",
    )

    passed = sum([c1, c2, c3, c4, c5, c6, c7])
    print("-" * 78)
    print(f"  {passed}/7 criteria passed")
    if passed < 7:
        print("\n  Tuning guide:")
        print("    pooled rate too low/high ...... shift mu_bio.log10_rate")
        print("    interval too narrow ........... raise mu_bio.sd")
        print("    absurd rates .................. lower mu_bio.sd or tau_type.scale")
        print("    G->A share too tight .......... raise delta_ga.sd")
        print("    splines too flat/wild ......... adjust sigma_rw.scale")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--spec", type=Path, default=None, help="JSON file overriding the default spec")
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-spec", action="store_true", help="print the default spec and exit")
    args = ap.parse_args()

    if args.dump_spec:
        print(json.dumps(DEFAULT_SPEC, indent=2, default=float))
        return

    spec = dict(DEFAULT_SPEC)
    if args.spec:
        spec.update(json.loads(args.spec.read_text()))

    d = load_data(args.data)
    print(f"loaded {len(d):,} rows from {args.data}")
    print("\nprior spec:")
    print(json.dumps(spec, indent=2, default=float))

    sim = simulate(spec, d, args.draws, args.seed)
    report(sim, d, spec)


if __name__ == "__main__":
    main()
