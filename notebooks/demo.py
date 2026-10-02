# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "genotox-food-migrants[notebook] @ git+https://github.com/E4CE-UA/genotox-food-migrants",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import html
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    _candidates = [Path.cwd(), Path.cwd().parent]
    try:
        _nb = mo.notebook_dir()
        if _nb is not None:
            _candidates.insert(0, Path(_nb).parent)
    except Exception:
        pass

    ROOT = next((c for c in _candidates if (c / "src" / "paths.py").exists()), None)
    if ROOT is not None and str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src import (
        candidate_view,
        chemistry,
        data,
        environment,
        external,
        figures,
        integration,
        labels,
        models,
        paths,
        public_features,
    )

    if ROOT is None:
        ROOT = Path(paths.ROOT)

    environment.set_seeds()
    figures.apply_style()

    return (
        ROOT,
        candidate_view,
        chemistry,
        data,
        environment,
        external,
        figures,
        html,
        integration,
        labels,
        mo,
        models,
        np,
        pd,
        paths,
        public_features,
    )


@app.cell
def _(public_features):
    # Public food-contact-material GC-MS study used as the guided story.
    #
    # IMPORTANT: the MAF is a feature table. Its mass_to_charge field is kept
    # with the depositor's meaning. We DO NOT calculate a neutral exact mass
    # from a candidate SMILES and present it as an experimental observation.
    STUDY = "FCM2018"
    HOOK_FEATURE = "FCM2018-0010"
    NIAS = {"FCM2018-0010", "FCM2018-0004"}
    N_CANDIDATES = 20

    gcms_candidates, gcms_features, gcms_info = public_features.build(
        STUDY, n_candidates=N_CANDIDATES
    )

    # Read the deposited MAF as well so the demo can distinguish what the
    # deposit actually reports from what this notebook derives later.
    _maf = public_features.load_maf(STUDY).copy()
    _maf.insert(
        0,
        "feature_id",
        [f"{STUDY}-{i:04d}" for i in range(1, len(_maf) + 1)],
    )
    feature_meta = _maf.set_index("feature_id", drop=False)

    return (
        HOOK_FEATURE,
        NIAS,
        N_CANDIDATES,
        STUDY,
        feature_meta,
        gcms_candidates,
        gcms_features,
        gcms_info,
    )


@app.cell
def _(candidate_view, mo):
    # One visual language for the whole story. Molecules are intentionally
    # absent from scene 1; the CSS does not change that scientific ordering.
    _BASE = candidate_view.CSS.replace("<style>", "").replace("</style>", "")
    _INSPECTOR = getattr(candidate_view, "INSPECTOR_CSS", "")
    THEME = (
        _BASE
        + _INSPECTOR
        + r"""
:root {
  --paper:#faf8f4;
  --panel:#fffdf9;
  --ink:#252525;
  --muted:#77736d;
  --line:#e4dfd6;
  --petrol:#1f6f78;
  --petrol-soft:#dcecef;
  --amber:#b7791f;
  --amber-soft:#f7ead0;
  --bad:#9f4f43;
}
.story-shell { max-width: 1040px; margin: 0 auto; }
.story-head {
  display:flex; align-items:baseline; gap:12px;
  border-bottom:1px solid var(--line);
  padding:2px 0 8px 0; margin-bottom:10px;
}
.story-kicker {
  font:600 12px ui-monospace,Menlo,monospace;
  color:var(--petrol); letter-spacing:.04em; text-transform:uppercase;
}
.story-study {
  margin-left:auto; color:var(--muted);
  font:11px ui-monospace,Menlo,monospace;
}
.story-title {
  font:600 29px/1.12 Georgia,"Iowan Old Style",serif;
  color:var(--ink); margin:10px 0 4px 0;
}
.story-sub {
  color:var(--muted); max-width:72ch; font-size:14px; line-height:1.5;
  margin:0 0 12px 0;
}
.story-panel {
  background:var(--panel); border:1px solid var(--line);
  border-radius:14px; padding:16px 18px;
}
.story-panel + .story-panel { margin-top:10px; }
.story-grid2 {
  display:grid; grid-template-columns:1fr 1fr; gap:12px;
}
.story-grid3 {
  display:grid; grid-template-columns:repeat(3,1fr); gap:10px;
}
.story-eyebrow {
  font:600 11px ui-monospace,Menlo,monospace;
  text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
  margin-bottom:6px;
}
.story-big {
  font:600 25px/1.15 Georgia,"Iowan Old Style",serif;
  color:var(--ink);
}
.story-mono {
  font:12px ui-monospace,Menlo,monospace; color:var(--ink);
}
.story-note {
  color:var(--muted); font-size:12px; line-height:1.45;
}
.story-good { color:var(--petrol); }
.story-warn { color:var(--amber); }
.story-pill {
  display:inline-block; border-radius:999px; padding:3px 8px;
  font:600 10px ui-monospace,Menlo,monospace; margin-right:4px;
}
.story-pill.ok { color:var(--petrol); background:var(--petrol-soft); }
.story-pill.warn { color:#815516; background:var(--amber-soft); }
.story-pill.dim { color:#666; background:#efede8; }
.story-observed {
  display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px;
}
.story-list { margin:0; padding-left:18px; line-height:1.7; font-size:13px; }
.story-signal {
  height:160px; position:relative; overflow:hidden;
  border:1px solid var(--line); border-radius:12px;
  background:linear-gradient(to bottom,#fffdf9,#fbf8f2);
}
.story-signal .axis {
  position:absolute; left:7%; right:5%; bottom:26px; height:1px; background:#a7a29a;
}
.story-signal .trace {
  position:absolute; left:8%; right:6%; bottom:28px; height:110px;
}
.story-signal svg { width:100%; height:100%; }
.story-signal .caption {
  position:absolute; left:12px; top:9px; color:var(--muted);
  font:10px ui-monospace,Menlo,monospace; text-transform:uppercase;
}
.story-signal .fid {
  position:absolute; right:18%; top:26px;
  color:var(--petrol); font:600 11px ui-monospace,Menlo,monospace;
}
.identity-hero {
  display:grid; grid-template-columns:minmax(260px,42%) 1fr; gap:18px; align-items:center;
}
.identity-mol {
  min-height:220px; display:flex; align-items:center; justify-content:center;
  background:#fff; border:1px solid var(--line); border-radius:12px; padding:8px;
}
.identity-mol svg { max-width:100%; height:auto; }
.alt-grid {
  display:grid; grid-template-columns:repeat(auto-fill,minmax(145px,1fr)); gap:9px;
}
.alt-card {
  border:1px solid var(--line); border-radius:10px; padding:8px;
  background:#fff; position:relative; min-height:145px;
}
.alt-card.assigned { border:2px solid var(--petrol); }
.alt-card .tag {
  position:absolute; top:6px; left:6px; z-index:2;
  font:600 9px ui-monospace,Menlo,monospace;
  padding:2px 5px; border-radius:99px; background:#efede8; color:#555;
}
.alt-card.assigned .tag { color:var(--petrol); background:var(--petrol-soft); }
.alt-card .draw {
  min-height:105px; display:flex; justify-content:center; align-items:center;
}
.alt-card .draw svg { max-width:100%; height:auto; }
.alt-card .cap {
  display:flex; justify-content:space-between;
  color:var(--muted); font:10px ui-monospace,Menlo,monospace;
}
.nn-grid {
  display:grid; grid-template-columns:1fr 130px 1fr; gap:12px; align-items:center;
}
.nn-mol {
  min-height:210px; display:flex; align-items:center; justify-content:center;
  border:1px solid var(--line); border-radius:12px; background:#fff; padding:8px;
}
.nn-mol svg { max-width:100%; height:auto; }
.nn-mid { text-align:center; }
.nn-score {
  font:700 30px Georgia,serif; color:var(--petrol);
}
.domain-scale { margin:18px 0 8px; }
.domain-track {
  height:13px; background:#ebe8e2; border-radius:99px; position:relative;
}
.domain-fill {
  position:absolute; left:0; top:0; bottom:0; background:var(--petrol-soft);
  border-radius:99px;
}
.domain-dot {
  width:18px; height:18px; border-radius:50%; background:var(--petrol);
  border:3px solid #fff; box-shadow:0 0 0 1px var(--petrol);
  position:absolute; top:50%; transform:translate(-50%,-50%);
}
.domain-cut {
  width:2px; background:var(--amber); position:absolute; top:-8px; bottom:-8px;
}
.domain-labels {
  display:flex; justify-content:space-between; margin-top:7px;
  color:var(--muted); font:10px ui-monospace,Menlo,monospace;
}
.cov-row { margin:8px 0; }
.cov-top {
  display:flex; align-items:baseline; gap:8px; margin-bottom:3px; font-size:11px;
}
.cov-id { font:11px ui-monospace,Menlo,monospace; min-width:106px; }
.cov-spacer { flex:1; }
.cov-track { height:11px; background:#efede8; border-radius:99px; overflow:hidden; }
.cov-fill { height:100%; background:var(--petrol); border-radius:99px; }
.cov-row.zero .cov-track { background:var(--amber-soft); }
.stat-strip {
  display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:10px 0;
}
.stat-box {
  border:1px solid var(--line); border-radius:10px; padding:12px; background:#fff;
}
.stat-box b { display:block; font:600 24px Georgia,serif; color:var(--ink); }
.stat-box span { color:var(--muted); font-size:11px; }
.final-line {
  border-left:3px solid var(--petrol); padding:10px 14px; margin-top:14px;
  background:#f7fbfb; font:600 18px/1.4 Georgia,serif;
}
@media (max-width:760px) {
  .story-grid2,.story-observed,.identity-hero,.nn-grid { grid-template-columns:1fr; }
  .nn-grid { gap:6px; }
  .stat-strip { grid-template-columns:1fr; }
}
"""
    )
    mo.Html(f"<style>{THEME}</style>")
    return (THEME,)


@app.cell
def _(data, external, labels, chemistry, models, np, paths):
    # Reuse the project's default EFSA -> structure -> substance -> standardize
    # -> ECFP4 -> LightGBM path. The long notebook remains the place where the
    # scaffold-split validation, calibration and ablations are shown.
    _efsa = data.load_efsa()
    _with_struct = data.with_structure_only(_efsa)
    _agg = labels.aggregate_by_substance(
        _with_struct,
        key="inchikey",
        rule="any_positive",
        ambiguous="exclude",
    )
    _std = chemistry.standardize_table(
        _agg,
        smiles_col="smiles",
        cache_path=paths.CACHE_STANDARDIZATION,
    )

    _external_loaded = external.load(cache_path=paths.CACHE_STANDARDIZATION)
    _merged, _source_conflicts = external.merge(_std, _external_loaded)

    modelable = _merged[
        _merged["y"].notna() & _merged["ok"].fillna(False)
    ].copy()
    modelable = (
        modelable
        .drop_duplicates(subset="inchikey_std")
        .reset_index(drop=True)
    )

    y_train = modelable["y"].astype(int).to_numpy()
    train_fps = chemistry.fingerprints(
        modelable["smiles_std"], radius=2, n_bits=2048
    )
    X_train = chemistry.matrix(train_fps, n_bits=2048)

    demo_model = models.build("lgbm", 0).fit(X_train, y_train)
    train_keys = set(modelable["inchikey_std"].dropna().astype(str))

    model_note = {
        "n": int(len(modelable)),
        "positive": int(y_train.sum()),
        "prevalence": float(y_train.mean()) if len(y_train) else float("nan"),
        "external_rows": int(len(_merged) - len(_std)),
        "source_conflicts": int(len(_source_conflicts)),
    }

    return (
        X_train,
        demo_model,
        model_note,
        modelable,
        train_fps,
        train_keys,
        y_train,
    )


@app.cell
def _(
    chemistry,
    demo_model,
    gcms_candidates,
    modelable,
    np,
    train_fps,
    train_keys,
):
    # Fixed, expensive layer: score every candidate once and find its closest
    # training compound. The applicability cutoff is applied later reactively.
    _fps = chemistry.fingerprints(
        gcms_candidates["smiles"], radius=2, n_bits=2048
    )
    _valid = [i for i, fp in enumerate(_fps) if fp is not None]

    _p = np.full(len(_fps), np.nan)
    _sim = np.full(len(_fps), np.nan)
    _nn_smiles = np.array([""] * len(_fps), dtype=object)
    _nn_name = np.array([""] * len(_fps), dtype=object)

    _name_col = next(
        (
            c
            for c in ("name", "compound", "substance", "chemical_name")
            if c in modelable.columns
        ),
        "inchikey_std",
    )

    if _valid:
        _mat = chemistry.matrix([_fps[i] for i in _valid], n_bits=2048)
        _p[_valid] = demo_model.predict_proba(_mat)[:, 1]
        _s, _idx = chemistry.nearest_neighbor(
            [_fps[i] for i in _valid], train_fps
        )
        _sim[_valid] = _s
        for _k, _i in enumerate(_valid):
            _j = int(_idx[_k])
            if _j >= 0:
                _nn_smiles[_i] = str(modelable["smiles_std"].iloc[_j])
                _nn_name[_i] = str(modelable[_name_col].iloc[_j])

    scored_base = gcms_candidates.assign(
        p_genotox=_p,
        tanimoto_neighbour=_sim,
        nn_smiles=_nn_smiles,
        nn_name=_nn_name,
    )

    # Is the assigned identity itself already present in the training table?
    _assigned = scored_base[
        scored_base["is_assigned"].fillna(False)
    ].reset_index(drop=True)
    if len(_assigned):
        _assigned_std = chemistry.standardize_table(
            _assigned[["smiles"]], smiles_col="smiles", cache_path=None
        )
        assigned_train = (
            _assigned.assign(
                assigned_in_training=_assigned_std["inchikey_std"]
                .astype(str)
                .isin(train_keys)
                .to_numpy()
            )
            .drop_duplicates("feature_id")
            .set_index("feature_id")["assigned_in_training"]
        )
    else:
        assigned_train = {}

    return assigned_train, scored_base


@app.cell
def _(HOOK_FEATURE, NIAS, gcms_features, mo):
    _steps = {
        "1 · Observe": 1,
        "2 · Identify": 2,
        "3 · Challenge": 3,
        "4 · Extrapolate": 4,
        "5 · Propagate": 5,
    }
    scene = mo.ui.radio(
        options=_steps,
        value="1 · Observe",
        label="",
        inline=True,
    )

    ad_cut = mo.ui.slider(
        start=0.20,
        stop=0.70,
        step=0.05,
        value=0.40,
        show_value=True,
        label="How far are you willing to extrapolate?  Tanimoto cutoff",
        full_width=True,
    )

    identity_mode = mo.ui.radio(
        options={
            "Trust the published assignment": "top1_only",
            "Keep structural uncertainty": "uniform",
        },
        value="Keep structural uncertainty",
        label="Identity assumption",
    )

    _feat = gcms_features[
        gcms_features["n_isomers_total"].fillna(0) > 0
    ].copy()
    _feat = _feat.set_index("feature_id")
    _order = [HOOK_FEATURE] + [x for x in _feat.index if x != HOOK_FEATURE]
    _feat = _feat.loc[_order].reset_index()

    _opts = {}
    for _, _r in _feat.iterrows():
        _fid = str(_r["feature_id"])
        _mark = "⚠ " if _fid in NIAS else ""
        _msi = _r.get("msi_level")
        _id_txt = (
            "confirmed with standard"
            if _msi == _msi and int(_msi) == 1
            else "library-only"
        )
        _opts[
            f"{_mark}{_r.get('chemical_formula') or '?'} · "
            f"{int(_r.get('n_isomers_total') or 0):,} isomers · "
            f"{_id_txt} ({_fid})"
        ] = _fid

    explore_feature = mo.ui.dropdown(
        options=_opts,
        value=next(iter(_opts)),
        label="Explore another feature",
        searchable=True,
    )

    return ad_cut, explore_feature, identity_mode, scene


@app.cell
def _(HOOK_FEATURE, scored_base, mo):
    # Candidate selector used only after identity has been introduced.
    _sub = (
        scored_base[scored_base["feature_id"] == HOOK_FEATURE]
        .sort_values("rank")
        .reset_index(drop=True)
    )
    _candidate_opts = {}
    for _i, _r in _sub.iterrows():
        _label = f"#{int(_r.get('rank', _i + 1))}"
        if bool(_r.get("is_assigned", False)):
            _label += " — published assignment"
        _candidate_opts[_label] = int(_i)

    _assigned_idx = next(
        (
            _i
            for _i, _r in _sub.iterrows()
            if bool(_r.get("is_assigned", False))
        ),
        0,
    )
    _assigned_label = next(
        (
            k
            for k, v in _candidate_opts.items()
            if v == _assigned_idx
        ),
        next(iter(_candidate_opts)),
    )

    candidate_choice = mo.ui.dropdown(
        options=_candidate_opts,
        value=_assigned_label,
        label="Structure to test",
    )
    return (candidate_choice,)


@app.cell
def _(candidate_view, html, np):
    def safe_text(value, fallback="—"):
        if value is None:
            return fallback
        try:
            if value != value:
                return fallback
        except Exception:
            pass
        s = str(value).strip()
        return html.escape(s) if s and s.lower() != "nan" else fallback

    def fmtfmt_num(value, digits=2, fallback="—"):
        try:
            x = float(value)
            if not np.isfinite(x):
                return fallback
            return f"{x:.{digits}f}"
        except Exception:
            return fallback

    def signal_html(feature_id):
        # Schematic only: no raw chromatogram/spectrum is fabricated.
        return f"""
        <div class="story-signal">
          <div class="caption">schematic chromatographic feature · not raw study data</div>
          <div class="fid">{html.escape(str(feature_id))}</div>
          <div class="axis"></div>
          <div class="trace">
            <svg viewBox="0 0 700 150" preserveAspectRatio="none" aria-label="schematic peak">
              <path d="M0,125 C90,124 160,126 225,123
                       C275,120 300,116 325,88
                       C345,62 357,18 371,10
                       C386,22 397,72 416,98
                       C444,126 520,123 700,125"
                    fill="none" stroke="#1f6f78" stroke-width="3"/>
              <line x1="371" y1="8" x2="371" y2="130"
                    stroke="#b7791f" stroke-width="1.5" stroke-dasharray="5 5"/>
            </svg>
          </div>
        </div>
        """

    def mol_svg(smiles, width=330, height=230, highlight=False):
        try:
            return candidate_view.molecule_svg(
                smiles, width=width, height=height, highlight=highlight
            )
        except Exception:
            return '<div class="story-note">structure unavailable</div>'

    def identity_grid(sub, n_show=12):
        d = sub.sort_values("rank").head(n_show)
        cards = []
        for _, r in d.iterrows():
            assigned = bool(r.get("is_assigned", False))
            rank = int(r.get("rank", 0))
            svg = mol_svg(r.get("smiles"), width=170, height=112, highlight=False)
            cards.append(
                '<div class="alt-card%s">'
                '<div class="tag">%s</div>'
                '<div class="draw">%s</div>'
                '<div class="cap"><span>#%d</span><span>%s</span></div>'
                '</div>'
                % (
                    " assigned" if assigned else "",
                    "★ published assignment" if assigned else "same formula",
                    svg,
                    rank,
                    "assigned" if assigned else "alternative",
                )
            )
        more = ""
        if len(sub) > len(d):
            more = (
                f'<div class="story-note" style="margin-top:8px">'
                f'Showing {len(d)} of {len(sub)} structures carried by the demo.'
                f'</div>'
            )
        return '<div class="alt-grid">' + "".join(cards) + "</div>" + more

    def domain_html(sim, cutoff):
        sim = 0.0 if not np.isfinite(sim) else float(np.clip(sim, 0, 1))
        cutoff = float(np.clip(cutoff, 0, 1))
        return f"""
        <div class="domain-scale">
          <div class="story-eyebrow">Similarity to training chemistry</div>
          <div class="domain-track">
            <div class="domain-fill" style="width:{100*sim:.1f}%"></div>
            <div class="domain-dot" style="left:{100*sim:.1f}%"></div>
            <div class="domain-cut" style="left:{100*cutoff:.1f}%"></div>
          </div>
          <div class="domain-labels">
            <span>0 · remote</span>
            <span>candidate Tc = {sim:.2f}</span>
            <span>cutoff = {cutoff:.2f}</span>
            <span>1 · identical</span>
          </div>
        </div>
        """

    def coverage_row(fid, coverage, level, is_nias=False):
        c = float(np.clip(coverage, 0, 1)) if np.isfinite(coverage) else 0.0
        cls = " zero" if c <= 0 else ""
        tag = (
            '<span class="story-pill ok">confirmed</span>'
            if level == 1
            else '<span class="story-pill dim">library-only</span>'
        )
        if is_nias:
            tag += '<span class="story-pill warn">NIAS</span>'
        return f"""
        <div class="cov-row{cls}">
          <div class="cov-top">
            <span class="cov-id">{html.escape(str(fid))}</span>
            {tag}
            <span class="cov-spacer"></span>
            <b>{100*c:.0f}% evaluable</b>
          </div>
          <div class="cov-track"><div class="cov-fill" style="width:{100*c:.1f}%"></div></div>
        </div>
        """

    return (
        coverage_row,
        domain_html,
        identity_grid,
        mol_svg,
        fmt_num,
        signal_html,
        safe_text,
    )


@app.cell
def _(
    HOOK_FEATURE,
    NIAS,
    ad_cut,
    explore_feature,
    gcms_features,
    identity_mode,
    integration,
    np,
    scored_base,
):
    # Fast reactive layer. Moving the domain threshold or changing the identity
    # assumption does NOT refit the model.
    AD_CUT = float(ad_cut.value)
    scored = scored_base.assign(
        in_domain=(
            scored_base["tanimoto_neighbour"].fillna(0.0)
            >= AD_CUT - 1e-12
        )
    )

    identity_method = str(identity_mode.value)
    scored = scored.assign(
        weight=integration.gated_weights(
            scored,
            gcms_features,
            method=identity_method,
            temperature=1.0,
        )
    )

    feature_scores = integration.probability_per_feature(
        scored,
        scored["p_genotox"],
        scored["weight"],
        in_domain=scored["in_domain"],
    )

    # Attach the evidence level and the assigned-identity similarity.
    _levels = gcms_features.set_index("feature_id")["msi_level"]
    _assigned_sim = (
        scored[
            scored["is_assigned"].fillna(False)
        ]
        .drop_duplicates("feature_id")
        .set_index("feature_id")["tanimoto_neighbour"]
    )
    wrapper = feature_scores.copy()
    wrapper["msi_level"] = wrapper["feature_id"].map(_levels)
    wrapper["assigned_similarity"] = wrapper["feature_id"].map(_assigned_sim)
    wrapper["is_nias"] = wrapper["feature_id"].isin(NIAS)

    _lvl = wrapper["msi_level"]
    _confirmed = wrapper.loc[_lvl == 1, "covered_weight"]
    _library = wrapper.loc[_lvl != 1, "covered_weight"]

    wrapper_stats = {
        "n_features": int(len(wrapper)),
        "n_confirmed": int((_lvl == 1).sum()),
        "n_library": int((_lvl != 1).sum()),
        "confirmed_coverage": (
            float(_confirmed.mean()) if len(_confirmed) else float("nan")
        ),
        "library_coverage": (
            float(_library.mean()) if len(_library) else float("nan")
        ),
        "lowest_assigned": set(
            wrapper.dropna(subset=["assigned_similarity"])
            .nsmallest(2, "assigned_similarity")["feature_id"]
        ),
        "explore_feature": str(explore_feature.value),
        "hook_feature": HOOK_FEATURE,
    }

    return AD_CUT, feature_scores, identity_method, scored, wrapper, wrapper_stats


@app.cell
def _(
    AD_CUT,
    HOOK_FEATURE,
    NIAS,
    STUDY,
    coverage_row,
    domain_html,
    identity_grid,
    mol_svg,
    fmt_num,
    signal_html,
    safe_text,
    assigned_train,
    candidate_choice,
    feature_meta,
    gcms_features,
    gcms_info,
    identity_mode,
    mo,
    model_note,
    scene,
    scored,
    scored_base,
    wrapper,
    wrapper_stats,
):
    _s = int(scene.value)
    _feat = gcms_features.set_index("feature_id").loc[HOOK_FEATURE]
    _raw = (
        feature_meta.loc[HOOK_FEATURE]
        if HOOK_FEATURE in feature_meta.index
        else None
    )
    _hook = (
        scored[scored["feature_id"] == HOOK_FEATURE]
        .sort_values("rank")
        .reset_index(drop=True)
    )
    _assigned = _hook[_hook["is_assigned"].fillna(False)]
    _assigned = _assigned.iloc[0] if len(_assigned) else _hook.iloc[0]

    _msi = _feat.get("msi_level")
    _msi_int = int(_msi) if _msi == _msi else None
    _confirmed = _msi_int == 1

    _raw_name = (
        _raw.get("metabolite_identification")
        if _raw is not None
        else None
    )
    _raw_reliability = (
        _raw.get("reliability")
        if _raw is not None
        else None
    )
    _raw_mz = (
        _raw.get("mass_to_charge")
        if _raw is not None
        else None
    )
    _raw_rt = (
        _raw.get("retention_time")
        if _raw is not None
        else None
    )

    _titles = {
        1: "What did the experiment actually give us?",
        2: "Interpretation turns a feature into a molecule",
        3: "How dependent are we on that identity?",
        4: "Has the model seen chemistry like this before?",
        5: "Propagate uncertainty through the whole wrapper",
    }
    _subs = {
        1: (
            "Start with the analytical object, not with a structure. "
            "A molecular drawing would already assume the answer."
        ),
        2: (
            "Now — and only now — reveal the structure assigned by the published "
            "GC–MS identification."
        ),
        3: (
            "The same molecular formula can correspond to many structures. "
            "These are a sensitivity set, not competing library hits."
        ),
        4: (
            "Accept one identity provisionally, then ask whether the toxicity "
            "model has training chemistry close enough to support extrapolation."
        ),
        5: (
            "Change the identity assumption or the applicability-domain cutoff. "
            "The ML model stays fixed; what changes is how much of each feature "
            "the evidence lets us evaluate."
        ),
    }

    _header = mo.vstack(
        [
            mo.Html(
                '<div class="story-shell"><div class="story-head">'
                '<span class="story-kicker">Identity-aware genotoxicity screening</span>'
                f'<span class="story-study">{STUDY} · step {_s}/5</span>'
                '</div></div>'
            ),
            scene,
            mo.Html(
                '<div class="story-shell">'
                f'<div class="story-title">{_titles[_s]}</div>'
                f'<div class="story-sub">{_subs[_s]}</div>'
                '</div>'
            ),
        ]
    )

    if _s == 1:
        _mz = safe_text(_raw_mz)
        _rt = safe_text(_raw_rt)
        _body = mo.Html(
            '<div class="story-shell">'
            + signal_html(HOOK_FEATURE)
            + f"""
            <div class="story-observed">
              <div class="story-panel">
                <div class="story-eyebrow">Reported for this feature in the deposit</div>
                <ul class="story-list">
                  <li><b>feature</b> <code>{safe_text(HOOK_FEATURE)}</code></li>
                  <li><b>mass_to_charge</b> {_mz} <span class="story-note">(upstream MAF field; here it is not treated as neutral exact mass)</span></li>
                  <li><b>retention field</b> {_rt}</li>
                  <li><b>an assigned identity exists</b> — but we have not revealed it yet</li>
                </ul>
              </div>
              <div class="story-panel">
                <div class="story-eyebrow">Not directly observed by this notebook</div>
                <ul class="story-list">
                  <li>✗ a unique molecular structure</li>
                  <li>✗ a neutral monoisotopic exact mass derived from the experiment</li>
                  <li>✗ genotoxicity</li>
                  <li>✗ an identity probability for every PubChem isomer</li>
                </ul>
                <div class="story-note" style="margin-top:8px">
                  The bundled MAF is a feature/assignment table. The raw EI spectrum is not
                  fabricated here; the peak drawing above is explicitly schematic.
                </div>
              </div>
            </div>
            <div class="final-line">
              The next step is interpretation: somebody has to turn this feature into a structure hypothesis.
            </div>
            </div>
            """
        )

    elif _s == 2:
        _svg = mol_svg(
            _assigned.get("smiles"),
            width=390,
            height=260,
            highlight=False,
        )
        _status = (
            '<span class="story-pill ok">confirmed with authentic standard</span>'
            if _confirmed
            else '<span class="story-pill warn">tentative library assignment</span>'
        )
        _standard = "YES" if _confirmed else "NO"
        _level = (
            f"MSI {_msi_int}"
            if _msi_int is not None
            else safe_text(_raw_reliability)
        )
        _body = mo.Html(
            f"""
            <div class="story-shell">
              <div class="story-panel identity-hero">
                <div class="identity-mol">{_svg}</div>
                <div>
                  <div class="story-eyebrow">Published assignment</div>
                  <div class="story-big">{safe_text(_raw_name, "Assigned structure")}</div>
                  <div style="margin:9px 0">{_status}</div>
                  <div class="story-grid2">
                    <div>
                      <div class="story-eyebrow">Identification level</div>
                      <div class="story-mono">{safe_text(_level)}</div>
                    </div>
                    <div>
                      <div class="story-eyebrow">Authentic standard?</div>
                      <div class="story-mono">{_standard}</div>
                    </div>
                    <div>
                      <div class="story-eyebrow">Formula reported</div>
                      <div class="story-mono">{safe_text(_feat.get("chemical_formula"))}</div>
                    </div>
                    <div>
                      <div class="story-eyebrow">Feature</div>
                      <div class="story-mono">{safe_text(HOOK_FEATURE)}</div>
                    </div>
                  </div>
                </div>
              </div>
              <div class="final-line">
                Tentative identification ≠ confirmed compound.
              </div>
            </div>
            """
        )

    elif _s == 3:
        _n_total = int(_feat.get("n_isomers_total") or len(_hook))
        _n_carried = len(_hook)
        _grid = identity_grid(_hook, n_show=12)
        _body = mo.Html(
            f"""
            <div class="story-shell">
              <div class="story-panel">
                <div class="story-eyebrow">Published assignment versus formula-compatible structures</div>
                <div class="stat-strip">
                  <div class="stat-box">
                    <b>1</b><span>published GC–MS assignment</span>
                  </div>
                  <div class="stat-box">
                    <b>{_n_total:,}</b><span>PubChem structures reported for the same molecular formula</span>
                  </div>
                  <div class="stat-box">
                    <b>{_n_carried}</b><span>structures sampled/carried by this demo</span>
                  </div>
                </div>
                {_grid}
              </div>
              <div class="story-panel">
                <b>These are not {_n_carried} competing library hits.</b>
                <div class="story-note" style="margin-top:5px">
                  They are structures consistent with the same molecular formula and sampled
                  from PubChem. Only the highlighted structure is the identity assigned by the
                  published GC–MS workflow. The alternatives are used here as a structural
                  sensitivity analysis; equal weighting is not claimed to be an experimentally
                  measured identity posterior.
                </div>
              </div>
              <div class="final-line">
                Before asking “is it genotoxic?”, ask how much the answer depends on which structure we mean.
              </div>
            </div>
            """
        )

    elif _s == 4:
        _idx = int(candidate_choice.value)
        _r = _hook.iloc[max(0, min(_idx, len(_hook) - 1))]
        _sim = float(_r.get("tanimoto_neighbour"))
        _inside = np.isfinite(_sim) and _sim >= AD_CUT
        _query_svg = mol_svg(
            _r.get("smiles"),
            width=310,
            height=220,
            highlight=False,
        )
        _nn_svg = mol_svg(
            _r.get("nn_smiles"),
            width=310,
            height=220,
            highlight=False,
        )
        _score = _r.get("p_genotox")
        _score_box = (
            f"""
            <div class="story-panel">
              <span class="story-pill ok">inside domain</span>
              <b>Uncalibrated model score: {fmt_num(_score, 3)}</b>
              <div class="story-note">
                The score is shown only because the candidate passes the current
                similarity criterion. The long notebook contains the scaffold-split
                validation and calibration analysis.
              </div>
            </div>
            """
            if _inside
            else f"""
            <div class="story-panel">
              <span class="story-pill warn">outside domain</span>
              <b>Score withheld from the scientific interpretation.</b>
              <div class="story-note">
                The classifier can still output {fmt_num(_score, 3)}, but computing a
                number is not the same as having chemical support for it.
              </div>
            </div>
            """
        )
        _in_train = (
            HOOK_FEATURE in assigned_train.index
            and bool(assigned_train.loc[HOOK_FEATURE])
            if hasattr(assigned_train, "index")
            else False
        )
        _body = mo.vstack(
            [
                candidate_choice,
                mo.Html(
                    f"""
                    <div class="story-shell">
                      <div class="story-panel">
                        <div class="nn-grid">
                          <div>
                            <div class="story-eyebrow">Structure we are asking about</div>
                            <div class="nn-mol">{_query_svg}</div>
                            <div class="story-mono" style="margin-top:5px">
                              candidate #{int(_r.get("rank", _idx + 1))}
                              {" · published assignment" if bool(_r.get("is_assigned", False)) else ""}
                            </div>
                          </div>
                          <div class="nn-mid">
                            <div class="story-eyebrow">nearest training chemistry</div>
                            <div class="nn-score">Tc {fmt_num(_sim, 2)}</div>
                          </div>
                          <div>
                            <div class="story-eyebrow">Closest EFSA training structure</div>
                            <div class="nn-mol">{_nn_svg}</div>
                            <div class="story-mono" style="margin-top:5px">{safe_text(_r.get("nn_name"))}</div>
                          </div>
                        </div>
                        {domain_html(_sim, AD_CUT)}
                      </div>
                      {_score_box}
                      <div class="story-note" style="margin-top:8px">
                        Assigned identity already in the model table: <b>{"yes" if _in_train else "no"}</b>.
                        This is reported so memorisation/leakage is not hidden.
                      </div>
                      <div class="final-line">
                        A model can always return a score. The scientific question is whether it had grounds to return one.
                      </div>
                    </div>
                    """
                ),
            ]
        )

    else:
        _rows = []
        _sorted = wrapper.sort_values(
            ["msi_level", "covered_weight"],
            ascending=[True, False],
            na_position="last",
        )
        for _, _r in _sorted.iterrows():
            _lvl = _r.get("msi_level")
            _lvl_i = int(_lvl) if _lvl == _lvl else 0
            _rows.append(
                coverage_row(
                    _r["feature_id"],
                    float(_r["covered_weight"]),
                    _lvl_i,
                    bool(_r["is_nias"]),
                )
            )

        _c1 = wrapper_stats["confirmed_coverage"]
        _c2 = wrapper_stats["library_coverage"]
        _lowest = wrapper_stats["lowest_assigned"]
        _nias_pattern = _lowest == set(NIAS)

        _pattern_text = (
            "At the current data snapshot, the two NIAS assignments are also the "
            "two assigned structures furthest from the EFSA training chemistry."
            if _nias_pattern
            else "The NIAS are marked explicitly; their position relative to the "
            "training chemistry is recomputed rather than assumed."
        )

        _body = mo.vstack(
            [
                ad_cut,
                identity_mode,
                mo.Html(
                    f"""
                    <div class="story-shell">
                      <div class="stat-strip">
                        <div class="stat-box">
                          <b>{wrapper_stats["n_features"]}</b>
                          <span>GC–MS features</span>
                        </div>
                        <div class="stat-box">
                          <b>{100*_c1:.0f}%</b>
                          <span>mean identity weight evaluable · confirmed ({wrapper_stats["n_confirmed"]})</span>
                        </div>
                        <div class="stat-box">
                          <b>{100*_c2:.0f}%</b>
                          <span>mean identity weight evaluable · library-only ({wrapper_stats["n_library"]})</span>
                        </div>
                      </div>
                      <div class="story-panel">
                        <div class="story-eyebrow">Whole-wrapper evidence map</div>
                        {"".join(_rows)}
                      </div>
                      <div class="story-panel">
                        <b>{_pattern_text}</b>
                        <div class="story-note" style="margin-top:5px">
                          The bars are <i>coverage</i>, not safety. A missing bar does not
                          mean “not genotoxic”; it means the current structure-based model
                          cannot evaluate that part of the feature's identity under the chosen
                          domain criterion.
                        </div>
                      </div>
                      <div class="final-line">
                        The prediction does not begin with the model. It begins with deciding what molecule the analytical feature actually represents.
                      </div>
                    </div>
                    """
                ),
            ]
        )

    _footer = mo.Html(
        f"""
        <div class="story-shell" style="margin-top:14px">
          <div class="story-note">
            Demo model: LightGBM on ECFP4, fitted on {model_note["n"]:,} modelable
            substances ({model_note["positive"]:,} positives; prevalence
            {100*model_note["prevalence"]:.1f}%). This guided notebook is for the
            presentation story; use <code>notebooks/genotox_marimo.py</code> for the
            full scaffold-split validation, calibration, ablations, TTC analysis and
            provenance audit.
          </div>
        </div>
        """
    )

    mo.vstack([_header, _body, _footer])
    return


if __name__ == "__main__":
    app.run()
