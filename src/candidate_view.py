"""Rendering of a feature's candidate set as drawn structures.

The point of this module is that the structural alert stops being a boolean in
a table and becomes the atoms it fired on, drawn on the molecule. Everything
else here -- the weight pill, the domain ring, the collapse/marginalize
comparison -- exists so that one feature's identity uncertainty is legible at a
glance instead of by reading three tables side by side.

Nothing in here computes chemistry. It consumes the columns the notebook has
already produced (`p_genotox`, `weight`, `in_domain`) and the alert patterns
from `chemistry.DNA_REACTIVE_ALERTS`, so a change to either shows up here
without a second implementation to keep in sync.
"""

from __future__ import annotations

import html as _html

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from . import chemistry

# One colour per alert class, so that a viewer who sees two cards lit up the
# same way is looking at the same mechanism. Muted on purpose: the highlight
# has to read as a chemist's annotation, not as a warning banner.
ALERT_COLORS = {
    "aromatic nitro": (0.85, 0.33, 0.31),
    "primary aromatic amine": (0.29, 0.56, 0.78),
    "epoxide": (0.90, 0.62, 0.20),
    "aziridine": (0.80, 0.52, 0.25),
    "N-nitroso": (0.65, 0.35, 0.72),
    "hydrazine": (0.45, 0.62, 0.35),
    "organic azide": (0.35, 0.60, 0.60),
    "aromatic azo": (0.75, 0.45, 0.55),
    "alkyl halide": (0.55, 0.55, 0.60),
    "Michael acceptor": (0.88, 0.50, 0.35),
}
_DEFAULT_COLOR = (0.60, 0.60, 0.65)

_PATTERNS = None


def _patterns():
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = chemistry.alert_patterns()
    return _PATTERNS


def alert_hits(smiles: str):
    """Which alert classes a structure carries, and on which atoms.

    Returns a list of (class name, atom indices, bond indices). An unparseable
    SMILES returns an empty list, which is not the same as 'no alert' -- the
    caller distinguishes the two through the mol being None.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return []
    hits = []
    for name, patt in _patterns():
        atoms = set()
        for match in mol.GetSubstructMatches(patt):
            atoms.update(match)
        if not atoms:
            continue
        bonds = [b.GetIdx() for b in mol.GetBonds()
                 if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
        hits.append((name, sorted(atoms), bonds))
    return hits


def molecule_svg(smiles: str, width: int = 210, height: int = 150,
                 highlight: bool = True) -> str:
    """Draw one structure, colouring the atoms that trip each alert class.

    When several classes overlap on the same atom the first one wins, which is
    a display choice and not a statement about which mechanism dominates; the
    card lists every class that fired underneath the drawing.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return ('<div class="cv-nostruct">no structure</div>')

    atom_colors, bond_colors = {}, {}
    if highlight:
        for name, atoms, bonds in alert_hits(smiles):
            colour = ALERT_COLORS.get(name, _DEFAULT_COLOR)
            for a in atoms:
                atom_colors.setdefault(a, colour)
            for b in bonds:
                bond_colors.setdefault(b, colour)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 1.4
    opts.highlightRadius = 0.32
    opts.fillHighlights = True
    try:
        mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    except Exception:
        pass
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors) or None,
        highlightBonds=list(bond_colors) or None,
        highlightAtomColors=atom_colors or None,
        highlightBondColors=bond_colors or None,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("svg:", "")


CSS = """
<style>
.cv-wrap { font: 13px/1.4 ui-sans-serif, system-ui, sans-serif; }
.cv-grid { display: flex; flex-wrap: wrap; gap: 10px; }
.cv-card { width: 212px; border: 1px solid rgba(128,128,128,.35);
           border-radius: 10px; padding: 8px; box-sizing: border-box; }
.cv-card.cv-out { border-style: dashed; opacity: .62; }
.cv-card.cv-assigned { border-color: rgba(70,130,180,.9); border-width: 2px; }
.cv-tile { background: #fbfbf9; border-radius: 6px; display: flex;
           justify-content: center; align-items: center; min-height: 150px; }
.cv-nostruct { color: #999; font-size: 12px; padding: 60px 0; text-align: center; }
.cv-row { display: flex; justify-content: space-between; align-items: center;
          margin-top: 6px; gap: 6px; }
.cv-p { font-variant-numeric: tabular-nums; font-weight: 600; }
.cv-bar { height: 5px; border-radius: 3px; background: rgba(128,128,128,.22);
          margin-top: 5px; overflow: hidden; }
.cv-bar > span { display: block; height: 100%; background: #b8433f; }
.cv-pill { font-size: 11px; padding: 1px 6px; border-radius: 999px;
           border: 1px solid rgba(128,128,128,.4); white-space: nowrap; }
.cv-alerts { margin-top: 6px; font-size: 11px; display: flex;
             flex-wrap: wrap; gap: 4px; }
.cv-alert { padding: 1px 6px; border-radius: 4px; color: #fff; }
.cv-none { color: #8a8a8a; font-size: 11px; margin-top: 6px; }
.cv-cmp { display: flex; gap: 22px; flex-wrap: wrap; align-items: flex-end;
          margin: 4px 0 12px; }
.cv-stat small { display: block; font-size: 11px; opacity: .7; }
.cv-stat b { font-size: 22px; font-variant-numeric: tabular-nums; }
.cv-ttc { margin-top: 8px; }
.cv-note { font-size: 12px; opacity: .75; margin-top: 6px; }
</style>
"""


def _rgb(c):
    return "rgb(%d,%d,%d)" % tuple(int(255 * v) for v in c)


def candidate_card_html(row) -> str:
    """One candidate: structure, probability, weight, domain, alert classes."""
    smi = row.get("smiles")
    p = pd.to_numeric(row.get("p_genotox"), errors="coerce")
    w = pd.to_numeric(row.get("weight"), errors="coerce")
    in_dom = bool(row.get("in_domain")) if pd.notna(row.get("in_domain")) else False
    assigned = bool(row.get("is_assigned", False))

    classes = ["cv-card"]
    if not in_dom:
        classes.append("cv-out")
    if assigned:
        classes.append("cv-assigned")

    hits = alert_hits(smi) if isinstance(smi, str) else []
    if hits:
        chips = "".join(
            '<span class="cv-alert" style="background:%s">%s</span>'
            % (_rgb(ALERT_COLORS.get(n, _DEFAULT_COLOR)), _html.escape(n))
            for n, _, _ in hits
        )
        alerts = '<div class="cv-alerts">%s</div>' % chips
    else:
        alerts = '<div class="cv-none">none of the ten classes</div>'

    # Outside the applicability domain the model still returns a number, and a
    # large one read next to a structure is exactly the claim this notebook
    # refuses to make. The score is withheld rather than shown greyed out.
    p_txt = "--" if pd.isna(p) else ("not evaluable" if not in_dom else "%.3f" % p)
    w_txt = "--" if pd.isna(w) else "%.3f" % w
    # Log scale: scores span 1e-5 to 0.7, and a linear bar is invisible for
    # everything but the top few.
    if pd.isna(p) or not in_dom or p <= 0:
        bar = 0.0
    else:
        bar = float(np.clip((np.log10(max(p, 1e-5)) + 5.0) / 5.0, 0, 1))
    rank = row.get("rank")
    label = "rank %s" % rank if pd.notna(rank) else ""
    if assigned:
        label += " &middot; assigned"

    return (
        '<div class="%s">'
        '<div class="cv-tile">%s</div>'
        '<div class="cv-row"><span class="cv-p">P %s</span>'
        '<span class="cv-pill">w %s</span></div>'
        '<div class="cv-bar"><span style="width:%.1f%%"></span></div>'
        '<div class="cv-row"><span class="cv-pill">%s</span>'
        '<span class="cv-pill">%s</span></div>'
        "%s</div>"
    ) % (
        " ".join(classes),
        molecule_svg(smi) if isinstance(smi, str) else '<div class="cv-nostruct">no structure</div>',
        p_txt, w_txt, 100 * bar,
        label or "&nbsp;",
        "in domain" if in_dom else "out of domain",
        alerts,
    )


def candidate_grid_html(cand: pd.DataFrame, feature_id, n_max: int = 12) -> str:
    """One feature's candidates: evaluable ones first by model score, the rest
    grouped after as not-evaluable, capped for rendering."""
    sub = cand[cand["feature_id"] == feature_id]
    # A high score OUTSIDE the applicability domain is not evidence of higher
    # hazard, so out-of-domain candidates must not lead on score. Order in-domain
    # candidates first (by score), then the not-evaluable group.
    if "p_genotox" in sub.columns:
        _dom = (
            sub["in_domain"].fillna(False).astype(int)
            if "in_domain" in sub.columns
            else 0
        )
        sub = (
            sub.assign(_dom=_dom)
            .sort_values(["_dom", "p_genotox"], ascending=[False, False], na_position="last")
            .drop(columns="_dom")
        )
    elif "rank" in sub.columns:
        sub = sub.sort_values("rank")
    shown = sub.head(n_max)
    cards = "".join(candidate_card_html(r) for _, r in shown.iterrows())
    more = ""
    if len(sub) > len(shown):
        more = ('<div class="cv-note">Showing %d of %d candidates carried in the '
                "sum; the rest are drawn from the same isomer space.</div>"
                % (len(shown), len(sub)))
    return '<div class="cv-wrap"><div class="cv-grid">%s</div>%s</div>' % (cards, more)


def collapse_comparison(cand: pd.DataFrame, feature_id,
                        baseline_ttc: float = 90.0,
                        genotoxic_ttc: float = 0.15,
                        threshold: float = 0.5) -> dict:
    """What committing to candidate 1 gives you, against marginalizing.

    Returns both scores, the weight actually covered, and the TTC limit each
    route lands on. The limit is a step function of the call, which is why the
    two numbers are 90 and 0.15 and never anything in between.
    """
    sub = cand[cand["feature_id"] == feature_id].copy()
    p = pd.to_numeric(sub["p_genotox"], errors="coerce")
    w = pd.to_numeric(sub["weight"], errors="coerce").fillna(0.0)
    usable = p.notna()
    if "in_domain" in sub.columns:
        usable &= sub["in_domain"].fillna(False).astype(bool)

    covered = float(w[usable].sum())
    marginal = float((w[usable] * p[usable]).sum())

    top = sub.sort_values("rank").head(1) if "rank" in sub.columns else sub.head(1)
    p_top = pd.to_numeric(top["p_genotox"], errors="coerce")
    top1 = float(p_top.iloc[0]) if len(p_top) and pd.notna(p_top.iloc[0]) else float("nan")

    def _limit(score):
        if pd.isna(score):
            return float("nan")
        return genotoxic_ttc if score >= threshold else baseline_ttc

    return {
        "top1": top1,
        "marginal": marginal,
        "marginal_norm": marginal / covered if covered > 0 else float("nan"),
        "covered_weight": covered,
        "n_candidates": int(len(sub)),
        "ttc_top1": _limit(top1),
        "ttc_marginal": _limit(marginal),
        "threshold": float(threshold),
    }


def ttc_bar_html(cmp: dict, lo: float = 0.01, hi: float = 2000.0) -> str:
    """The two limits on a log axis, because the gap is the whole argument."""
    def pos(v):
        if not np.isfinite(v) or v <= 0:
            return 0.0
        f = (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        return 100 * float(np.clip(f, 0, 1))

    marks = ""
    for v, label, colour in ((cmp["ttc_top1"], "top-1", "#4a7ba7"),
                             (cmp["ttc_marginal"], "marginalized", "#b8433f")):
        if not np.isfinite(v):
            continue
        marks += (
            '<div style="position:absolute;left:%.1f%%;top:0;transform:translateX(-50%%);'
            'text-align:center"><div style="width:2px;height:22px;background:%s;'
            'margin:0 auto"></div><div style="font-size:11px;color:%s;white-space:nowrap">'
            "%s<br>%g &micro;g/day</div></div>" % (pos(v), colour, colour, label, v)
        )
    return ('<div class="cv-ttc" style="position:relative;height:66px;'
            'border-bottom:1px solid rgba(128,128,128,.4)">%s</div>' % marks)


def comparison_html(cmp: dict) -> str:
    """Headline numbers for one feature, big enough to read from a video."""
    def fmt(v):
        return "--" if not np.isfinite(v) else "%.3f" % v

    return (
        '<div class="cv-wrap"><div class="cv-cmp">'
        '<div class="cv-stat"><small>commit to candidate 1</small><b>%s</b></div>'
        '<div class="cv-stat"><small>marginalized over %d candidates</small><b>%s</b></div>'
        '<div class="cv-stat"><small>identity mass evaluated</small><b>%s</b></div>'
        "</div>%s"
        '<div class="cv-note">The limit is a step function of the call at '
        "threshold %.3f: the genotoxic TTC replaces the Cramer-class one outright "
        "rather than scaling it, so there is nothing between the two marks.</div></div>"
    ) % (
        fmt(cmp["top1"]), cmp["n_candidates"], fmt(cmp["marginal"]),
        fmt(cmp["covered_weight"]), ttc_bar_html(cmp), cmp["threshold"],
    )


def feature_panel(cand: pd.DataFrame, feature_id, threshold: float = 0.5,
                  n_max: int = 12, baseline_ttc: float = 90.0):
    """Everything for one feature, as a marimo element."""
    import marimo as mo

    cmp = collapse_comparison(cand, feature_id, baseline_ttc=baseline_ttc,
                              threshold=threshold)
    return mo.Html(CSS + comparison_html(cmp) + candidate_grid_html(cand, feature_id, n_max))


# --- inspector panel --------------------------------------------------------
#
# Textbook mechanism for each of the ten DNA-reactive structural classes. This
# is a one-line description of the class chemistry, NOT the model's reasoning
# and NOT a validated alert rulebase (see chemistry.DNA_REACTIVE_ALERTS: "none
# of these ten" is not "no alert", and a hit is a flag, not a conclusion).
ALERT_MECHANISM = {
    "aromatic nitro": "Nitroreduction can yield nitrenium / hydroxylamine species that form covalent DNA adducts.",
    "primary aromatic amine": "Metabolic N-oxidation can give a nitrenium ion that binds DNA.",
    "epoxide": "The strained oxirane ring can alkylate DNA bases directly.",
    "aziridine": "The strained three-membered nitrogen ring can alkylate DNA directly.",
    "N-nitroso": "Metabolic activation generates an alkyldiazonium ion that alkylates DNA.",
    "hydrazine": "Oxidation can produce diazonium species and radicals that damage DNA.",
    "organic azide": "Can generate reactive nitrene intermediates.",
    "aromatic azo": "Reductive cleavage regenerates aromatic amines (see that class).",
    "alkyl halide": "A leaving-group carbon can alkylate DNA nucleophiles by substitution.",
    "Michael acceptor": "The activated alkene adds DNA / protein nucleophiles by conjugate addition.",
}

DOMAIN_CUTOFF = 0.40

INSPECTOR_CSS = """
.insp { font: 13px/1.5 ui-sans-serif, system-ui, sans-serif; color: #2b2b2b; }
.insp-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
  border-bottom: 1px solid #e4dfd6; padding-bottom: 6px; margin-bottom: 8px; }
.insp-idx { font: 600 12px ui-monospace, Menlo, monospace; color: #6b6b6b; }
.insp-name { font: 600 16px ui-monospace, Menlo, monospace; color: #2b2b2b; }
.insp-ikey { font: 11px ui-monospace, Menlo, monospace; color: #9a9a9a; }
.insp-tag { margin-left: auto; font: 600 10px ui-sans-serif, system-ui, sans-serif;
  padding: 2px 8px; border-radius: 999px; }
.insp-tag.assigned { background: #1f6f78; color: #fff; }
.insp-tag.alt { background: #efece6; color: #6b6b6b; }
.insp-hero { display: flex; justify-content: center; padding: 6px 0 10px 0; }
.insp-hero svg { max-width: 100%; height: auto; }
.insp-sec { margin-top: 12px; }
.insp-sec-h { font: 600 11px ui-sans-serif, system-ui, sans-serif;
  text-transform: uppercase; letter-spacing: .06em; color: #8a8a8a; margin-bottom: 6px; }
.insp-nn { display: flex; align-items: center; gap: 8px; justify-content: center; }
.insp-nn-col { text-align: center; flex: 1; }
.insp-nn-lbl { font-size: 10px; color: #8a8a8a; margin-bottom: 2px; min-height: 26px; }
.insp-nn-lbl span { color: #2b2b2b; font-weight: 600; }
.insp-nn-arrow { font: 600 12px ui-monospace, Menlo, monospace; color: #1f6f78; white-space: nowrap; }
.insp-dom { margin-top: 8px; padding: 8px 10px; border-radius: 6px; font-size: 12px; }
.insp-dom.indom { background: #e7f1f2; color: #16535a; border-left: 3px solid #1f6f78; }
.insp-dom.outdom { background: #fbeccb; color: #7a5104; border-left: 3px solid #b7791f; }
.insp-score { display: flex; align-items: baseline; gap: 10px; padding: 8px 10px;
  border-radius: 6px; background: #fffdf9; border: 1px solid #e4dfd6; }
.insp-score .lbl { font-size: 11px; color: #8a8a8a; text-transform: uppercase; letter-spacing: .05em; }
.insp-score .val { font: 600 20px ui-monospace, Menlo, monospace; color: #1f6f78; }
.insp-score.noeval .val { font-size: 15px; color: #b7791f; font-style: italic; }
.insp-score .note { font-size: 11px; color: #8a8a8a; margin-left: auto; text-align: right; max-width: 55%; }
.insp-weight { margin-top: 8px; font-size: 12px; color: #4a4a4a; }
.insp-weight b { color: #1f6f78; font-family: ui-monospace, Menlo, monospace; }
.insp-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 6px; }
.insp-chip { font: 600 10px ui-sans-serif, system-ui, sans-serif; color: #fff;
  padding: 2px 7px; border-radius: 999px; }
.insp-mech { font-size: 12px; color: #4a4a4a; margin: 2px 0; }
.insp-mech b { color: #2b2b2b; }
.insp-none { font-size: 12px; color: #6b6b6b; font-style: italic; }
.insp-foot { font-size: 10.5px; color: #8a8a8a; margin-top: 8px; line-height: 1.5;
  border-top: 1px dashed #e4dfd6; padding-top: 6px; }
"""


def inspector_html(row, weight_txt: str = None, cutoff: float = None) -> str:
    """The right-column detail panel for one selected candidate structure.

    ``row`` is a single row of the scored table, carrying ``name``, ``smiles``,
    ``is_assigned``, ``in_domain``, ``p_genotox``, ``tanimoto_neighbour``,
    ``nn_smiles`` and ``nn_name``. ``weight_txt`` is an optional pre-formatted
    line stating how much of the peak this candidate carries under the current
    weighting assumption.

    Honesty rules enforced here:
      - a model score is shown only inside the applicability domain, and always
        labelled uncalibrated;
      - outside the domain the panel abstains ("Not evaluable") and shows no
        number;
      - structural classes are flagged with their textbook mechanism, framed
        as a structural flag, never as the model's reasoning or a conclusion.
    """
    def g(key, default=None):
        try:
            v = row[key]
        except Exception:
            v = default
        return v

    smi = g("smiles")
    smi = smi if isinstance(smi, str) else ""
    # This dataset carries no trivial name for a candidate -- the structures are
    # isomers enumerated by molecular formula. Anchor the header on the formula
    # (computed from the structure) and the InChIKey, never an invented name.
    inchikey = g("inchikey") or ""
    formula = ""
    if smi:
        try:
            from rdkit.Chem import rdMolDescriptors
            _m = Chem.MolFromSmiles(smi)
            if _m is not None:
                formula = rdMolDescriptors.CalcMolFormula(_m)
        except Exception:
            formula = ""
    headline = formula or (inchikey.split("-")[0] if inchikey else "candidate")
    is_assigned = bool(g("is_assigned", False))
    in_dom = bool(g("in_domain", False))
    p = g("p_genotox")
    tan = g("tanimoto_neighbour")
    nn_smi = g("nn_smiles")
    nn_smi = nn_smi if isinstance(nn_smi, str) else ""
    nn_name = g("nn_name") or "training structure"

    tan_txt = "--" if tan is None or tan != tan else "%.2f" % float(tan)
    tag = ('<span class="insp-tag assigned">assigned identity</span>'
           if is_assigned else
           '<span class="insp-tag alt">alternative (same formula)</span>')

    hero = molecule_svg(smi, width=300, height=210) if smi else '<div class="cv-nostruct">no structure</div>'
    cand_small = molecule_svg(smi, width=150, height=120) if smi else ""
    nn_small = molecule_svg(nn_smi, width=150, height=120, highlight=False) if nn_smi else '<div class="cv-nostruct">--</div>'

    _cut = DOMAIN_CUTOFF if cutoff is None else float(cutoff)
    if in_dom:
        dom = (
            '<div class="insp-dom indom">Inside the model\'s applicability domain &mdash; '
            'nearest training structure at Tanimoto %s (criterion &ge; %.2f).</div>'
            % (tan_txt, _cut)
        )
    else:
        dom = (
            '<div class="insp-dom outdom">Outside the model\'s applicability domain &mdash; '
            'nearest training structure at Tanimoto %s, below the %.2f cutoff. '
            'The model has no comparable example, so it does not score this structure.</div>'
            % (tan_txt, _cut)
        )

    if in_dom and p is not None and p == p:
        score = (
            '<div class="insp-sec"><div class="insp-score">'
            '<span class="lbl">uncalibrated model score</span>'
            '<span class="val">%.3f</span>'
            '<span class="note">a relative ranking score, not a calibrated probability</span>'
            '</div></div>' % float(p)
        )
    else:
        score = (
            '<div class="insp-sec"><div class="insp-score noeval">'
            '<span class="lbl">model score</span>'
            '<span class="val">Not evaluable</span>'
            '<span class="note">no score shown &mdash; outside the applicability domain</span>'
            '</div></div>'
        )

    weight = ('<div class="insp-weight">%s</div>' % weight_txt) if weight_txt else ""

    hits = alert_hits(smi) if smi else []
    if hits:
        chips = "".join(
            '<span class="insp-chip" style="background:%s">%s</span>'
            % (_rgb(ALERT_COLORS.get(nm, _DEFAULT_COLOR)), nm)
            for nm, _a, _b in hits
        )
        mech = "".join(
            '<div class="insp-mech"><b>%s.</b> %s</div>'
            % (nm, ALERT_MECHANISM.get(nm, ""))
            for nm, _a, _b in hits
        )
        alerts_body = '<div class="insp-chips">%s</div>%s' % (chips, mech)
    else:
        alerts_body = '<div class="insp-none">None of the ten DNA-reactive classes.</div>'

    return (
        '<div class="insp">'
        '<div class="insp-head"><span class="insp-name">%s</span>'
        '<span class="insp-ikey">%s</span>%s</div>'
        '<div class="insp-hero">%s</div>'
        '<div class="insp-sec"><div class="insp-sec-h">Applicability domain</div>'
        '<div class="insp-nn">'
        '<div class="insp-nn-col"><div class="insp-nn-lbl">this candidate</div>%s</div>'
        '<div class="insp-nn-arrow">Tanimoto %s</div>'
        '<div class="insp-nn-col"><div class="insp-nn-lbl">nearest training structure<br><span>%s</span></div>%s</div>'
        '</div>%s</div>'
        '%s%s'
        '<div class="insp-sec"><div class="insp-sec-h">Structural classes present</div>'
        '%s'
        '<div class="insp-foot">A structural-class flag from a textbook mechanism, '
        'computed from the structure alone. It is not this model\'s reasoning, not a '
        'validated alert rulebase, and not a genotoxicity conclusion; '
        '&ldquo;none of these ten&rdquo; is not &ldquo;no alert&rdquo;.</div>'
        '</div>'
        '</div>'
    ) % (headline, inchikey, tag, hero, cand_small, tan_txt, nn_name, nn_small, dom,
         score, weight, alerts_body)
