"""A clickable molecule grid for the candidate set of one GC-MS peak.

This is the one interactive component the rest of the notebook is built around:
the reader sees every structure the library match could not distinguish, clicks
one, and the inspector panel on the right redraws for that choice.

The widget is deliberately thin. It owns exactly one piece of reactive state --
``selected``, the 0-based position of the clicked card within the feature's
candidate list -- and nothing else. Everything a card shows (the drawing, the
alert flags, whether the structure sits inside the model's applicability
domain) is computed in Python by :func:`build_cards` and handed over as plain
data, so the JavaScript never touches chemistry and the Python side never
touches the DOM.

Decoupling note: ``selected`` starts at ``-1`` (no click yet). A notebook cell
can therefore treat the widget as an *override* -- read ``selected`` when it is
>= 0, and otherwise fall back to a plain ``mo.ui.dropdown``. If the front-end
fails to render in a hosted runtime, the dropdown still drives the inspector
and the notebook stays usable.
"""

from __future__ import annotations

import pathlib

import anywidget
import traitlets

# --- palette (kept in sync with the notebook's editorial theme) -------------
PETROL = "#1f6f78"   # selection / assigned identity / in-domain accent
AMBER = "#b7791f"    # uncertainty / structural alert / abstention
GRAPHITE = "#2b2b2b"
MUTED = "#8a8a8a"


def build_cards(sub, assigned_mask=None):
    """Turn one feature's candidate rows into the card dicts the widget renders.

    ``sub`` is the slice of the scored table for a single ``feature_id`` (each
    row a candidate structure), already carrying ``smiles``, ``p_genotox`` and
    ``in_domain``. ``assigned_mask`` is an optional boolean aligned to ``sub``
    marking the structure the library actually named; if omitted, no card is
    flagged as assigned.

    Out-of-domain candidates carry ``score_txt = ""`` -- the grid never prints a
    model score the model has no basis for. Structural alerts are reported as a
    list of class names, described elsewhere by their own rule, never as the
    model's reasoning.
    """
    # Imported lazily so importing the widget never drags in RDKit.
    from genotox_food_migrants import candidate_view

    sub = sub.reset_index(drop=True)
    if assigned_mask is not None:
        assigned = list(assigned_mask)
    else:
        assigned = [False] * len(sub)

    cards = []
    for i, row in sub.iterrows():
        smi = row.get("smiles")
        smi = smi if isinstance(smi, str) else ""
        try:
            svg = candidate_view.molecule_svg(smi) if smi else ""
        except Exception:
            svg = ""
        try:
            hits = candidate_view.alert_hits(smi) if smi else []
        except Exception:
            hits = []
        alert_labels = [h[0] for h in hits]
        in_dom = bool(row.get("in_domain", False))
        p = row.get("p_genotox")
        # Score is shown only inside the domain, and always as an uncalibrated
        # model score -- see the inspector caption for the full wording.
        tan = row.get("tanimoto_neighbour")
        tan_ok = tan is not None and tan == tan
        # Every card shows its Tanimoto to the nearest training structure - the
        # quantity that actually varies between candidates and decides whether
        # the model can judge them. Out of domain: Tc only, never a score.
        tc = ("Tc %.2f" % float(tan)) if tan_ok else "Tc n/a"
        if in_dom and p is not None and p == p:  # p == p screens out NaN
            score_txt = "%s \u00b7 score %.3f" % (tc, float(p))
        else:
            score_txt = "%s \u00b7 no score" % tc
        cards.append(
            {
                "idx": int(i),
                "svg": svg,
                "in_domain": in_dom,
                "has_alert": bool(alert_labels),
                "alert_labels": alert_labels,
                "is_assigned": bool(assigned[i]) if i < len(assigned) else False,
                "score_txt": score_txt,
                "tanimoto": float(tan) if tan is not None and tan == tan else None,
            }
        )
    return cards


_ESM = r"""
function badge(text, cls) {
  const b = document.createElement("span");
  b.className = "cg-badge " + cls;
  b.textContent = text;
  return b;
}

function render({ model, el }) {
  el.classList.add("cg-root");
  const grid = document.createElement("div");
  grid.className = "cg-grid";

  function paint() {
    grid.innerHTML = "";
    const cards = model.get("cards") || [];
    const sel = model.get("selected");
    cards.forEach((c) => {
      const card = document.createElement("div");
      card.className = "cg-card";
      if (!c.in_domain) card.classList.add("cg-ood");
      if (c.is_assigned) card.classList.add("cg-assigned");
      if (c.idx === sel) card.classList.add("cg-sel");
      card.dataset.idx = c.idx;

      // top row of flags
      const flags = document.createElement("div");
      flags.className = "cg-flags";
      if (c.is_assigned) flags.appendChild(badge("assigned", "cg-b-assigned"));
      if (c.has_alert) flags.appendChild(badge("alert", "cg-b-alert"));
      if (!c.in_domain) flags.appendChild(badge("outside domain", "cg-b-ood"));
      card.appendChild(flags);

      // structure drawing (RDKit SVG markup, injected as-is)
      const draw = document.createElement("div");
      draw.className = "cg-draw";
      draw.innerHTML = c.svg || "";
      card.appendChild(draw);

      // caption: index + score (only inside domain)
      const cap = document.createElement("div");
      cap.className = "cg-cap";
      const tag = document.createElement("span");
      tag.className = "cg-tag";
      tag.textContent = "#" + (c.idx + 1);
      cap.appendChild(tag);
      const sc = document.createElement("span");
      sc.className = "cg-score";
      sc.textContent = c.score_txt || (c.in_domain ? "" : "no score");
      cap.appendChild(sc);
      card.appendChild(cap);

      card.addEventListener("click", () => {
        model.set("selected", c.idx);
        model.save_changes();
      });
      grid.appendChild(card);
    });
  }

  paint();
  model.on("change:cards", paint);
  model.on("change:selected", paint);
  el.appendChild(grid);
}

export default { render };
"""

_CSS = """
.cg-root { width: 100%; }
.cg-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;
}
.cg-card {
  border: 1px solid #e4dfd6;
  border-radius: 8px;
  background: #fffdf9;
  padding: 6px 6px 4px 6px;
  cursor: pointer;
  transition: box-shadow .12s ease, transform .12s ease, border-color .12s ease;
  display: flex;
  flex-direction: column;
}
.cg-card:hover { box-shadow: 0 2px 8px rgba(31,111,120,.14); transform: translateY(-1px); }
.cg-card.cg-assigned { border-left: 3px solid #1f6f78; }
.cg-card.cg-sel { border-color: #1f6f78; box-shadow: 0 0 0 2px #1f6f78 inset; }
.cg-card.cg-ood { opacity: .62; }
.cg-card.cg-ood .cg-draw { filter: grayscale(.55); }
.cg-flags { display: flex; gap: 4px; flex-wrap: wrap; min-height: 16px; margin-bottom: 2px; }
.cg-badge { font: 600 9px/1.4 ui-sans-serif, system-ui, sans-serif;
  padding: 1px 5px; border-radius: 999px; letter-spacing: .02em; }
.cg-b-assigned { background: #1f6f78; color: #fff; }
.cg-b-alert { background: #fbeccb; color: #8a5a06; }
.cg-b-ood { background: #efece6; color: #8a8a8a; }
.cg-draw { display: flex; justify-content: center; align-items: center; min-height: 92px; }
.cg-draw svg { max-width: 100%; height: auto; }
.cg-cap { display: flex; justify-content: space-between; align-items: baseline;
  margin-top: 2px; }
.cg-tag { font: 600 11px ui-monospace, "SF Mono", Menlo, monospace; color: #2b2b2b; }
.cg-score { font: 11px ui-monospace, "SF Mono", Menlo, monospace; color: #6b6b6b; }
.cg-card.cg-ood .cg-score { color: #b7791f; font-style: italic; }
"""


class CandidateGrid(anywidget.AnyWidget):
    """Clickable grid of the candidate structures for one peak.

    Traits synced to the front-end:
      ``cards``    -- list of dicts from :func:`build_cards`.
      ``selected`` -- 0-based index of the clicked card; ``-1`` before any click.
    """

    _esm = _ESM
    _css = _CSS
    cards = traitlets.List().tag(sync=True)
    selected = traitlets.Int(-1).tag(sync=True)
