"""Shared, versioned competition demo component. No compatibility implementation."""

def build_demo_view():
        """Guided presentation for the real-data competition figure."""

        from functools import lru_cache
        from html import escape
        from io import StringIO
        import xml.etree.ElementTree as ET
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        PAPER = "#faf8f4"
        INK = "#253b43"
        PETROL = "#126b78"
        VIOLET = "#6550a8"
        AMBER = "#aa6916"
        MUTED = "#61737b"
        LINE = "#dce2df"

        CSS = """
        <style>
        :root { --background:#faf8f4; --color-background:#faf8f4; }
        body, .marimo, #root { background:#faf8f4; }
        .gx { color:#253b43; max-width:1120px; margin-inline:auto; font-family:ui-sans-serif,system-ui,-apple-system,sans-serif; line-height:1.5; }
        .gx * { box-sizing:border-box; }
        .gx h1,.gx h2,.gx p { margin:0; }
        .gx h1 { font:400 clamp(36px,4.4vw,57px)/1.06 Georgia,"Iowan Old Style",serif; letter-spacing:-1.5px; margin:13px 0 15px; max-width:880px; }
        .gx h1 em { color:#6550a8; font-style:normal; }
        .gx h2 { font:400 29px/1.15 Georgia,"Iowan Old Style",serif; letter-spacing:-.45px; }
        .gx-kicker,.gx-label,.gx-step-name { font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; text-transform:uppercase; letter-spacing:.07em; color:#61737b; }
        .gx-kicker { color:#126b78; border-top:3px solid #126b78; padding-top:14px; }
        .gx-mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
        .gx-small { font-size:12px; color:#61737b; line-height:1.5; }
        .gx-intro { background:#eeeafb; border-left:5px solid #6550a8; border-radius:0 9px 9px 0; padding:17px 21px; }
        .gx-intro p { font-size:15px; max-width:940px; }
        .gx-intro p+p { margin-top:8px; }
        .gx-intro strong { color:#513d90; }
        .gx-task { display:flex; flex-wrap:wrap; align-items:baseline; gap:9px; margin-top:13px; font-size:14px; }
        .gx-task b { color:#126b78; }
        .gx-data-scale { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; margin:18px 0 14px; }
        .gx-data-scale > div { background:#fffdf9; border:1px solid #dce2df; border-top:3px solid #126b78; border-radius:6px; padding:11px 13px; }
        .gx-data-scale > div:nth-child(3) { border-top-color:#426ca8; background:#edf3f9; }
        .gx-data-scale > div:last-child { border-top-color:#6550a8; background:#eeeafb; }
        .gx-data-scale strong { display:block; color:#126b78; font:400 29px/1.2 ui-monospace,monospace; margin:3px 0 4px; }
        .gx-data-scale > div:last-child strong { color:#6550a8; }
        .gx-case-intro { color:#40555c; background:#f0ede5; border-radius:6px; padding:11px 14px; font-size:13px; }
        .gx-case-intro b { color:#6550a8; }
        .gx-route { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; margin:18px 0 5px; }
        .gx-route-item { padding:11px 13px; border:1px solid #dce2df; border-radius:6px; font-size:12px; }
        .gx-route-item b { display:block; color:#126b78; font-size:13px; margin-bottom:3px; }
        .gx-route-item:nth-child(2) { background:#fff3db; border-color:#e8cfa3; }
        .gx-route-item:nth-child(2) b { color:#915811; }
        .gx-route-item:nth-child(3) { background:#eeeafb; border-color:#d9cfee; }
        .gx-route-item:nth-child(3) b { color:#6550a8; }
        .gx-section { border-top:1px solid #dce2df; padding:21px 0 9px; margin-top:17px; }
        .gx-step { display:flex; align-items:flex-start; gap:12px; }
        .gx-step-number { display:flex; align-items:center; justify-content:center; flex:0 0 32px; height:32px; border-radius:50%; background:#126b78; color:#fff; font:600 14px/1 ui-monospace,monospace; }
        .gx-step.amber .gx-step-number { background:#aa6916; }
        .gx-step.violet .gx-step-number { background:#6550a8; }
        .gx-step p { font-size:13px; color:#61737b; margin-top:7px; max-width:900px; }
        .gx-hero { display:grid; grid-template-columns:1.15fr 1.15fr .85fr; gap:12px; margin-top:9px; }
        .gx-hero > section { min-width:0; padding:16px; border:1px solid #c5ddd9; border-top:4px solid #126b78; border-radius:8px; background:#e9f4f1; }
        .gx-hero > section:nth-child(2) { background:#edf3f9; border-color:#ccdae9; border-top-color:#426ca8; }
        .gx-hero > section:last-child { background:#e9f4f1; }
        .gx-hero > section.warn { background:#fff3db; border-color:#e8cfa3; border-top-color:#aa6916; }
        .gx-hero .gx-label { color:#365762; }
        .gx-structure { height:183px; display:flex; justify-content:center; align-items:center; }
        .gx-structure svg { width:100%; max-height:183px; }
        .gx-name { font-size:14px; font-weight:600; line-height:1.4; min-height:40px; }
        .gx-evidence { display:inline-block; padding:3px 7px; font:10px/1.4 ui-monospace,monospace; background:#ffffff8c; border:1px solid #b7d7d0; color:#126b78; margin:8px 5px 7px 0; border-radius:3px; }
        .gx-evidence.warn { color:#915811; border-color:#dfbf86; background:#fff7e8; }
        .gx-tc { display:flex; align-items:baseline; gap:10px; margin:8px 0 3px; }
        .gx-tc b { color:#126b78; font:400 35px/1.1 ui-monospace,monospace; letter-spacing:-1.4px; }
        .gx-status { display:inline-block; padding:6px 9px; border-radius:4px; font:600 19px/1.2 ui-monospace,monospace; margin:13px 0 10px; color:#fff; background:#126b78; }
        .gx-status.warn { background:#aa6916; }
        .gx-score { font:400 25px/1.2 ui-monospace,monospace; margin:7px 0!important; }
        .gx-output { margin-top:22px; padding-top:13px; border-top:1px solid #00000017; }
        .gx-note { border-left:4px solid #126b78; background:#e9f4f1; padding:12px 15px; margin:12px 0 2px; border-radius:0 5px 5px 0; font-size:14px; }
        .gx-note.warn { background:#fff3db; border-color:#aa6916; }
        .gx-note.violet { background:#eeeafb; border-color:#6550a8; }
        .gx-note strong { display:block; margin-bottom:3px; }
        .gx-inline { display:flex; align-items:center; justify-content:space-between; gap:20px; }
        .gx-metric { font:400 27px/1.1 ui-monospace,monospace; color:#126b78; }
        .gx-coverage { display:flex; gap:3px; margin:10px 0 8px; }
        .gx-coverage span { flex:1; height:8px; border-radius:2px; background:#126b78; }
        .gx-coverage span.out { background:#dcb477; }
        .gx-guide { font-size:13px; background:#f1f0e9; padding:11px 14px; margin:9px 0 2px; border-radius:6px; color:#40555c; }
        .gx-guide strong { color:#915811; }
        .gx-guide.violet { background:#eeeafb; }
        .gx-guide.violet strong { color:#6550a8; }
        .gx-weight { display:grid; grid-template-columns:1fr 1fr; gap:24px; padding:18px 0; }
        .gx-weight-value { font:400 52px/1.07 Georgia,serif; color:#126b78; letter-spacing:-1.8px; margin:10px 0; }
        .gx-weight-value.violet { color:#6550a8; }
        .gx-weight-value.warn { color:#aa6916; }
        .gx-weight-value .gx-before { color:#61737b; font-size:32px; }
        .gx-weight-value .gx-arrow { color:#6550a8; font:24px/1 sans-serif; margin:0 8px; }
        .gx-bar { margin:7px 0 15px; }
        .gx-bar-label { display:flex; justify-content:space-between; gap:8px; font-size:12px; margin-bottom:6px; }
        .gx-track { height:12px; border:1px solid #ccd7d3; background:#edf0e9; border-radius:3px; overflow:hidden; }
        .gx-fill { height:100%; background:#126b78; transition:width .18s; }
        .gx-bar.alt .gx-fill { background:#6550a8; }
        .gx-bar.unrevealed .gx-track { background:repeating-linear-gradient(135deg,#f0ecf8,#f0ecf8 5px,#e5dff2 5px,#e5dff2 10px); }
        .gx-bar.inactive { opacity:.65; }
        .gx-bar.active .gx-bar-label { font-weight:650; }
        .gx-weight-box { background:#f2f0fa; border:1px solid #ddd5ef; border-radius:8px; padding:14px 17px; }
        .gx-residual { font-size:12px; color:#915811; margin-top:10px!important; }
        .gx-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:9px; margin-top:15px; }
        .gx-card { background:#e9f4f1; border:1px solid #bcd9d1; padding:9px 10px; border-radius:6px; min-width:0; }
        .gx-card.out { background:#fff7e8; border-color:#dfc498; border-style:dashed; }
        .gx-card.assigned { border:2px solid #6550a8; padding:8px 9px; }
        .gx-card-head { display:flex; justify-content:space-between; gap:4px; font:10px/1.3 ui-monospace,monospace; color:#61737b; }
        .gx-card-mol { height:109px; display:flex; align-items:center; justify-content:center; }
        .gx-card-mol svg { width:100%; max-height:109px; }
        .gx-card-score { display:flex; justify-content:space-between; gap:5px; font:11px/1.4 ui-monospace,monospace; }
        .gx-card-state { color:#126b78; font-weight:700; }
        .gx-card.out .gx-card-state { color:#915811; }
        .gx-card a { color:#546770; text-decoration:underline; font-size:10px; }
        .gx-plot svg { width:100%; height:auto; display:block; }
        .gx-plot { padding:10px 0 0; }
        .gx-feature { margin:14px auto; padding:14px 17px; background:#eeeafb; border-left:4px solid #6550a8; border-radius:0 7px 7px 0; }
        .gx-feature-top { display:flex; justify-content:space-between; align-items:baseline; gap:14px; flex-wrap:wrap; }
        .gx-feature h2 { font-size:23px; }
        .gx-feature-meta { display:flex; gap:24px; flex-wrap:wrap; margin-top:10px; font-size:12px; }
        .gx-feature-meta b { display:block; color:#253b43; font:16px/1.4 ui-monospace,monospace; }
        .gx-workspace { display:grid; grid-template-columns:minmax(0,1.06fr) minmax(0,1fr); gap:20px; align-items:start; margin:16px auto; }
        .gx-choices { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-top:12px; }
        .gx-choices.single { grid-template-columns:minmax(150px,240px); }
        .gx-choice { min-width:0; }
        .gx-choice button { width:100%; height:auto!important; padding:0!important; display:block!important; white-space:normal; border:0!important; border-radius:6px; background:transparent!important; }
        .gx-choice button:hover { box-shadow:0 0 0 2px #6550a8; }
        .gx-choice button:focus-visible { outline:3px solid #6550a8; outline-offset:2px; }
        .gx-choice .prose { max-width:none; margin:0; }
        .gx-choice .prose p { margin:0; }
        .gx-choice-card { background:#e9f4f1; border:1px solid #bddbd4; padding:9px 8px; border-radius:6px; text-align:left; color:#253b43; }
        .gx-choice-card.out { background:#fff7e8; border-color:#dfc498; border-style:dashed; }
        .gx-choice-card.selected { border:2px solid #6550a8; padding:8px 7px; box-shadow:0 0 0 1px #6550a8 inset; }
        .gx-choice-card svg { display:block; width:100%; height:98px; }
        .gx-choice-id { font:10px/1.4 ui-monospace,monospace; overflow-wrap:anywhere; }
        .gx-choice-values { display:flex; justify-content:space-between; gap:5px; margin-top:4px; font:11px/1.4 ui-monospace,monospace; }
        .gx-choice-action { margin-top:5px; font-size:10px; color:#6550a8; }
        .gx-inspector { position:sticky; top:18px; padding:17px; background:#fffdf9; border:1px solid #dce2df; border-top:4px solid #6550a8; border-radius:8px; }
        .gx-inspector h2 { font-size:23px; margin:5px 0; }
        .gx-inspector-pair { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:13px 0; }
        .gx-inspector-pair > div { min-width:0; }
        .gx-inspector-pair .gx-name { font-size:12px; overflow-wrap:anywhere; }
        .gx-inspector-pair .gx-structure { height:145px; }
        .gx-inspector-pair svg { max-height:145px; }
        .gx-results { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
        .gx-results > div { min-width:0; padding:12px; border:1px solid #dce2df; border-radius:6px; background:#edf3f9; }
        .gx-results > div:first-child { background:#e9f4f1; }
        .gx-result-value { font:400 25px/1.2 ui-monospace,monospace; color:#126b78; margin:5px 0!important; overflow-wrap:anywhere; }
        .gx-inspector .gx-status { font-size:13px; margin:8px 0 6px; }
        .gx-input-note { font-size:12px; color:#61737b; margin-top:8px!important; }
        .gx-plot-window { position:relative; width:100%; isolation:isolate; z-index:1; }
        .gx-plot-window > svg { width:100%; height:auto; display:block; pointer-events:none; }
        .gx-plot-hit { position:absolute; transform:translate(-50%,-50%); width:26px; height:26px; z-index:50; overflow:hidden; opacity:0; pointer-events:auto; }
        .gx-plot-hit:focus-within { opacity:.5; outline:2px solid #6550a8; border-radius:50%; }
        .gx-plot-hit button { width:26px!important; min-width:0!important; height:26px!important; padding:0!important; border:0!important; border-radius:50%; opacity:0; pointer-events:auto; }
        .gx-plot-hit button:focus-visible { opacity:.4; background:#6550a8; outline:2px solid #6550a8; }
        .gx-point { cursor:help; }
        .gx-point:hover use,.gx-point:focus use { stroke-width:2.2; }
        .gx-control-title { font-size:14px; font-weight:650; color:#126b78; margin-top:6px!important; }
        .gx-rule-readout { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:8px 0; font:14px/1.5 ui-monospace,monospace; }
        .gx-rule-readout b { color:#126b78; }
        .gx-rule-readout.warn b { color:#aa6916; }
        .gx-finding { border-left:4px solid #aa6916; background:#fff3db; padding:13px 16px; font-size:15px; line-height:1.5; margin:10px 0 12px!important; }
        .gx-close { padding:25px 24px; border-radius:9px; margin-top:24px; background:#193e48; color:#fff; }
        .gx-close h2 { max-width:900px; }
        .gx-close p { max-width:880px; margin-top:12px; font-size:14px; color:#e2efed; }
        .gx-close .gx-label { color:#a4d6cb; margin-bottom:10px; }
        .gx-routes { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:18px auto; }
        .gx-route-col { padding:16px 18px; border-radius:8px; min-width:0; }
        .gx-route-col.trad { background:#f1f0e9; border:1px solid #ddd9c9; border-top:4px solid #aa6916; }
        .gx-route-col.ours { background:#eeeafb; border:1px solid #d9cfee; border-top:4px solid #6550a8; }
        .gx-route-col h3 { font:400 20px/1.2 Georgia,serif; margin:6px 0 3px; }
        .gx-route-col .gx-label { color:#915811; }
        .gx-route-col.ours .gx-label { color:#6550a8; }
        .gx-route-step { padding:9px 0; border-top:1px solid #0000001a; font-size:13px; }
        .gx-route-step:first-of-type { border-top:0; }
        .gx-route-step b { display:block; font:10px/1.5 ui-monospace,monospace; text-transform:uppercase; letter-spacing:.06em; color:#61737b; margin-bottom:2px; }
        .gx-verdict { margin-top:11px; padding:11px 13px; border-radius:6px; background:#fff; font-size:14px; }
        .gx-verdict strong { display:block; margin-bottom:3px; }
        .gx-route-col.trad .gx-verdict strong { color:#915811; }
        .gx-route-col.ours .gx-verdict strong { color:#6550a8; }
        @media(max-width:800px) { .gx-routes { grid-template-columns:1fr; } }
        .gx-conclusion { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:20px; }
        .gx-conclusion div { border-top:2px solid #77c3b7; padding-top:10px; font-size:12px; color:#e2efed; }
        .gx-conclusion div:nth-child(2) { border-color:#b5a2e0; }
        .gx-conclusion div:nth-child(3) { border-color:#e4b875; }
        .gx-conclusion b { display:block; color:#fff; margin-bottom:4px; font-size:13px; }
        .gx-details p { margin:9px 0; font-size:13px; }
        .gx-details a { color:#126b78; text-decoration:underline; }
        .gx-details h3 { font-size:14px; margin-top:18px; }
        @media(max-width:800px) {
         .gx-workspace { grid-template-columns:1fr; }
         .gx-inspector { position:static; grid-row:1; }
         .gx-hero { grid-template-columns:1fr 1fr; }
         .gx-hero > section:last-child { grid-column:1/-1; }
         .gx-grid { grid-template-columns:repeat(4,minmax(0,1fr)); }
         .gx-data-scale { grid-template-columns:repeat(2,minmax(0,1fr)); }
        }
        @media(max-width:540px) {
         .gx-choices { grid-template-columns:repeat(2,minmax(0,1fr)); }
         .gx h1 { font-size:38px; }
         .gx-route,.gx-hero,.gx-weight,.gx-conclusion { grid-template-columns:1fr; }
         .gx-hero > section:last-child { grid-column:auto; }
         .gx-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
         .gx-inline { flex-wrap:wrap; gap:9px; }
         .gx-weight-value { font-size:46px; }
         .gx-intro,.gx-close { padding:16px; }
        }

        .gx-selection-banner {display:grid;grid-template-columns:105px minmax(0,1fr) auto;gap:16px;align-items:center;padding:12px 17px;border:1px solid #d9cfee;border-left:4px solid #6550a8;border-radius:7px;background:#fffdf9;margin:15px auto;}
        .gx-selection-banner strong {display:block;font:400 23px/1.2 Georgia,serif;margin:5px 0;overflow-wrap:anywhere;}
        .gx-selection-thumb {width:105px;height:78px;flex:0 0 105px;}
        .gx-selection-thumb svg {display:block;width:100%;height:100%;}
        .gx-selection-numbers {display:grid;gap:5px;font:12px/1.5 ui-monospace,monospace;color:#6550a8;}
        .gx-reference-molecule {height:130px;background:#fffdf98c;border-radius:5px;margin:12px 0;}
        .gx-reference-molecule svg {display:block;width:100%;height:100%;}
        .gx-comparison-values {display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0;font-size:11px;color:#61737b;}
        .gx-comparison-values b {display:block;font:20px/1.3 ui-monospace,monospace;color:#253b43;overflow-wrap:anywhere;}
        .gx-identity-transition {display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin:14px 0 10px;font-size:12px;}
        .gx-identity-transition b {display:block;font:29px/1.2 ui-monospace,monospace;color:#6550a8;}
        .gx-identity-map {display:flex;flex-wrap:wrap;gap:5px;margin:12px 0 8px;}
        .gx-identity-tile {padding:4px 7px;min-width:37px;text-align:center;border:1px solid #bddbd4;background:#e9f4f1;color:#126b78;border-radius:3px;font:11px/1.4 ui-monospace,monospace;}
        .gx-identity-tile.out {background:#fff3db;color:#915811;border-color:#e8cfa3;border-style:dashed;}
        .gx-identity-tile.no-output {background:#eeefef;color:#61737b;border-color:#bfc8cb;border-style:dotted;}
        .gx-identity-tile.zero-weight {opacity:.45;}
        .gx-identity-tile.selected {outline:2px solid #6550a8;outline-offset:1px;opacity:1;}
        .gx-set-result {font-size:13px;margin:12px 0 9px!important;}
        .gx-grain-note {padding-top:8px;border-top:1px solid #d9cfee;}
        .gx-motif-heading {display:flex;gap:15px;align-items:center;margin:7px 0;}
        .gx-sensitivity-panels {display:grid;grid-template-columns:1fr 1fr;gap:18px;}
        .gx-sensitivity-panels > section {min-width:0;border:1px solid #dce2df;border-radius:8px;padding:14px;background:#fffdf9;}
        .gx-sensitivity-panels > section:first-child {border-top:3px solid #6550a8;}
        .gx-sensitivity-panels > section:last-child {border-top:3px solid #126b78;}
        .gx-sensitivity-panels h3 {font:400 20px/1.25 Georgia,serif;margin:5px 0 7px;overflow-wrap:anywhere;}
        .gx-sensitivity-panels svg {display:block;width:100%;height:auto;margin-top:9px;}
        @media(max-width:800px){.gx-sensitivity-panels {grid-template-columns:1fr;}.gx-selection-banner {grid-template-columns:85px minmax(0,1fr);gap:10px;}.gx-selection-thumb {width:85px;flex-basis:85px;}.gx-selection-numbers {grid-column:2;font-size:11px;}.gx-selection-banner strong {font-size:20px;}}
        </style>
        """


        def escaped(value):
            return escape(str(value), quote=True)


        def fmt_score(value):
            if value is None or not np.isfinite(float(value)):
                return "Unavailable"
            if 0 < float(value) < .0001:
                return f"{100*float(value):.2g}%"
            return f"{100*float(value):.2f}%"


        def scenario_summary(view):
            """Immediate, reactive comparison: molecule result vs active scenario."""
            assigned = view['assigned']
            equal = view['assumption'] == 'same_formula' and not view['confirmed']
            title = 'Equal-weight scenario' if equal else 'Assigned-only scenario'
            weight_text = (f"Equal weights: 1/{view['n_retained']} per retained structure. Hypothetical assumptions."
                          if equal else 'All weight stays on the published assigned identity.'
                          + (' Standard confirmation is preserved.' if view['confirmed'] else ' The assignment is tentative.'))
            scope = ('Fitted training identity; this is not a held-out prediction.' if assigned['in_training']
                     else 'Assigned identity absent from the fitted training set.')
            pool = view['listed_formula_pool_size']
            pool_text = (f"{view['n_retained']:,} / {pool:,} listed formula structures retained ({100*view['retained_pool_fraction']:.2f}%)."
                         if pool else 'Listed formula-pool size unavailable.')
            no_average = ('Zero included weight: the denominator is zero, so no average exists.'
                          if view['covered_weight'] == 0 else
                          f"Denominator: {view['n_weighted_evaluable']} weighted structures; {100*view['covered_weight']:.0f}% of active scenario weight included.")
            return f'''<div class="gx gx-live-summary" id="scenario-comparison" data-assumption="{view['assumption']}"
                data-average="{view['conditional_score']}" data-n-evaluable="{view['n_evaluable']}" aria-live="polite"
                style="margin:12px auto 24px">
              <div class="gx-label">Identity uncertainty experiment · {escaped(view['feature']['feature_id'])}</div>
              <div class="gx-results" style="margin-top:10px;align-items:stretch">
                <div><div class="gx-label">Published assigned structure · {escaped(view['feature']['display_name'])}</div>
                  <p class="gx-result-value" data-result="assigned">{fmt_score(assigned['score'])}</p>
                  <p class="gx-small">Estimated probability of an aggregated positive EFSA label, conditional on this structure.</p>
                  <p class="gx-small"><strong>{scope}</strong> {'Meets' if view['assigned_evaluable'] else 'Does not meet'} your Tc ≥ {view['cutoff']:.2f} rule.</p></div>
                <div style="background:#eeeafb;border-color:#d9cfee"><div class="gx-label">{title} · active result</div>
                  <p class="gx-result-value" data-result="scenario">{fmt_score(view['conditional_score'])}</p>
                  <p class="gx-small"><strong>Scenario average over retained structures</strong> with scores that meet your similarity rule.</p>
                  <p class="gx-small">{weight_text}<br>{no_average}</p></div>
              </div>
              <div class="gx-note" style="margin-top:12px;padding:12px 16px">
                <p class="gx-small"><strong>{view['n_evaluable']} / {view['n_retained']} retained structures meet Tc ≥ {view['cutoff']:.2f}:
                  {100*view['scenario_structure_coverage']:.0f}% coverage of this {view['n_retained']}-structure scenario.</strong><br>{pool_text}</p>
                <p class="gx-small">The published identity is forcibly retained alongside sampled formula alternatives.
                  This listed PubChem pool is not all chemically possible identities. The retained fraction does not measure the chance of omitting the true identity.</p>
                <p class="gx-small">The alternatives are structural stress tests without individual EI spectra, retention indices or search scores.
                  Equal weights are hypothetical. The scenario average is not the probability that the GC–MS peak is genotoxic.</p>
              </div>
            </div>'''


        @lru_cache(maxsize=4)
        def validation_plot(name):
            import base64
            from genotox_food_migrants import paths
            path = paths.VALIDATION / name
            if not path.exists():
                return ''
            return '<img style="width:100%;height:auto;margin:16px 0" alt="Generated internal validation figure" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '">'


        def validation_panel(provenance):
            report = provenance.get('validation', {})
            if not report.get('summary'):
                return '<p>Generated validation outputs are unavailable; no performance numbers are substituted.</p>'
            rows = []
            for entry in report['summary']:
                cells = [f"{entry[c]['median']:.4f} [{entry[c]['q1']:.4f}–{entry[c]['q3']:.4f}]"
                         for c in ('pr_auc', 'brier', 'log_loss', 'ece')]
                policy = 'Ring + individual acyclic' if entry['protocol'] == 'scaffold_identity' else 'Ring + acyclic series ≥0.70'
                rows.append(f"<tr><td>{policy}</td><td>{escaped(entry['model'])}</td>" + ''.join(f'<td>{x}</td>' for x in cells) + '</tr>')
            return '''<div class="gx gx-details"><h3>Reproduced internal validation · five outer seeds</h3>
              <p>Raw and sigmoid ensemble predictions reuse exactly the same three fitted classifiers.
                The single-classifier result is a separate reference. A calibrated logistic baseline is included.</p>
              <div style="overflow-x:auto"><table style="font-size:12px;width:100%;border-collapse:collapse">
                <thead><tr><th>Split protocol</th><th>Model</th><th>AP (PR area)</th><th>Brier</th><th>Log loss</th><th>ECE</th></tr></thead>
                <tbody>''' + ''.join(rows) + '''</tbody></table></div>
              <p class="gx-small">Values are median [Q1–Q3]. The repeated splits share identities; this spread is not a confidence interval.
                Lower Brier, log loss and ECE are better. AP is average precision. ECE uses ten equal-width bins.</p>
              <p>Train/test and classifier/calibrator identity lists, held-out predictions, overlap audits and figures are saved in the project.
                The acyclic-series protocol groups connected ECFP4 neighbours at Tc ≥ 0.70 using structures only.
                Similarity-cutoff selection uses nested predictions inside each outer training set; the outer test is never used to choose it.
                The visitor's 0.40 control remains exploratory.</p>
              <p><strong>Independent compatible endpoint validation: not performed.</strong> The confirmed FCM identities validate identity, not genotoxicity.
                The 10,827 bundled label rows match the verified Genotoxicity_KJ_2023.xlsx reference export, DOI 10.5281/zenodo.8120114. The historical download date and original CAS/CID enrichment logs were not recorded.
                This project is a transparent compound-identity uncertainty screening demonstration.</p>''' + validation_plot('validation_metrics.png') + validation_plot('calibration_scaffold_identity.png') + validation_plot('calibration_scaffold_acyclic_series_070.png') + '</div>'


        @lru_cache(maxsize=1800)
        def structure_svg(smiles, width=320, height=190, highlights=()):
            molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            if molecule is None:
                return '<span class="gx-small">Structure unavailable</span>'
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            options = drawer.drawOptions()
            options.clearBackground = False
            options.useDefaultAtomPalette()
            options.bondLineWidth = 1.6
            options.padding = .09
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, highlightAtoms=list(highlights), highlightAtomColors={atom:(0.94,0.76,0.44) for atom in highlights})
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            return svg[svg.find("<svg"):]


        def header(features, provenance):
            return f'''<div class="gx">
              <div class="gx-kicker">Food-contact migrants · Genotoxicity screening · EFSA OpenFoodTox</div>
              <h1>What if the chemical <em>identity</em> is wrong?</h1>
              <div class="gx-intro"><p><strong>One molecular formula can describe very different chemistry.</strong> Start with the reported oleic-acid assignment. Compare an estimated EFSA-positive probability with the same-formula identity scenario and its coverage.</p>
                <p><strong>1. Inspect the assignment. 2. Compare molecular structures. 3. Check covered identity weight.</strong> The fitted sigmoid-calibrated model stays fixed while marimo updates the probability estimates and coverage.</p></div>
              <div class="gx-data-scale">
                <div><div class="gx-label">EFSA records</div><strong>{provenance['n_efsa_records']:,}</strong><span class="gx-small">Substance–output records</span></div>
                <div><div class="gx-label">Resolved identities</div><strong>{provenance['n_efsa_resolved_identities']:,}</strong><span class="gx-small">Distinct raw InChIKeys</span></div>
                <div><div class="gx-label">Model reference</div><strong>{provenance['n_training_substances']:,}</strong><span class="gx-small">Standardized EFSA training structures</span></div>
                <div><div class="gx-label">This identity experiment</div><strong>{len(features)}</strong><span class="gx-small">Reported GC–MS assignments</span></div>
              </div></div>'''


        def feature_context(view):
            f = view["feature"]
            evidence = "Confirmed with a standard · recorded MSI 1" if view["confirmed"] else "Library assignment · recorded MSI 2"
            return f'''<div class="gx gx-feature">
              <div class="gx-feature-top"><div><div class="gx-label">GC–MS entry · reported assignment</div><h2>{escaped(f['feature_id'])}</h2></div>
                <div><b>{escaped(f['display_name'])}</b><div class="gx-small">Assigned identity · {evidence}</div></div></div>
              <div class="gx-feature-meta">
                <span>Reported retention time<b>{f['rt_mean']:.2f} min</b></span>
                <span>Reported fragment ion<b>m/z {f['mass_to_charge']:.0f}</b></span>
                <span>Formula of assigned structure<b>{escaped(f['chemical_formula'])}</b></span>
                <span>Assigned identifier<b>{escaped(f['database_identifier'])}</b></span>
              </div>
              <p class="gx-input-note">One entry in the 33-assignment table; its FCM2018 identifier is a project label. Retention time and m/z describe the reported observation. The selected molecular structure is what the model evaluates.</p>
            </div>'''


        def structure_name(row, feature):
            if bool(row["is_assigned"]):
                return str(feature["display_name"])
            name = row.get("name")
            if isinstance(name, str) and name.strip():
                return name
            cid = row.get("cid")
            return f"PubChem CID {int(cid)}" if cid is not None and cid == cid else f"Retained structure #{int(row['rank'])}"


        def structure_uri(row, feature):
            cid = row.get("cid")
            return f"https://pubchem.ncbi.nlm.nih.gov/compound/{int(cid)}" if cid is not None and cid == cid else str(feature["assigned_uri"])


        def choice_label(row, view, selected_key, control_html=""):
            key = f"{view['feature']['feature_id']}:{int(row['rank'])}"
            inside = bool(row["in_domain"])
            cid = row.get("cid")
            tag = "Assigned identity" if bool(row["is_assigned"]) else f"CID {int(cid)}"
            classes = "gx-choice-card" + (" out" if not inside else "") + (" selected" if key == selected_key else "")
            has_output = np.isfinite(float(row['score'])) and np.isfinite(float(row['tanimoto_neighbour']))
            tc_text = f"{row['tanimoto_neighbour']:.3f}" if has_output else "—"
            state = ('IN' if inside else 'OUT') if has_output else 'NO OUTPUT'
            smiles = row['smiles_std'] if has_output else row['smiles']
            return f'''<div class="{classes}"><div class="gx-choice-id">#{int(row['rank']):02d} · {escaped(tag)}</div>
              {structure_svg(smiles,200,135)}
              <div class="gx-choice-values"><span>Tc {tc_text}</span><b>{state}</b></div>
              <div class="gx-choice-id">P(EFSA+) {fmt_score(row['score'])}</div>
              <div class="gx-choice-action">{control_html or ('Selected · results shown' if key == selected_key else 'Inspect this structure')}</div>
            </div>'''


        def comparison(view, molecule_a, molecule_b):
            cards = []
            for label, molecule in [('A', molecule_a), ('B', molecule_b)]:
                row = molecule['selected']
                key = molecule['selection_key']
                name = escaped(structure_name(row, view['feature']))
                training = 'Fitted training identity; not a held-out prediction.' if row['in_training'] else 'Query identity absent from fitted training set.'
                cards.append(f'<section data-comparison-key="{key}"><h3>Molecule {label} · {name}</h3>' +
                    choice_label(row, view, '', 'Comparison selection') +
                    f'<p class="gx-small">{training}</p></section>')
            same = '<p class="gx-note">A and B currently select the same structure.</p>' if molecule_a['selection_key'] == molecule_b['selection_key'] else ''
            return ('<div class="gx gx-section" id="molecule-comparison">' +
                '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px">' + ''.join(cards) + '</div>' + same +
                '<p class="gx-small">P(EFSA+) estimates an aggregated positive EFSA label conditional on each structure. Tc is similarity to its own nearest EFSA training neighbour, not similarity between A and B. IN/OUT follows the active cutoff. Missing outputs remain unavailable. Neither score establishes genotoxicity.</p></div>')

        def selection_banner(view, inspected):
            row = inspected['selected']
            f = view['feature']
            has_output = np.isfinite(float(row['score']))
            smiles = row['smiles_std'] if has_output else row['smiles']
            tc = f"{row['tanimoto_neighbour']:.3f}" if has_output else 'Unavailable'
            state = 'IN' if inspected['selected_evaluable'] else ('OUT' if has_output else 'NO OUTPUT')
            return f'''<div class="gx gx-selection-banner" data-selection-key="{inspected['selection_key']}" aria-live="polite">
              <div class="gx-selection-thumb">{structure_svg(smiles,150,95)}</div>
              <div><div class="gx-label">Currently inspected · {escaped(f['feature_id'])} · structure #{int(row['rank']):02d}</div>
                <strong>{escaped(structure_name(row,f))}</strong><p class="gx-small">{'Assigned identity' if bool(row['is_assigned']) else 'Same-formula alternative'} · all individual-result panels below follow this molecule.</p></div>
              <div class="gx-selection-numbers"><b>Tc {tc} · {state}</b><span>P(EFSA+) {fmt_score(row['score'])}</span><p class="gx-small" style="max-width:250px">Estimated probability of an aggregated positive EFSA label, conditional on this structure.</p></div></div>'''


        def choices_intro(view):
            f = view['feature']
            if view['assumption'] == 'published':
                title = 'Assigned molecular identity'
                text = 'Open Explore same-formula structures to inspect the alternatives. Each displayed value estimates an aggregated positive EFSA label conditional on that structure.'
            else:
                title = f"{view['n_retained']} molecular structures · {escaped(f['chemical_formula'])}"
                text = 'Click a molecule. Its drawing, nearest EFSA neighbour, score and structural motifs update together. These sampled structures are stress tests without individual spectral support. Each value estimates an aggregated positive EFSA label conditional on that structure.'
            return f'''<div><div class="gx-label">Molecules for {escaped(f['feature_id'])}</div>
              <h2 style="font-size:24px;margin:5px 0">{title}</h2><p class="gx-small">{text}</p></div>'''


        def inspector(view, inspected):
            row = inspected["selected"]
            f = view["feature"]
            name = structure_name(row, f)
            accepted = inspected["selected_evaluable"]
            tc = float(row["tanimoto_neighbour"])
            has_output = np.isfinite(float(row['score'])) and np.isfinite(tc)
            source = "Assigned identity" if bool(row["is_assigned"]) else "Other retained same-formula structure"
            state = "MEETS YOUR SIMILARITY RULE" if accepted else "BELOW YOUR SIMILARITY RULE"
            op = "≥" if accepted else "<"
            membership = ('<div class="gx-status warn">TRAINING IDENTITY · fitted estimate, not a held-out assessment</div>'
                          if bool(row["in_training"]) else
                          '<p class="gx-small">Query identity absent from the fitted EFSA training set. Its unknown experimental outcome is not an external validation result.</p>')
            if not has_output:
                state = "MODEL INPUT COULD NOT BE READ"
                interpretation = "No supported standardized model input is available. The retained raw structure is shown; this pipeline produced no score or similarity for it. " + str(row.get("structure_note", ""))
            elif accepted:
                interpretation = "The selected structure meets your similarity criterion. Its score contributes only when this identity scenario gives the structure weight."
            else:
                interpretation = "The score is calculated for this structure. Screening interpretation is withheld under your current similarity criterion."
            tc_text = f"{tc:.3f}" if has_output else "Unavailable"
            comparison = f"Tc {tc:.6f} {op} {view['cutoff']:.2f}" if has_output else "Fingerprint unavailable for standardized model input"
            model_label = "Model input" if has_output else "Retained raw structure"
            model_smiles = row['smiles_std'] if has_output else row['smiles']
            neighbour = (structure_svg(row['nn_smiles'],300,200) if has_output
                         else '<span class="gx-small">No fingerprint to compare</span>')
            return f'''<aside id="current-molecular-result" class="gx-inspector" data-selection-key="{inspected['selection_key']}" aria-live="polite">
              <div class="gx-label">Selected molecular result · {escaped(f['feature_id'])} · #{int(row['rank']):02d}</div>
              <h2>{escaped(name)}</h2><p class="gx-small">{source}</p>
              <div class="gx-inspector-pair">
                <div><div class="gx-label">{model_label}</div><div class="gx-structure">{structure_svg(model_smiles,300,200,alert_details(row)["atoms"])}</div>
                  <a class="gx-small" href="{escaped(structure_uri(row,f))}" target="_blank" rel="noopener noreferrer">Open molecular record in PubChem</a></div>
                <div><div class="gx-label">Closest EFSA training structure</div><div class="gx-structure">{neighbour}</div>
                  <div class="gx-name">{escaped(row['nn_name'])}</div></div>
              </div>
              <div class="gx-results">
                <div><div class="gx-label">Estimated EFSA-positive probability</div><p class="gx-result-value">{fmt_score(row['score'])}</p>
                  <p class="gx-small">Estimated probability of an aggregated positive EFSA label, conditional on this structure.</p></div>
                <div><div class="gx-label">Nearest-structure similarity</div><p class="gx-result-value">{tc_text}</p>
                  <p class="gx-small">Tanimoto · ECFP4<br>Your selected minimum: {view['cutoff']:.2f}</p></div>
              </div>
              <div class="gx-status{'' if accepted else ' warn'}">{state}</div>
              <p class="gx-small gx-mono">{comparison}</p>
              <p style="font-size:13px;margin-top:8px">{interpretation}</p>{membership}
              <p class="gx-input-note">Selecting a molecule changes the structure, its nearest neighbour and its score. Moving the Tc criterion changes applicability while those calculations stay fixed.</p>
            </aside>'''


        def two_routes(view, inspected):
            f, assigned, chosen = view['feature'], view['assigned'], inspected['selected']
            def molecule(row):
                return structure_svg(row['smiles_std'] if np.isfinite(float(row['score'])) else row['smiles'], 340, 150)
            def similarity(row):
                return f"{row['tanimoto_neighbour']:.3f}" if np.isfinite(float(row['tanimoto_neighbour'])) else 'Unavailable'
            baseline_state = 'IN' if view['assigned_evaluable'] else 'OUT'
            selection_state = 'IN' if inspected['selected_evaluable'] else 'OUT'
            if not np.isfinite(float(chosen['score'])):
                selection_state = 'NO MODEL OUTPUT'
            relation = 'The assigned identity is currently selected.' if bool(chosen['is_assigned']) else 'Same formula; a different molecular structure.'
            return f'''<div class="gx gx-routes" data-selection-key="{inspected['selection_key']}" data-feature-id="{escaped(f['feature_id'])}">
              <section class="gx-route-col trad" data-reference-key="{escaped(f['feature_id'])}:{int(assigned['rank'])}">
                <div class="gx-label">Recorded assignment · reference · {escaped(f['feature_id'])}</div>
                <h3>{escaped(f['display_name'])}</h3><div class="gx-reference-molecule">{molecule(assigned)}</div>
                <p class="gx-small">{'Standard-confirmed · MSI 1' if view['confirmed'] else 'Library assignment · MSI 2'}</p>
                <div class="gx-comparison-values"><span>Similarity Tc<b>{similarity(assigned)}</b></span><span>Estimated P(EFSA+)<b>{fmt_score(assigned['score'])}</b></span></div>
                <p class="gx-small">Estimated probability of an aggregated positive EFSA label, conditional on this structure.</p><div class="gx-verdict"><strong>{baseline_state} at Tc ≥ {view['cutoff']:.2f}</strong>Reference identity from this GC–MS entry. Choose another entry to change it.</div></section>
              <section class="gx-route-col ours" data-selection-key="{inspected['selection_key']}">
                <div class="gx-label">Selected molecule · {escaped(f['feature_id'])} · #{int(chosen['rank']):02d}</div>
                <h3>{escaped(structure_name(chosen, f))}</h3><div class="gx-reference-molecule">{molecule(chosen)}</div>
                <p class="gx-small">{relation}</p>
                <div class="gx-comparison-values"><span>Similarity Tc<b>{similarity(chosen)}</b></span><span>Estimated P(EFSA+)<b>{fmt_score(chosen['score'])}</b></span></div>
                <p class="gx-small">Estimated probability of an aggregated positive EFSA label, conditional on this structure.</p><div class="gx-verdict"><strong>{selection_state} at Tc ≥ {view['cutoff']:.2f}</strong>Individual molecular result. Click another structure to change it.</div></section>
            </div>'''


        def weight_summary(view, inspected):
            f, row = view['feature'], inspected['selected']
            covered = view['covered_weight']
            if view['confirmed']:
                scope = 'Confirmed identity'
                scenario = '100% of identity weight stays on the standard-confirmed assignment. Other molecules remain available for inspection.'
            elif view['assumption'] == 'same_formula':
                scope = f"All {view['n_retained']} structures · equal weights"
                scenario = f"Each retained structure receives {100/view['n_retained']:.0f}% in this hypothetical identity scenario."
            else:
                scope = 'Assigned identity only'
                scenario = '100% of identity weight is on the assigned structure.'
            tiles = []
            for _, candidate in view['candidates'].iterrows():
                rank = int(candidate['rank'])
                key = f"{f['feature_id']}:{rank}"
                selected = key == inspected['selection_key']
                classes = 'gx-identity-tile' + (' no-output' if not np.isfinite(float(candidate['score'])) else (' in' if candidate['in_domain'] else ' out'))
                if float(candidate['weight']) == 0:
                    classes += ' zero-weight'
                if selected:
                    classes += ' selected'
                tc = f"{candidate['tanimoto_neighbour']:.3f}" if np.isfinite(float(candidate['tanimoto_neighbour'])) else 'unavailable'
                title = f"Structure #{rank:02d} · Tc {tc} · scenario weight {100*float(candidate['weight']):.0f}%"
                tiles.append(f'<span class="{classes}" data-key="{key}" title="{escaped(title)}">#{rank:02d}</span>')
            transition = ''
            if not view['confirmed'] and view['assumption'] == 'same_formula':
                transition = f'''<div class="gx-identity-transition"><span>Assigned identity <b>{100*view['published_covered_weight']:.0f}%</b></span><span aria-hidden="true">→</span><span>Equal-weight set <b>{100*covered:.0f}%</b></span></div>'''
            if view['assumption'] == 'same_formula' and not view['confirmed']:
                headline = f"{view['n_evaluable']} of {view['n_retained']} structures meet Tc ≥ {view['cutoff']:.2f}"
            else:
                headline = f"{100*covered:.0f}% of the active identity weight meets Tc ≥ {view['cutoff']:.2f}"
            score = fmt_score(view['conditional_score'])
            return f'''<div class="gx gx-note violet gx-weight-live" data-feature-id="{escaped(f['feature_id'])}" data-selection-key="{inspected['selection_key']}">
              <div class="gx-label">Identity set · {escaped(f['feature_id'])} · {escaped(f['display_name'])}</div>
              <h2 style="font-size:23px;margin:6px 0">{headline}</h2><p class="gx-small">{scope}. {scenario}</p>
              {transition}<div class="gx-track" style="margin-top:10px"><div class="gx-fill" style="width:{100*covered:.6f}%;background:#6550a8"></div></div>
              <div class="gx-identity-map">{''.join(tiles)}</div>
              <p class="gx-small">Violet outline: inspected molecule #{int(row['rank']):02d}. Teal: meets Tc. Amber: below Tc. Grey: no model output. Faded tiles carry no identity weight.</p>
              <p class="gx-set-result"><b>{100*covered:.0f}% active scenario weight included</b> · {100*view['uncovered_weight']:.0f}% excluded · scenario average over retained structures {score}.</p>
              <p class="gx-small">This average describes the EFSA-label estimates of the included structures; it is not the probability that the peak is genotoxic.</p>
              <p class="gx-small gx-grain-note">This summary uses the complete identity set. It changes with the GC–MS entry, Tc or identity assumption; inspecting one member changes the outlined tile and its individual result above.</p>
            </div>'''


        def whole_intro(features, provenance, view, inspected):
            row = inspected['selected']
            tc = f"{row['tanimoto_neighbour']:.3f}" if np.isfinite(float(row['tanimoto_neighbour'])) else 'Unavailable'
            overlay = 'The violet diamond is the inspected alternative.' if not bool(row['is_assigned']) else 'The ring marks the inspected assignment.'
            return f'''<div class="gx gx-section"><div class="gx-label">{len(features)} reported GC–MS assignments · assigned structures</div>
              <h2>Choose an identity. Explore its molecular result.</h2>
              <p class="gx-small" style="margin-top:8px"><b>{view['n_assigned_evaluable']} of {len(features)}</b> assigned structures meet Tc ≥ {view['cutoff']:.2f}.
                Each point represents one assigned structure from the input table, compared with {provenance['n_training_substances']:,} EFSA training structures.
                Click a point to select that GC–MS entry.</p>
              <p class="gx-small" style="margin-top:8px;padding:8px 11px;border-left:3px solid #6550a8;background:#eeeafb"><b>Inspected: {escaped(view['feature']['feature_id'])} · structure #{int(row['rank']):02d} · Tc {tc}.</b> {overlay} The 33 source points represent the original assignments.</p>
            </div>'''


        def closing(view, inspected):
            row = inspected['selected']
            name = structure_name(row,view['feature'])
            has_output = np.isfinite(float(row['score']))
            if not has_output:
                outcome = 'This standardized model input produced no score or similarity.'
            elif inspected['selected_evaluable']:
                outcome = f"This molecule meets your criterion Tc ≥ {view['cutoff']:.2f}."
            else:
                outcome = f"This molecule is outside your criterion Tc ≥ {view['cutoff']:.2f}; its score is withheld from screening interpretation."
            tc = f"{row['tanimoto_neighbour']:.3f}" if has_output else 'Unavailable'
            set_result = (f"{100*view['covered_weight']:.0f}% of the active identity weight is included; "
                          f"scenario average over retained structures {fmt_score(view['conditional_score'])}."
                          if view['covered_weight'] > 0 else
                          '0% included identity weight: no aggregate screening score under these settings.')
            next_step = ('Next step: confirm the tentative identity with independent analytical evidence.'
                         if not view['confirmed'] else
                         'Assess the confirmed identity using the original conclusions and experimental evidence.')
            return f'''<div class="gx gx-close" data-selection-key="{inspected['selection_key']}">
              <div class="gx-label">Selected molecular result · {escaped(view['feature']['feature_id'])} · #{int(row['rank']):02d}</div>
              <h2>Compound identity changes the screening scenario.</h2><p>{escaped(name)} · {outcome}</p><p>Estimated P(EFSA+) {fmt_score(row['score'])} · similarity Tc {tc}.</p>
              <p class="gx-small">Estimated probability of an aggregated positive EFSA label, conditional on this structure.</p>
              <p><strong>{set_result}</strong></p><p>{next_step}</p>
              <p>Identity, applicability and structural motifs guide follow-up. Neither a small score nor an absent motif establishes safety.</p></div>'''


        def plot_positions(features):
            positions = []
            for x, group in enumerate(("Standard-confirmed", "Library-only")):
                rows = features.loc[features["evidence_group"].eq(group)].sort_values("feature_id")
                slots = np.linspace(-.3,.3,len(rows)) if len(rows)>1 else [0.0]
                for slot, (_, row) in zip(slots, rows.iterrows()):
                    px = .075 + ((x + slot + .45) / 1.9) * (.98 - .075)
                    py = 1 - (.19 + float(row["tanimoto_neighbour"]) / 1.055 * (.94 - .19))
                    positions.append((str(row["feature_id"]), float(px), float(py), str(row["display_name"]), float(row["tanimoto_neighbour"])))
            return positions


        def figure_svg(fig):
            buf = StringIO()
            fig.savefig(buf, format="svg", facecolor=PAPER)
            plt.close(fig)
            svg = buf.getvalue()
            return svg[svg.find("<svg"):]

        def study_plot(features, view, inspected=None):
            with plt.rc_context({"font.family":"DejaVu Sans", "font.size":10,
                                 "text.color":INK, "axes.labelcolor":MUTED,
                                 "xtick.color":MUTED, "ytick.color":MUTED,
                                 "svg.fonttype":"none"}):
                fig, ax = plt.subplots(figsize=(10.8,3.3))
                fig.subplots_adjust(left=.075,right=.98,bottom=.19,top=.94)
                ax.set_facecolor(PAPER)
                ax.axhspan(0, view["cutoff"], color="#efece4", alpha=.65, zorder=0)
                point_information = []
                for x, group in enumerate(("Standard-confirmed", "Library-only")):
                    rows = features.loc[features["evidence_group"].eq(group)].sort_values("feature_id")
                    slots = np.linspace(-.3,.3,len(rows)) if len(rows)>1 else [0.0]
                    for slot, (_, row) in zip(slots, rows.iterrows()):
                        pos = x+slot
                        marker = "*" if row["is_nias"] else "o"
                        color = AMBER if row["is_nias"] else (PETROL if group == "Standard-confirmed" else VIOLET)
                        face = color if group == "Standard-confirmed" or row["is_nias"] else PAPER
                        size = 140 if row["is_nias"] else 47
                        dot = ax.scatter(pos,row["tanimoto_neighbour"],s=size,marker=marker,
                                         facecolors=face,edgecolors=color,linewidths=1.1,zorder=3)
                        point_id = "compound-" + str(row["feature_id"])
                        dot.set_gid(point_id)
                        passes = bool(row["score"] == row["score"] and row["tanimoto_neighbour"] >= view["cutoff"])
                        evidence = "Confirmed in input data" if row["msi_level"] == 1 else "Tentative identity"
                        tooltip = (f"{row['feature_id']} · {row['display_name']}\n"
                                   f"Similarity Tc: {row['tanimoto_neighbour']:.6f}\n"
                                   f"Selected minimum: {view['cutoff']:.2f} · {'PASSES' if passes else 'BELOW MINIMUM'}\n"
                                   f"{evidence}{' · NIAS: non-intentionally added substance' if row['is_nias'] else ''}\n"
                                   f"Closest training structure: {row['nn_name']}")
                        point_information.append((point_id, tooltip))
                        if row["feature_id"] == view["feature"]["feature_id"]:
                            ax.scatter(pos,row["tanimoto_neighbour"],s=size+150,
                                       facecolors="none",edgecolors=INK,linewidths=1,zorder=4)
                if inspected is not None:
                    selected_row = inspected['selected']
                    if not bool(selected_row['is_assigned']) and np.isfinite(float(selected_row['tanimoto_neighbour'])):
                        point = next(p for p in plot_positions(features) if p[0] == str(view['feature']['feature_id']))
                        x_value = (point[1]-.075)/(.98-.075)*1.9-.45
                        ax.plot([x_value,x_value], [view['assigned']['tanimoto_neighbour'],selected_row['tanimoto_neighbour']],
                                color=VIOLET,linewidth=1.1,linestyle=(0,(2,2)),zorder=4)
                        marker = ax.scatter(x_value,selected_row['tanimoto_neighbour'],s=115,marker='D',
                                            facecolor=VIOLET,edgecolor='white',linewidth=.8,zorder=5)
                        marker.set_gid('inspected-structure')
                ax.axhline(view["cutoff"],color=INK,linestyle=(0,(4,3)),linewidth=1,zorder=2)
                ax.text(1.38,view["cutoff"]+.025,f"Minimum similarity: {view['cutoff']:.2f}",ha="right",va="bottom",fontsize=9,color=INK)
                ax.set_ylim(0,1.055); ax.set_xlim(-.45,1.45)
                ax.set_yticks([0,.2,.4,.6,.8,1.0]); ax.set_ylabel("Nearest-EFSA similarity · Tc",fontsize=10)
                n1=int(features["msi_level"].eq(1).sum()); n2=len(features)-n1
                ax.set_xticks([0,1],[f"Confirmed in input data · {n1}",f"Tentative identity · {n2}"])
                ax.spines[["top","right"]].set_visible(False)
                for side in ("left","bottom"): ax.spines[side].set_color(LINE)
                ax.tick_params(axis="both",length=0,pad=8)
                ax.yaxis.grid(True,color=LINE,linewidth=.5); ax.set_axisbelow(True)
                ET.register_namespace("", "http://www.w3.org/2000/svg")
                ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
                root = ET.fromstring(figure_svg(fig))
                by_id = {element.get("id"): element for element in root.iter() if element.get("id")}
                for point_id, tooltip in point_information:
                    point = by_id[point_id]
                    point.set("class", "gx-point")
                    point.set("tabindex", "0")
                    point.set("role", "img")
                    point.set("aria-label", tooltip)
                    title = ET.Element("{http://www.w3.org/2000/svg}title")
                    title.text = tooltip
                    point.insert(0, title)
                return ET.tostring(root, encoding="unicode")

        def study_finding(features, view, inspected):
            passing = features['score'].notna() & features['tanimoto_neighbour'].ge(view['cutoff'])
            confirmed = int((passing & features['msi_level'].eq(1)).sum())
            tentative = int((passing & features['msi_level'].ne(1)).sum())
            row = inspected['selected']
            tc = f"{row['tanimoto_neighbour']:.3f}" if np.isfinite(float(row['tanimoto_neighbour'])) else 'unavailable'
            extra = 'The diamond shows your inspected alternative; the 33 original dots remain assigned structures.' if not bool(row['is_assigned']) and np.isfinite(float(row['tanimoto_neighbour'])) else 'The ring marks your selected assignment.'
            return f'''<div class="gx gx-study-readout" aria-live="polite" data-selection-key="{inspected['selection_key']}">
              <p class="gx-finding"><b>{int(passing.sum())}/{len(features)}</b> assigned structures meet Tc ≥ {view['cutoff']:.2f}: {confirmed} confirmed and {tentative} tentative identities.
                Current entry: <span class="gx-mono">{escaped(view['feature']['feature_id'])}</span> · inspected structure #{int(row['rank']):02d} · Tc {tc}.</p>
              <p class="gx-small">Filled circle: confirmed. Open circle: library assignment. Star: flagged NIAS. {extra} Horizontal spacing separates entries.</p></div>'''

        def sensitivity_plot(view, inspected):
            candidates = view['candidates']
            similarity = candidates['tanimoto_neighbour'].to_numpy()
            valid = candidates['score'].notna().to_numpy() & np.isfinite(similarity)
            assigned_weights = candidates['is_assigned'].fillna(False).astype(float).to_numpy()
            alternative_weights = assigned_weights if view['confirmed'] else np.ones(len(candidates))/len(candidates)
            boundaries = similarity[valid & (similarity >= .1) & (similarity <= .9)]
            thresholds = np.unique(np.r_[np.linspace(.1,.9,161), boundaries, np.nextafter(boundaries, np.inf)])
            thresholds = thresholds[thresholds <= .9]
            assigned_curve = np.array([float(assigned_weights[valid & (similarity >= t)].sum()) for t in thresholds])
            alternative_curve = np.array([float(alternative_weights[valid & (similarity >= t)].sum()) for t in thresholds])
            row = inspected['selected']
            tc = float(row['tanimoto_neighbour'])
            has_output = np.isfinite(tc) and np.isfinite(float(row['score']))
            name = structure_name(row,view['feature'])
            rank = int(row['rank'])
            cutoff = view['cutoff']
            with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'}):
                def base_axis():
                    fig, ax = plt.subplots(figsize=(5.25,2.6))
                    fig.subplots_adjust(left=.16,right=.96,bottom=.24,top=.87)
                    ax.set_facecolor(PAPER)
                    ax.set_xlim(.1,.9); ax.set_ylim(-.06,1.08)
                    ax.set_xticks([.1,.3,.5,.7,.9])
                    ax.axvline(cutoff,color=AMBER,linewidth=1.2)
                    ax.set_xlabel('Minimum similarity · Tc',color=MUTED)
                    ax.spines[['top','right']].set_visible(False)
                    ax.spines[['left','bottom']].set_color(LINE)
                    ax.tick_params(colors=MUTED)
                    ax.yaxis.grid(True,color=LINE,linewidth=.5)
                    return fig,ax
                fig, ax = base_axis()
                if has_output:
                    selected_curve = (tc >= thresholds).astype(float)
                    ax.plot(thresholds,selected_curve,color=PETROL,linewidth=2,drawstyle='steps-post')
                    ax.scatter([cutoff],[float(inspected['selected_evaluable'])],color=AMBER,s=45,zorder=5)
                    if .1 <= tc <= .9:
                        marker = ax.axvline(tc,color=VIOLET,linewidth=1,linestyle=(0,(2,3)))
                        marker.set_gid('inspected-sensitivity')
                    note = f'Structure #{rank:02d} · Tc {tc:.3f}'
                else:
                    note = 'No model output for this structure'
                    ax.text(.5,.48,'No inclusion curve',ha='center',transform=ax.transAxes,color=MUTED)
                ax.set_yticks([0,1],['OUT','IN'])
                ax.set_ylabel('Selected molecule',color=MUTED)
                ax.text(.98,1.1,note,ha='right',transform=ax.transAxes,color=VIOLET,fontsize=9)
                left = figure_svg(fig)
                fig, ax = base_axis()
                fig.subplots_adjust(top=.74)
                ax.plot(thresholds,assigned_curve,color=PETROL,linewidth=1.7,drawstyle='steps-post',linestyle='--',label='Assigned identity')
                if not view['confirmed']:
                    ax.plot(thresholds,alternative_curve,color=VIOLET,linewidth=1.8,drawstyle='steps-post',label=f"Equal weights · {view['n_retained']} structures")
                ax.scatter([cutoff],[view['covered_weight']],color=AMBER,s=45,zorder=5)
                ax.set_yticks([0,.5,1],['0%','50%','100%'])
                ax.set_ylabel('Included identity weight',color=MUTED)
                ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),frameon=False,fontsize=8)
                right = figure_svg(fig)
            state = 'IN' if inspected['selected_evaluable'] else ('OUT' if has_output else 'NO OUTPUT')
            return f'''<div class="gx-sensitivity-panels">
              <section class="gx-selected-sensitivity" data-selection-key="{inspected['selection_key']}"><div class="gx-label">One molecule · #{rank:02d}</div>
                <h3>{escaped(name)}</h3><p class="gx-small">At Tc ≥ {cutoff:.2f}: <b>{state}</b>. Click a molecule to change this curve.</p>{left}</section>
              <section class="gx-ensemble-sensitivity" data-feature-id="{escaped(view['feature']['feature_id'])}"><div class="gx-label">Whole identity set · {escaped(view['feature']['feature_id'])}</div>
                <h3>{escaped(view['feature']['display_name'])}</h3><p class="gx-small">At Tc ≥ {cutoff:.2f}: <b>{100*view['covered_weight']:.0f}% identity weight included</b>. Choose another GC–MS entry to change this set.</p>{right}</section>
            </div>'''

        @lru_cache(maxsize=1800)
        def rule_hits(smiles):
            from genotox_food_migrants import candidate_view
            if not isinstance(smiles, str) or Chem.MolFromSmiles(smiles) is None:
                return None
            return candidate_view.alert_hits(smiles)

        def alert_details(row):
            valid = np.isfinite(float(row['score']))
            smiles = row['smiles_std'] if valid else row['smiles']
            hits = rule_hits(smiles)
            return dict(
                names=[h[0] for h in hits] if hits is not None else None,
                atoms=tuple(sorted({a for h in (hits or []) for a in h[1]})),
                representation='standardized model input' if valid else 'retained raw structure',
            )

        def alert_panel(view, inspected):
            from genotox_food_migrants import chemistry
            row = inspected['selected']
            details = alert_details(row)
            names = details['names']
            motif_background = '#fff3db' if names is None or names else '#e9f4f1'
            motif_border = '#e8cfa3' if names is None or names else '#bddbd4'
            motif_color = '#915811' if names is None or names else '#126b78'
            if names is None:
                match = 'Structure could not be read for motif matching'
                explanation = 'No structural-alert result was produced for this representation.'
            elif names:
                match = ' · '.join(names)
                explanation = f"{len(names)} matched motif class{'es' if len(names)!=1 else ''}. Highlighted atoms in the inspector are the matched atoms."
            else:
                match = f"No match in the {len(chemistry.DNA_REACTIVE_ALERTS)}-rule set"
                explanation = 'This describes the listed motif rules; it is not a genotoxicity classification.'
            domain = 'meets' if inspected['selected_evaluable'] else 'does not meet'
            if not np.isfinite(float(row['score'])):
                model_text = 'No supported standardized model input is available; no classifier result was produced.'
            else:
                model_text = f"Estimated P(EFSA+) {fmt_score(row['score'])}; this structure {domain} your similarity criterion Tc ≥ {view['cutoff']:.2f}."
            return f'''<div class="gx gx-alert-live gx-section" data-selection-key="{inspected['selection_key']}" aria-live="polite">
              <div class="gx-label">Structural motifs · {escaped(view['feature']['feature_id'])} · structure #{int(row['rank']):02d}</div>
              <div class="gx-motif-heading"><div class="gx-selection-thumb">{structure_svg(row['smiles_std'] if np.isfinite(float(row['score'])) else row['smiles'],150,95,details['atoms'])}</div><h2 style="font-size:25px;margin:6px 0">{escaped(structure_name(row,view['feature']))}</h2></div>
              <div class="gx-results" style="margin-top:12px"><div style="background:{motif_background};border-color:{motif_border}"><div class="gx-label">Project SMARTS rules</div>
                <p class="gx-result-value" style="font-size:18px;color:{motif_color}">{escaped(match)}</p><p class="gx-small">{explanation}</p></div>
                <div><div class="gx-label">Structure-specific classifier output</div><p style="font-size:14px;margin:8px 0">{model_text}</p>
                  <p class="gx-small">A motif match and a classifier score measure different things. Neither determines genotoxicity on its own.</p></div></div>
              <p class="gx-small" style="margin-top:8px">Rules applied to the {details['representation']}. Changing the selected molecule recomputes its matched motifs; changing Tc updates the model-inclusion rule.</p>
            </div>'''

        from types import SimpleNamespace
        return SimpleNamespace(
            CSS=CSS, header=header, comparison=comparison, feature_context=feature_context,
            scenario_summary=scenario_summary, validation_panel=validation_panel,
            choices_intro=choices_intro, choice_label=choice_label,
            structure_name=structure_name, inspector=inspector, selection_banner=selection_banner,
            weight_summary=weight_summary, two_routes=two_routes, whole_intro=whole_intro, closing=closing,
            study_plot=study_plot, study_finding=study_finding,
            plot_positions=plot_positions, sensitivity_plot=sensitivity_plot, alert_panel=alert_panel, alert_details=alert_details,
        )
