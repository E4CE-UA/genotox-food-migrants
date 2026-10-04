"""Shared, versioned competition demo component. No compatibility implementation."""

def build_widgets():
        """Local widget definitions embedded in the delivered Marimo notebook."""
        import anywidget
        import traitlets

        STUDY_JS = r'''
        export default { render({model, el}) {
          el.classList.add('gx-study-widget');
          const host=document.createElement('div'); host.className='gx-chart-svg';
          const hint=document.createElement('div'); hint.className='gx-chart-hint'; hint.setAttribute('aria-live','polite');
          el.append(host,hint);
          function paint() {
            const active=el.getRootNode().activeElement || document.activeElement;
            const focusId=host.contains(active) ? active.id : '';
            host.innerHTML=model.get('svg');
            host.querySelectorAll('.gx-point').forEach(node=>{
              const id=node.id.replace(/^compound-/, '');
              node.setAttribute('role','button');node.setAttribute('tabindex','0');
              node.setAttribute('aria-label','Select '+id+' · '+node.getAttribute('aria-label'));
              const title=node.querySelector('title')?.textContent || id;
              function select(){model.set('click_key',id);model.set('click_seq',(model.get('click_seq')||0)+1);model.save_changes();}
              node.addEventListener('click',select);
              node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});
              node.addEventListener('pointerenter',()=>{hint.textContent=title.replaceAll('\n',' · ');});
              node.addEventListener('focus',()=>{hint.textContent=title.replaceAll('\n',' · ');});
            });
            if(focusId) host.querySelector('[id="'+focusId+'"]')?.focus({preventScroll:true});
            hint.textContent='Select a plotted assignment. Its molecular results appear directly below.';
          }
          model.on('change:svg',paint);paint();
          return ()=>model.off('change:svg',paint);
        }};
        '''

        GRID_JS = r'''
        export default { render({model,el}) {
          el.classList.add('gx-grid-widget');
          const summary=document.createElement('div');summary.className='gx-grid-selection';
          const label=document.createElement('span');label.setAttribute('aria-live','polite');
          const jump=document.createElement('button');jump.type='button';jump.textContent='Show selected result';
          function findResult(root){const direct=root.querySelector('#current-molecular-result');if(direct)return direct;for(const node of root.querySelectorAll('*')){if(node.shadowRoot){const found=findResult(node.shadowRoot);if(found)return found;}}return null;}
          jump.addEventListener('click',()=>findResult(document)?.scrollIntoView({block:'center',behavior:'smooth'}));
          summary.append(label,jump);
          const grid=document.createElement('div');grid.className='gx-molecule-grid';el.append(summary,grid);
          function paint(){
            grid.innerHTML='';const cards=model.get('cards')||[];
            const selected=cards.find(card=>card.key===model.get('selected_key'));
            label.textContent=selected ? 'Selected: '+selected.key.split(':')[0]+' · #'+selected.key.split(':')[1].padStart(2,'0')+' · '+selected.name : 'Select a molecule';
            grid.classList.toggle('single',cards.length===1);
            for(const card of cards){
              const button=document.createElement('button');button.type='button';button.className='gx-mol-button';
              button.dataset.key=card.key;button.setAttribute('aria-label','Inspect '+card.name);
              button.setAttribute('aria-pressed',String(card.key===model.get('selected_key')));
              button.innerHTML=card.html;
              button.addEventListener('click',()=>{model.set('click_key',card.key);model.set('click_seq',(model.get('click_seq')||0)+1);model.save_changes();});grid.append(button);
            }
          }
          model.on('change:cards',paint);model.on('change:selected_key',paint);paint();
          return ()=>{model.off('change:cards',paint);model.off('change:selected_key',paint);};
        }};
        '''

        TRACE_JS = r'''
        export default {render({model,el}){
          el.classList.add('gx-trace-widget');
          const tabs=document.createElement('div');tabs.className='gx-trace-tabs';tabs.setAttribute('role','group');tabs.setAttribute('aria-label','Source figures');
          const controls=document.createElement('div');controls.className='gx-trace-controls';
          const viewport=document.createElement('div');viewport.className='gx-trace-viewport';
          const img=document.createElement('img');img.draggable=false;viewport.append(img);
          const meter=document.createElement('span');meter.className='gx-trace-meter';
          let z=model.get('zoom')||1, px=model.get('pan_x')||0, py=model.get('pan_y')||0, timer, dragging=false, last;
          function transform(){img.style.transform=`translate(${px}px,${py}px) scale(${z})`;meter.textContent=z.toFixed(1)+'×';}
          function commit(){model.set('zoom',z);model.set('pan_x',px);model.set('pan_y',py);model.save_changes();}
          function zoom(factor){z=Math.max(1,Math.min(6,z*factor));if(z===1){px=0;py=0;}transform();clearTimeout(timer);timer=setTimeout(commit,160);}
          function action(label,fn){const b=document.createElement('button');b.type='button';b.textContent=label;b.addEventListener('click',fn);controls.append(b);}
          action('Zoom +',()=>zoom(1.35));action('Zoom −',()=>zoom(1/1.35));action('Reset view',()=>{z=1;px=0;py=0;transform();commit();});controls.append(meter);
          function paint(){
            tabs.innerHTML='';const panels=model.get('panels')||[];const layer=model.get('layer')||0;tabs.hidden=panels.length<=1;
            panels.forEach((p,i)=>{const b=document.createElement('button');b.type='button';b.textContent=p.label;b.setAttribute('aria-pressed',String(i===layer));
              b.addEventListener('click',()=>{model.set('layer',i);model.save_changes();});tabs.append(b);});
            const p=panels[layer]||panels[0];if(p){img.src='data:'+(p.mime||'image/png')+';base64,'+p.image;img.alt=p.alt||p.label;if(p.width&&p.height)viewport.style.aspectRatio=p.width+'/'+p.height;}
          }
          viewport.addEventListener('wheel',e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();zoom(e.deltaY<0?1.12:1/1.12);}},{passive:false});
          viewport.addEventListener('pointerdown',e=>{if(z<=1)return;dragging=true;last=[e.clientX,e.clientY];viewport.setPointerCapture(e.pointerId);});
          viewport.addEventListener('pointermove',e=>{if(!dragging)return;px+=e.clientX-last[0];py+=e.clientY-last[1];last=[e.clientX,e.clientY];
            const limitX=viewport.clientWidth*(z-1)/2,limitY=viewport.clientHeight*(z-1)/2;px=Math.max(-limitX,Math.min(limitX,px));py=Math.max(-limitY,Math.min(limitY,py));transform();});
          viewport.addEventListener('pointerup',()=>{if(dragging){dragging=false;commit();}});
          el.append(tabs,controls,viewport);model.on('change:layer',paint);paint();transform();
          return ()=>{clearTimeout(timer);model.off('change:layer',paint);};
        }};
        '''

        STUDY_CSS = '''
        .gx-study-widget {max-width:1120px;margin:auto;color:#253b43;}
        .gx-chart-svg svg {width:100%;height:auto;display:block;pointer-events:none;}
        .gx-chart-svg .gx-point {cursor:pointer;pointer-events:bounding-box;}
        .gx-chart-svg .gx-point use {pointer-events:all;}
        .gx-chart-svg .gx-point:focus use,.gx-chart-svg .gx-point:hover use {stroke-width:2.5px;}
        .gx-chart-hint {font:12px/1.5 system-ui;color:#61737b;min-height:36px;padding:6px 0;}
        '''
        GRID_CSS = '''
        .gx-grid-selection {display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;background:#eeeafb;border:1px solid #d9cfee;border-radius:5px;padding:9px;margin:0 0 11px;color:#6550a8;font:11px/1.5 ui-monospace,monospace;}
        .gx-grid-selection button {cursor:pointer;border:1px solid #6550a8;background:#fffdf9;color:#6550a8;border-radius:4px;padding:5px 8px;font:11px/1.4 system-ui;}
        .gx-molecule-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;}
        .gx-molecule-grid.single {grid-template-columns:minmax(150px,240px);}
        .gx-mol-button {background:transparent;border:0;padding:0;cursor:pointer;text-align:left;min-width:0;color:#253b43;font-family:system-ui;}
        .gx-mol-button:hover {box-shadow:0 0 0 1px #253b4355;border-radius:6px;}
        .gx-mol-button:focus-visible {outline:2px dotted #253b43;outline-offset:2px;border-radius:6px;}
        .gx-choice-card {height:100%;box-sizing:border-box;background:#e9f4f1;border:1px solid #bddbd4;padding:9px 8px;border-radius:6px;}
        .gx-choice-card.out {background:#fff7e8;border-color:#dfc498;border-style:dashed;}
        .gx-choice-card.selected {border:2px solid #6550a8;padding:8px 7px;box-shadow:0 0 0 1px #6550a8 inset;}
        .gx-choice-card > .gx-choice-id:first-child {min-height:28px;}
        .gx-choice-id {font:10px/1.4 ui-monospace,monospace;overflow-wrap:anywhere;}
        .gx-choice-card svg {display:block;width:100%;height:98px;}
        .gx-choice-values {display:flex;justify-content:space-between;gap:5px;font:11px/1.4 ui-monospace,monospace;margin-top:4px;}
        .gx-choice-action {font:11px/1.5 system-ui;color:#6550a8;margin-top:8px;padding-top:5px;border-top:1px solid #00000014;}
        @media(max-width:540px){.gx-molecule-grid {grid-template-columns:repeat(2,minmax(0,1fr));}}
        '''
        TRACE_CSS = '''
        .gx-trace-widget {max-width:1120px;margin:auto;font:13px/1.5 system-ui;color:#253b43;}
        .gx-trace-tabs[hidden] {display:none!important;}
        .gx-trace-tabs,.gx-trace-controls {display:flex;flex-wrap:wrap;gap:7px;margin:9px 0;align-items:center;}
        .gx-trace-widget button {border:1px solid #dce2df;border-radius:5px;background:#fffdf9;color:#253b43;padding:7px 11px;cursor:pointer;font:12px/1.4 system-ui;}
        .gx-trace-tabs button[aria-pressed=true] {background:#126b78;color:white;border-color:#126b78;}
        .gx-trace-widget button:focus-visible {outline:3px solid #6550a8;}
        .gx-trace-viewport {position:relative;aspect-ratio:980/552;overflow:hidden;border:1px solid #dce2df;border-radius:7px;background:white;touch-action:pan-y;}
        .gx-trace-viewport img {width:100%;height:100%;object-fit:contain;transform-origin:center;user-select:none;}
        .gx-trace-meter {font:12px/1.4 ui-monospace,monospace;color:#126b78;}
        '''


        class StudyChart(anywidget.AnyWidget):
            _esm = STUDY_JS
            _css = STUDY_CSS
            svg = traitlets.Unicode('').tag(sync=True)
            click_key = traitlets.Unicode('').tag(sync=True)
            click_seq = traitlets.Int(0).tag(sync=True)


        class MoleculeGrid(anywidget.AnyWidget):
            _esm = GRID_JS
            _css = GRID_CSS
            cards = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
            selected_key = traitlets.Unicode('').tag(sync=True)
            click_key = traitlets.Unicode('').tag(sync=True)
            click_seq = traitlets.Int(0).tag(sync=True)


        class TraceViewer(anywidget.AnyWidget):
            _esm = TRACE_JS
            _css = TRACE_CSS
            panels = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
            layer = traitlets.Int(0).tag(sync=True)
            zoom = traitlets.Float(1).tag(sync=True)
            pan_x = traitlets.Float(0).tag(sync=True)
            pan_y = traitlets.Float(0).tag(sync=True)
        from types import SimpleNamespace
        return SimpleNamespace(StudyChart=StudyChart, MoleculeGrid=MoleculeGrid, TraceViewer=TraceViewer)
