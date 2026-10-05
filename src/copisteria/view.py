"""One self-contained HTML page per read page: the scan with every symbol box, the symbols whose reading the
context changed (what the detector said, what the model reads, the evidence in nats, and -- on hover -- the
symbols the model looked at most when it decided), and the written MusicXML rendered by Verovio.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from .read import Reading, changes

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>copisteria {title}</title>
<style>
:root {{ --bg:#fafaf7; --fg:#222; --muted:#777; --chg:#d62728; --add:#2ca02c; --del:#9467bd; --ctx:#ff9f1c; }}
body {{ margin:0; font:13px system-ui, sans-serif; background:var(--bg); color:var(--fg); }}
header {{ padding:8px 14px; border-bottom:1px solid #ddd; display:flex; gap:18px; align-items:baseline; flex-wrap:wrap }}
header b {{ font-size:15px }}
.wrap {{ display:flex; height:calc(100vh - 42px); }}
.pane {{ overflow:auto; position:relative; }}
#left {{ flex:1.1; border-right:1px solid #ddd; }}
#right {{ flex:1; padding:6px; }}
#stage {{ position:relative; transform-origin:0 0; }}
#stage img {{ display:block; }}
#ov {{ position:absolute; left:0; top:0; }}
rect {{ fill:none; stroke-width:1.5; vector-effect:non-scaling-stroke; }}
rect.base {{ stroke:#1f77b4; opacity:.25 }}
rect.chg {{ stroke:var(--chg); stroke-width:2.5; cursor:pointer }}
rect.add {{ stroke:var(--add); stroke-width:2.5; cursor:pointer }}
rect.del {{ stroke:var(--del); stroke-width:2.5; stroke-dasharray:4 3; cursor:pointer }}
rect.ctx {{ stroke:var(--ctx); stroke-width:3; opacity:1 }}
#tip {{ position:fixed; pointer-events:none; background:#fff; border:1px solid #999; padding:6px 8px; font:12px ui-monospace,monospace;
  white-space:pre; display:none; z-index:9; box-shadow:0 2px 6px rgba(0,0,0,.15) }}
table {{ border-collapse:collapse; font:12px ui-monospace,monospace; margin-top:10px; width:100% }}
td,th {{ border-bottom:1px solid #e3e3e3; padding:2px 6px; text-align:left }}
tr:hover {{ background:#fff3d6; cursor:pointer }}
.svgpage svg {{ width:100%; height:auto; background:#fff; margin-bottom:8px; border:1px solid #e3e3e3 }}
.zoom {{ position:sticky; top:4px; left:4px; z-index:5; background:#fff; border:1px solid #ccc; padding:2px 6px }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1d1d1b; --fg:#ddd; }} .svgpage svg {{ background:#fff }} }}
</style></head><body>
<header><b>{title}</b><span>{summary}</span>
<span><span style="color:var(--chg)">&#9632;</span> reading changed &nbsp;<span style="color:var(--add)">&#9632;</span> made real
&nbsp;<span style="color:var(--del)">&#9632;</span> judged not real &nbsp;<span style="color:var(--ctx)">&#9632;</span> context it looked at</span></header>
<div class="wrap"><div id="left" class="pane"><div class="zoom">zoom <input id="z" type="range" min="0.2" max="2" step="0.05" value="{z0}"></div>
<div id="stage"><img src="data:image/jpeg;base64,{img}" width="{w}" height="{h}">
<svg id="ov" width="{w}" height="{h}" viewBox="0 0 {w} {h}"></svg></div></div>
<div id="right" class="pane"><div id="score">{svgs}</div><table id="tab"><tr><th>bar</th><th>symbol</th><th>change</th><th>evidence</th></tr></table></div></div>
<div id="tip"></div>
<script>
const S={scale}, SYMS={syms}, CHG={chg};
const ov=document.getElementById('ov'), tip=document.getElementById('tip'), stage=document.getElementById('stage');
const NS='http://www.w3.org/2000/svg', rects={{}};
function mk(b,cls){{const r=document.createElementNS(NS,'rect');r.setAttribute('x',b[0]*S);r.setAttribute('y',b[1]*S);
 r.setAttribute('width',(b[2]-b[0])*S);r.setAttribute('height',(b[3]-b[1])*S);r.setAttribute('class',cls);ov.appendChild(r);return r;}}
for(const s of SYMS) rects[s.i]=mk(s.box,'base');
const tab=document.getElementById('tab');
function fmt(c){{return Object.entries(c.diffs).map(([k,v])=>k+': '+v[0]+' → '+v[1]+(v[2]!=null?'  ('+(v[2]>0?'+':'')+v[2]+' nats)':'')).join('\\n');}}
function show(c,on,ev){{for(const [j,w] of c.context){{const r=rects[j]; if(r) r.classList.toggle('ctx',on);}}
 if(on&&ev){{tip.style.display='block';tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+14)+'px';
 tip.textContent=c.cls+'  staff '+c.staff+' bar '+c.bar+'\\n'+fmt(c)+(c.context.length?'\\nlooked at: '+c.context.map(([j,w])=>(SYMS.find(s=>s.i==j)||{{cls:'?'}}).cls+' '+w).join(', '):'');}}
 else tip.style.display='none';}}
for(const c of CHG){{const kind=('real' in c.diffs)?(c.diffs.real[1]>=0.5?'add':'del'):'chg';
 const r=rects[c.sym]||mk(c.box,kind); r.setAttribute('class',kind); ov.appendChild(r);
 r.addEventListener('mouseenter',e=>show(c,true,e)); r.addEventListener('mouseleave',e=>show(c,false));
 const tr=document.createElement('tr'); const ev=Object.values(c.diffs).map(v=>v[2]).filter(v=>v!=null);
 tr.innerHTML='<td>'+c.staff+':'+c.bar+'</td><td>'+c.cls+'</td><td>'+fmt(c).replace(/\\n/g,'<br>')+'</td><td>'+(ev.length?Math.max(...ev.map(Math.abs)).toFixed(1):'')+'</td>';
 tr.addEventListener('click',()=>{{const L=document.getElementById('left'); L.scrollTo({{left:c.box[0]*S*zoom-200,top:c.box[1]*S*zoom-200,behavior:'smooth'}});
  r.classList.add('ctx'); setTimeout(()=>r.classList.remove('ctx'),1200);}});
 tr.addEventListener('mouseenter',()=>show(c,true,null)); tr.addEventListener('mouseleave',()=>show(c,false)); tab.appendChild(tr);}}
let zoom={z0}; const zi=document.getElementById('z');
function setz(){{zoom=+zi.value; stage.style.transform='scale('+zoom+')'; stage.style.width=({w}*zoom)+'px'; stage.style.height=({h}*zoom)+'px';}}
zi.addEventListener('input',setz); setz();
</script></body></html>"""


def page_html(rd: Reading, image_path: str | Path, musicxml: str, title: str, max_w: int = 1800) -> str:
    from PIL import Image
    im = Image.open(image_path).convert("L")
    scale = min(1.0, max_w / im.width)
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    img = base64.b64encode(buf.getvalue()).decode()
    syms = [{"i": s.i, "box": [round(v, 1) for v in s.box], "cls": rd.tok[s.i]["cls"]["v"]}
            for s in rd.layout.syms if s.i in rd.tok and rd.tok[s.i]["real"]["p"] >= 0.5]
    chg = changes(rd)
    for c in chg:
        c["box"] = [round(v, 1) for v in c["box"]]
        c["diffs"] = {k: [v[0] if not isinstance(v[0], float) else round(v[0], 3),
                          v[1] if not isinstance(v[1], float) else round(v[1], 3), v[2]] for k, v in c["diffs"].items()}
    svgs = ""
    try:
        import verovio
        tk = verovio.toolkit()
        tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 40, "adjustPageHeight": True,
                       "footer": "none", "header": "auto"})
        if tk.loadData(musicxml):
            svgs = "".join(f'<div class="svgpage">{tk.renderToSVG(p)}</div>' for p in range(1, tk.getPageCount() + 1))
    except Exception as e:                                       # the score still shows as a table
        svgs = f"<p>verovio failed: {html.escape(str(e))}</p>"
    n_real = len(syms)
    kinds = {"changed": 0, "made real": 0, "not real": 0}
    for c in chg:
        if "real" in c["diffs"]:
            kinds["made real" if c["diffs"]["real"][1] >= 0.5 else "not real"] += 1
        else:
            kinds["changed"] += 1
    summary = f"{n_real} symbols read; context changed {kinds['changed']}, made real {kinds['made real']}, " \
              f"judged not real {kinds['not real']}"
    z0 = round(min(1.0, 900 / max(1, im.width)), 2)
    return HTML.format(title=html.escape(title), summary=summary, img=img, w=im.width, h=im.height, scale=scale,
                       syms=json.dumps(syms), chg=json.dumps(chg), svgs=svgs, z0=z0)
