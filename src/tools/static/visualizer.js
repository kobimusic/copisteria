// The detections page of src/tools/visualizer.py: the boxes over the image, their hover, the class checkboxes.
// DATA holds what the template rendered: syms (a row of [class index, conf, x0, y0, x1, y1] per detection, in the
// file's order), classes, cols (a colour per class), counts.
const S = DATA.syms, C = DATA.classes, P = DATA.cols, K = DATA.counts;
const NS = 'http://www.w3.org/2000/svg';
const ov = document.getElementById('ov'), tip = document.getElementById('tip'), img = document.getElementById('img');
const rects = {}, checks = {};                         // per class: its rects and its checkbox
let hot = null;

function draw() {
  ov.setAttribute('viewBox', '0 0 ' + img.naturalWidth + ' ' + img.naturalHeight);
  for (const s of S) {
    const r = document.createElementNS(NS, 'rect');
    r.setAttribute('x', s[2]); r.setAttribute('y', s[3]);
    r.setAttribute('width', s[4] - s[2]); r.setAttribute('height', s[5] - s[3]);
    r.setAttribute('stroke', P[s[0]]);
    r.addEventListener('mouseenter', () => show(r, s));
    r.addEventListener('mouseleave', hide);
    ov.appendChild(r);
    (rects[s[0]] = rects[s[0]] || []).push(r);
  }
}

function show(r, s) {                                 // class and confidence, beside the box
  if (hot) hot.setAttribute('stroke-width', 2);
  hot = r; r.setAttribute('stroke-width', 4);
  tip.textContent = C[s[0]] + '   conf ' + s[1];
  tip.style.display = 'block';
  const b = r.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
  let x = b.right + 6;  if (x + w > innerWidth - 4) x = Math.max(4, b.left - w - 6);
  let y = Math.max(4, b.top);  if (y + h > innerHeight - 4) y = innerHeight - h - 4;
  tip.style.left = x + 'px';  tip.style.top = y + 'px';
}

function hide() { tip.style.display = 'none'; if (hot) { hot.setAttribute('stroke-width', 2); hot = null; } }
function on(i, v) { if (!v) hide(); for (const r of rects[i] || []) r.style.display = v ? '' : 'none'; }

for (let i = 0; i < C.length; i++) {                  // one checkbox per class, all checked to start
  const lab = document.createElement('label'); lab.className = 'cls';
  const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = true;
  cb.addEventListener('change', () => on(i, cb.checked));
  const sw = document.createElement('span'); sw.className = 'sw'; sw.style.background = P[i];
  const nm = document.createElement('span'); nm.textContent = C[i];
  const n = document.createElement('span'); n.className = 'n'; n.textContent = K[i];
  lab.append(cb, sw, nm, n); document.getElementById('boxes').appendChild(lab);
  checks[i] = cb;
}

function all(v) { for (let i = 0; i < C.length; i++) { checks[i].checked = v; on(i, v); } }
document.getElementById('on').onclick = () => all(true);
document.getElementById('off').onclick = () => all(false);
if (img.complete) draw(); else img.addEventListener('load', draw);
