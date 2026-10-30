"""Self-contained, operator-triggered Web Audio approximation of the MIDI scores."""
import json
from pathlib import Path

from .replay import demo_events, plan
from .score import CUES, score


def write_preview(path, mapping):
    packets = {name: plan(demo_events(name), mapping) for name in ("captured", "missed", "unknown")}
    for cue in CUES:
        packets[cue] = [] if mapping["muted"] else score(cue, gain=mapping["gain"], channel=mapping["channel"])
    encoded = json.dumps(packets, allow_nan=False).replace("<", "\\u003c")
    Path(path).write_text(HTML.replace("__PACKETS__", encoded))


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sircus · Sound desk</title>
<style>
:root {color-scheme:dark;font:17px/1.5 system-ui;background:#101519;color:#edf4ed}
*{box-sizing:border-box} body{max-width:1050px;margin:auto;padding:48px 24px 80px}
header{border-bottom:1px solid #3a4846;padding-bottom:28px;margin-bottom:28px}
.eyebrow{color:#b5d9c6;font-size:.78rem;letter-spacing:.16em;text-transform:uppercase}
h1{font-size:clamp(2.5rem,7vw,4.4rem);line-height:1.04;margin:14px 0 20px;letter-spacing:-.04em}
h2{font-size:1.2rem;margin-top:32px}.muted{color:#aebdb9}.tag{display:inline-block;border:1px solid #54645e;border-radius:20px;padding:4px 12px;font-size:.75rem}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{background:#1a2527;border:1px solid #35464a;border-radius:16px;padding:22px;text-align:left;color:inherit;cursor:pointer;font:inherit;min-height:135px}
.card:hover{background:#253638;border-color:#91c4ab}.card b{display:block;font-size:1.3rem;color:#d5f3a9;margin-bottom:6px}.card span{font-size:.87rem;color:#b7c6c1}
.controls{position:sticky;bottom:14px;display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-top:28px;padding:18px;background:#283432;border:1px solid #56655e;border-radius:16px}
button:focus-visible,input:focus-visible{outline:3px solid #d5f3a9;outline-offset:4px}
#stop{font:inherit;border:0;border-radius:9px;padding:10px 20px;background:#f3c7b5;color:#241b15;cursor:pointer}label{display:flex;gap:12px;align-items:center}#status{flex-basis:100%;font-size:.85rem;color:#d4e1d9}
footer{font-size:.82rem;color:#9daea7;margin-top:28px} @media(max-width:650px){.grid{grid-template-columns:1fr 1fr}body{padding-top:28px}.scenarios{grid-template-columns:1fr}}
</style>
<header><div class="eyebrow">Simple Circus / audio study 01</div><h1>A little physics.<br>A lot of character.</h1>
<span class="tag">Synthetic demo · no robot connection</span>
<p class="muted">Review the cues, then play a short imagined trial. Sound starts only when you click.</p></header>
<h2>Sound palette</h2><div class="grid">
<button class="card" data-play="boing"><b>Boing!</b><span>First observed board impact</span></button>
<button class="card" data-play="squeak"><b>Squeak</b><span>A quick sliding bounce</span></button>
<button class="card" data-play="plop"><b>Plop</b><span>A soft downward drop</span></button>
<button class="card" data-play="spring"><b>Spring</b><span>A smaller elastic bounce</span></button>
<button class="card" data-play="celebration"><b>Ta-da!</b><span>Known capture outcome</span></button>
<button class="card" data-play="wong"><b>Wong × 3</b><span>Known miss outcome</span></button></div>
<h2>Replay a synthetic trial</h2><div class="grid scenarios">
<button class="card" data-play="captured"><b>Capture</b><span>Four contacts, then celebration</span></button>
<button class="card" data-play="missed"><b>Miss</b><span>Four contacts, then three wongs</span></button>
<button class="card" data-play="unknown"><b>Unknown</b><span>Four contacts. No outcome cue.</span></button></div>
<div class="controls"><button id="stop">Stop & release</button><label>Preview level <input id="level" aria-label="Preview level" type="range" min="0" max="0.5" step="0.01" value="0.15"></label>
<label><input id="mute" type="checkbox">Mute</label><div id="status" role="status" aria-live="polite">Ready · silent</div></div>
<footer>Original procedural tones; no downloaded samples or plugins. This browser approximation uses the same MIDI cue plan as Logic, with a simple triangle oscillator. Logic's patch changes the timbre. Synthetic contacts do not prove physical contact or capture. ODD MIDI alone cannot identify board or bucket.</footer>
<script>
const scores = __PACKETS__;
let ctx, master, compressor, nodes = [], timer, generation = 0;
const level = document.querySelector('#level'), mute = document.querySelector('#mute'), status = document.querySelector('#status');
function stop() {
  generation++; clearTimeout(timer);
  for (const {osc, amp} of nodes) { try { amp.disconnect(); osc.stop(); } catch (_) {} }
  nodes = []; status.textContent = 'Stopped · all pending tones cancelled';
}
function setLevel() { if(master) master.gain.setValueAtTime(mute.checked ? 0 : +level.value, ctx.currentTime); }
async function play(name) {
  stop(); const token = generation;
  if (mute.checked) { status.textContent = 'Muted · click Mute to enable sound'; return; }
  try {
    if (!ctx) {
      ctx = new AudioContext(); master = ctx.createGain(); compressor = ctx.createDynamicsCompressor();
      compressor.threshold.value = -18; compressor.ratio.value = 12;
      master.connect(compressor).connect(ctx.destination);
    }
    await ctx.resume(); if (token !== generation) return;
    setLevel(); const start = ctx.currentTime + .04, active = new Map(); let bend = 0;
    function finish(voice, t) { voice.amp.gain.setValueAtTime(voice.volume, t); voice.amp.gain.linearRampToValueAtTime(0, t+.025); voice.osc.stop(t+.03); }
    for (const [offset, bytes] of scores[name]) {
      const t = start+offset, kind = bytes[0]&0xf0, pitch = bytes[1], velocity = bytes[2];
      if (kind === 0x90 && velocity > 0) {
        if (active.has(pitch)) finish(active.get(pitch), t);
        const osc = ctx.createOscillator(), amp = ctx.createGain(), volume = velocity/127*.7;
        osc.type = 'triangle'; osc.frequency.setValueAtTime(440*2**((pitch-69+bend)/12), t);
        amp.gain.setValueAtTime(0,t); amp.gain.linearRampToValueAtTime(volume,t+.012);
        osc.connect(amp).connect(master); osc.start(t);
        const voice = {osc,amp,pitch,volume}; nodes.push(voice); active.set(pitch,voice);
      } else if (kind === 0x80 || (kind === 0x90 && velocity === 0)) {
        if (active.has(pitch)) { finish(active.get(pitch),t); active.delete(pitch); }
      } else if (kind === 0xe0) {
        bend = ((pitch+(velocity<<7))-8192)/8192*12;
        for (const voice of active.values()) voice.osc.frequency.setValueAtTime(440*2**((voice.pitch-69+bend)/12),t);
      } else if (kind === 0xb0 && [120,123].includes(pitch)) {
        for (const voice of active.values()) finish(voice,t); active.clear(); bend=0;
      }
    }
    const end = start + (scores[name].at(-1)?.[0] || 0) + .1;
    for (const voice of active.values()) finish(voice,end);
    status.textContent = 'Playing '+name+' · synthetic preview';
    timer = setTimeout(() => { stop(); status.textContent = 'Finished '+name+' · ready'; }, Math.max(0,(end-ctx.currentTime)*1000+50));
  } catch (error) { stop(); status.textContent = 'Audio unavailable: '+error.message; }
}
document.querySelectorAll('[data-play]').forEach(button => button.onclick = () => play(button.dataset.play));
document.querySelector('#stop').onclick = stop;
level.oninput = setLevel; mute.onchange = () => { if(mute.checked) stop(); setLevel(); };
document.addEventListener('visibilitychange', () => { if(document.hidden) stop(); });
window.addEventListener('pagehide',stop);
</script></html>'''
