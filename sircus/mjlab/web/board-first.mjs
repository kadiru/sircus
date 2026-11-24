// Read-only view. Audited historical relabeling is separate from new test drops.
const VERSION = 'sircus-board-first-display-v1';
const integer = n => Number.isInteger(n) && n >= 0;
const hash = h => typeof h === 'string' && /^[a-f0-9]{64}$/.test(h);
export function validCounts(c) {
  return !!c && ['scorable_native','old_captures','new_captures','gained','lost','unchanged_captures','unchanged_misses','invalid_native'].every(k => integer(c[k]))
    && c.old_captures === c.unchanged_captures + c.lost
    && c.new_captures === c.unchanged_captures + c.gained
    && c.scorable_native === c.unchanged_captures + c.gained + c.lost + c.unchanged_misses + c.invalid_native;
}
const audited = x => x?.status === 'completed' && x.audit_status === 'passed'
  && hash(x.provenance?.report_sha256) && hash(x.provenance?.audit_sha256);
export function rescoreReady(x) {
  if (!audited(x) || !validCounts(x.counts) || !Array.isArray(x.positions)) return false;
  const keys = new Set();
  for (const p of x.positions) {
    if (!Array.isArray(p.bucket_xy_m) || p.bucket_xy_m.length !== 2 || !p.bucket_xy_m.every(Number.isFinite) || !validCounts(p.counts)) return false;
    const key = JSON.stringify(p.bucket_xy_m); if (keys.has(key)) return false; keys.add(key);
  }
  return ['scorable_native','old_captures','new_captures','gained','lost','unchanged_captures','unchanged_misses','invalid_native'].every(k => x.positions.reduce((a,p)=>a+p.counts[k],0) === x.counts[k]);
}
export function comparisonReady(x) {
  return audited(x) && x.physics_substeps === 64 && x.positions === 50 && x.frozen_before_test === true
    && Array.isArray(x.methods) && x.methods.length === 2 && new Set(x.methods.map(m=>m.id)).size === 2
    && x.methods.every(m => hash(m.checkpoint_sha256) && m.attempts === 50
      && ['captures','setup_rejections','released_misses','invalid_physics'].every(k=>integer(m[k]))
      && m.captures + m.setup_rejections + m.released_misses + m.invalid_physics === 50);
}
function table(root, id, headings, rows) {
  const node = root.querySelector(`[data-board-first="${id}"]`); node.replaceChildren();
  if (!rows.length) return;
  const doc = root.ownerDocument, tab = doc.createElement('table'), head = doc.createElement('thead'), body = doc.createElement('tbody');
  const line = (values, tag) => { const tr=doc.createElement('tr'); for(const value of values){const td=doc.createElement(tag);td.textContent=String(value);tr.append(td);}return tr; };
  head.append(line(headings,'th')); for(const row of rows)body.append(line(row,'td'));
  tab.append(head,body);node.append(tab);
}
export function renderBoardFirst(root, raw) {
  root.hidden = raw?.version !== VERSION; if(root.hidden)return;
  const text = (id,value) => {root.querySelector(`[data-board-first="${id}"]`).textContent=value;};
  const r=raw.rescore, c=raw.comparison, ready=rescoreReady(r), tested=comparisonReady(c);
  text('status', ready ? 'Saved-shot rescoring complete · original labels preserved' : 'Checking saved trajectories · final counts await the independent audit');
  text('scope', raw.scope ?? 'Simulation only. Historical relabeling creates no new drops.');
  table(root,'counts',['Saved drops scored','Previously accepted','Accepted under board-first rule','Newly accepted','No longer accepted'],ready?[[r.counts.scorable_native,r.counts.old_captures,r.counts.new_captures,r.counts.gained,r.counts.lost]]:[]);
  text('changes',ready?`${r.counts.gained} saved captures are newly accepted; ${r.counts.lost} no longer meet the board-first requirement. ${r.counts.invalid_native} invalid-physics traces remain excluded. Contact presence does not prove that the contact caused a bounce.`:'No new capture total is claimed yet.');
  const changed=ready?r.positions.filter(p=>p.counts.gained||p.counts.lost).sort((a,b)=>b.counts.gained-a.counts.gained||b.counts.lost-a.counts.lost):[];
  table(root,'positions',['Bucket X, Y (mm)','Saved drops','Old captures','New captures','Gained / lost'],changed.map(p=>[p.bucket_xy_m.map(v=>(1000*v).toFixed(1)).join(', '),p.counts.scorable_native,p.counts.old_captures,p.counts.new_captures,`${p.counts.gained} / ${p.counts.lost}`]));
  text('training',raw.training_text??'The same small network ranks proposed shots. It is retrained on corrected TRAIN labels, selected using validation data, then frozen before new test drops.');
  text('test-status',tested?'Fresh 50-position comparison complete · both checkpoints frozen':`Fresh 50-position comparison: ${c?.phase??'pending'}. Results appear after the independent audit.`);
  table(root,'comparison',['Frozen selector','Captures / 50','Setup refused','Released misses','Invalid physics','Mean planning (s)'],tested?c.methods.map(m=>[m.label,`${m.captures} / 50`,m.setup_rejections,m.released_misses,m.invalid_physics,Number.isFinite(m.mean_planning_s)?m.mean_planning_s.toFixed(2):'—']):[]);
  text('conclusion',tested?c.conclusion:'Re-scoring alone is not evidence that a newly trained model generalizes better.');
  text('proof',ready?`Rescore report SHA-256: ${r.provenance.report_sha256}. Independent audit: ${r.provenance.audit_sha256}.`:'');
}
