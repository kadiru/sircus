/* Read-only, run-matched multi-cell report. No robot or simulator command API. */
let parallelChoice=null, parallelRun=null, parallelFrame=null;
const parallelEl=id=>document.getElementById(id);
const parallelMM=x=>Number.isFinite(x)?(x*1000).toFixed(0):'—';
function parallelRow(table,label,value){const r=document.createElement('tr');for(const text of [label,value]){const td=document.createElement('td');td.textContent=String(text);r.appendChild(td);}table.appendChild(r);}
function parallelRender(r){
 const fresh=r.phase==='completed'||r.phase==='failed'||Date.now()-Date.parse(r.timestamp_utc)<30000;
 parallelEl('status').textContent=`${r.phase}${fresh?'':' · updates stale'} · ${r.run}`;
 parallelEl('status').className='pill'+(r.phase==='failed'?' error':'');
 parallelEl('count').textContent=r.contract.num_envs;parallelEl('arms').textContent=r.contract.num_envs*2;
 parallelEl('attempts').textContent=r.attempts;parallelEl('progress').textContent=`${r.batches_completed}/${r.contract.batches}`;
 const c=r.cells.find(c=>c.id===parallelChoice)||r.cells[0];parallelChoice=c.id;
 const buttons=parallelEl('cells');buttons.replaceChildren();
 for(const row of r.cells){const b=document.createElement('button');b.className='cell';b.setAttribute('aria-pressed',String(row.id===c.id));
  for(const [tag,txt] of [['strong',`Table ${row.index+1}`],['small',`X ${parallelMM(row.bucket_xy_m[0])} · Y ${parallelMM(row.bucket_xy_m[1])} mm`],['small',`${parallelMM(row.distance_m)} mm from release line`]]){const e=document.createElement(tag);e.textContent=txt;b.appendChild(e);}
  const s=document.createElement('div');s.className='score';const tests=row.confirmations||[];
  s.textContent=tests.length?`${tests.filter(t=>t.route_success).length}/${tests.length} fresh captures`:(row.rows.length?`${row.rows.filter(t=>t.route_success).length}/${row.rows.length} search captures`:'Awaiting trials');
  if(tests.length&&tests.filter(t=>t.route_success).length<tests.length)s.className+=' warn';b.appendChild(s);b.onclick=()=>{parallelChoice=row.id;parallelRender(r);};buttons.appendChild(b);
 }
 const detail=parallelEl('detail');detail.replaceChildren();parallelEl('selectedId').textContent=c.id;
 parallelEl('selectedTitle').textContent=`Table ${c.index+1} · fixed bucket`;
 parallelRow(detail,'Bucket X / Y',`${parallelMM(c.bucket_xy_m[0])} / ${parallelMM(c.bucket_xy_m[1])} mm`);
 parallelRow(detail,'Horizontal distance',`${parallelMM(c.distance_m)} mm`);
 parallelRow(detail,'Board/release X / Y',`${parallelMM(c.board_xy_m[0])} / ${parallelMM(c.board_xy_m[1])} mm`);
 parallelRow(detail,'Heading toward bucket',`${c.heading_deg.toFixed(1)}° from −Y`);
 if(c.winner){parallelRow(detail,'Selected board height',`${parallelMM(c.winner.height_m)} mm`);parallelRow(detail,'Selected tilt',`${c.winner.tilt_deg.toFixed(1)}°`);parallelRow(detail,'Selected drop gap',`${parallelMM(c.winner.gap_m)} mm`);parallelRow(detail,'Ball-bottom release height',`${parallelMM(c.winner.height_m+c.winner.gap_m)} mm`);}
 else parallelRow(detail,'Selected setting',r.phase==='completed'?'None found in this search':'Pending');
 const tests=c.confirmations||[],last=c.rows.at(-1);
 if(last)parallelRow(detail,'Last explored H / gap / tilt',`${parallelMM(last.height_m)} / ${parallelMM(last.gap_m)} mm / ${last.tilt_deg.toFixed(1)}°`);
 const successes=tests.filter(t=>t.route_success).length;
 parallelEl('outcome').textContent=tests.length?`Fresh repeats: ${successes}/${tests.length} captured; ${tests.filter(t=>t.single_bounce_success).length} with exactly one board bounce; ${tests.filter(t=>t.route_quality===2).length} clean entries; ${tests.filter(t=>t.invalid).length} invalid. Small-sample simulation results, not a reliability guarantee.`:c.winner?'Fresh confirmation pending.':'No fresh confirmation yet. A finite search finding no setting does not prove impossibility.';
 if(r.view?.overview && (parallelRun!==r.run||parallelFrame!==r.view.render_frame)){
  parallelEl('overview').src=`${r.view.overview}?frame=${r.view.render_frame}`;parallelEl('close').src=`${r.view.close}?frame=${r.view.render_frame}`;
  parallelRun=r.run;parallelFrame=r.view.render_frame;
 }
 parallelEl('closeCaption').textContent=`Close camera · table ${r.cells[0].index+1}. This camera stays fixed while you inspect other cells below.`;
 const labels=parallelEl('labels');labels.replaceChildren();for(const p of r.view?.labels||[]){if(p.x<0||p.x>1||p.y<0||p.y>1)continue;const cell=r.cells.find(c=>c.id===p.id);if(!cell)continue;const b=document.createElement('button');b.className='label';b.textContent=String(cell.index+1);b.style.left=p.x*100+'%';b.style.top=p.y*100+'%';b.onclick=()=>{parallelChoice=cell.id;parallelRender(r);};labels.appendChild(b);}
 parallelEl('scope').textContent=r.scope;
 parallelEl('validation').textContent=`Bucket displacement: ${Number.isFinite(r.max_bucket_drift_m)?(r.max_bucket_drift_m*1000).toFixed(4)+' mm':'not yet checked'}. Contact/isolation checks: ${r.probes.length?r.probes.filter(p=>p.all_passed).length+'/'+r.probes.length+' passed':'pending'}. ${r.error||''}`;
 parallelEl('provenance').textContent=JSON.stringify({run:r.run,contract:r.contract,probes:r.probes,verification:r.verification,elapsed_s:r.elapsed_s,artifact_sha256:r.artifact_sha256},null,2);
 if(r.video?.run===r.run){parallelEl('videoPanel').className='panel media';const v=parallelEl('video');if(v.getAttribute('src')!==r.video.url)v.setAttribute('src',r.video.url);}else{parallelEl('videoPanel').className='panel media hidden';parallelEl('video').removeAttribute('src');}
 if(r.verification && r.csv_url){parallelEl('csvLink').className='';parallelEl('csvLink').href=r.csv_url;}else parallelEl('csvLink').className='hidden';
}
async function parallelUpdate(){try{const response=await fetch('/parallel-world.json',{cache:'no-store'});if(!response.ok)throw Error('Report unavailable');parallelRender(await response.json());}catch(e){parallelEl('status').textContent='Parallel-world report unavailable · last displayed data may be stale';}}
let parallelLabels=false;
parallelEl('showLabels').onclick=()=>{parallelLabels=!parallelLabels;parallelEl('labels').className=parallelLabels?'':'hidden';parallelEl('showLabels').textContent=parallelLabels?'Hide table IDs':'Show table IDs';parallelEl('showLabels').setAttribute('aria-pressed',String(parallelLabels));};
let parallelZoomed=true;
parallelEl('zoom').onclick=()=>{parallelZoomed=!parallelZoomed;parallelEl('sceneLayer').style.transform=parallelZoomed?'scale(1.6)':'scale(1)';parallelEl('zoom').textContent=parallelZoomed?'Show full frame':'Enlarge world';parallelEl('zoom').setAttribute('aria-pressed',String(parallelZoomed));};
setInterval(parallelUpdate,4000);parallelUpdate();
