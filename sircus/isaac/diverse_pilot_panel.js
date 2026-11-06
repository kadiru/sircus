/* Read-only evidence display: no policy, simulator or robot commands. */
const METHODS=['broad','guided','diverse'];
const SPLITS=['train','validation','test'];
const COLORS={broad:'#aebfd0',guided:'#78d0b7',diverse:'#77b9ee'};
export const percent=value=>Number.isFinite(value)?`${(value*100).toFixed(1)}%`:'pending';

export function frozenRows(comparison){
 const rows=[];
 for(const key of METHODS)for(const split of SPLITS)for(const stage of ['initial','final']){
  const method=comparison.methods[key],row=method.frozen[split][stage];
  rows.push([method.label,split,stage,row?.attempts??'pending',row?.released??'pending',
   row?`${row.captures} · ${percent(row.capture_rate)}`:'pending',row?.clean??'pending',row?.rejected??'pending']);
 }
 return rows;
}

export function comparisonSummary(comparison){
 const lines=METHODS.map(key=>{const m=comparison.methods[key],r=m.training;return `${m.label}: ${r.rejected}/${r.attempts} rejected (${percent(r.attempts?r.rejected/r.attempts:null)}).`;});
 lines.push('All rejected active attempts remain in the training denominator.');
 if(comparison.status!=='completed')lines.push('New run is incomplete; final held-out comparison is pending.');
 else for(const split of ['validation','test'])for(const previous of ['broad','guided']){
  const delta=comparison.final_policy_difference[split][`diverse_minus_${previous}_percentage_points`];
  lines.push(`${split} versus ${previous}: ${Number.isFinite(delta)?`${delta>=0?'+':''}${delta.toFixed(1)} percentage points`:'unavailable'}.`);
 }
 return lines.join(' ');
}

export function failureRows(comparison){
 const reasons=[...new Set(METHODS.flatMap(key=>Object.keys(comparison.methods[key].training_failure_counts||{})))].sort();
 return reasons.map(reason=>[reason,...METHODS.map(key=>comparison.methods[key].training_failure_counts?.[reason]??0)]);
}

function range(regions,axis,scale=1000){
 if(!regions.length)return '—';
 const low=Math.min(...regions.map(r=>r.low[axis])),high=Math.max(...regions.map(r=>r.high[axis]));
 return `${(low*scale).toFixed(1)}–${(high*scale).toFixed(1)}`;
}

export function supportRows(support,usage){
 const used=new Map((usage?.cells||[]).map(c=>[c.cell_id,c]));
 return (support?.cells||[]).map(cell=>{
  const regions=cell.regions||[],observed=used.get(cell.cell_id);
  const sampled=usage?.status!=='logged'?'not logged':observed?.split!=='train'?'held out':observed.sampled_regions;
  return [cell.cell_id,cell.status,regions.length,sampled,range(regions,0,180/Math.PI),range(regions,1),
   range(regions,2),range(regions,3),range(regions,4,180/Math.PI)];
 });
}

export function regionRows(support){
 return (support?.cells||[]).flatMap(cell=>(cell.regions||[]).map(r=>[
  cell.cell_id,r.id,...[0,1,2,3,4].map(i=>range([r],i,i===0||i===4?180/Math.PI:1000)),
  Number.isFinite(r.probe_acceptance)?`${percent(r.probe_acceptance)} sampled${r.qualification_target_met===false?' · below target':''}`:'anchor/probe receipt only'
 ]));
}

export function readyForFinalClaim(comparison){
 return comparison.status==='completed'&&comparison.independent_audit?.report?.status==='completed'
  &&METHODS.every(key=>comparison.methods[key].status==='completed'
   &&SPLITS.every(split=>comparison.methods[key].frozen[split].initial&&comparison.methods[key].frozen[split].final)
   &&['native','checkpoint','feasibility'].every(a=>comparison.methods[key].audit_status[a]==='passed'));
}

function node(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}
function table(target,headers,rows){
 target.replaceChildren();const t=node('table'),thead=node('thead'),heading=node('tr');
 for(const value of headers)heading.appendChild(node('th',value));thead.appendChild(heading);t.appendChild(thead);
 const body=node('tbody');for(const row of rows){const tr=node('tr');for(const value of row)tr.appendChild(node('td',String(value)));body.appendChild(tr);}t.appendChild(body);target.appendChild(t);
}
function svg(parent,tag,attributes,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attributes))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;parent.appendChild(n);return n;}
function curves(comparison){
 const root=document.getElementById('comparisonCurve');root.replaceChildren();
 const max=Math.max(1,...METHODS.flatMap(key=>comparison.methods[key].learning_curve.map(p=>p.completed_attempts)));
 const x=v=>58+590*v/max,y=v=>280-220*v;
 for(const value of [0,.25,.5,.75,1]){svg(root,'line',{x1:58,x2:648,y1:y(value),y2:y(value),stroke:'#405063'});svg(root,'text',{x:48,y:y(value)+4,fill:'#aebfd0','font-size':12,'text-anchor':'end'},percent(value));}
 for(const key of METHODS){const points=comparison.methods[key].learning_curve;if(points.length>1)svg(root,'polyline',{points:points.map(p=>`${x(p.completed_attempts)},${y(p.rolling_capture_rate)}`).join(' '),fill:'none',stroke:COLORS[key],'stroke-width':3});}
 for(const value of [0,.25,.5,.75,1])svg(root,'text',{x:x(max*value),y:308,fill:'#aebfd0','font-size':12,'text-anchor':'middle'},Math.round(max*value));
 svg(root,'text',{x:350,y:335,fill:'#aebfd0','text-anchor':'middle'},'Attempted training shots, including rejected requests');
}

function render(report){
 const c=report.comparison,status=document.getElementById('comparisonStatus');
 if(!c){status.textContent='Verified comparison is unavailable; no improvement claim.';return;}
 status.textContent=readyForFinalClaim(c)?'Completed comparison · native/policy/geometry audit receipts attached':'Preliminary comparison · complete native run or independent audits still pending';
 document.getElementById('comparisonSummary').textContent=comparisonSummary(c);
 table(document.getElementById('comparisonChecks'),['Method','Split','Stage','Attempted','Released','Captured / attempted','Clean','Rejected'],frozenRows(c));
 table(document.getElementById('comparisonFailures'),['Training result / failure','Broad','One local region','Diverse + settling'],failureRows(c));
 document.getElementById('comparisonScope').textContent=c.scientific_scope;
 document.getElementById('comparisonActionScope').textContent=c.action_scope;
 document.getElementById('comparisonSettling').textContent=JSON.stringify(Object.fromEntries(METHODS.map(key=>[key,c.methods[key].setup_protocol])),null,2);
 const prep=c.geometry_preparation,support=prep.support,usage=c.region_usage,cost=prep.process_resource_receipt?.report;
 const count=support?`${support.retained_region_count} regions at ${support.retained_supported_cells}/35 sites`:'support not attached';
 let resource='';
 if(cost){const f=value=>Number.isFinite(value)?value.toFixed(2):'unknown';resource=` Resource receipt: ${f(cost.wall_s)} s wall, ${f(cost.user_cpu_s)} s user CPU, ${f(cost.system_cpu_s)} s system CPU; ${Number.isFinite(cost.maximum_resident_set_kib)?(cost.maximum_resident_set_kib/1024).toFixed(1):'unknown'} MiB process peak RSS (not aggregate worker peak; original map cost excluded).`;}
 document.getElementById('comparisonPreparation').textContent=`${prep.cost_scope} Original map: ${prep.original_sobol_requests} requests, ${prep.original_map_admitted} qualified endpoints. New support: ${count}.${resource} ${usage?.scope||'Region-choice usage is not attached.'}`;
 table(document.getElementById('comparisonSupports'),['Cell','Support','Regions','Requested regions sampled in training','Bearing envelope °','Radius envelope mm','Board H envelope mm','Gap envelope mm','Tilt envelope °'],supportRows(support,usage));
 table(document.getElementById('comparisonRegions'),['Cell','Region','Bearing °','Radius mm','Board H mm','Gap mm','Tilt °','Geometry qualification'],regionRows(support));
 document.getElementById('comparisonSupport').textContent=JSON.stringify({geometry_preparation:prep,region_usage:usage,inference_scope:c.inference_scope},null,2);
 curves(c);
}

if(typeof document!=='undefined'){
 let previous=null;
 async function refresh(){try{const res=await fetch('/report.json',{cache:'no-store'});if(!res.ok)throw Error(`HTTP ${res.status}`);const data=await res.json(),identity=JSON.stringify(data.comparison);if(identity!==previous){previous=identity;render(data);}}catch(error){document.getElementById('comparisonStatus').textContent=`Comparison unavailable · ${error.message}`;}}
 refresh();setInterval(refresh,10000);
}
