/* Read-only comparison: no physics, policy inference or robot controls. */
export const percent=value=>Number.isFinite(value)?`${(value*100).toFixed(1)}%`:'pending';
export function frozenRows(comparison){
 const rows=[];
 for(const key of ['broad','guided'])for(const split of ['train','validation','test']){
  const method=comparison.methods[key],stages=method.frozen[split];
  for(const stage of ['initial','final']){
   const row=stages[stage];
   rows.push([method.label,split,stage,row?.attempts??'pending',row?.released??'pending',
     row?`${row.captures} · ${percent(row.capture_rate)}`:'pending',row?.clean??'pending',row?.rejected??'pending']);
  }
 }
 return rows;
}
export function comparisonSummary(comparison){
 const a=comparison.methods.broad.training,b=comparison.methods.guided.training;
 const rejectsA=a.attempts?a.rejected/a.attempts:null,rejectsB=b.attempts?b.rejected/b.attempts:null;
 const lines=[`Broad: ${a.rejected}/${a.attempts} requests rejected (${percent(rejectsA)}). Guided: ${b.rejected}/${b.attempts} (${percent(rejectsB)}). All rejected requests remain in the training denominator.`];
 if(comparison.status!=='completed')lines.push('Guided run is incomplete. Final held-out comparison is pending.');
 else for(const split of ['validation','test']){
  const d=comparison.final_policy_difference[split].guided_minus_broad_percentage_points;
  lines.push(`${split}: ${Number.isFinite(d)?`${d>=0?'+':''}${d.toFixed(1)} percentage points versus the broad final policy`:'not available'}.`);
 }
 return lines.join(' ');
}
export function supportRows(support){
 return (support?.cells||[]).map(cell=>{
  if(cell.status!=='supported')return [cell.cell_id,'unresolved','—','—','—','—','—','No local support; fallback attempts still counted'];
  const range=(i,scale=1000)=>`${(cell.low[i]*scale).toFixed(1)}–${(cell.high[i]*scale).toFixed(1)}`;
  return [cell.cell_id,'supported',range(0,180/Math.PI),range(1),range(2),range(3),range(4,180/Math.PI),`${percent(cell.probe_acceptance)} sampled qualification${cell.qualification_target_met===false?' · BELOW 75% TARGET':''}`];
 });
}
export function supportExclusionNote(comparison){
 const receipt=comparison.posthoc_support_exclusion?.report;
 if(!receipt)return null;
 const all=receipt.old_successful_sites,excluded=receipt.old_successful_settings_excluded;
 const finding=all>0&&excluded===all?`Geometry guidance excludes all ${all} previously successful settings.`:`Geometry guidance excludes ${excluded}/${all} previously successful settings.`;
 return `POSTHOC EVALUATION · ${finding} Lower reach/IK failure alone does not improve aiming. This does not prove the new boxes have no successful shots. The diagnostic was not used to tune anchors, policy, rewards or training support.`;
}
function node(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}
function table(target,headers,rows){target.replaceChildren();const t=node('table'),head=node('thead'),tr=node('tr');for(const v of headers)tr.appendChild(node('th',v));head.appendChild(tr);t.appendChild(head);const body=node('tbody');for(const row of rows){const r=node('tr');for(const v of row)r.appendChild(node('td',String(v)));body.appendChild(r);}t.appendChild(body);target.appendChild(t);}
function svg(parent,tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;parent.appendChild(n);return n;}
function curves(comparison){
 const root=document.getElementById('comparisonCurve');root.replaceChildren();
 const methods=comparison.methods,max=Math.max(1,...Object.values(methods).flatMap(m=>m.learning_curve.map(p=>p.completed_attempts)));
 const x=v=>58+590*v/max,y=v=>280-220*v;
 for(const v of[0,.25,.5,.75,1]){svg(root,'line',{x1:58,x2:648,y1:y(v),y2:y(v),stroke:'#405063'});svg(root,'text',{x:48,y:y(v)+4,fill:'#aebfd0','font-size':12,'text-anchor':'end'},percent(v));}
 for(const key of ['broad','guided']){const points=methods[key].learning_curve;if(points.length>1)svg(root,'polyline',{points:points.map(p=>`${x(p.completed_attempts)},${y(p.rolling_capture_rate)}`).join(' '),fill:'none',stroke:key==='guided'?'#78d0b7':'#aebfd0','stroke-width':3});}
 for(const v of[0,.25,.5,.75,1])svg(root,'text',{x:x(max*v),y:308,fill:'#aebfd0','font-size':12,'text-anchor':'middle'},Math.round(max*v));
 svg(root,'text',{x:350,y:335,fill:'#aebfd0','text-anchor':'middle'},'Attempted training shots, including rejected requests');
}
export function readyForFinalClaim(comparison){return comparison.status==='completed'&&comparison.independent_audit?.report?.status==='completed'&&['broad','guided'].every(key=>['native','checkpoint','feasibility'].every(a=>comparison.methods[key].audit_status[a]==='passed'));}
function render(report){
 const c=report.comparison,status=document.getElementById('comparisonStatus');
 if(!c){status.textContent='Verified baseline comparison is unavailable; no improvement claim.';return;}
 status.textContent=readyForFinalClaim(c)?'Completed comparison · native/policy/geometry audit receipts attached':'Preliminary comparison · complete audit receipts or native run still pending';
 document.getElementById('comparisonSummary').textContent=comparisonSummary(c);
 let posthoc=document.getElementById('comparisonPosthoc');if(!posthoc){posthoc=node('p');posthoc.id='comparisonPosthoc';posthoc.className='note';document.getElementById('comparisonSummary').after(posthoc);}
 const exclusion=supportExclusionNote(c);posthoc.hidden=!exclusion;posthoc.textContent=exclusion||'';
 table(document.getElementById('comparisonChecks'),['Method','Split','Stage','Attempted','Released','Captured / attempted','Clean','Rejected'],frozenRows(c));
 document.getElementById('comparisonScope').textContent=c.scientific_scope;
 document.getElementById('comparisonActionScope').textContent=c.action_scope;
 const prep=c.geometry_preparation,support=prep.support;
 const cost=prep.process_resource_receipt?.report,resource=cost?` Measured support preprocessing: ${cost.wall_s.toFixed(2)} s wall, ${cost.user_cpu_s.toFixed(2)} s user CPU, ${cost.system_cpu_s.toFixed(2)} s system CPU, ${(cost.maximum_resident_set_kib/1024).toFixed(1)} MiB GNU time maximum RSS (not aggregate worker peak; excludes prior map cost).`:'';
 document.getElementById('comparisonPreparation').textContent=`${prep.cost_scope} Original map: ${prep.original_sobol_requests} requests, ${prep.original_map_admitted} admitted. Local support: ${support?`${support.supported_cells}/35 cells; ${support.probe_accepted}/${support.probe_requests} accepted probes; ${Number.isFinite(support.elapsed_s)?support.elapsed_s.toFixed(1)+' s':'time unknown'}`:'not attached'}.${resource}`;
 table(document.getElementById('comparisonSupports'),['Cell','Support','Bearing °','Radius mm','Board H mm','Gap mm','Tilt °','Qualification'],supportRows(support));
 document.getElementById('comparisonSupport').textContent=JSON.stringify({geometry_preparation:c.geometry_preparation,posthoc_support_exclusion:c.posthoc_support_exclusion||'not attached'},null,2);
 curves(c);
}
if(typeof document!=='undefined'){
 let previous=null;
 async function refresh(){try{const res=await fetch('/report.json',{cache:'no-store'});if(!res.ok)throw Error(`HTTP ${res.status}`);const data=await res.json(),identity=JSON.stringify(data.comparison);if(identity!==previous){previous=identity;render(data);}}catch(e){document.getElementById('comparisonStatus').textContent=`Comparison unavailable · ${e.message}`;}}
 refresh();setInterval(refresh,10000);
}
