/* Read-only synthetic shot catalogue. No command or hardware endpoints. */
function catalogueRender(r) {
 const n=v=>Number.isFinite(v)?v.toFixed(1):'—';
 document.getElementById('catalogueStatus').textContent=`${r.phase} · ${r.search_completed}/${r.grid_planned+r.refinement_planned} search settings (${r.grid_planned} coarse + ${r.refinement_planned} refinement) · ${r.run}`;
 const s=r.search_summary;
 document.getElementById('catalogueSearch').textContent=`Search: ${s.clean} clean captures · ${s.assisted} assisted/marginal captures · ${s.misses} misses · ${s.invalid} rejected. These are searched settings, not policy performance.`;
 document.getElementById('catalogueDecision').textContent=r.phase==='completed'?'Confirmation complete. These are finite simulation results, not guaranteed shots or physical calibration. Unchanged repeats test repeatability; perturbed repeats test only the stated small variations.':'Study incomplete: selection and confirmation results are provisional.';
 const cell=(e,mode)=>{const q=e[mode+'_summary'];return q?`${q.clean} / ${q.assisted} / ${q.misses} / ${q.invalid} (${q.trials}/20)`:'pending';};
 studyTable(document.getElementById('catalogueFinalists'),['Board H mm','Tilt °','Ball bottom Z mm','Ball center Z mm','Flange Z mm','Distance mm','Nominal: clean / assisted / miss / invalid','Perturbed: clean / assisted / miss / invalid'],(r.confirmed||[]).map(e=>[n(e.board_height_mm),n(e.candidate.tilt_deg),n(e.ball_bottom_release_height_mm),n(e.ball_center_release_height_mm),n(e.dropper_flange_height_mm),n(e.distance_mm),cell(e,'nominal'),cell(e,'perturbed')]));
 const root=document.getElementById('catalogueHeatmaps');root.replaceChildren();
 const grid=r.contract.grid;
 for(const h of grid.heights_m){
 const wrap=document.createElement('div'),title=document.createElement('h3');title.textContent=`Board H ${Math.round(h*1000)} mm`;wrap.appendChild(title);
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 470 235');svg.setAttribute('role','img');svg.setAttribute('aria-label',title.textContent+' gap versus tilt quality scores');wrap.appendChild(svg);
 for(const [j,g] of grid.gaps_m.entries()){
 studySvg(svg,'text',{x:37,y:40+j*33,fill:'#c5d9db','font-size':12,'text-anchor':'end'},Math.round(g*1000));
 for(const [i,t] of grid.tilts_deg.entries()){
 const v=r.rows.find(c=>c.height_m===h&&c.gap_m===g&&c.tilt_deg===t);
 const fill=!v?'#294047':v.invalid?'#74828a':v.quality_score===2?'#55c9ad':v.quality_score===1?'#ffbb59':'#bc7862';
 studySvg(svg,'rect',{x:47+i*37,y:20+j*33,width:32,height:28,fill});
 studySvg(svg,'text',{x:63+i*37,y:40+j*33,fill:'#10262b','font-size':12,'text-anchor':'middle'},!v?'':v.invalid?'×':v.quality_score);
 if(j===0)studySvg(svg,'text',{x:63+i*37,y:204,fill:'#c5d9db','font-size':11,'text-anchor':'middle'},t+'°');
 }}studySvg(svg,'text',{x:47,y:230,fill:'#c5d9db','font-size':12},'Rows: gap mm · columns: tilt from horizontal');root.appendChild(wrap);
 }
 const success=(r.rows||[]).filter(v=>v.success&&!v.invalid).sort((a,b)=>b.quality_score-a.quality_score||(b.minimum_mouth_clearance_mm??-999)-(a.minimum_mouth_clearance_mm??-999));
 const radius=r.contract.parent.scene.ball.radius_m*1000;
 studyTable(document.getElementById('catalogueAll'),['Board H mm','Gap mm','Ball center Z mm','Tilt °','Score','Mouth clearance mm','Trial'],success.map(v=>[n(v.height_m*1000),n(v.gap_m*1000),n((v.height_m+v.gap_m)*1000+radius),n(v.tilt_deg),v.quality_score,n(v.minimum_mouth_clearance_mm),v.trial_id]));
 const measured=(r.rows||[]).concat(...(r.confirmed||[]).map(e=>(e.nominal||[]).concat(e.perturbed||[]))).filter(v=>v.settled);
 document.getElementById('catalogueDetails').textContent=JSON.stringify({contract:r.contract,contact_probes:r.probes,max_settled_position_error_mm:measured.length?Math.max(...measured.map(v=>v.settled.position_error_mm)):null,max_settled_normal_error_deg:measured.length?Math.max(...measured.map(v=>v.settled.normal_error_deg)):null,artifact_sha256:r.artifact_sha256},null,2);
}
function catalogueFresh(r, d) {
 const el=document.getElementById('catalogueFresh');
 if(!d || d.contract.source_run!==r.run){el.textContent='No matching independent confirmation report available.';return;}
 const s=d.summary,c=d.contract.candidate;
 el.textContent=`Fresh 100-trial check: ${d.phase} · board H ${(c.height_m*1000).toFixed(0)} mm, gap ${(c.gap_m*1000).toFixed(0)} mm, tilt ${c.tilt_deg}° · ${s.captures}/${s.trials} captures, ${s.clean} clean, ${s.assisted} assisted, ${s.misses} misses, ${s.invalid} rejected. Same bounded variations; independent seeds, no retuning.`;
}
async function catalogueUpdate(){try{const response=await fetch('/shot-catalogue.json',{cache:'no-store'});if(!response.ok)throw Error('unavailable');const r=await response.json();catalogueRender(r);let d=null;try{const fresh=await fetch('/catalogue-confirmation.json',{cache:'no-store'});if(fresh.ok)d=await fresh.json();}catch(e){}catalogueFresh(r,d);}catch(e){document.getElementById('catalogueStatus').textContent='No current catalogue report available';}}
setInterval(catalogueUpdate,5000);catalogueUpdate();
