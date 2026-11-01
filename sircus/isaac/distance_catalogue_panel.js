/* Read-only distance search and frozen confirmation. Never issues motion commands. */
function distanceCatalogueRender(r) {
 const n=v=>Number.isFinite(v)?v.toFixed(1):'—';
 const entries=[...r.contract.distances_m].sort((a,b)=>a-b).map(d=>r.distances.find(e=>e.distance_m===d)||{distance_m:d,phase:'pending',rows:[],screened:[]});
 document.getElementById('distanceCatalogueStatus').textContent=`${r.phase} · ${r.attempts} parameter attempts · ${r.run}`;
 const rows=entries.map(e=>{const c=e.winner,s=e.confirmation_summary,search=e.search_summary;return [Math.round(e.distance_m*1000),e.phase,search?`${search.captures}/${search.trials} (${search.invalid} rejected)`:'pending',c?n(c.height_m*1000):'—',c?n(c.tilt_deg):'—',c?n(c.gap_m*1000):'—',c?n((c.height_m+c.gap_m)*1000+37.5):'—',s?`${s.captures}/${s.trials} of 50`:e.phase==='no_capture_found'?'No captured setting found':'pending',s?`${s.clean} / ${s.assisted} / ${s.misses} / ${s.invalid}`:'—'];});
 studyTable(document.getElementById('distanceCatalogueTable'),['Distance mm','Stage','Search captures / attempts','Board H mm','Tilt °','Gap mm','Ball-center Z mm','Fresh captures','Clean / assisted / miss / invalid'],rows);
 const svg=document.getElementById('distanceCatalogueChart');svg.replaceChildren();
 studySvg(svg,'line',{x1:45,y1:220,x2:680,y2:220,stroke:'#8aafb4'});
 for(const v of [0,.5,1]){const y=220-v*170;studySvg(svg,'line',{x1:45,y1:y,x2:680,y2:y,stroke:'#35545b'});studySvg(svg,'text',{x:38,y:y+4,fill:'#c5d9db','font-size':12,'text-anchor':'end'},Math.round(v*100)+'%');}
 for(const [i,e] of entries.entries()){
 const x=75+i*100,s=e.confirmation_summary;
 if(s&&s.trials){const clean=170*s.clean/s.trials,assisted=170*s.assisted/s.trials;studySvg(svg,'rect',{x,y:220-clean,width:44,height:clean,fill:'#55c9ad'});studySvg(svg,'rect',{x,y:220-clean-assisted,width:44,height:assisted,fill:'#ffbb59'});studySvg(svg,'text',{x:x+22,y:232,fill:'#c5d9db','text-anchor':'middle','font-size':11},`${s.trials}/50`);}
 else studySvg(svg,'text',{x:x+22,y:180,fill:'#c5d9db','text-anchor':'middle','font-size':11},e.phase==='no_capture_found'?'none found':'pending');
 studySvg(svg,'text',{x:x+22,y:255,fill:'#c5d9db','text-anchor':'middle','font-size':13},Math.round(e.distance_m*1000)+' mm');
 }
 document.getElementById('distanceCatalogueDecision').textContent=r.phase==='completed'?'Search and frozen confirmation complete. Judge each distance by its fresh 50-trial results, not its best searched shot. No captured setting found means no solution found within this search—not proof of impossibility.':'Study incomplete. Completed distance rows are usable evidence; pending distances have no conclusion yet.';
 const details=document.getElementById('distanceCatalogueCandidates');details.replaceChildren();
 for(const e of entries){if(!e.screened.length)continue;const wrap=document.createElement('div'),h=document.createElement('h3');h.textContent=Math.round(e.distance_m*1000)+' mm candidate screening';wrap.appendChild(h);studyTable(wrap,['Board H / gap mm / tilt °','5 unchanged: captures','20 varied: clean / assisted / miss / invalid'],e.screened.map(s=>[`${n(s.candidate.height_m*1000)} / ${n(s.candidate.gap_m*1000)} / ${n(s.candidate.tilt_deg)}`,s.nominal_summary?`${s.nominal_summary.captures}/${s.nominal_summary.trials}`:'pending',s.perturbed_summary?`${s.perturbed_summary.clean} / ${s.perturbed_summary.assisted} / ${s.perturbed_summary.misses} / ${s.perturbed_summary.invalid} (${s.perturbed_summary.trials}/20)`:'pending']));details.appendChild(wrap);}
 document.getElementById('distanceCatalogueDetails').textContent=JSON.stringify({contract:r.contract,probes:r.distances.map(e=>({distance_m:e.distance_m,probes:e.probes})),verification:r.verification,artifact_sha256:r.artifact_sha256},null,2);
 document.getElementById('distanceCatalogueDownload').style.display=r.phase==='completed'?'block':'none';
}
async function distanceCatalogueUpdate(){try{const response=await fetch('/distance-catalogue.json',{cache:'no-store'});if(!response.ok)throw Error('unavailable');distanceCatalogueRender(await response.json());}catch(e){document.getElementById('distanceCatalogueStatus').textContent='No current distance catalogue report available';}}
setInterval(distanceCatalogueUpdate,5000);distanceCatalogueUpdate();
