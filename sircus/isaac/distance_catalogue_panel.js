/* Read-only distance search and frozen confirmation. Never issues motion commands. */
let distanceSetupChoice=null;
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
 if(r.route_audit){document.getElementById('distanceRouteWarning').textContent='Trajectory warning — captured trials with a low external rebound: '+r.route_audit.distances.filter(e=>e.confirmed_captures).sort((a,b)=>a.distance_m-b.distance_m).map(e=>`${Math.round(e.distance_m*1000)} mm: ${e.low_external_rebound_flags}/${e.confirmed_captures}${e.missing_capture_paths?` (missing paths: ${e.missing_capture_paths})`:""}`).join('; ')+'. Likely table/other contact; no table contact sensor was recorded. Absence of a flag does not certify direct travel. Original scores are unchanged.';}else{document.getElementById('distanceRouteWarning').textContent='Direct board-to-bucket travel is not certified: intermediate contacts are permitted, and the saved-path audit is not yet available for this report.';}
 distanceSetupRender(r);
 document.getElementById('distanceCatalogueDetails').textContent=JSON.stringify({contract:r.contract,probes:r.distances.map(e=>({distance_m:e.distance_m,probes:e.probes})),verification:r.verification,route_audit:r.route_audit,artifact_sha256:r.artifact_sha256},null,2);
 document.getElementById('distanceCatalogueDownload').style.display=r.phase==='completed'&&r.verification?'block':'none';
}
async function distanceCatalogueUpdate(){try{const response=await fetch('/distance-catalogue.json',{cache:'no-store'});if(!response.ok)throw Error('unavailable');distanceCatalogueRender(await response.json());}catch(e){document.getElementById('distanceCatalogueStatus').textContent='No current distance catalogue report available';}}
setInterval(distanceCatalogueUpdate,5000);distanceCatalogueUpdate();

function distanceSetupRender(r) {
 const available=r.distances.filter(e=>e.winner).sort((a,b)=>a.distance_m-b.distance_m);
 const buttons=document.getElementById('distanceSetupButtons');buttons.replaceChildren();
 const svg=document.getElementById('distanceSetupDiagram');svg.replaceChildren();
 const caption=document.getElementById('distanceSetupCaption');
 const cfg=r.contract.parent_quality?.parent?.scene;
 if(!available.length||!cfg){caption.textContent='Waiting for a selected setting and matching scene geometry.';return;}
 const e=available.find(e=>e.distance_m===distanceSetupChoice)||available[0];distanceSetupChoice=e.distance_m;
 for(const row of available){const b=document.createElement('button');b.textContent=Math.round(row.distance_m*1000)+' mm';b.setAttribute('aria-pressed',String(row===e));b.style.cssText='margin:4px;padding:9px 14px;color:#f9f3df;border:1px solid #65b8b1;border-radius:6px;background:'+(row===e?'#376169':'#24474c');b.onclick=()=>{distanceSetupChoice=row.distance_m;distanceSetupRender(r);};buttons.appendChild(b);}
 const c=e.winner,scale=.65,origin=160,base=555,mm=v=>v*1000;
 const X=v=>origin+mm(v)*scale,Z=v=>base-mm(v)*scale;
 const line=(x1,y1,x2,y2,color,dash)=>studySvg(svg,'line',{x1,y1,x2,y2,stroke:color,'stroke-width':2,...(dash?{'stroke-dasharray':dash}:{})});
 const label=(x,y,text,color='#c5d9db')=>studySvg(svg,'text',{x,y,fill:color,'font-size':17},text);
 line(50,base,805,base,'#8aafb4');label(55,base+26,'Tabletop Z = 0');
 line(origin,25,origin,base,'#58777c','6 5');label(35,20,'Fixed release plane');
 const a=c.tilt_deg*Math.PI/180,half=cfg.board.size_m[0]/2;
 line(X(-half*Math.cos(a)),Z(c.height_m+half*Math.sin(a)),X(half*Math.cos(a)),Z(c.height_m-half*Math.sin(a)),'#55c9ad');
 const normal=.1;line(origin,Z(c.height_m),X(normal*Math.sin(a)),Z(c.height_m+normal*Math.cos(a)),'#55c9ad');
 label(X(.13),Z(c.height_m)+5,`Board H ${mm(c.height_m).toFixed(1)} mm · tilt ${c.tilt_deg}°`,'#55c9ad');
 const radius=cfg.ball.radius_m,center=c.height_m+c.gap_m+radius;
 studySvg(svg,'circle',{cx:origin,cy:Z(center),r:mm(radius)*scale,fill:'#ef795c'});
 line(origin,Z(center-radius),origin,Z(c.height_m),'#ef795c','4 4');
 label(X(.09),Z(center)-15,`Ball-center Z ${mm(center).toFixed(1)} mm`,'#ffab92');
 label(X(.09),Z(center)+10,`Bottom Z ${mm(center-radius).toFixed(1)} mm · gap ${mm(c.gap_m).toFixed(1)} mm`);
 const b=cfg.bucket,top=b.assumed_interior_depth_m+b.assumed_bottom_thickness_m,d=e.distance_m;
 const points=[[d-b.outer_top_diameter_m/2,top],[d-b.outer_bottom_diameter_m/2,0],[d+b.outer_bottom_diameter_m/2,0],[d+b.outer_top_diameter_m/2,top]];
 studySvg(svg,'polyline',{points:points.map(([x,z])=>X(x)+','+Z(z)).join(' '),fill:'none',stroke:'#ffbb59','stroke-width':4});
 line(X(d-b.inner_top_diameter_m/2),Z(top),X(d+b.inner_top_diameter_m/2),Z(top),'#ffbb59','4 4');
 label(X(d)-90,Z(top)-15,'177 mm opening','#ffbb59');
 line(origin,base+52,X(d),base+52,'#c5d9db');line(origin,base+44,origin,base+60,'#c5d9db');line(X(d),base+44,X(d),base+60,'#c5d9db');
 label(origin+15,base+79,`Horizontal plane-to-bucket-center distance: ${mm(d).toFixed(0)} mm (toward −Y)`);
 caption.textContent='Geometry schematic of nominal selected targets, not a trajectory or camera measurement. Bucket uses the provisional 175 mm interior depth + 8 mm base; reported-height interpretation remains unresolved. Arms and travel paths are omitted.';
}
