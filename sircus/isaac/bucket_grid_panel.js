/* Read-only spatial feasibility study. No robot or training command API. */
let bucketGridChoice=null;
function bucketGridRender(r){
 const el=id=>document.getElementById(id), mm=v=>(v*1000).toFixed(0);
 const stale=r.timestamp_utc&&r.phase!=='completed'&&(Date.now()-Date.parse(r.timestamp_utc)>30000);
 el('bucketGridStatus').textContent=`${r.phase}${stale?' · updates stale':''} · ${r.attempts} attempts · ${r.run}`;
 const cells=r.contract.cells.map(c=>({...c,...r.cells.find(x=>x.id===c.id)}));
 studyTable(el('bucketGridTable'),['Bucket X / Y mm','Distance mm','Stage','Search board-only / attempts','Board H / gap mm / tilt °','Fresh board-only captures','Single-bounce / clean / assisted / invalid','Old-rule captures'],cells.map(c=>{
  const s=c.confirmation_summary,w=c.winner;
  return [`${mm(c.bucket_xy_m[0])} / ${mm(c.bucket_xy_m[1])}`,mm(c.distance_m),c.phase==='no_route_setting_found'?'no setting found':c.phase||'pending',c.search_summary?`${c.search_summary.route_captures}/${c.search_summary.trials}`:'pending',w?`${mm(w.height_m)} / ${mm(w.gap_m)} / ${w.tilt_deg.toFixed(1)}`:'—',s?`${s.route_captures}/${s.trials} (${s.trials}/30 tested)`:c.phase==='no_route_setting_found'?'none found':'pending',s?`${s.single_bounce_captures} / ${s.clean} / ${s.assisted} / ${s.invalid}`:'—',s?`${s.historical_captures}/${s.trials}`:'—'];
 }));
 const svg=el('bucketGridMap');svg.replaceChildren();
 const X=x=>60+(x-.35)*750,Y=y=>70+(-.1-y)*470;
 studySvg(svg,'rect',{x:15,y:35,width:510,height:360,fill:'#163238',stroke:'#496a70'});
 studySvg(svg,'text',{x:20,y:23,fill:'#b7d1d1','font-size':15},'Top view · X → · −Y ↓ · targets on table');
 studySvg(svg,'circle',{cx:X(.67),cy:Y(-.1),r:9,fill:'#68d4c6'});
 studySvg(svg,'text',{x:X(.67)+15,y:Y(-.1)+5,fill:'#68d4c6','font-size':14},'Board / release XY');
 for(const c of cells){
  const s=c.confirmation_summary;
  const color=s?(s.route_captures/s.trials>=.9?'#68d4c6':s.route_captures?'#ffbd69':'#ee8064'):c.phase==='no_route_setting_found'?'#ee8064':'#73878c';
  studySvg(svg,'circle',{cx:X(c.bucket_xy_m[0]),cy:Y(c.bucket_xy_m[1]),r:30,fill:color,opacity:.85});
  studySvg(svg,'text',{x:X(c.bucket_xy_m[0]),y:Y(c.bucket_xy_m[1])+4,fill:'#10262b','font-size':13,'text-anchor':'middle'},s?`${s.route_captures}/${s.trials}`:c.phase==='no_route_setting_found'?'none':'pending');
 }
 const buttons=el('bucketGridButtons');buttons.replaceChildren();
 const c=cells.find(c=>c.id===bucketGridChoice)||cells[0];bucketGridChoice=c.id;
 for(const row of cells){const b=document.createElement('button');b.textContent=`${mm(row.bucket_xy_m[0])}, ${mm(row.bucket_xy_m[1])}`;b.setAttribute('aria-pressed',String(c.id===row.id));b.onclick=()=>{bucketGridChoice=row.id;bucketGridRender(r);};buttons.appendChild(b);}
 el('bucketGridDetail').textContent=`Bucket (${mm(c.bucket_xy_m[0])}, ${mm(c.bucket_xy_m[1])}) mm; ${mm(c.distance_m)} mm from board, heading ${c.heading_deg.toFixed(1)}° from −Y. `+(c.winner?`Selected board center H ${mm(c.winner.height_m)} mm, tilt ${c.winner.tilt_deg.toFixed(1)}°, gap ${mm(c.winner.gap_m)} mm; release ball-bottom Z ${mm(c.winner.height_m+c.winner.gap_m)} mm. `:'No frozen setting yet. ')+(c.phase==='no_route_setting_found'?'This bounded search found no confirmed board-only setting; not proof of impossibility.':'');
 if(c.confirmation_summary?.trials){const s=c.confirmation_summary,n=s.trials,p=s.route_captures/n,z=1.96,den=1+z*z/n,mid=(p+z*z/(2*n))/den,half=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;el('bucketGridDetail').textContent+=` Fresh confirmation ${s.route_captures}/${n}; approximate 95% Wilson interval ${((mid-half)*100).toFixed(1)}–${((mid+half)*100).toFixed(1)}%. Applies only to these small variations.`;}
 el('bucketGridProvenance').textContent=JSON.stringify({contract:r.contract,probes:r.probes,controls:r.controls,verification:r.verification},null,2);
 el('bucketGridDownload').style.display=r.phase==='completed'&&r.verification?'block':'none';
}
async function bucketGridUpdate(){try{const response=await fetch('/bucket-grid.json',{cache:'no-store'});if(!response.ok)throw Error('unavailable');bucketGridRender(await response.json());}catch(e){document.getElementById('bucketGridStatus').textContent='No spatial grid report available yet';}}
setInterval(bucketGridUpdate,5000);bucketGridUpdate();
