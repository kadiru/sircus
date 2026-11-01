/* Read-only fixed-plane development results; no physical controls. */
let planeReport = null, planeReplay = 0, planeFrame = 0;
function planeDraw() {
 const r=planeReport;if(!r)return;
 const c=r.replays?.[planeReplay]?.candidate || r.rows?.find(x=>x.success) || {height_m:.4,gap_m:.16,tilt_deg:22};
 const path=r.replays?.[planeReplay]?.trajectory_m || [];
 const svg=document.getElementById('planeDiagram');svg.replaceChildren();
 const X=d=>170+d*500,Z=h=>435-h*500;
 const line=(x1,y1,x2,y2,color,width=2,dash='')=>studySvg(svg,'line',{x1,y1,x2,y2,stroke:color,'stroke-width':width,'stroke-dasharray':dash});
 const label=(x,y,t,color='#c5d9db')=>studySvg(svg,'text',{x,y,fill:color,'font-size':14},t);
 line(30,Z(0),650,Z(0),'#c5d9db');label(35,460,'Table · Z=0');
 line(X(0),Z(.77),X(0),Z(0),'#65d2d3',2,'6 5');label(35,32,'Transparent release plane');
 const bucket=r.contract.scene.bucket, top=bucket.assumed_bottom_thickness_m+bucket.assumed_interior_depth_m;
 const d=r.contract.distance_m,rt=bucket.inner_top_diameter_m/2,rb=bucket.inner_bottom_diameter_m/2;
 line(X(d-rt),Z(top),X(d-rb),Z(bucket.assumed_bottom_thickness_m),'#ffca76',4);line(X(d+rt),Z(top),X(d+rb),Z(bucket.assumed_bottom_thickness_m),'#ffca76',4);line(X(d-rb),Z(bucket.assumed_bottom_thickness_m),X(d+rb),Z(bucket.assumed_bottom_thickness_m),'#ffca76',4);
 label(X(d)-60,Z(top)-15,'Bucket opening');
 const a=c.tilt_deg*Math.PI/180,half=(r.contract.scene.board?.size_m?.[0] || .2)/2;
 line(X(-half*Math.cos(a)),Z(c.height_m+half*Math.sin(a)),X(half*Math.cos(a)),Z(c.height_m-half*Math.sin(a)),'#66ddd0',7);
 const radius=r.contract.scene.ball.radius_m, release=c.height_m+c.gap_m+radius;
 studySvg(svg,'circle',{cx:X(0),cy:Z(release),r:radius*500,fill:'none',stroke:'#ff8b62','stroke-width':2});
 label(X(0)+28,Z(release),'Release');label(35,Z(c.height_m)+27,'Board H='+Math.round(c.height_m*1000)+' mm');
 line(X(0),Z(.025),X(d),Z(.025),'#a2b7ed',2);label(X(.10),Z(.025)-8,'D = '+Math.round(d*1000)+' mm toward −Y');
 if(path.length){
 const xy=path.map(p=>[X(r.contract.board_xy_m[1]-p[1]),Z(p[2])]);
 studySvg(svg,'polyline',{points:xy.map(p=>p.join(',')).join(' '),stroke:'#ff8b62','stroke-width':2,fill:'none'});
 const p=xy[planeFrame%xy.length];studySvg(svg,'circle',{cx:p[0],cy:p[1],r:radius*500,fill:'#ff8b62',opacity:.8});
 }
 label(35,490,'Side view: −Y forward, Z up. Ball path measured in PhysX; arms omitted.');
 document.getElementById('planeShot').textContent=`H ${Math.round(c.height_m*1000)} mm · gap ${Math.round(c.gap_m*1000)} mm · tilt ${c.tilt_deg}°`+(path.length?' · fresh replay: '+(r.replays[planeReplay].result.success?'capture':'miss'):' · schematic, replay pending');
}
function planeRender(r) {
 planeReport=r;const status=document.getElementById('planeStatus');
 status.textContent=`${r.phase} · ${r.completed}/${r.planned} grid combinations · ${r.captures} captures · ${r.invalid} rejected · ${r.run}`;
 document.getElementById('planeDecision').textContent=r.conclusion||(r.phase==='completed'?'Development search complete. Selected repeats establish same-scene repeatability only; no learned policy or distance generalization yet.':'Experiment incomplete; no feasibility conclusion yet.');
 const root=document.getElementById('planeHeatmaps');root.replaceChildren();
 for(const h of r.contract.heights_m){
 const wrap=document.createElement('div'),title=document.createElement('h3');title.textContent=`Board height ${Math.round(h*1000)} mm`;wrap.appendChild(title);
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 390 245');svg.setAttribute('role','img');svg.setAttribute('aria-label',title.textContent+' gap versus tilt');wrap.appendChild(svg);
 for(const [j,g] of r.contract.gaps_m.entries()){
 studySvg(svg,'text',{x:42,y:42+j*35,fill:'#c5d9db','font-size':12,'text-anchor':'end'},Math.round(g*1000));
 for(const [i,t] of r.contract.tilts_deg.entries()){
 const v=r.rows.find(c=>c.height_m===h&&c.gap_m===g&&c.tilt_deg===t);
 const fill=!v?'#294047':v.invalid?'#74828a':v.success?'#55c9ad':'#bc7862';
 studySvg(svg,'rect',{x:55+i*42,y:22+j*35,width:37,height:30,fill});
 studySvg(svg,'text',{x:73+i*42,y:42+j*35,fill:'#10262b','font-size':12,'text-anchor':'middle'},!v?'':v.invalid?'×':v.success?'✓':'·');
 if(j===0)studySvg(svg,'text',{x:73+i*42,y:215,fill:'#c5d9db','font-size':12,'text-anchor':'middle'},t+'°');
 }}studySvg(svg,'text',{x:55,y:240,fill:'#c5d9db','font-size':12},'Rows: gap (mm) · columns: board tilt');root.appendChild(wrap);
 }
 studyTable(document.getElementById('planeRepeats'),['Combination','Fresh repeats','Invalid'],(r.repeats||[]).map(e=>[`${Math.round(e.candidate.height_m*1000)} mm / ${Math.round(e.candidate.gap_m*1000)} mm / ${e.candidate.tilt_deg}°`,e.trials.filter(x=>x.success).length+'/'+e.trials.length,e.trials.filter(x=>x.invalid).length]));
 const buttons=document.getElementById('planeReplayButtons');buttons.replaceChildren();
 for(const [i] of (r.replays||[]).entries()){const b=document.createElement('button');b.textContent='View measured shot '+(i+1);b.onclick=()=>{planeReplay=i;planeFrame=0;planeDraw();};buttons.appendChild(b);}
 const img=document.getElementById('planeSetup');img.style.display=r.replays?.length?'block':'none';if(r.replays?.length)img.src='/plane-setup.jpg?run='+encodeURIComponent(r.run);
 document.getElementById('planeDetails').textContent=JSON.stringify({contract:r.contract,pose_summary:r.pose_summary,artifact_sha256:r.artifact_sha256},null,2);
 planeDraw();
}
async function planeUpdate(){try{planeRender(await(await fetch('/plane-sweep.json',{cache:'no-store'})).json());}catch(e){document.getElementById('planeStatus').textContent='No current fixed-plane report available';}}
setInterval(()=>{planeFrame++;if(planeReport?.replays?.length)planeDraw();},1000/30);
setInterval(planeUpdate,5000);planeUpdate();
