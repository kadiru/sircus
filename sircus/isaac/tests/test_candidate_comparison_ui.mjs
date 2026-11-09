import fs from 'node:fs';
import assert from 'node:assert/strict';
import test from 'node:test';

const source=fs.readFileSync(new URL('../candidate_comparison_panel.js',import.meta.url),'utf8');
const html=fs.readFileSync(new URL('../candidate_comparison.html',import.meta.url),'utf8');
const panel=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const fixture=()=>({schema:'sircus-candidate-native-comparison-report-v1',status:'passed',audit:{status:'passed',sha256:'a'.repeat(64)},source_binding:{checkpoint_sha256:'b'.repeat(64)},methods:{
  parent:{requests:35,planned_requests:35,captures:13,clean:4,rejected:7,released:28,compute:{wall_s:102.5,cpu_s:97.2,peak_rss_mib:7448.7,cuda_peak_allocated_mib:950,cuda_peak_reserved_mib:1200,geometry_s:2.1,selection_s:0.05}},
  bank:{requests:35,planned_requests:35,captures:16,clean:5,rejected:2,released:33,compute:{wall_s:108,cpu_s:105,peak_rss_mib:7455,cuda_peak_allocated_mib:970,cuda_peak_reserved_mib:1200,geometry_s:2.5,selection_s:0.1}}},
  paired:{gains:4,losses:1,changed_actions:7},compute:{initialization_shared_wall_s:55},preprocessing:{candidate_bank_wall_s:45.7,candidate_bank_audit_wall_s:117},
  cases:[{cell_id:'train-000',bucket_xy_m:[.2,-.1],action_changed:true,parent:{capture:false,clean:false,rejection:'endpoint_ik_unresolved_Arm14',action:{board_x_m:.4,board_y_m:-.2,height_m:.35,gap_m:.24,tilt_deg:29}},bank:{capture:true,clean:true,rejection:null,action:{board_x_m:.41,board_y_m:-.2,height_m:.35,gap_m:.24,tilt_deg:29}}}],
  scope:['Synthetic test fixture only. No measured performance.']});

class Element{
  constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.textContent='';this.listeners={};}
  appendChild(node){this.children.push(node);return node;}
  replaceChildren(...nodes){this.children=nodes;this.textContent='';}
  setAttribute(key,value){this.attributes[key]=value;}
  addEventListener(key,callback){this.listeners[key]=callback;}
}
function dom(){const nodes=new Map();globalThis.document={createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);}};return nodes;}
const text=node=>node.textContent+node.children.map(text).join(' ');

test('only complete internally consistent independently audited results are final',()=>{
  assert.equal(panel.resultsAudited(fixture()),true);
  for(const status of ['prepared','queued','running','native_complete','audit_pending','failed']){const report=fixture();report.status=status;assert.equal(panel.resultsAudited(report),false);assert.equal(panel.methodRows(report)[0][1],'pending audit');}
  for(const mutate of [r=>r.audit.status='pending',r=>r.methods.bank.requests=34,r=>r.methods.bank.clean=17,r=>r.methods.parent.released=29,r=>r.schema='legacy']){const report=fixture();mutate(report);assert.equal(panel.resultsAudited(report),false);}
});

test('queued and native-complete states distinguish execution from passed audit',()=>{
  assert.match(panel.statusView({...fixture(),status:'queued',queue:{reason:'A different project owns the GPU.'}}).detail,/different project/);
  assert.match(panel.statusView({...fixture(),status:'native_complete'}).title,/audit pending/);
  assert.match(panel.statusView({...fixture(),audit:{status:'pending'}}).title,/not verified/);
  assert.match(panel.statusView({schema:'unknown'}).title,/unavailable/);
});

test('requested position denominators survive rejection accounting',()=>{
  const rows=panel.methodRows(fixture());assert.match(rows[0][1],/13 \/ 35 · 37.1%/);assert.equal(rows[0][3],'7 / 35');assert.equal(rows[1][4],'33 / 35');
  assert.match(panel.pairedText(fixture()),/gains: 4.*losses: 1.*choices: 7 \/ 35/);
});

test('phase-separated resource values keep missing measurements unknown',()=>{
  const report=fixture();delete report.methods.bank.compute.peak_rss_mib;report.methods.bank.compute.cpu_s=-1;
  const rows=panel.computeRows(report);assert.equal(rows[0][1],'102.50 s');assert.equal(rows[0][3],'7,448.7 MiB');assert.equal(rows[1][2],'unknown');assert.equal(rows[1][3],'unknown');
  assert.equal(panel.computeRows({...report,status:'audit_pending'})[0][1],'pending audit');
});

test('DOM renders audited results and exact paired settings safely',()=>{
  const nodes=dom();try{const report=fixture();report.cases[0].parent.rejection='<script>never executable</script>';panel.render(report);
    assert.match(nodes.get('statusTitle').textContent,/completed/);assert.match(text(nodes.get('metrics')),/13 \/ 35.*16 \/ 35/);assert.equal(nodes.get('rateChart').children[0].tag,'svg');
    assert.match(text(nodes.get('caseDetail')),/<script>never executable<\/script>/);assert.match(text(nodes.get('caseDetail')),/Captured · clean/);assert.match(text(nodes.get('caseDetail')),/350.00 mm/);
    assert.match(text(nodes.get('sharedCompute')),/55.00 s.*45.70 s.*117.00 s/);assert.match(text(nodes.get('sourceIdentity')),/Audit:\s+passed/);
  }finally{delete globalThis.document;}
});

test('queued render hides previously visible final metrics and cases',()=>{
  const nodes=dom();try{panel.render(fixture());const report=fixture();report.status='queued';report.methods={parent:{requests:0,planned_requests:35},bank:{requests:0,planned_requests:35}};panel.render(report);
    assert.match(nodes.get('statusTitle').textContent,/Waiting/);assert.match(nodes.get('progress').textContent,/0 \/ 70/);assert.doesNotMatch(text(nodes.get('metrics')),/13 \/ 35/);assert.match(text(nodes.get('rateChart')),/Pending/);assert.match(text(nodes.get('caseDetail')),/pending audit/);
  }finally{delete globalThis.document;}
});

test('report fetch is GET-only and compatible page exposes all result sections',async()=>{
  const nodes=dom();try{let called;await panel.loadReport(async(url,options)=>{called={url,options};return {ok:true,json:async()=>fixture()};});assert.equal(called.url,'/report.json');assert.equal(called.options.cache,'no-store');assert.equal(called.options.method,undefined);
    await assert.rejects(panel.loadReport(async()=>({ok:false,status:404})),/Report HTTP 404/);
  }finally{delete globalThis.document;}
  assert.doesNotMatch(source,/innerHTML|WebSocket|XMLHttpRequest|method:\s*['"]POST/);
  for(const id of ['statusCard','metrics','rateChart','paired','computeTable','sharedCompute','caseList','caseDetail','sourceIdentity'])assert.ok(html.includes(`id="${id}"`));
  assert.match(html,/TRAIN positions/);assert.match(html,/No learning or held-out evaluation/);assert.match(html,/127.0.0.1:18797/);
});
