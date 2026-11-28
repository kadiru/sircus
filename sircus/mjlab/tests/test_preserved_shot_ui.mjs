import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {PRESERVED_SHOT_VERSION, preservedDiagnosisReady, preservedDevelopmentReady, preservedFreshReady, preservedPaired, renderPreservedShot} from '../web/interactive-physics.mjs';

const hash = 'a'.repeat(64);
const provenance = {audit_sha256:hash, manifest_sha256:hash, report_sha256:hash, contract_sha256:hash};
const ids = [1,16,28,32,33,42,43,44,46].map(n=>`approach-comparison-test-${String(n).padStart(2,'0')}`);
const layouts = () => ids.map((id,i)=>({id,bucket_xy_m:[i/10,-.4]}));
function diagnosis(){return {audit_status:'passed',checks_count:196747,provenance,positions:layouts().map(r=>({...r,current_checks:3,current_admitted:2,expanded_checks:16,expanded_local_checks:15,expanded_decision_s:4,rejections:[{reason:'finite_ik_unresolved',count:16}],current_geometry_rank:70}))};}
function development(){return {status:'completed',audit_status:'passed',physics_substeps:64,provenance,controls:{passed:18,total:18},positions:layouts().map(r=>({...r,current:{attempts:3,captures:2,releases:3},preserved:{attempts:3,captures:3,releases:3,used_fallback:3}}))};}
function fresh(){
 const results = Array.from({length:50},(_,i)=>({id:`fresh-${i}`,bucket_xy_m:[i/100,0],current:{outcome:i<40?'capture':i<48?'flight miss':'setup refused',request_s:12},preserved:{outcome:i<42?'capture':i<49?'flight miss':'setup refused',request_s:15,used_fallback:i<20}}));
 return {status:'completed',audit_status:'passed',provenance,positions:50,physics_substeps:64,frozen_before_test:true,attempts_per_position:1,checks_count:999,results,
 methods:[{id:'current',attempts:50,captures:40,releases:48,setup_rejections:2,released_misses:8,invalid_physics:0,request_s:{mean:12,median:12,p90:12}},
 {id:'preserved',attempts:50,captures:42,releases:49,setup_rejections:1,released_misses:7,invalid_physics:0,request_s:{mean:15,median:15,p90:15}}]};
}
const payload = () => ({version:PRESERVED_SHOT_VERSION,diagnosis:diagnosis(),development:development(),fresh:fresh(),observed_at_unix:1795795200});

test('historical diagnosis requires exact known cases, admitted parent choices and complete rejection totals',()=>{
 assert.equal(preservedDiagnosisReady(diagnosis()),true);
 for(const mutate of [x=>x.audit_status='pending',x=>x.checks_count=0,x=>x.positions[0].current_admitted=0,x=>x.positions[0].expanded_checks=15,x=>x.positions[0].rejections[0].count=15,x=>x.positions[0].expanded_decision_s=30,x=>x.positions[1].id=x.positions[0].id,x=>x.provenance={audit_sha256:hash,manifest_sha256:''}]){const x=diagnosis();mutate(x);assert.equal(preservedDiagnosisReady(x),false);}
});
test('development results require matched native64 triples and passed controls',()=>{
 assert.equal(preservedDevelopmentReady(development()),true);
 for(const mutate of [x=>x.status='running',x=>x.audit_status='pending',x=>x.physics_substeps=128,x=>x.controls.passed=17,x=>x.positions[0].preserved.attempts=2,x=>x.positions[0].preserved.captures=4,x=>x.positions[0].preserved.used_fallback=4]){const x=development();mutate(x);assert.equal(preservedDevelopmentReady(x),false);}
});
test('fresh comparison recomputes counts, paired results and actual request time',()=>{
 const f=fresh();assert.equal(preservedFreshReady(f),true);
 assert.deepEqual(preservedPaired(f.results),{both:40,current_only:0,preserved_only:2,neither:8,fallback:20});
 for(const mutate of [x=>x.status='running',x=>x.audit_status='pending',x=>x.positions=49,x=>x.physics_substeps=128,x=>x.frozen_before_test=false,x=>x.attempts_per_position=2,x=>x.methods[1].captures=45,x=>x.results[1].bucket_xy_m=x.results[0].bucket_xy_m,x=>x.results[0].current.request_s=9,x=>x.results[49].preserved.used_fallback=true,x=>x.methods[1].request_s.mean=12,x=>x.provenance={...provenance,contract_sha256:''}]){const x=fresh();mutate(x);assert.equal(preservedFreshReady(x),false);}
});

class Node {constructor(tag,doc){this.tagName=tag;this.ownerDocument=doc;this.children=[];this.textContent='';}append(...n){this.children.push(...n);}replaceChildren(...n){this.children=n;}querySelector(s){return this.slots[s.match(/="([^"]+)"/)[1]];}}
function root(){const doc={createElement:tag=>new Node(tag,doc)},r=new Node('section',doc);r.slots=Object.fromEntries(['status','progress','diagnosis-status','diagnosis','development-status','development','fresh-status','outcomes','paired','times','timing-note','positions','proof'].map(k=>[k,new Node('div',doc)]));return r;}
function nodes(n){return [n,...n.children.flatMap(nodes)];}
test('each section is independently gated; incomplete fresh results cannot hide verified diagnosis or repeats',()=>{
 const r=root(),x=payload();x.fresh.status='running';renderPreservedShot(r,x);
 assert.match(r.slots['diagnosis-status'].textContent,/135 of 144/);
 assert.match(r.slots['development-status'].textContent,/18\/18 controls passed/);
 assert.match(r.slots['fresh-status'].textContent,/await completed simulation/);
 assert.equal(r.slots.outcomes.children.length,0);assert.equal(r.slots.times.children.length,0);assert.equal(r.slots.paired.textContent,'');
 assert.equal(r.slots.diagnosis.children.length,1);assert.equal(r.slots.development.children.length,1);
 renderPreservedShot(r,{...x,development:{...x.development,audit_status:'pending'}});
 assert.equal(r.slots.development.children.length,0);assert.equal(r.slots.diagnosis.children.length,1);
});
test('completed display uses safe text, distinguishes measured request wall time, and keeps cached DOM stable',()=>{
 const r=root(),x=payload();x.progress_text='<img src=x onerror=alert(1)>';renderPreservedShot(r,x);
 assert.equal(r.slots.progress.textContent,x.progress_text);assert.match(r.slots.paired.textContent,/gained 2, lost 0.*20\/50/);
 assert.match(r.slots['timing-note'].textContent,/full request wall time/);assert.match(r.slots.proof.textContent,/not a live process check/);
 assert.ok(nodes(r.slots.times).some(n=>n.textContent==='15.00'));
 assert.equal(nodes(r.slots.outcomes).filter(n=>n.tagName==='button').length,0);
 const first=r.slots.outcomes.children[0];renderPreservedShot(r,structuredClone(x));assert.equal(r.slots.outcomes.children[0],first);
 renderPreservedShot(r,{...x,fresh:{...x.fresh,audit_status:'pending'}});assert.equal(r.slots.outcomes.children.length,0);assert.equal(r.slots.paired.textContent,'');
});
test('additive existing-module integration preserves media slots and contains no runtime controls',()=>{
 const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8'),app=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
 for(const id of ['preservedShot','approachComparison','expandedSelectorReplayVideoAssets','boardPoseVideoAssets'])assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
 const panel=html.split('<section id="preservedShot"')[1].split('</section>')[0];assert.doesNotMatch(panel,/<button|<input|<select/);assert.match(panel,/Later table, rim or robot rebounds are allowed/);
 assert.equal((app.match(/renderPreservedShot\(\$\('preservedShot'\), report\?\.preserved_shot\)/g)||[]).length,2);
 const source=readFileSync(new URL('../web/interactive-physics.mjs',import.meta.url),'utf8').split('// Preserve checked shots:')[1];assert.doesNotMatch(source,/fetch\(|innerHTML|\.drop\(|\.plan\(/);
 assert.doesNotMatch(html,/data:video/);
});
