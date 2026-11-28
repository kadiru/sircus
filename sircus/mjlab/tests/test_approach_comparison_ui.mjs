import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {approachComparisonReady,renderApproachComparison} from '../web/interactive-physics.mjs';
const hash='a'.repeat(64);
function example(){
 const methods=['current','approach'].map(id=>({id,checkpoint_sha256:hash,attempts:50,releases:48,setup_rejections:2,captures:40,released_misses:7,invalid_physics:1,decision_s:{mean:10,median:10,p90:10},request_s:{mean:11,median:11,p90:11}}));
 const results=Array.from({length:50},(_,i)=>({id:`p${i}`,bucket_xy_m:[i/100,0],...Object.fromEntries(['current','approach'].map(id=>[id,{outcome:i<40?'capture':i<47?'flight miss':i===47?'invalid physics':'setup refused',decision_s:10,request_s:11}]))}));
 return {version:'sircus-approach-comparison-display-v1',status:'completed',audit_status:'passed',positions:50,physics_substeps:64,frozen_before_test:true,attempts_per_position:1,planning_budget_s:30,checks_count:100,
  provenance:{report_sha256:hash,audit_sha256:hash,contract_sha256:hash,manifest_sha256:hash},methods,results,paired:{both:40,current_only:0,approach_only:0,neither:10},native_releases:114,gpu_wait_s:300,gpu_lease_s:600,controls_passed:18,conclusion:'Keep live planner unchanged.',protocol:'Whole-method comparison, no retraining.'};
}
test('completed results require fifty matched frozen native64 attempts and consistent counts/times',()=>{
 assert.equal(approachComparisonReady(example()),true);
 for(const change of [{status:'running'},{audit_status:'pending'},{positions:49},{physics_substeps:128},{frozen_before_test:false},{attempts_per_position:2},{checks_count:0},{native_releases:100},{gpu_lease_s:0},{controls_passed:17}])assert.equal(approachComparisonReady({...example(),...change}),false);
 for(const change of [x=>x.results[0].approach.outcome='flight miss',x=>x.results[1].bucket_xy_m=x.results[0].bucket_xy_m,x=>x.methods[0].decision_s.mean=9,x=>x.methods[1].request_s.p90=10,x=>x.paired.both=39,x=>x.methods[0].checkpoint_sha256='',x=>x.results[0].current.request_s=9]){const x=example();change(x);assert.equal(approachComparisonReady(x),false);}
});
class Node {constructor(tag,doc){this.tagName=tag;this.ownerDocument=doc;this.children=[];this.textContent='';}append(...n){this.children.push(...n);}replaceChildren(...n){this.children=n;}querySelector(s){return this.slots[s.match(/="([^"]+)"/)[1]];}}
function root(){const d={createElement:t=>new Node(t,d)},r=new Node('section',d);r.slots=Object.fromEntries(['status','progress','outcomes','paired','times','timing-note','gpu','conclusion','positions','protocol','proof'].map(k=>[k,new Node('div',d)]));return r;}
function walk(n){return [n,...n.children.flatMap(walk)];}
test('display keeps decision/request/GPU times separate, safely renders text and preserves details during polls',()=>{
 const r=root(),x=example();x.conclusion='<script>untrusted</script>';renderApproachComparison(r,x);
 assert.match(r.slots.status.textContent,/50-position comparison complete/);assert.match(r.slots.gpu.textContent,/queue wait: 5.0 min.*lease: 10.0 min/);
 assert.match(r.slots.paired.textContent,/Both captured 40.*gained 0, lost 0/);assert.match(r.slots['timing-note'].textContent,/same 30-second/);
 assert.equal(r.slots.conclusion.textContent,'<script>untrusted</script>');assert.equal(walk(r.slots.times).filter(n=>n.tagName==='button').length,0);
 assert.ok(walk(r.slots.times).some(n=>n.textContent==='90th percentile full request'));
 const first=r.slots.outcomes.children[0];renderApproachComparison(r,structuredClone(x));assert.equal(r.slots.outcomes.children[0],first);
 renderApproachComparison(r,{...x,audit_status:'pending'});assert.equal(r.slots.outcomes.children.length,0);assert.equal(r.slots.paired.textContent,'');
});
test('queued view is explicitly a timestamped snapshot and displays no capture rates',()=>{
 const r=root();renderApproachComparison(r,{version:'sircus-approach-comparison-display-v1',status:'pending',phase:'waiting_for_4090',gpu_wait_s:65,observed_at_unix:1795795200,progress_text:'Prepared 100/100 planning requests.'});
 assert.match(r.slots.status.textContent,/Last reported/);assert.match(r.slots.proof.textContent,/not a live process check/);
 assert.match(r.slots.gpu.textContent,/1.1 min/);assert.equal(r.slots.outcomes.children.length,0);assert.equal(r.slots.times.children.length,0);
});
test('expired queue does not imply an active GPU waiter',()=>{
 const r=root();renderApproachComparison(r,{version:'sircus-approach-comparison-display-v1',status:'pending',phase:'waiting_for_4090',queue_expired:true,gpu_wait_s:65,observed_at_unix:1795795200});
 assert.match(r.slots.status.textContent,/expired before simulation started/);assert.equal(r.slots.outcomes.children.length,0);
});
test('additive read-only panel is refreshed twice and leaves existing media scripts intact',()=>{
 const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8'),app=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
 for(const id of ['approachComparison','bucketApproachStudy','expandedSelectorReplayVideoAssets','boardPoseVideoAssets'])assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
 const panel=html.split('<section id="approachComparison"')[1].split('</section>')[0];assert.doesNotMatch(panel,/<button|<input|<select/);assert.match(panel,/Later table, rim, robot or other rebounds are allowed/);
 assert.equal((app.match(/renderApproachComparison\(\$\('approachComparison'\), report\?\.approach_comparison\)/g)||[]).length,2);
 const source=readFileSync(new URL('../web/interactive-physics.mjs',import.meta.url),'utf8').split('// Additive result display:')[1];assert.doesNotMatch(source,/fetch\(|innerHTML|\.drop\(|\.plan\(/);
});
