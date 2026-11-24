import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bucketApproachReady,renderBucketApproach} from '../web/interactive-physics.mjs';
const hash='a'.repeat(64);
const setting=()=>({values:[.5,0,.35,28,.2],q10_rad:Array(6).fill(0),q14_rad:Array(6).fill(0),release_ball_center_m:[.5,0,.5875],heading_offset_deg:0,twist_deg:0});
const forecast=()=>({summary:'Forecast clears rising approach',limits:'One board and one table rebound only',collision:false,first_collision_s:null,minimum_margin_m:.01,opening_margin_m:.02});
const counts=(captures=0)=>({requested:3,completed:3,released:3,captures,flight_misses:3-captures,setup_rejections:0,invalid_physics:0,robot_contacts:0,bucket_contacts:3,settings:setting(),forecast:forecast()});
const example=()=>({version:'sircus-bucket-approach-display-v1',status:'completed',audit_status:'passed',scope:'development',physics_substeps:64,frozen_before_native:true,
 provenance:{report_sha256:hash,audit_sha256:hash,preparation_sha256:hash},checks_count:500,native_releases:51,
 controls:{controls_pre:{status:'passed',requested:9},controls_post:{status:'passed',requested:9}},
 cases:['04','10','19'].map((id,i)=>({id:`expanded-selector-test-${id}`,bucket_xy_m:[.6+i/10,0],original_method:i===1?'current':'expanded',methods:{original:counts(),corrected:counts(3)}})),
 diagnostic_controls:[{id:'47-current',bucket_xy_m:[1.17,.21],...counts(0)}],
 preservation:['04-current','10-expanded','19-current','47-expanded'].map(id=>({id,bucket_xy_m:[.5,0],...counts(3)})),
 conclusion:'Known-case improvement; no live promotion.',limits:'Held endpoints; no new policy accuracy.'});
test('only audited complete matched native64 drops produce final rates',()=>{
 assert.equal(bucketApproachReady(example()),true);
 for(const change of [{audit_status:'pending'},{status:'running'},{scope:'test'},{physics_substeps:128},{frozen_before_native:false},{checks_count:0},{native_releases:47},{provenance:{}}])assert.equal(bucketApproachReady({...example(),...change}),false);
 for(const change of [x=>x.cases.pop(),x=>x.cases[0].methods.original.captures=1,x=>x.cases[0].methods.corrected.settings.q10_rad=[NaN],x=>x.preservation[1].id=x.preservation[0].id,x=>x.controls.controls_post.status='failed',x=>x.cases[0].methods.corrected.forecast.minimum_margin_m=NaN]){const x=example();change(x);assert.equal(bucketApproachReady(x),false);}
});
test('withheld corrections count as attempted failures; preservation regressions remain visible',()=>{
 const x=example(),m=x.cases[0].methods.corrected;
 Object.assign(m,{released:0,captures:0,flight_misses:0,setup_rejections:3,robot_contacts:0,bucket_contacts:0,settings:null});x.native_releases-=3;
 assert.equal(bucketApproachReady(x),true);
 x.preservation[0].captures=2;x.preservation[0].flight_misses=1;assert.equal(bucketApproachReady(x),true);
});
class Node {constructor(tag,doc){this.tagName=tag;this.ownerDocument=doc;this.children=[];this.textContent='';}append(...n){this.children.push(...n);}replaceChildren(...n){this.children=n;}querySelector(s){return this.slots[s.match(/="([^"]+)"/)[1]];}}
function root(){const d={createElement:t=>new Node(t,d)},r=new Node('section',d);r.slots=Object.fromEntries(['status','summary','cases','preservation','diagnostic','controls','conclusion','proof','limits'].map(k=>[k,new Node('div',d)]));return r;}
function walk(n){return [n,...n.children.flatMap(walk)];}
test('read-only rendering separates forecast and native outcomes and preserves open details during polls',()=>{
 const r=root(),x=example();x.cases[0].methods.corrected.forecast.summary='<script>untrusted</script>';
 renderBucketApproach(r,x);assert.match(r.slots.summary.textContent,/unchanged shots 0\/9.*selection 9\/9/);
 assert.match(r.slots.controls.textContent,/12\/12.*18.*51.*500/);
 assert.ok(walk(r.slots.cases).some(n=>n.textContent==='<script>untrusted</script>'));
 assert.equal(walk(r.slots.cases).filter(n=>n.tagName==='button').length,0);
 const first=r.slots.cases.children[0];renderBucketApproach(r,structuredClone(x));assert.equal(r.slots.cases.children[0],first);
 renderBucketApproach(r,{...x,audit_status:'pending'});assert.equal(r.slots.cases.children.length,0);assert.equal(r.slots.summary.textContent,'');
});
test('panel is additive and wired into both refresh paths without planning calls',()=>{
 const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8'),app=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
 for(const id of ['bucketApproachStudy','expandedSelectorReplays','expandedSelectorReplayVideoAssets','expandedSelectorStudy'])assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
 const panel=html.split('<section id="bucketApproachStudy"')[1].split('</section>')[0];assert.doesNotMatch(panel,/<button|<input|<select/);assert.match(panel,/table, rim and robot bounces remain allowed/);
 assert.equal((app.match(/renderBucketApproach\(\$\('bucketApproachStudy'\), report\?\.bucket_approach_study\)/g)||[]).length,2);
 const source=readFileSync(new URL('../web/interactive-physics.mjs',import.meta.url),'utf8').split('// Read-only native64 diagnostic;')[1];assert.doesNotMatch(source,/fetch\(|innerHTML|\.drop\(|\.plan\(/);
});
