import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {expandedSelectorReady} from '../web/interactive-physics.mjs';
const example=()=>({version:'sircus-expanded-selector-display-v1',status:'completed',audit_status:'passed',positions:50,physics_substeps:64,frozen_before_test:true,
 provenance:{report_sha256:'a'.repeat(64),audit_sha256:'b'.repeat(64)},
 library_rows:[{label:'Training',positions:100,settings:1000,capturing_settings:500},{label:'Validation',positions:20,settings:200,capturing_settings:50}],
 methods:['current','expanded'].map(id=>({id,checkpoint_sha256:'c'.repeat(64),attempts:50,captures:40,setup_rejections:5,released_misses:4,invalid_physics:1,mean_decision_s:20,mean_request_s:21})),
 results:Array.from({length:50},(_,i)=>({bucket_xy_m:[i/100,0],...Object.fromEntries(['current','expanded'].map(id=>[id,i<40?'capture':i<45?'setup refused':i<49?'flight miss':'invalid physics']))}))});
test('final rates require audited fifty-position native64 comparison and consistent per-position outcomes',()=>{
 assert.equal(expandedSelectorReady(example()),true);
 for(const change of [{status:'running'},{audit_status:'pending'},{positions:49},{physics_substeps:128},{frozen_before_test:false},{provenance:{}}]) assert.equal(expandedSelectorReady({...example(),...change}),false);
 const changed=example();changed.results[0].expanded='flight miss';assert.equal(expandedSelectorReady(changed),false);
 const duplicate=example();duplicate.results[1].bucket_xy_m=duplicate.results[0].bucket_xy_m;assert.equal(expandedSelectorReady(duplicate),false);
 const timing=example();timing.methods[0].mean_request_s=10;assert.equal(expandedSelectorReady(timing),false);
 assert.equal(expandedSelectorReady({...example(),library_rows:undefined}),false);
 const library=example();library.library_rows[0].capturing_settings=1001;assert.equal(expandedSelectorReady(library),false);
});
test('result panel is read-only and wired into both refresh paths',()=>{
 const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
 const app=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
 const panel=html.split('<section id="expandedSelectorStudy"')[1].split('</section>')[0];
 assert.doesNotMatch(panel,/<button|<input|<select/);
 assert.match(panel,/Later|later table or robot contacts/);
 for(const id of ['expandedSelectorStudy','boardFirstStudy','boardPoseVideoAssets'])assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
 assert.equal((app.match(/renderExpandedSelector\(\$\('expandedSelectorStudy'\), report\?\.expanded_selector_study\)/g)||[]).length,2);
});
