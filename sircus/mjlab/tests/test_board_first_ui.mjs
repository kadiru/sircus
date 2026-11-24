import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { boardFirstValidCounts as validCounts, boardFirstRescoreReady as rescoreReady, boardFirstComparisonReady as comparisonReady } from '../web/interactive-physics.mjs';
const provenance={report_sha256:'a'.repeat(64),audit_sha256:'b'.repeat(64)};
const counts={scorable_native:10,old_captures:4,new_captures:5,gained:2,lost:1,unchanged_captures:3,unchanged_misses:3,invalid_native:1};
test('rescoring separates gains, losses and invalid physics on one denominator',()=>{
 const r={status:'completed',audit_status:'passed',provenance,counts,positions:[{bucket_xy_m:[1,.025],counts}]};
 assert.equal(rescoreReady(r),true);
 assert.equal(rescoreReady({...r,audit_status:'pending'}),false);
 assert.equal(validCounts({...counts,new_captures:6}),false);
 assert.equal(rescoreReady({...r,positions:[]}),false);
 assert.equal(rescoreReady({...r,positions:[...r.positions,...r.positions]}),false);
});
test('only complete independently audited frozen50 comparisons display final capture rates',()=>{
 const m={checkpoint_sha256:'c'.repeat(64),attempts:50,captures:38,setup_rejections:2,released_misses:9,invalid_physics:1};
 const c={status:'completed',audit_status:'passed',provenance,positions:50,physics_substeps:64,frozen_before_test:true,methods:[{...m,id:'original'},{...m,id:'corrected'}]};
 assert.equal(comparisonReady(c),true);
 for(const change of [{frozen_before_test:false},{positions:49},{physics_substeps:128},{audit_status:'pending'}, {methods:[{...m,id:'original'},{...m,id:'original'}]}])assert.equal(comparisonReady({...c,...change}),false);
 assert.equal(comparisonReady({...c,methods:[c.methods[0],{...c.methods[1],captures:39}]}),false);
});
test('new panel is read-only and keeps old studies and both refresh calls',()=>{
 const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
 const app=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
 const source=readFileSync(new URL('../web/interactive-physics.mjs',import.meta.url),'utf8').split('// Read-only view.')[1];
 const panel=html.split('<section id="boardFirstStudy"')[1].split('</section>')[0];
 assert.doesNotMatch(panel,/<button|<input|<select/);
 assert.doesNotMatch(source,/fetch\(|innerHTML|physics\.drop|physics\.plan/);
 assert.match(panel,/Historical results retain their original definitions/);
 for(const id of ['clearShotRange','remainingLinkClearance','hybridMissVideoAssets','boardPoseVideoAssets'])assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
 assert.equal((app.match(/renderBoardFirst\(\$\('boardFirstStudy'\), report\?\.board_first_study\)/g)||[]).length,2);
});
