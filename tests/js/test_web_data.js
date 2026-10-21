'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
global.EconBenchModelIds = require('../../web/model_ids.js');
const D = require('../../web/data.js');
const web = path.resolve(__dirname, '../../web');
const read = file => JSON.parse(fs.readFileSync(path.join(web, file), 'utf8'));

// Missing and invalid values never become competitive behavior or zero claims.
for (const [experiment] of D.experiments) assert.equal(D.hasData(experiment, {}), false, experiment);
assert.equal(D.metrics({}).stag, null);
assert.equal(D.metrics({trials: [{decision: 'invalid'}, {decision: null}]}).stag, null);
assert.equal(D.metrics({trials: [{decision: 'A'}, {decision: 'invalid'}]}).stag, 0);
assert.equal(D.metrics({ultimatum_responder: [{offer_percentage: 50, decision: 'REJECT'}]}).unfairRejection, null);
assert.equal(D.metrics({sender_trials: [{send_rate: null}, {send_rate: 0}]}).trust, 0);
assert.equal(D.mean([null, undefined, '0', NaN, Infinity]), null);
assert.equal(D.mean([0, null, 1]), 0.5);
assert.equal(D.claimOnDisplayScale({decision: null, low: 2, high: 100}), null);
assert.equal(D.claimOnDisplayScale({claim_100_scale: null, relative_claim: 0}), 2);
assert.equal(D.claimOnDisplayScale({decision: 600, low: 200, high: 1000}), 51);
assert.equal(D.claimOnDisplayScale({decision: 100, low: 2, high: 100}), 100);

// Canonical and prototype projections share the same observation-based metrics.
assert.equal(D.hasData('time', {datasets: [{label:'$10',data:[null, 0.6]}]}), true);
assert.equal(D.hasData('independence', {results:[{axis:'X',indifference_value:null}]}), false);
assert.equal(D.hasData('independence', {results:[{axis:'X',indifference_value:0}]}), true);
assert.equal(D.metrics({benchmark_version:'1.0.0', metrics:{overall_take_rate:1}, trials:[{decision:'TAKE'}]}).pass, 0);
assert.equal(D.metrics({benchmark_version:'1.0.0', trials:[{contribution_pct:0.25}]}).contribution, 25);
assert.equal(D.sourceFor({social:{dictator_proposer:[1]}}, 'dictator').dictator_proposer.length, 1);

const catalog = read('catalog.json');
let pairs = 0;
const counts = Object.fromEntries(D.experiments.map(([id]) => [id,0]));
for (const entry of catalog.models) {
    const bundle = Object.fromEntries(Object.entries(entry.files).map(([kind,file]) => [kind,read(`data/${file}`)]));
    let covered = 0;
    for (const [experiment] of D.experiments) {
        if (D.hasData(experiment, D.sourceFor(bundle, experiment))) {covered++; pairs++; counts[experiment]++;}
    }
    assert.ok(covered > 0, `${entry.id} must have observations`);
}
const traveller = read('data/travellers_dilemma_experiment_gpt-4o-mini.json');
assert.equal(D.metrics(traveller).claim, 51);
assert.equal(D.hasData('travellers_dilemma', traveller), true);
assert.equal(D.metrics(read('data/trust_game_experiment_gpt-4o-mini.json')).trust.toFixed(1), '83.3');
console.log(JSON.stringify({models:catalog.models.length,pairs,counts},null,2));
console.log('Website missing-data and projection regression checks passed.');

(async () => {
    const originalFetch = global.fetch;
    try {
        const calls = [];
        global.fetch = async url => {calls.push(url); return {ok:true,json:async()=>({trials:[{decision:0}]})};};
        const loaded = await D.loadModel({id:'test',files:{beauty_contest:'beauty_contest_experiment_test.json'}});
        assert.deepEqual(calls, ['data/beauty_contest_experiment_test.json']);
        assert.equal(D.metrics(loaded.beauty_contest).beauty, 0);
        await assert.rejects(D.loadModel({id:'test',files:{time:'../other.json'}}), /Invalid result filename/);
        global.fetch = async () => ({ok:false});
        await assert.rejects(D.loadCatalog(), /catalog could not be loaded/);
        await assert.rejects(D.loadModel({id:'test',files:{time:'time_experiment_test.json'}}), /Could not load time results/);
    } finally {global.fetch = originalFetch;}
    console.log('Catalog-only requests and load-error checks passed.');
})().catch(error => {console.error(error); process.exitCode = 1;});
