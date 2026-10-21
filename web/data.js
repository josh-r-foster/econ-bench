(function (root) {
    'use strict';
    const finite = Number.isFinite;
    const mean = values => {
        const valid = values.filter(finite);
        return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null;
    };
    const rows = (data, key = 'trials') => Array.isArray(data?.[key]) ? data[key] : [];
    const percent = value => finite(value) ? value * 100 : null;
    const fieldMean = (data, key, field) => mean(rows(data, key).map(t => t[field]));
    const rate = (data, accepted, valid) => {
        const trials = rows(data).filter(t => valid.includes(t.decision));
        return trials.length ? 100 * trials.filter(t => t.decision === accepted).length / trials.length : null;
    };
    const experiments = [
        ['time', 'Time discounting', 'time-discounting'],
        ['independence', 'Risky choice', 'independence'],
        ['dictator', 'Dictator game', 'dictator-game'],
        ['ultimatum', 'Ultimatum game', 'ultimatum-game'],
        ['trust_game', 'Trust game', 'trust-game'],
        ['stag_hunt', 'Stag Hunt', 'stag-hunt-game'],
        ['beauty_contest', 'Beauty Contest', 'beauty-contest-game'],
        ['centipede_game', 'Centipede game', 'centipede-game'],
        ['public_goods', 'Public Goods', 'public-goods-game'],
        ['travellers_dilemma', "Traveller’s Dilemma", 'travellers-dilemma'],
        ['matching_pennies', 'Matching Pennies', 'matching-pennies-game'],
    ];

    function claimOnDisplayScale(t) {
        if (finite(t.claim_100_scale)) return t.claim_100_scale;
        if (finite(t.relative_claim)) return 2 + t.relative_claim * 98;
        if (finite(t.decision) && finite(t.low) && finite(t.high) && t.high > t.low) {
            return 2 + (t.decision - t.low) / (t.high - t.low) * 98;
        }
        return null;
    }

    // Each experiment remains separate. Missing observations never become zero.
    function metrics(data) {
        const m = data?.metrics || {};
        const social = data || {};
        const unfair = rows(social, 'ultimatum_responder').filter(t => finite(t.offer_percentage) && t.offer_percentage < 50 && ['ACCEPT', 'REJECT'].includes(t.decision));
        return {
            dictator: fieldMean(data, 'dictator_proposer', 'offer_percentage'),
            ultimatumOffer: fieldMean(data, 'ultimatum_proposer', 'offer_percentage'),
            unfairRejection: unfair.length ? 100 * unfair.filter(t => t.decision === 'REJECT').length / unfair.length : null,
            trust: percent(fieldMean(data, 'sender_trials', 'send_rate')),
            reciprocity: percent(fieldMean(data, 'receiver_trials', 'return_rate_of_received')),
            stag: rate(data, 'B', ['A', 'B']),
            beauty: mean(rows(data).map(t => t.decision)),
            pass: rate(data, 'PASS', ['PASS', 'TAKE']),
            contribution: percent(fieldMean(data, 'trials', 'contribution_pct')),
            claim: mean(rows(data).map(claimOnDisplayScale)),
            heads: rate(data, 'HEADS', ['HEADS', 'TAILS']),
            patience: m.patience?.discount_factor ?? null,
            riskError: m.risk?.error_rate ?? null,
        };
    }

    function hasData(experiment, data) {
        if (!data) return false;
        if (experiment === 'time') {
            const sets = Array.isArray(data.datasets) ? data.datasets.map(d => d.data) : Object.values(data.datasets || {});
            return sets.some(values => Array.isArray(values) && values.some(finite));
        }
        if (experiment === 'independence') return rows(data, 'results').some(r => finite(r.indifference_value) && ['X', 'Y'].includes(r.axis));
        if (experiment === 'dictator') return finite(metrics(data).dictator);
        if (experiment === 'ultimatum') return finite(metrics(data).ultimatumOffer) && finite(metrics(data).unfairRejection);
        if (experiment === 'trust_game') return finite(metrics(data).trust) && finite(metrics(data).reciprocity);
        const key = {stag_hunt: 'stag', beauty_contest: 'beauty', centipede_game: 'pass', public_goods: 'contribution', travellers_dilemma: 'claim', matching_pennies: 'heads'}[experiment];
        return key ? finite(metrics(data)[key]) : false;
    }

    function sourceFor(bundle, experiment) {
        if (experiment === 'dictator') return bundle.dictator || bundle.social;
        if (experiment === 'ultimatum') return bundle.ultimatum || bundle.social;
        return bundle[experiment];
    }

    async function loadCatalog() {
        const response = await fetch('catalog.json');
        if (!response.ok) throw new Error('The results catalog could not be loaded. Please reload the page.');
        return response.json();
    }

    async function loadModel(entry) {
        const safeId = EconBenchModelIds.toPathComponent(entry.id);
        const results = await Promise.all(Object.entries(entry.files).map(async ([kind, filename]) => {
            const prefix = kind === 'independence' ? 'independence_results' : `${kind}_experiment`;
            const expected = kind === 'rationality' ? `${safeId}_rationality.json` : `${prefix}_${safeId}.json`;
            if (filename !== expected) throw new Error(`Invalid result filename for ${entry.id}.`);
            const response = await fetch(`data/${filename}`);
            if (!response.ok) throw new Error(`Could not load ${kind} results for ${entry.id}. Please reload the page.`);
            return [kind, await response.json()];
        }));
        return Object.fromEntries(results);
    }

    const api = {finite, mean, rows, experiments, metrics, hasData, sourceFor, claimOnDisplayScale, loadCatalog, loadModel};
    root.EconBenchData = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
}(typeof globalThis !== 'undefined' ? globalThis : this));
