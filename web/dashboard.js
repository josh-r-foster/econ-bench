/* Tables contain observations from one experiment, with no imputed scores. */
(async function () {
    'use strict';
    const D = EconBenchData;
    const format = value => value.toFixed(1);
    const pct = value => `${format(value)}%`;
    const countField = (data, key, field) => D.rows(data, key).filter(t => D.finite(t[field])).length;
    const longestDelay = data => {
        const datasets = Array.isArray(data.datasets) ? data.datasets.map(d => d.data) : Object.values(data.datasets);
        let i = data.labels.length - 1;
        while (i >= 0 && !datasets.some(values => D.finite(values[i]))) i--;
        return {delay: data.labels[i], factor: D.mean(datasets.map(values => values[i])), count: datasets.flat().filter(D.finite).length};
    };
    const riskDeviation = data => D.mean(D.rows(data, 'results').filter(r => D.finite(r.indifference_value) && ['X', 'Y'].includes(r.axis)).map(r => {
        const ref = r.reference_point;
        const ev = ref.p_H + ref.p_M / 2;
        const predicted = Math.min(1, Math.max(0, r.axis === 'Y' ? ev * 2 - 1 : 1 - ev * 2));
        return Math.abs(r.indifference_value - predicted) * 100;
    }));
    const views = {
        time: {
            description: 'Discount factors at the longest observed delay, averaged across stake sizes. Higher values indicate greater patience. Patience alone does not establish rationality.',
            headers: ['Mean discount factor', 'Delay', 'Estimated points'],
            values: data => {const t = longestDelay(data); return [t.factor.toFixed(3), `${t.delay} ${data.benchmark_version ? 'days' : 'months'}`, t.count];},
        },
        independence: {
            description: 'Mean absolute deviation from a risk-neutral expected-value benchmark, in percentage points. Risk aversion can also produce deviations; this is not a pure test of rationality.',
            headers: ['Benchmark deviation', 'Indifference points'],
            values: data => [`${format(riskDeviation(data))} pp`, D.rows(data, 'results').filter(r => D.finite(r.indifference_value) && ['X', 'Y'].includes(r.axis)).length],
        },
        dictator: {
            description: 'Average share of the endowment offered to a passive recipient.',
            headers: ['Share offered', 'Offers'],
            values: data => [pct(D.metrics(data).dictator), countField(data, 'dictator_proposer', 'offer_percentage')],
        },
        ultimatum: {
            description: 'Average proposer offer and rejection rate for offers below an equal split. Responder choices are conditional vignettes, not realized bargaining.',
            headers: ['Share offered', 'Unequal offers rejected', 'Proposer / responder observations'],
            values: data => [pct(D.metrics(data).ultimatumOffer), pct(D.metrics(data).unfairRejection), `${countField(data, 'ultimatum_proposer', 'offer_percentage')} / ${D.rows(data, 'ultimatum_responder').filter(t => ['ACCEPT', 'REJECT'].includes(t.decision)).length}`],
        },
        trust_game: {
            description: 'Mean share sent by the sender and mean share of received funds returned by the receiver. The roles are elicited separately.',
            headers: ['Share sent', 'Share returned', 'Sender / receiver observations'],
            values: data => [pct(D.metrics(data).trust), pct(D.metrics(data).reciprocity), `${countField(data, 'sender_trials', 'send_rate')} / ${countField(data, 'receiver_trials', 'return_rate_of_received')}`],
        },
        stag_hunt: {
            description: 'Share of choices favoring the risky, payoff-dominant Stag option over the safe Hare option.',
            headers: ['Stag choices', 'Choices'],
            values: data => [pct(D.metrics(data).stag), D.rows(data).filter(t => ['A', 'B'].includes(t.decision)).length],
        },
        beauty_contest: {
            description: 'Mean guess in the 2/3-of-the-average game (0–100). The Nash benchmark is zero; proximity alone does not identify reasoning depth.',
            headers: ['Mean guess', 'Guesses'],
            values: data => [format(D.metrics(data).beauty), countField(data, 'trials', 'decision')],
        },
        centipede_game: {
            description: 'Share of stated choices to pass at isolated hypothetical nodes. These choices do not form a played game path.',
            headers: ['Pass choices', 'Choices'],
            values: data => [pct(D.metrics(data).pass), D.rows(data).filter(t => ['PASS', 'TAKE'].includes(t.decision)).length],
        },
        public_goods: {
            description: 'Average share of the endowment contributed to the shared pool.',
            headers: ['Share contributed', 'Contributions'],
            values: data => [pct(D.metrics(data).contribution), countField(data, 'trials', 'contribution_pct')],
        },
        travellers_dilemma: {
            description: 'Mean claim normalized to a common 2–100 scale across monetary levels. Two is the lower-bound Nash benchmark.',
            headers: ['Mean claim (2–100)', 'Claims'],
            values: data => [format(D.metrics(data).claim), D.rows(data).filter(t => D.finite(D.claimOnDisplayScale(t))).length],
        },
        matching_pennies: {
            description: 'Sampled choice balance. A rate near 50% alone does not establish a mixed strategy; consult the source records and condition-level uncertainty.',
            headers: ['Heads choices', 'Choices'],
            values: data => [pct(D.metrics(data).heads), D.rows(data).filter(t => ['HEADS', 'TAILS'].includes(t.decision)).length],
        },
    };
    const create = (tag, text, className) => {
        const node = document.createElement(tag);
        if (text !== undefined) node.textContent = text;
        if (className) node.className = className;
        return node;
    };
    const creator = model => model.startsWith('claude') ? 'Anthropic' : model.startsWith('gemini') ? 'Google' : /^(gpt|o\d)/.test(model) ? 'OpenAI' : 'Other';
    const status = document.getElementById('loadStatus');
    try {
        const catalog = await D.loadCatalog();
        const results = await Promise.all(catalog.models.map(async entry => ({id: entry.id, data: await D.loadModel(entry)})));
        const fragment = document.createDocumentFragment();
        const links = document.createDocumentFragment();
        const unpublished = [];
        let pairs = 0;
        let published = 0;
        const coverage = new Map(results.map(model => [model.id, 0]));
        for (const [experiment, title, anchor] of D.experiments) {
            const entries = results.map(model => ({id: model.id, data: D.sourceFor(model.data, experiment)})).filter(model => D.hasData(experiment, model.data));
            if (!entries.length) { unpublished.push(title); continue; }
            entries.sort((a, b) => creator(a.id).localeCompare(creator(b.id)) || a.id.localeCompare(b.id, undefined, {numeric: true}));
            published++;
            pairs += entries.length;
            entries.forEach(entry => coverage.set(entry.id, coverage.get(entry.id) + 1));
            const view = views[experiment];
            const section = create('section', undefined, 'result-section');
            section.id = anchor;
            section.append(create('h2', title), create('p', view.description));
            const wrapper = create('div', undefined, 'table-container');
            const table = create('table');
            table.append(create('caption', `${entries.length} ${entries.length === 1 ? 'model' : 'models'} with published observations · grouped by provider and model`));
            const header = create('tr');
            ['Creator', 'Model', ...view.headers].forEach(label => {
                const th = create('th', label); th.scope = 'col'; header.append(th);
            });
            const thead = create('thead'); thead.append(header); table.append(thead);
            const tbody = create('tbody');
            for (const entry of entries) {
                const tr = create('tr');
                tr.append(create('td', creator(entry.id)));
                const cell = create('td');
                const link = create('a', entry.id);
                link.href = `card.html?model=${encodeURIComponent(entry.id)}#${anchor}`;
                cell.append(link); tr.append(cell);
                view.values(entry.data).forEach(value => tr.append(create('td', String(value), 'numeric')));
                tbody.append(tr);
            }
            table.append(tbody); wrapper.append(table); section.append(wrapper); fragment.append(section);
            const jump = create('a', title); jump.href = `#${anchor}`; links.append(jump);
        }
        if (!published) throw new Error('No published observations are available yet.');
        document.getElementById('results').replaceChildren(fragment);
        const nav = document.getElementById('experimentNav'); nav.replaceChildren(links); nav.hidden = false;
        const summary = document.getElementById('coverageSummary');
        summary.textContent = `${results.length} models with result files · ${published} of 11 experiment types represented · ${pairs} model–experiment pairs. ${[...coverage.values()].filter(count => count === 11).length} models have complete coverage.`;
        summary.hidden = false;
        if (unpublished.length) {
            const note = document.getElementById('unpublished');
            note.textContent = `No observations published yet: ${unpublished.join(' and ')}.`; note.hidden = false;
        }
        status.hidden = true;
    } catch (error) {
        status.textContent = `${error.message} You can also browse the source data in the GitHub repository linked below.`;
        status.setAttribute('role', 'alert');
    }
}());
