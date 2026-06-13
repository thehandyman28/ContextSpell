async function checkText() {
    const text = document.getElementById('inputText').value.trim();
    if (!text) { alert('Please enter some text.'); return; }

    document.getElementById('loadingMsg').style.display     = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('noErrors').style.display       = 'none';

    try {
        const response = await fetch('/check', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ text })
        });
        const data = await response.json();
        document.getElementById('loadingMsg').style.display = 'none';

        if (data.ContextSpell.length === 0 && data.hunspell.length === 0) {
            document.getElementById('noErrors').style.display = 'block';
            return;
        }

        renderResults(text, data.ContextSpell, data.hunspell);

    } catch (err) {
        document.getElementById('loadingMsg').style.display = 'none';
        alert('Error — make sure the server is running.');
    }
}

function renderResults(originalText, ContextSpellResults, hunspellResults) {

    document.getElementById('ContextSpellHighlighted').innerHTML =
        buildHighlightedText(originalText, ContextSpellResults, 'ContextSpell');

    document.getElementById('hunspellHighlighted').innerHTML =
        buildHighlightedText(originalText, hunspellResults, 'hunspell');

    const ourEl = document.getElementById('ContextSpellResults');
    ourEl.innerHTML = '';
    ContextSpellResults.forEach(r => ourEl.appendChild(buildContextSpellCard(r)));  // ← fixed

    const hunspellEl = document.getElementById('hunspellResults');
    hunspellEl.innerHTML = '';
    hunspellResults.forEach(r => hunspellEl.appendChild(buildHunspellCard(r)));

    buildSummary(ContextSpellResults, hunspellResults);

    document.getElementById('resultsSection').style.display = 'block';
}

function buildHighlightedText(text, results, system) {
    let html = text;
    const sorted = [...results].sort((a, b) => b.word.length - a.word.length);
    sorted.forEach(r => {
        const cls     = system === 'hunspell' ? 'highlight-hunspell'
                      : r.type === 'context'  ? 'highlight-context'
                      : 'highlight-unknown';
        const escaped = r.word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const tip     = r.suggestions.join(', ');
        html = html.replace(
            new RegExp(`\\b${escaped}\\b`, 'g'),
            `<span class="${cls}" title="${tip}">${r.word}</span>`
        );
    });
    return html;
}

function buildContextSpellCard(r) {
    const card = document.createElement('div');
    card.className = `card ${r.type}`;

    const typeLabel = r.type === 'context'
        ? 'Context error — wrong word for this position'
        : 'Unknown word — spelling error';

    const pillsHtml = r.suggestions
        .map(s => `<span class="sugg-pill">${s}</span>`)
        .join('');

    const s = r.scores;
    const w = r.weights;

    const freqPct     = Math.min(100, Math.round((s.frequency || 0) * 10000));
    const bigramPct   = s.bigram > 0 ? Math.min(100, Math.round(s.bigram / 10)) : 0;
    const combinedPct = Math.min(100, Math.round((s.combined || 0) * 100));

    let candidatesHtml = '';
    if (r.all_candidates && r.all_candidates.length > 0) {
        const rows = r.all_candidates.map(c =>
            `<tr>
                <td>${c.word}</td>
                <td>${c.distance}</td>
                <td>${c.score}</td>
            </tr>`
        ).join('');
        candidatesHtml = `
            <table class="candidates-table">
                <tr>
                    <th>Candidate</th>
                    <th>Distance</th>
                    <th>Score</th>
                </tr>
                ${rows}
            </table>`;
    }

    const uid = Math.random().toString(36).slice(2);

    card.innerHTML = `
        <div class="card-type">${typeLabel}</div>
        <div class="card-word">${r.word}</div>
        <div class="sugg-row">${pillsHtml}</div>
        <div class="scores-section">
            <button class="scores-toggle" onclick="toggleScores('${uid}')">
                Show scoring details
            </button>
            <div class="scores-detail" id="scores-${uid}">

                <div class="weight-label">
                    Weights: frequency ${w.frequency} &nbsp;|&nbsp; bigram context ${w.bigram}
                </div>

                <div class="score-row">
                    <span class="score-label">Global frequency</span>
                    <span class="score-value">${(s.frequency || 0).toExponential(2)}</span>
                </div>
                <div class="score-bar-wrap">
                    <div class="score-bar freq" style="width:${freqPct}%"></div>
                </div>

                <div class="score-row">
                    <span class="score-label">Bigram context score</span>
                    <span class="score-value">${s.bigram || 0}</span>
                </div>
                <div class="score-bar-wrap">
                    <div class="score-bar bigram" style="width:${bigramPct}%"></div>
                </div>

                <div class="score-row">
                    <span class="score-label">Combined score</span>
                    <span class="score-value">${s.combined || 0}</span>
                </div>
                <div class="score-bar-wrap">
                    <div class="score-bar combined" style="width:${combinedPct}%"></div>
                </div>

                ${r.type === 'unknown' ? candidatesHtml : ''}
            </div>
        </div>`;

    return card;
}

function buildHunspellCard(r) {
    const card = document.createElement('div');
    card.className = 'card hunspell-card';
    const pillsHtml = r.suggestions
        .map(s => `<span class="sugg-pill">${s}</span>`)
        .join('');
    card.innerHTML = `
        <div class="card-type">Unknown word — dictionary lookup</div>
        <div class="card-word">${r.word}</div>
        <div class="sugg-row">${pillsHtml}</div>`;
    return card;
}

function toggleScores(uid) {
    const el  = document.getElementById(`scores-${uid}`);
    const btn = el.previousElementSibling;
    if (el.style.display === 'block') {
        el.style.display = 'none';
        btn.textContent  = 'Show scoring details';
    } else {
        el.style.display = 'block';
        btn.textContent  = 'Hide scoring details';
    }
}

function buildSummary(ContextSpell, hunspell) {
    const ContextSpellWords = new Set(ContextSpell.map(r => r.word.toLowerCase()));
    const hunspellWords     = new Set(hunspell.map(r => r.word.toLowerCase()));

    const onlyContextSpell = [...ContextSpellWords].filter(w => !hunspellWords.has(w)).length;       // ← fixed
    const onlyHunspell     = [...hunspellWords].filter(w => !ContextSpellWords.has(w)).length;       // ← fixed
    const both             = [...ContextSpellWords].filter(w => hunspellWords.has(w)).length;        // ← fixed
    const contextOnly      = ContextSpell.filter(r => r.type === 'context').length;                  // ← fixed

    document.getElementById('summaryRow').innerHTML = `
        <div class="summary-item">
            <span class="summary-num">${ContextSpell.length}</span>
            <span class="summary-lbl">Our system flagged</span>
        </div>
        <div class="summary-item">
            <span class="summary-num">${hunspell.length}</span>
            <span class="summary-lbl">Hunspell flagged</span>
        </div>
        <div class="summary-item">
            <span class="summary-num">${both}</span>
            <span class="summary-lbl">Flagged by both</span>
        </div>
        <div class="summary-item">
            <span class="summary-num" style="color:#3498db">${onlyContextSpell}</span>   <!-- ← fixed -->
            <span class="summary-lbl">Only our system</span>
        </div>
        <div class="summary-item">
            <span class="summary-num" style="color:#27ae60">${onlyHunspell}</span>
            <span class="summary-lbl">Only Hunspell</span>
        </div>
        <div class="summary-item">
            <span class="summary-num" style="color:#e74c3c">${contextOnly}</span>
            <span class="summary-lbl">Context errors caught</span>
        </div>`;
}

function clearAll() {
    document.getElementById('inputText').value              = '';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('noErrors').style.display       = 'none';
    document.getElementById('loadingMsg').style.display     = 'none';
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('inputText').addEventListener('keydown', e => {
        if (e.ctrlKey && e.key === 'Enter') checkText();
    });
});