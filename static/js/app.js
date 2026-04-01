function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function buildEvidence(items) {
  if (!items || items.length === 0) {
    return '<div class="evidence">No specific manipulative line detected.</div>';
  }

  const html = items
    .map((item) => {
      const terms = item.matched_terms && item.matched_terms.length
        ? ` | terms: ${item.matched_terms.join(', ')}`
        : '';
      return `
        <div class="evidence-item">
          <strong>Line ${item.line_number}</strong>: ${escapeHtml(item.text)}<br/>
          Score: ${item.score}${escapeHtml(terms)}
        </div>
      `;
    })
    .join('');

  return `<div class="evidence">${html}</div>`;
}

function updateNeedle(gaugeFill, gaugeValue, percent) {
  if (!gaugeFill || !gaugeValue) {
    return;
  }

  const bounded = Math.max(0, Math.min(100, percent));
  const angle = -90 + (bounded * 1.8);
  gaugeFill.style.transform = `translateX(-50%) rotate(${angle}deg)`;
  gaugeValue.textContent = `${bounded.toFixed(1)}%`;
}

function renderSignalBars(categories) {
  const signalBars = document.getElementById('signalBars');
  if (!signalBars) {
    return;
  }

  const bars = Object.values(categories).map((cat) => {
    const score = Math.max(0, Math.min(1, Number(cat.score || 0)));
    const threshold = Math.max(0, Math.min(1, Number(cat.threshold || 0)));

    return `
      <div class="signal-row">
        <div class="signal-row-head">
          <strong>${cat.display_name}</strong>
          <span>${(score * 100).toFixed(1)}%</span>
        </div>
        <div class="signal-track">
          <div class="signal-fill" style="width:${(score * 100).toFixed(1)}%"></div>
          <div class="signal-threshold" style="left:${(threshold * 100).toFixed(1)}%"></div>
        </div>
        <div class="signal-meta">
          <span>Score ${(score).toFixed(4)}</span>
          <span>Threshold ${(threshold).toFixed(2)}</span>
        </div>
      </div>
    `;
  }).join('');

  signalBars.innerHTML = bars;
}

function renderResults(result) {
  const resultPanel = document.getElementById('resultPanel');
  const riskBadge = document.getElementById('riskBadge');
  const categoryGrid = document.getElementById('categoryGrid');
  const riskGaugeFill = document.getElementById('riskGaugeFill');
  const riskGaugeValue = document.getElementById('riskGaugeValue');

  if (!resultPanel || !riskBadge || !categoryGrid) {
    return;
  }

  resultPanel.hidden = false;

  const riskScore = Math.max(0, Math.min(1, Number(result.risk_score || 0)));
  const riskPercent = riskScore * 100;
  riskBadge.innerHTML = `<div class="risk">Manipulation Risk: ${result.risk_level} (score: ${riskScore.toFixed(4)} | ${riskPercent.toFixed(1)}%)</div>`;

  updateNeedle(riskGaugeFill, riskGaugeValue, riskPercent);
  renderSignalBars(result.categories || {});

  const cards = Object.values(result.categories).map((cat) => {
    const yesNo = cat.flagged ? 'Yes' : 'No';
    const badgeClass = cat.flagged ? 'badge-yes' : 'badge-no';
    const score = Math.max(0, Math.min(1, Number(cat.score || 0)));
    const threshold = Math.max(0, Math.min(1, Number(cat.threshold || 0)));

    return `
      <article class="category-card">
        <h3>${cat.display_name}</h3>
        <p><span class="badge ${badgeClass}">${yesNo}</span></p>
        <div class="signal-inline">
          <div class="signal-track">
            <div class="signal-fill" style="width:${(score * 100).toFixed(1)}%"></div>
            <div class="signal-threshold" style="left:${(threshold * 100).toFixed(1)}%"></div>
          </div>
          <div class="signal-meta">
            <span>Score: <strong>${score.toFixed(4)}</strong></span>
            <span>Threshold: <strong>${threshold.toFixed(2)}</strong></span>
          </div>
        </div>
        ${buildEvidence(cat.evidence_lines)}
      </article>
    `;
  }).join('');

  categoryGrid.innerHTML = cards;
}

async function submitForAnalysis(article) {
  const resp = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ article }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.error || 'Analysis failed.');
  }
  return data;
}

function initAnalyzePage() {
  const articleInput = document.getElementById('articleInput');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const statusEl = document.getElementById('status');

  if (!articleInput || !analyzeBtn || !statusEl) {
    return;
  }

  const stored = sessionStorage.getItem('last_article_text');
  if (stored) {
    articleInput.value = stored;
  }

  analyzeBtn.addEventListener('click', async () => {
    const article = articleInput.value.trim();
    if (!article) {
      statusEl.textContent = 'Please paste narration text first.';
      return;
    }

    statusEl.textContent = 'Analyzing and preparing report page...';
    analyzeBtn.disabled = true;

    try {
      const result = await submitForAnalysis(article);
      sessionStorage.setItem('last_article_text', article);
      sessionStorage.setItem('last_analysis_result', JSON.stringify(result));
      window.location.href = '/results';
    } catch (err) {
      statusEl.textContent = err.message || 'Server error while analyzing.';
      analyzeBtn.disabled = false;
    }
  });
}

function initResultsPage() {
  const raw = sessionStorage.getItem('last_analysis_result');
  const errorPanel = document.getElementById('resultError');
  const errorText = document.getElementById('resultErrorText');

  if (!document.getElementById('resultPanel')) {
    return;
  }

  if (!raw) {
    if (errorPanel && errorText) {
      errorPanel.hidden = false;
      errorText.textContent = 'No analysis found. Please run Check Narration first.';
    }
    return;
  }

  try {
    const result = JSON.parse(raw);
    renderResults(result);
  } catch {
    if (errorPanel && errorText) {
      errorPanel.hidden = false;
      errorText.textContent = 'Saved analysis data is invalid. Please re-run analysis.';
    }
  }
}

function updateGauge(accuracyPercent) {
  const gaugeFill = document.getElementById('gaugeFill');
  const gaugeValue = document.getElementById('gaugeValue');
  updateNeedle(gaugeFill, gaugeValue, accuracyPercent);
}

function renderAccuracyPage(metrics) {
  const summary = document.getElementById('metricSummary');
  const grid = document.getElementById('metricsGrid');

  if (!summary || !grid) {
    return;
  }

  if (!metrics || !metrics.available) {
    summary.textContent = 'Model not trained yet. Run `python train_model.py` first.';
    updateGauge(0);
    return;
  }

  const overall = metrics.overall || {};
  const macroAcc = Number(overall.macro_accuracy || 0) * 100;
  const macroF1 = Number(overall.macro_f1 || 0);

  updateGauge(macroAcc);
  summary.innerHTML = `
    Samples: <strong>${metrics.sample_count ?? '-'}</strong> | 
    Macro Accuracy: <strong>${(macroAcc / 100).toFixed(4)}</strong> | 
    Macro F1: <strong>${macroF1.toFixed(4)}</strong>
  `;

  const fold = metrics.grouped_5fold_summary || {};
  const cards = [
    {
      title: 'Grouped 5-Fold Accuracy',
      value: fold.macro_accuracy_mean ?? '-',
      desc: `Std: ${fold.macro_accuracy_std ?? '-'}`,
    },
    {
      title: 'Grouped 5-Fold F1',
      value: fold.macro_f1_mean ?? '-',
      desc: `Std: ${fold.macro_f1_std ?? '-'}`,
    },
    {
      title: 'Unique Sources',
      value: metrics.unique_source_count ?? '-',
      desc: metrics.split_strategy || '-',
    },
  ];

  grid.innerHTML = cards.map((c) => `
    <article class="category-card">
      <h3>${c.title}</h3>
      <p><strong>${c.value}</strong></p>
      <p>${c.desc}</p>
    </article>
  `).join('');
}

async function initAccuracyPage() {
  if (!document.getElementById('speedometer')) {
    return;
  }

  try {
    const resp = await fetch('/api/metrics');
    const metrics = await resp.json();
    renderAccuracyPage(metrics);
  } catch {
    const summary = document.getElementById('metricSummary');
    if (summary) {
      summary.textContent = 'Unable to load model metrics.';
    }
    updateGauge(0);
  }
}

initAnalyzePage();
initResultsPage();
initAccuracyPage();
