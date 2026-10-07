/* ============================================================
   TradeSense V2 — Shared JavaScript Utilities
   All chart rendering, API communication, and shared helpers.
   The frontend NEVER contains mathematical logic —
   all calculations are done by the backend.
   ============================================================ */

window.TradeSense = (function() {
  'use strict';

  // ── State ──────────────────────────────────────────────────────────────
  let currentAnalysis = null;

  // ── API Client ──────────────────────────────────────────────────────────
  async function apiCall(endpoint, method = 'GET', body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(endpoint, opts);
    if (!res.ok) throw new Error(`${method} ${endpoint} → HTTP ${res.status}`);
    return res.json();
  }

  // ── Loading helpers ─────────────────────────────────────────────────────
  function showLoading(id) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '<div class="flex justify-center py-12"><div class="spinner"></div></div>';
  }
  function hideLoading(id) {
    const el = document.getElementById(id);
    if (el && el.querySelector('.spinner')) el.innerHTML = '';
  }

  // ── Formatters ──────────────────────────────────────────────────────────
  function formatNumber(n, decimals = 2) {
    if (n == null || isNaN(n)) return '—';
    return Number(n).toLocaleString('en-IN', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  }
  function formatPercent(n) {
    if (n == null || isNaN(n)) return '—';
    const sign = n >= 0 ? '+' : '';
    const cls  = n >= 0 ? 'badge-positive' : 'badge-negative';
    return `<span class="${cls}">${sign}${Number(n).toFixed(2)}%</span>`;
  }
  function formatCurrency(n) {
    if (n == null || isNaN(n)) return '—';
    return '₹' + Number(n).toLocaleString('en-IN', { minimumFractionDigits: 2 });
  }
  function setDataModeBadge(mode) {
    const badge = document.getElementById('data-mode-badge');
    const sidebar = document.getElementById('sidebar-data-mode');
    const m = (mode || '').toUpperCase();
    let text = '● LIVE • NSE';
    let cls = 'badge-live';
    
    if (m.includes('LIVE') || m.includes('HIST')) {
      text = '● LIVE • NSE';
      cls = 'badge-live';
    } else if (m.includes('DEMO')) {
      text = 'DEMO DATA';
      cls = 'badge-demo';
    } else if (m.includes('UNAVAIL') || m.includes('FAIL')) {
      text = 'Live data unavailable';
      cls = 'px-2 py-0.5 rounded font-bold text-xs bg-rose-100 text-rose-800 border border-rose-300';
    }
    
    if (badge) { badge.textContent = text; badge.className = cls + ' cursor-pointer text-xs'; }
    if (sidebar) { 
      const txtSpan = document.getElementById('sidebar-mode-text');
      if (txtSpan) txtSpan.textContent = text;
    }
  }

  async function triggerRefresh() {
    const btn = document.getElementById('btn-refresh-live');
    const icon = btn ? btn.querySelector('.refresh-icon') : null;
    if (icon) icon.classList.add('animate-spin');
    try {
      const mode = (document.getElementById('ctrl-mode')?.value || 'LIVE');
      const sector = (document.getElementById('ctrl-sector')?.value || document.getElementById('lab-sector')?.value || 'All Sectors');
      const period = (document.getElementById('ctrl-period')?.value || document.getElementById('lab-period')?.value || '1y');
      const threshold = parseFloat(document.getElementById('ctrl-threshold')?.value || 0.70);
      const k = parseInt(document.getElementById('ctrl-k')?.value || document.getElementById('lab-k')?.value || 3);
      
      const res = await apiCall('/api/refresh', 'POST', {
        data_mode: mode,
        sector: sector,
        period: period,
        corr_threshold: threshold,
        portfolio_k: k
      });
      
      const lastUp = document.getElementById('last-updated');
      if (lastUp && (res.last_updated || res.created_at)) {
        lastUp.textContent = res.last_updated || res.created_at;
      }
      if (res.data_mode) {
        setDataModeBadge(res.data_mode);
      }
      
      window.dispatchEvent(new CustomEvent('tradesense:refreshed', { detail: res }));
    } catch(err) {
      console.error('Refresh failed:', err);
      setDataModeBadge('UNAVAILABLE');
    } finally {
      if (icon) icon.classList.remove('animate-spin');
    }
  }

  // ── Plotly: Correlation Heatmap ──────────────────────────────────────────
  function renderCorrelationHeatmap(containerId, matrix, labels) {
    const el = document.getElementById(containerId);
    if (!el || !matrix || !labels) return;
    const z  = labels.map(r => labels.map(c => (matrix[r] && matrix[r][c] != null) ? matrix[r][c] : 0));
    Plotly.newPlot(el, [{
      type: 'heatmap', z, x: labels, y: labels,
      colorscale: 'RdYlGn', zmin: -1, zmax: 1,
      text: z.map(row => row.map(v => v.toFixed(2))),
      texttemplate: '%{text}', showscale: true,
      hovertemplate: '%{y} × %{x}: %{z:.3f}<extra></extra>',
    }], {
      margin: { t: 40, l: 100, r: 20, b: 100 },
      paper_bgcolor: 'white', plot_bgcolor: 'white',
      xaxis: { tickangle: -45 }, font: { family: 'Inter, sans-serif', size: 11 },
    }, { responsive: true });
  }

  // ── Plotly: Network Graph ────────────────────────────────────────────────
  function renderNetworkGraph(containerId, graphData) {
    const el = document.getElementById(containerId);
    if (!el || !graphData) return;
    const { edges = [], stats = {}, coloring_dict = {} } = graphData;

    // Use simple circular layout for nodes
    const tickers = Object.keys(coloring_dict).length > 0 ? Object.keys(coloring_dict) :
      [...new Set(edges.flatMap(e => [e.source, e.target]))];
    const n = tickers.length;
    const pos = {};
    tickers.forEach((t, i) => {
      const angle = (2 * Math.PI * i) / n;
      pos[t] = { x: Math.cos(angle), y: Math.sin(angle) };
    });

    // Edge traces
    const edgeX = [], edgeY = [];
    edges.forEach(e => {
      const s = pos[e.source], t = pos[e.target];
      if (s && t) { edgeX.push(s.x, t.x, null); edgeY.push(s.y, t.y, null); }
    });

    // Color palette for groups
    const palette = ['#4f46e5','#0ea5e9','#16a34a','#dc2626','#d97706','#9333ea','#0d9488','#db2777','#65a30d','#ea580c'];
    const nodeColors = tickers.map(t => palette[(coloring_dict[t] || 0) % palette.length]);

    const edgeTrace = { type: 'scatter', x: edgeX, y: edgeY, mode: 'lines',
      line: { width: 1.5, color: '#cbd5e1' }, hoverinfo: 'none' };
    const nodeTrace = { type: 'scatter', x: tickers.map(t => pos[t].x), y: tickers.map(t => pos[t].y),
      mode: 'markers+text', text: tickers.map(t => t.replace('.NS','')),
      textposition: 'top center', textfont: { size: 10, family: 'Inter' },
      marker: { size: 18, color: nodeColors, line: { width: 2, color: '#fff' } },
      hovertemplate: '%{text}<extra></extra>', };

    Plotly.newPlot(el, [edgeTrace, nodeTrace], {
      showlegend: false, margin: { t: 20, l: 20, r: 20, b: 20 },
      xaxis: { showgrid: false, zeroline: false, showticklabels: false },
      yaxis: { showgrid: false, zeroline: false, showticklabels: false },
      paper_bgcolor: 'white', plot_bgcolor: '#f8fafc',
    }, { responsive: true });
  }

  // ── Plotly: Hasse Diagram ────────────────────────────────────────────────
  function renderHasseDiagram(containerId, hasseData) {
    const el = document.getElementById(containerId);
    if (!el || !hasseData) return;
    const layout   = hasseData.layout || {};
    const edges    = hasseData.cover_relation || [];
    const tickers  = hasseData.tickers || Object.keys(layout);

    const edgeX = [], edgeY = [];
    edges.forEach(e => {
      const s = layout[e.from], t = layout[e.to];
      if (s && t) { edgeX.push(s[0], t[0], null); edgeY.push(s[1], t[1], null); }
    });
    const edgeTrace = { type: 'scatter', x: edgeX, y: edgeY, mode: 'lines',
      line: { width: 2, color: '#6366f1' }, hoverinfo: 'none' };
    const nodeTrace = {
      type: 'scatter',
      x: tickers.map(t => layout[t] ? layout[t][0] : 0.5),
      y: tickers.map(t => layout[t] ? layout[t][1] : 0.5),
      mode: 'markers+text', text: tickers.map(t => t.replace('.NS','')),
      textposition: 'top center', textfont: { size: 11 },
      marker: { size: 20, color: '#4f46e5', line: { width: 2, color: '#fff' } },
      hovertemplate: '%{text}<extra></extra>',
    };
    Plotly.newPlot(el, [edgeTrace, nodeTrace], {
      showlegend: false, margin: { t: 30, l: 60, r: 60, b: 30 },
      xaxis: { showgrid: false, zeroline: false, showticklabels: false },
      yaxis: { showgrid: false, zeroline: false, showticklabels: false },
      paper_bgcolor: 'white', plot_bgcolor: 'white',
    }, { responsive: true });
  }

  // ── Plotly: Radar Chart ──────────────────────────────────────────────────
  function renderPortfolioRadar(containerId, portA, portB) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const categories = ['Return', 'Risk', 'Diversification', 'Group Diversity', 'Dominance'];
    const keysA = [portA.return_score, portA.risk_score, portA.diversification_score, portA.group_score, portA.dominance_score];
    const keysB = portB ? [portB.return_score, portB.risk_score, portB.diversification_score, portB.group_score, portB.dominance_score] : null;
    const traces = [{
      type: 'scatterpolar', r: [...keysA, keysA[0]], theta: [...categories, categories[0]],
      fill: 'toself', name: 'Portfolio A', line: { color: '#4f46e5' }, fillcolor: 'rgba(79,70,229,.15)',
    }];
    if (keysB) traces.push({
      type: 'scatterpolar', r: [...keysB, keysB[0]], theta: [...categories, categories[0]],
      fill: 'toself', name: 'Portfolio B', line: { color: '#dc2626' }, fillcolor: 'rgba(220,38,38,.15)',
    });
    Plotly.newPlot(el, traces, {
      polar: { radialaxis: { range: [0, 100] } },
      showlegend: true, margin: { t: 30, l: 30, r: 30, b: 30 }, paper_bgcolor: 'white',
    }, { responsive: true });
  }

  // ── Plotly: Price Chart ──────────────────────────────────────────────────
  function renderPriceChart(containerId, priceHistory, symbol) {
    const el = document.getElementById(containerId);
    if (!el || !priceHistory || !priceHistory.length) return;
    const dates  = priceHistory.map(p => p.date);
    const prices = priceHistory.map(p => p.close);
    Plotly.newPlot(el, [{
      x: dates, y: prices, type: 'scatter', mode: 'lines',
      line: { color: '#4f46e5', width: 2 }, fill: 'tozeroy', fillcolor: 'rgba(79,70,229,.06)',
      hovertemplate: '%{x}<br>₹%{y:,.2f}<extra></extra>',
    }], {
      margin: { t: 20, l: 60, r: 20, b: 40 },
      yaxis: { title: 'Price (₹)', tickprefix: '₹' },
      paper_bgcolor: 'white', plot_bgcolor: '#f8fafc',
      font: { family: 'Inter', size: 11 },
    }, { responsive: true });
  }

  // ── Portfolio Cards Renderer ─────────────────────────────────────────────
  function renderPortfolioCards(portfolios, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const medals  = ['🥇', '🥈', '🥉'];
    const classes = ['rank-card-gold', 'rank-card-silver', 'rank-card-bronze'];
    el.innerHTML = portfolios.slice(0, 3).map((p, i) => `
      <div class="rank-card ${classes[i]} shadow-xs p-4 rounded-2xl bg-white border border-slate-200">
        <div class="flex items-center justify-between mb-3">
          <span class="text-2xl">${medals[i]}</span>
          <div class="text-right">
            <span class="text-2xl font-black text-indigo-700 font-mono">${p.dm_score?.toFixed(1) ?? '—'}</span>
            <span class="text-[10px] text-slate-400 font-bold block uppercase">DM Score / 100</span>
          </div>
        </div>
        <div class="flex flex-wrap gap-1.5 mb-3">
          ${(p.portfolio || []).map(s => `<span class="bg-indigo-50 border border-indigo-200 text-indigo-800 text-xs font-mono font-bold px-2 py-0.5 rounded-md">${s.replace('.NS','')}</span>`).join('')}
        </div>
        <div class="space-y-1.5 pt-1 border-t border-slate-100">
          ${[
            ['Return (30%)', p.return_score, '#4f46e5'],
            ['Risk Efficiency (25%)', p.risk_score, '#7c3aed'],
            ['Diversification (20%)', p.diversification_score, '#059669'],
            ['Stability (15%)', p.stability_score ?? p.dominance_score, '#d97706'],
            ['Network Benefit (10%)', p.network_score ?? p.group_score, '#0284c7']
          ].map(([label, val, col]) => `
            <div class="flex items-center gap-2 text-[11px]">
              <span class="text-slate-600 font-medium w-36 truncate">${label}</span>
              <div class="score-bar flex-1 bg-slate-100 rounded-full h-2 overflow-hidden"><div class="score-bar-fill h-full rounded-full" style="width:${Math.min(100, Math.max(0, val || 0)).toFixed(1)}%; background-color: ${col}"></div></div>
              <span class="font-mono text-slate-700 font-bold w-9 text-right">${val != null ? Number(val).toFixed(1) : '—'}</span>
            </div>`).join('')}
        </div>
        <div class="text-[10px] text-slate-400 mt-2.5 pt-2 border-t border-slate-100 leading-tight">
          Score = 0.30×Return + 0.25×RiskEff + 0.20×Div + 0.15×Stab + 0.10×Net
        </div>
      </div>`).join('');
  }

  // ── KPI Cards Updater ────────────────────────────────────────────────────
  function updateKpiCards(data) {
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('kpi-stocks',      data.stocks_analyzed ?? '—');
    set('kpi-edges',       data.graph_stats?.num_edges ?? '—');
    set('kpi-nondom',      data.non_dominated_count ?? '—');
    set('kpi-colors',      data.num_color_groups ?? '—');
    set('kpi-portfolios',  data.total_portfolios_evaluated ?? '—');
    set('kpi-topdm',       data.top_dm_score?.toFixed(2) ?? '—');
    const ts = document.getElementById('last-updated');
    if (ts) ts.textContent = 'Updated ' + new Date().toLocaleTimeString();
  }

  // ── Modal Handlers ──────────────────────────────────────────────────────
  function openDataSourceModal() {
    const modal = document.getElementById('data-source-modal');
    if (modal) {
      modal.classList.remove('hidden');
      document.body.style.overflow = 'hidden';
    }
  }

  function closeDataSourceModal() {
    const modal = document.getElementById('data-source-modal');
    if (modal) {
      modal.classList.add('hidden');
      document.body.style.overflow = '';
    }
  }

  // ── Mobile Responsive Sidebar Handlers ─────────────────────────────────
  function toggleSidebar(forceState) {
    const sidebar = document.getElementById('main-sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    if (!sidebar) return;
    const isClosed = sidebar.classList.contains('-translate-x-full');
    const shouldOpen = forceState !== undefined ? forceState : isClosed;
    if (shouldOpen) {
      sidebar.classList.remove('-translate-x-full');
      if (backdrop) backdrop.classList.remove('hidden');
      document.body.style.overflow = 'hidden';
    } else {
      sidebar.classList.add('-translate-x-full');
      if (backdrop) backdrop.classList.add('hidden');
      document.body.style.overflow = '';
    }
  }

  function closeSidebarMobile() {
    if (window.innerWidth < 768) {
      toggleSidebar(false);
    }
  }

  // ── Cross-Device Connect Modal Handlers ─────────────────────────────────
  async function openDeviceModal() {
    const modal = document.getElementById('device-modal');
    if (!modal) return;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    try {
      const info = await apiCall('/api/market/network-info');
      const networkUrl = info.network_url || `http://${window.location.hostname}:8000`;
      const netInput = document.getElementById('device-network-url');
      if (netInput) netInput.value = networkUrl;

      // Update QR Code with network URL for instant phone camera scanning
      const qrImg = document.getElementById('device-qr-image');
      if (qrImg) {
        qrImg.src = `https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=${encodeURIComponent(networkUrl)}`;
      }
    } catch(e) {
      console.warn("Could not fetch network info:", e);
    }
  }

  function closeDeviceModal() {
    const modal = document.getElementById('device-modal');
    if (modal) {
      modal.classList.add('hidden');
      document.body.style.overflow = '';
    }
  }

  function copyDeviceLink(inputId, btnId) {
    const input = document.getElementById(inputId);
    const btn = document.getElementById(btnId);
    if (input) {
      input.select();
      input.setSelectionRange(0, 99999);
      navigator.clipboard.writeText(input.value).then(() => {
        if (btn) {
          const original = btn.innerHTML;
          btn.innerHTML = '✓ Copied!';
          btn.classList.add('bg-emerald-600');
          setTimeout(() => {
            btn.innerHTML = original;
            btn.classList.remove('bg-emerald-600');
          }, 2000);
        }
      });
    }
  }

  // ── Global Chart Responsive Resize Handler ──────────────────────────────
  window.addEventListener('resize', () => {
    if (window.Plotly) {
      document.querySelectorAll('.js-plotly-plot').forEach(el => {
        try { Plotly.Plots.resize(el); } catch(e) {}
      });
    }
  });

  // ── Public API ───────────────────────────────────────────────────────────
  return {
    apiCall, showLoading, hideLoading,
    formatNumber, formatPercent, formatCurrency, setDataModeBadge,
    renderCorrelationHeatmap, renderNetworkGraph, renderHasseDiagram,
    renderPortfolioRadar, renderPriceChart, renderPortfolioCards,
    updateKpiCards, openDataSourceModal, closeDataSourceModal, triggerRefresh,
    toggleSidebar, closeSidebarMobile, openDeviceModal, closeDeviceModal, copyDeviceLink,
    get currentAnalysis() { return currentAnalysis; },
    set currentAnalysis(v) { currentAnalysis = v; },
  };
})();

