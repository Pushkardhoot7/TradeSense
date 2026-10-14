/**
 * market_relationships.js
 * -----------------------
 * Interactive force-directed market network and diverging correlation heatmap
 * with complete shared stock, group, and sector selection for TradeSense V2.1.
 *
 * Implements:
 * - Phase 2: Shared Stock, Multi-Stock, Cluster Group, and Sector Selection
 * - Phase 3: Force-directed network rendered only on the active selection
 * - Phase 4: Diverging correlation heatmap strictly synchronized with the selection
 * - Phase 5: Cross-visualization bidirectional interactivity (Network <-> Heatmap)
 * - Phase 8: Clean plain-language investor explanations without raw math formulas
 */

(function () {
  'use strict';

  // Sensible default selection (10 liquid multi-sector stocks)
  const DEFAULT_SUBSET = [
    'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'RELIANCE.NS',
    'HDFCBANK.NS', 'ICICIBANK.NS', 'BAJFINANCE.NS', 'TATAMOTORS.NS', 'MARUTI.NS'
  ];

  // Diverging color scale: Blue (-1.0) -> Neutral (#f8fafc) -> Red (+1.0)
  const DIVERGING_CORR_SCALE = [
    [0.0, '#1d4ed8'],  // Deep blue (-1.0)
    [0.25, '#60a5fa'], // Soft blue (-0.5)
    [0.48, '#f1f5f9'], // Off-white near zero (-0.04)
    [0.50, '#f8fafc'], // Neutral zero (0.00)
    [0.52, '#f1f5f9'], // Off-white near zero (+0.04)
    [0.75, '#fb7185'], // Soft rose/red (+0.5)
    [1.0, '#dc2626'],  // Deep red (+1.0)
  ];

  // Distinct qualitative colors for cluster groups
  const CLUSTER_PALETTE = [
    '#4f46e5', '#0ea5e9', '#10b981', '#f59e0b', '#ec4899',
    '#8b5cf6', '#06b6d4', '#84cc16', '#f97316', '#64748b'
  ];

  // Global Component State
  const State = {
    threshold: 0.70,
    selectionMode: 'multi', // 'multi' | 'single' | 'group' | 'sector' | 'all'
    isDefaultSelection: true,
    selectedTickers: new Set(DEFAULT_SUBSET),
    allUniverseTickers: [],
    allSectors: [],
    stockClusters: [],
    unclusteredStocks: [],
    focusedNode: null,
    focusedPair: null,
    rawCorrMatrix: {},
    cachedGraphRes: null,
    cachedCorrRes: null,
  };

  /**
   * Fast 2D force-directed layout (Fruchterman-Reingold variant).
   */
  function computeForceDirectedLayout(nodes, edges) {
    const N = nodes.length;
    if (N === 0) return {};

    const pos = {};
    nodes.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / N;
      const radius = 0.65 + (i % 3) * 0.08;
      pos[n] = {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
      };
    });

    const k = Math.sqrt(4.0 / Math.max(N, 1));
    const iterations = 50;
    let temp = 0.12;

    for (let it = 0; it < iterations; it++) {
      // 1. Repulsion
      for (let i = 0; i < N; i++) {
        const u = nodes[i];
        for (let j = i + 1; j < N; j++) {
          const v = nodes[j];
          const dx = pos[u].x - pos[v].x;
          const dy = pos[u].y - pos[v].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
          const f = (k * k) / dist;
          pos[u].vx += (dx / dist) * f;
          pos[u].vy += (dy / dist) * f;
          pos[v].vx -= (dx / dist) * f;
          pos[v].vy -= (dy / dist) * f;
        }
      }

      // 2. Attraction along edges
      edges.forEach((e) => {
        const u = e.source, v = e.target;
        if (!pos[u] || !pos[v]) return;
        const dx = pos[v].x - pos[u].x;
        const dy = pos[v].y - pos[u].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const f = ((dist * dist) / k) * 1.5;
        pos[u].vx += (dx / dist) * f;
        pos[u].vy += (dy / dist) * f;
        pos[v].vx -= (dx / dist) * f;
        pos[v].vy -= (dy / dist) * f;
      });

      // 3. Gravity towards center
      nodes.forEach((n) => {
        const p = pos[n];
        const distCenter = Math.sqrt(p.x * p.x + p.y * p.y) || 0.01;
        p.vx -= (p.x / distCenter) * 0.05 * distCenter;
        p.vy -= (p.y / distCenter) * 0.05 * distCenter;
        const disp = Math.sqrt(p.vx * p.vx + p.vy * p.vy) || 0.01;
        const lim = Math.min(disp, temp);
        p.x += (p.vx / disp) * lim;
        p.y += (p.vy / disp) * lim;
        p.vx = 0; p.vy = 0;
      });
      temp *= 0.94;
    }

    // Normalize coordinates to [-0.95, 0.95]
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    nodes.forEach((n) => {
      const p = pos[n];
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });

    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    nodes.forEach((n) => {
      pos[n].x = ((pos[n].x - minX) / spanX) * 1.8 - 0.9;
      pos[n].y = ((pos[n].y - minY) / spanY) * 1.8 - 0.9;
    });

    return pos;
  }

  /**
   * Render the Market Connection Network for active selected stocks.
   */
  function renderMarketNetwork(containerId, activeTickers, edges, degrees, focusedStock = null) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const N = activeTickers.length;
    if (N === 0) {
      el.innerHTML = `
        <div class="flex flex-col items-center justify-center h-full text-slate-400 text-xs p-6 text-center space-y-2">
          <span class="text-3xl">🔍</span>
          <strong class="text-slate-600">No stocks selected.</strong>
          <span>Choose stocks using the selection bar above to display market connections.</span>
        </div>`;
      return;
    }

    // Adjacency map for fast neighbor queries
    const adjMap = {};
    activeTickers.forEach((t) => { adjMap[t] = new Set(); });
    edges.forEach((e) => {
      if (adjMap[e.source]) adjMap[e.source].add(e.target);
      if (adjMap[e.target]) adjMap[e.target].add(e.source);
    });

    // Layout positions
    const pos = computeForceDirectedLayout(activeTickers, edges);

    const isFocused = Boolean(focusedStock && activeTickers.includes(focusedStock));
    const activeNeighbors = isFocused ? (adjMap[focusedStock] || new Set()) : null;

    // Edge lines
    const edgeX = [], edgeY = [];
    const edgeMidX = [], edgeMidY = [], edgeMidText = [];

    edges.forEach((e) => {
      const s = pos[e.source], t = pos[e.target];
      if (s && t) {
        edgeX.push(s.x, t.x, null);
        edgeY.push(s.y, t.y, null);

        edgeMidX.push((s.x + t.x) / 2);
        edgeMidY.push((s.y + t.y) / 2);
        const wStr = (e.weight !== undefined) ? Number(e.weight).toFixed(3) : State.threshold.toFixed(2);
        edgeMidText.push(`${e.source.replace('.NS', '')} ↔ ${e.target.replace('.NS', '')}<br><b>Pearson r = ${wStr}</b>`);
      }
    });

    const edgeTrace = {
      type: 'scatter',
      x: edgeX,
      y: edgeY,
      mode: 'lines',
      line: {
        width: isFocused ? 1.0 : 1.8,
        color: isFocused ? '#e2e8f0' : '#818cf8',
      },
      hoverinfo: 'none',
      showlegend: false,
    };

    // Highlighted incident edges if node is focused
    const selEdgeX = [], selEdgeY = [];
    if (isFocused) {
      edges.forEach((e) => {
        if (e.source === focusedStock || e.target === focusedStock) {
          const s = pos[e.source], t = pos[e.target];
          if (s && t) {
            selEdgeX.push(s.x, t.x, null);
            selEdgeY.push(s.y, t.y, null);
          }
        }
      });
    }

    const selEdgeTrace = {
      type: 'scatter',
      x: selEdgeX,
      y: selEdgeY,
      mode: 'lines',
      line: {
        width: 3.5,
        color: '#4f46e5',
      },
      hoverinfo: 'none',
      showlegend: false,
    };

    // Edge midpoint hover trace
    const edgeHoverTrace = {
      type: 'scatter',
      x: edgeMidX,
      y: edgeMidY,
      mode: 'markers',
      marker: { size: 6, color: 'rgba(0,0,0,0)' },
      hoverinfo: 'text',
      text: edgeMidText,
      showlegend: false,
    };

    // Node markers & labels
    const nodeX = [], nodeY = [];
    const nodeText = [], nodeHover = [];
    const nodeColors = [], nodeSizes = [], nodeBorders = [], nodeBorderWidths = [];

    // Map cluster colors
    const clusterMap = {};
    State.stockClusters.forEach((c, idx) => {
      c.members.forEach((m) => { clusterMap[m] = idx; });
    });

    activeTickers.forEach((t) => {
      const p = pos[t] || { x: 0, y: 0 };
      nodeX.push(p.x);
      nodeY.push(p.y);

      const cleanSym = t.replace('.NS', '');
      const deg = degrees[t] || 0;
      const cIdx = clusterMap[t] !== undefined ? clusterMap[t] : 9;
      const baseColor = CLUSTER_PALETTE[cIdx % CLUSTER_PALETTE.length];
      const baseSize = Math.min(14 + deg * 2.5, 26);

      let isDimmed = false;
      let isTarget = false;
      let isNeighbor = false;

      if (isFocused) {
        if (t === focusedStock) {
          isTarget = true;
        } else if (activeNeighbors && activeNeighbors.has(t)) {
          isNeighbor = true;
        } else {
          isDimmed = true;
        }
      }

      // Hide labels on large selections unless hovered or focused
      let label = '';
      if (N <= 20 || isTarget || isNeighbor || deg >= 3) {
        label = cleanSym;
      }
      nodeText.push(label);

      nodeHover.push(
        `<b>${cleanSym}</b> (${t})<br>` +
        `Active Connections: <b>${deg}</b><br>` +
        `<i>Click node to isolate co-moving partners</i>`
      );

      if (isTarget) {
        nodeColors.push('#4f46e5');
        nodeSizes.push(baseSize + 6);
        nodeBorders.push('#fbbf24'); // Amber focus ring
        nodeBorderWidths.push(3.5);
      } else if (isNeighbor) {
        nodeColors.push(baseColor);
        nodeSizes.push(baseSize + 2);
        nodeBorders.push('#4f46e5');
        nodeBorderWidths.push(2.5);
      } else if (isDimmed) {
        nodeColors.push('rgba(203, 213, 225, 0.4)');
        nodeSizes.push(Math.max(baseSize - 4, 8));
        nodeBorders.push('rgba(148, 163, 184, 0.25)');
        nodeBorderWidths.push(1);
      } else {
        nodeColors.push(baseColor);
        nodeSizes.push(baseSize);
        nodeBorders.push('#ffffff');
        nodeBorderWidths.push(2);
      }
    });

    const nodeTrace = {
      type: 'scatter',
      x: nodeX,
      y: nodeY,
      mode: 'markers+text',
      text: nodeText,
      textposition: 'top center',
      textfont: {
        family: 'Inter, sans-serif',
        size: 11,
        color: '#1e293b',
        weight: 'bold',
      },
      marker: {
        size: nodeSizes,
        color: nodeColors,
        line: { color: nodeBorders, width: nodeBorderWidths },
      },
      hoverinfo: 'text',
      hovertext: nodeHover,
      customdata: activeTickers,
      showlegend: false,
    };

    const traces = [edgeTrace];
    if (isFocused) traces.push(selEdgeTrace);
    traces.push(edgeHoverTrace, nodeTrace);

    const layout = {
      margin: { t: 15, l: 15, r: 15, b: 15 },
      hovermode: 'closest',
      dragmode: 'pan',
      paper_bgcolor: 'white',
      plot_bgcolor: '#f8fafc',
      xaxis: { showgrid: false, zeroline: false, showticklabels: false, range: [-1.15, 1.15] },
      yaxis: { showgrid: false, zeroline: false, showticklabels: false, range: [-1.15, 1.15] },
      annotations: edges.length === 0 ? [{
        text: `No connections qualify at threshold r ≥ ${State.threshold.toFixed(2)}.<br><span style="font-size:11px;color:#64748b;">Selected stocks exhibit low linear co-movement. Adjust the slider above to inspect weaker links.</span>`,
        showarrow: false,
        font: { family: 'Inter', size: 12, color: '#334155' },
        xref: 'paper', yref: 'paper',
        x: 0.5, y: 0.5,
        bgcolor: '#ffffff',
        bordercolor: '#cbd5e1',
        borderwidth: 1,
        borderpad: 12,
      }] : [],
    };

    const config = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
      displaylogo: false,
    };

    Plotly.react(el, traces, layout, config).then(() => {
      el.removeAllListeners && el.removeAllListeners('plotly_click');
      el.on('plotly_click', (data) => {
        if (!data || !data.points || data.points.length === 0) return;
        const pt = data.points[0];
        if (pt.customdata) {
          const clicked = pt.customdata;
          handleStockFocus(clicked === State.focusedNode ? null : clicked);
        }
      });
    });

    const badge = document.getElementById('network-edge-count-badge');
    if (badge) badge.textContent = `${edges.length} Active Edges`;
  }

  /**
   * Render the Correlation Matrix Heatmap for active selected stocks.
   */
  function renderCorrelationHeatmap(containerId, activeTickers, matrix) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const N = activeTickers.length;
    if (N === 0) {
      el.innerHTML = `
        <div class="flex flex-col items-center justify-center h-full text-slate-400 text-xs p-6 text-center space-y-2">
          <span class="text-3xl">📊</span>
          <strong class="text-slate-600">No stocks selected.</strong>
          <span>Select stocks to view their pairwise correlation matrix.</span>
        </div>`;
      return;
    }

    const cleanDisplayLabels = activeTickers.map((s) => s.replace('.NS', ''));
    const zValues = activeTickers.map((rowSym) =>
      activeTickers.map((colSym) => {
        if (matrix[rowSym] && matrix[rowSym][colSym] !== undefined) {
          const v = matrix[rowSym][colSym];
          return v !== null ? Number(v) : (rowSym === colSym ? 1.0 : 0.0);
        }
        return rowSym === colSym ? 1.0 : 0.0;
      })
    );

    const showTextValues = N <= 14;
    const textTemplates = showTextValues
      ? zValues.map((row) => row.map((v) => (v !== null ? v.toFixed(2) : '—')))
      : undefined;

    const heatmapTrace = {
      type: 'heatmap',
      z: zValues,
      x: cleanDisplayLabels,
      y: cleanDisplayLabels,
      colorscale: DIVERGING_CORR_SCALE,
      zmin: -1.0,
      zmax: 1.0,
      showscale: true,
      colorbar: {
        title: {
          text: 'Pearson r',
          side: 'right',
          font: { family: 'Inter', size: 10, color: '#64748b' },
        },
        thickness: 12,
        len: 0.85,
        tickvals: [-1.0, -0.5, 0.0, 0.5, 1.0],
        ticktext: ['-1.0', '-0.5', '0.0', '+0.5', '+1.0'],
        tickfont: { family: 'JetBrains Mono', size: 9 },
      },
      text: textTemplates,
      texttemplate: showTextValues ? '%{text}' : undefined,
      textfont: { family: 'JetBrains Mono', size: 9, color: '#0f172a' },
      hovertemplate:
        'Pair: <b>%{y} ↔ %{x}</b><br>' +
        'Pearson Correlation: <b>%{z:.3f}</b><br>' +
        '<extra></extra>',
      customdata: activeTickers.map((r) => activeTickers.map((c) => [r, c])),
    };

    const layout = {
      margin: { t: 25, l: 60, r: 25, b: 60 },
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      xaxis: {
        tickangle: -45,
        tickfont: { family: 'Inter', size: N > 20 ? 8 : 10 },
      },
      yaxis: {
        tickfont: { family: 'Inter', size: N > 20 ? 8 : 10 },
        autorange: 'reversed',
      },
      font: { family: 'Inter, sans-serif' },
    };

    const config = {
      responsive: true,
      displayModeBar: false,
    };

    Plotly.react(el, [heatmapTrace], layout, config).then(() => {
      el.removeAllListeners && el.removeAllListeners('plotly_click');
      el.on('plotly_click', (data) => {
        if (!data || !data.points || data.points.length === 0) return;
        const pt = data.points[0];
        const stockA = activeTickers[pt.yIndex || pt.pointIndex?.[0] || 0];
        const stockB = activeTickers[pt.xIndex || pt.pointIndex?.[1] || 0];
        const rVal = pt.z;
        if (stockA && stockB) {
          handleHeatmapCellClick(stockA, stockB, rVal);
        }
      });
    });

    const matrixBadge = document.getElementById('matrix-size-badge');
    if (matrixBadge) matrixBadge.textContent = `${N} × ${N} Matrix`;
  }

  /**
   * Handle stock focus from Network node click.
   */
  function handleStockFocus(ticker) {
    State.focusedNode = ticker;
    State.focusedPair = null;

    const activeList = Array.from(State.selectedTickers);
    const edges = extractQualifyingEdges(activeList, State.rawCorrMatrix, State.threshold);
    const degrees = computeDegrees(activeList, edges);

    renderMarketNetwork('market-network-plot', activeList, edges, degrees, ticker);
    updateStockInspector(ticker);
  }

  /**
   * Handle Heatmap cell click: highlight both stocks & connecting edge.
   */
  function handleHeatmapCellClick(stockA, stockB, rVal) {
    State.focusedPair = { stockA, stockB, r: rVal };
    State.focusedNode = stockA;

    const activeList = Array.from(State.selectedTickers);
    const edges = extractQualifyingEdges(activeList, State.rawCorrMatrix, State.threshold);
    const degrees = computeDegrees(activeList, edges);

    // Re-render network highlighting stockA
    renderMarketNetwork('market-network-plot', activeList, edges, degrees, stockA);

    // Update pair callout badge
    updatePairCallout(stockA, stockB, rVal);
    updateStockInspector(stockA);
  }

  /**
   * Extract qualifying edges where |r| >= threshold among active tickers.
   */
  function extractQualifyingEdges(tickers, matrix, threshold) {
    const edges = [];
    const n = tickers.length;
    for (let i = 0; i < n; i++) {
      const u = tickers[i];
      for (let j = i + 1; j < n; j++) {
        const v = tickers[j];
        if (matrix[u] && matrix[u][v] !== undefined && matrix[u][v] !== null) {
          const val = Number(matrix[u][v]);
          if (!isNaN(val) && Math.abs(val) >= threshold) {
            edges.push({ source: u, target: v, weight: val });
          }
        }
      }
    }
    return edges;
  }

  /**
   * Compute degrees per active ticker.
   */
  function computeDegrees(tickers, edges) {
    const degs = {};
    tickers.forEach((t) => { degs[t] = 0; });
    edges.forEach((e) => {
      if (degs[e.source] !== undefined) degs[e.source]++;
      if (degs[e.target] !== undefined) degs[e.target]++;
    });
    return degs;
  }

  /**
   * Update the Pair Callout Card under the Heatmap.
   */
  function updatePairCallout(stockA, stockB, rVal) {
    const card = document.getElementById('selected-pair-card');
    const title = document.getElementById('pair-title');
    const valEl = document.getElementById('pair-val');
    const status = document.getElementById('pair-status');
    const desc = document.getElementById('pair-desc');

    if (!card || !title || !valEl || !status || !desc) return;

    card.classList.remove('hidden');
    const cleanA = stockA.replace('.NS', '');
    const cleanB = stockB.replace('.NS', '');
    title.textContent = `${cleanA} ↔ ${cleanB}`;

    const numR = Number(rVal);
    valEl.textContent = `r = ${numR.toFixed(3)}`;

    const qualifies = Math.abs(numR) >= State.threshold;
    if (qualifies) {
      status.textContent = 'Active Connection';
      status.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-800';
      desc.textContent = `|r| = ${Math.abs(numR).toFixed(2)} ≥ ${State.threshold.toFixed(2)}. These stocks qualify for a network connection line.`;
    } else {
      status.textContent = 'No Connection';
      status.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-slate-200 text-slate-700';
      desc.textContent = `|r| = ${Math.abs(numR).toFixed(2)} < ${State.threshold.toFixed(2)}. Correlation does not meet the connection threshold.`;
    }
  }

  /**
   * Update Interactive Stock Inspector Panel.
   */
  function updateStockInspector(ticker) {
    const panel = document.getElementById('stock-inspector-panel');
    if (!panel) return;

    if (!ticker) {
      panel.innerHTML = `
        <div class="p-5 bg-white rounded-2xl border border-slate-200 shadow-xs text-center text-slate-400 text-xs">
          💡 Tip: Click any stock node in the network or cell in the heatmap to isolate its co-moving partners and diversifiers.
        </div>`;
      return;
    }

    const cleanSym = ticker.replace('.NS', '');
    const activeList = Array.from(State.selectedTickers);
    const edges = extractQualifyingEdges(activeList, State.rawCorrMatrix, State.threshold);
    const degs = computeDegrees(activeList, edges);
    const deg = degs[ticker] || 0;

    // Find correlated partners among active selection
    const partners = [];
    activeList.forEach((o) => {
      if (o !== ticker && State.rawCorrMatrix[ticker] && State.rawCorrMatrix[ticker][o] !== undefined) {
        const val = Number(State.rawCorrMatrix[ticker][o]);
        if (!isNaN(val)) {
          partners.push({ symbol: o, r: val });
        }
      }
    });

    partners.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
    const strongPartners = partners.filter((p) => Math.abs(p.r) >= State.threshold);
    const diversifiers = partners.filter((p) => Math.abs(p.r) < 0.30);

    panel.innerHTML = `
      <div class="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-4">
        <div class="flex items-center justify-between border-b border-slate-100 pb-3 flex-wrap gap-2">
          <div class="flex items-center gap-3">
            <span class="w-10 h-10 rounded-xl bg-indigo-50 border border-indigo-200 text-indigo-700 font-black text-sm flex items-center justify-center font-mono">
              ${cleanSym}
            </span>
            <div>
              <h4 class="text-sm font-bold text-slate-900">${cleanSym} (${ticker})</h4>
              <p class="text-xs text-slate-500">Active Connections: <strong>${deg} stocks</strong> (|r| ≥ ${State.threshold.toFixed(2)})</p>
            </div>
          </div>
          <button onclick="TradeSenseMarketRel.clearFocus()" class="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 text-xs font-bold transition-colors">
            Clear Focus
          </button>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <!-- Co-moving partners -->
          <div class="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <span class="font-bold text-slate-800 flex items-center gap-1.5">
              <span>🔗</span> Direct Co-Moving Partners (${strongPartners.length})
            </span>
            <div class="flex flex-wrap gap-1.5">
              ${strongPartners.length > 0 ? strongPartners.map((p) => `
                <span class="px-2 py-0.5 rounded bg-white border border-slate-200 font-mono font-bold text-indigo-700 text-[11px]">
                  ${p.symbol.replace('.NS', '')} (r = ${p.r.toFixed(2)})
                </span>
              `).join('') : '<span class="text-slate-400">No qualifying co-moving partners under this threshold.</span>'}
            </div>
          </div>

          <!-- Diversification candidates -->
          <div class="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <span class="font-bold text-slate-800 flex items-center gap-1.5">
              <span>🛡️</span> Low-Correlation Diversifiers (${diversifiers.length})
            </span>
            <div class="flex flex-wrap gap-1.5">
              ${diversifiers.slice(0, 8).map((p) => `
                <span class="px-2 py-0.5 rounded bg-white border border-slate-200 font-mono font-bold text-emerald-700 text-[11px]">
                  ${p.symbol.replace('.NS', '')} (r = ${p.r.toFixed(2)})
                </span>
              `).join('')}
            </div>
          </div>
        </div>
      </div>`;
  }

  /**
   * Render Selected Stocks Badges in Control Bar.
   */
  function renderSelectedBadges() {
    const container = document.getElementById('selected-badges-container');
    const badge = document.getElementById('selection-status-badge');
    if (!container) return;

    const list = Array.from(State.selectedTickers);
    if (badge) {
      if (State.isDefaultSelection) {
        badge.textContent = `Default Subset (${list.length} Stocks)`;
        badge.className = 'px-2.5 py-1 rounded-full text-[11px] font-bold bg-indigo-50 text-indigo-700 border border-indigo-200 font-mono';
      } else {
        badge.textContent = `Custom Selection (${list.length} Stocks)`;
        badge.className = 'px-2.5 py-1 rounded-full text-[11px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200 font-mono';
      }
    }

    if (list.length === 0) {
      container.innerHTML = `<span class="text-xs text-rose-500 font-medium">No stocks selected. Please add stocks.</span>`;
      return;
    }

    container.innerHTML = list.map((t) => `
      <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-50 border border-indigo-200 text-indigo-800 font-mono font-bold text-xs">
        ${t.replace('.NS', '')}
        <button onclick="TradeSenseMarketRel.removeStock('${t}')" class="text-indigo-400 hover:text-rose-600 font-sans font-bold leading-none" title="Remove">&times;</button>
      </span>
    `).join('');
  }

  /**
   * Synchronize all visualizations and KPI summary cards.
   */
  function updateVisualizations() {
    const activeList = Array.from(State.selectedTickers);
    const edges = extractQualifyingEdges(activeList, State.rawCorrMatrix, State.threshold);
    const degrees = computeDegrees(activeList, edges);

    // 1. Update KPI Summary Cards
    const kpiStocks = document.getElementById('kpi-rel-stocks');
    const kpiEdges = document.getElementById('kpi-rel-edges');
    const kpiDensity = document.getElementById('kpi-rel-density');
    const kpiAvgCorr = document.getElementById('kpi-rel-avg-corr');

    const N = activeList.length;
    const maxEdges = N > 1 ? (N * (N - 1)) / 2 : 1;
    const densityPct = N > 1 ? ((edges.length / maxEdges) * 100).toFixed(1) : '0.0';

    // Calculate average pairwise correlation across selection
    let sumCorr = 0, countCorr = 0;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const u = activeList[i], v = activeList[j];
        if (State.rawCorrMatrix[u] && State.rawCorrMatrix[u][v] !== undefined) {
          sumCorr += Number(State.rawCorrMatrix[u][v]);
          countCorr++;
        }
      }
    }
    const avgR = countCorr > 0 ? (sumCorr / countCorr).toFixed(2) : '+0.00';

    if (kpiStocks) kpiStocks.textContent = N;
    if (kpiEdges) kpiEdges.textContent = edges.length;
    if (kpiDensity) kpiDensity.textContent = `${densityPct}%`;
    if (kpiAvgCorr) kpiAvgCorr.textContent = Number(avgR) >= 0 ? `+${avgR}` : avgR;

    // 2. Render Market Network
    renderMarketNetwork('market-network-plot', activeList, edges, degrees, State.focusedNode);

    // 3. Render Correlation Matrix Heatmap
    renderCorrelationHeatmap('correlation-heatmap-plot', activeList, State.rawCorrMatrix);

    // 4. Update Badge controls
    renderSelectedBadges();
  }

  /**
   * Initialize and fetch data.
   */
  async function initMarketRelationships(threshold = 0.70) {
    State.threshold = threshold;
    const disp = document.getElementById('threshold-display-val');
    if (disp) disp.textContent = threshold.toFixed(2);

    try {
      const [corrRes, graphRes, clusterRes, sectorRes] = await Promise.all([
        TradeSense.apiCall('/api/correlation'),
        TradeSense.apiCall(`/api/graph?threshold=${threshold}`),
        TradeSense.apiCall(`/api/clusters?threshold=${threshold}`).catch(() => ({ clusters: [], unclustered: [] })),
        TradeSense.apiCall('/api/sectors/analysis').catch(() => ({ sectors: [] })),
      ]);

      State.cachedCorrRes = corrRes;
      State.cachedGraphRes = graphRes;
      State.rawCorrMatrix = corrRes?.matrix || {};
      State.allUniverseTickers = corrRes?.all_tickers || corrRes?.tickers || [];
      State.stockClusters = clusterRes?.clusters || graphRes?.clusters || [];
      State.unclusteredStocks = clusterRes?.unclustered || graphRes?.unclustered || [];
      State.allSectors = sectorRes?.sectors || [];

      // Populate dropdowns
      populateSelectionDropdowns();

      // Check for URL parameters (?subset=TCS.NS,INFY.NS or ?sector=Banking or ?stock=TCS.NS)
      const urlParams = new URLSearchParams(window.location.search);
      const urlSubset = urlParams.get('subset');
      const urlSector = urlParams.get('sector');
      const urlStock = urlParams.get('stock');

      if (urlSubset) {
        const tickers = urlSubset.split(',').map((s) => s.trim()).filter((s) => State.allUniverseTickers.includes(s));
        if (tickers.length > 0) {
          State.selectedTickers = new Set(tickers);
          State.isDefaultSelection = false;
        }
      } else if (urlSector) {
        const sectorObj = State.allSectors.find((s) => s.sector.toLowerCase() === urlSector.toLowerCase());
        if (sectorObj && sectorObj.stock_symbols) {
          State.selectedTickers = new Set(sectorObj.stock_symbols);
          State.isDefaultSelection = false;
          window.TradeSenseMarketRel.setSelectionMode('sector');
          const secSelect = document.getElementById('sector-dropdown');
          if (secSelect) secSelect.value = sectorObj.sector;
        }
      } else if (urlStock && State.allUniverseTickers.includes(urlStock)) {
        window.TradeSenseMarketRel.onSingleStockSelected(urlStock);
        return;
      } else if (State.isDefaultSelection) {
        const validDefault = DEFAULT_SUBSET.filter((s) => State.allUniverseTickers.includes(s));
        State.selectedTickers = new Set(validDefault.length >= 3 ? validDefault : State.allUniverseTickers.slice(0, 10));
      }

      updateVisualizations();
    } catch (err) {
      console.error('Market relationships load error:', err);
    }
  }

  /**
   * Populate group, sector, and single stock selectors.
   */
  function populateSelectionDropdowns() {
    // 1. Group Dropdown
    const grpSelect = document.getElementById('group-dropdown');
    if (grpSelect) {
      if (State.stockClusters.length === 0) {
        grpSelect.innerHTML = `<option value="">No multi-stock clusters detected at r ≥ ${State.threshold.toFixed(2)}</option>`;
      } else {
        grpSelect.innerHTML = `<option value="">-- Choose a Co-Movement Cluster --</option>` +
          State.stockClusters.map((c, i) => `
            <option value="${i}">
              Cluster #${c.cluster_id || (i + 1)} (${c.size} stocks, avg r = ${c.avg_correlation}) - Led by ${c.lead_stock?.replace('.NS', '')}
            </option>
          `).join('');
      }
    }

    // 2. Sector Dropdown
    const secSelect = document.getElementById('sector-dropdown');
    if (secSelect && State.allSectors.length > 0) {
      secSelect.innerHTML = `<option value="">-- Choose a Sector --</option>` +
        State.allSectors.map((s) => `
          <option value="${s.sector}">
            ${s.sector} (${s.stocks_analyzed} stocks)
          </option>
        `).join('');
    }

    // 3. Single Stock Dropdown
    const singleSelect = document.getElementById('single-stock-dropdown');
    if (singleSelect) {
      singleSelect.innerHTML = `<option value="">-- Choose Focus Stock --</option>` +
        State.allUniverseTickers.map((t) => `
          <option value="${t}">${t.replace('.NS', '')} (${t})</option>
        `).join('');
    }
  }

  // Public Interface
  window.TradeSenseMarketRel = {
    init: initMarketRelationships,

    setThreshold: (val) => {
      State.threshold = parseFloat(val);
      const disp = document.getElementById('threshold-display-val');
      if (disp) disp.textContent = State.threshold.toFixed(2);
      updateVisualizations();
    },

    setSelectionMode: (mode) => {
      State.selectionMode = mode;
      const modes = ['multi', 'single', 'group', 'sector', 'all'];
      modes.forEach((m) => {
        const btn = document.getElementById(`mode-btn-${m}`);
        if (btn) {
          if (m === mode) {
            btn.className = 'px-3 py-1 rounded-lg bg-indigo-600 text-white shadow-xs transition-colors';
          } else {
            btn.className = 'px-3 py-1 rounded-lg text-slate-600 hover:bg-slate-200 transition-colors';
          }
        }
      });

      // Toggle panel visibility
      const multiPanel = document.getElementById('multi-search-panel');
      const grpPanel = document.getElementById('group-select-panel');
      const secPanel = document.getElementById('sector-select-panel');
      const singlePanel = document.getElementById('single-select-panel');

      if (multiPanel) multiPanel.classList.toggle('hidden', mode !== 'multi');
      if (grpPanel) grpPanel.classList.toggle('hidden', mode !== 'group');
      if (secPanel) secPanel.classList.toggle('hidden', mode !== 'sector');
      if (singlePanel) singlePanel.classList.toggle('hidden', mode !== 'single');

      if (mode === 'all') {
        State.selectedTickers = new Set(State.allUniverseTickers);
        State.isDefaultSelection = false;
        updateVisualizations();
      }
    },

    onSearchInput: (query) => {
      const dropdown = document.getElementById('search-dropdown');
      if (!dropdown) return;

      const q = query.trim().toUpperCase();
      if (!q) {
        dropdown.classList.add('hidden');
        return;
      }

      const matches = State.allUniverseTickers.filter((t) => t.toUpperCase().includes(q));
      if (matches.length === 0) {
        dropdown.innerHTML = `<div class="p-3 text-xs text-slate-400">No matching stocks found.</div>`;
      } else {
        dropdown.innerHTML = matches.slice(0, 15).map((t) => {
          const isSelected = State.selectedTickers.has(t);
          return `
            <div onclick="TradeSenseMarketRel.toggleStock('${t}')" class="px-3.5 py-2 hover:bg-indigo-50 cursor-pointer flex items-center justify-between text-xs border-b border-slate-100 last:border-0">
              <span class="font-mono font-bold text-slate-900">${t.replace('.NS', '')} <span class="text-slate-400 font-sans font-normal">(${t})</span></span>
              <span class="text-[11px] font-bold ${isSelected ? 'text-rose-600' : 'text-indigo-600'}">
                ${isSelected ? '✓ Added (Click to remove)' : '+ Add to selection'}
              </span>
            </div>
          `;
        }).join('');
      }
      dropdown.classList.remove('hidden');
    },

    toggleStock: (ticker) => {
      State.isDefaultSelection = false;
      if (State.selectedTickers.has(ticker)) {
        State.selectedTickers.delete(ticker);
      } else {
        State.selectedTickers.add(ticker);
      }
      const searchInp = document.getElementById('stock-search-input');
      const dropdown = document.getElementById('search-dropdown');
      if (searchInp) searchInp.value = '';
      if (dropdown) dropdown.classList.add('hidden');
      updateVisualizations();
    },

    removeStock: (ticker) => {
      State.isDefaultSelection = false;
      State.selectedTickers.delete(ticker);
      updateVisualizations();
    },

    clearStockSelection: () => {
      State.isDefaultSelection = false;
      State.selectedTickers.clear();
      updateVisualizations();
    },

    resetToDefaultSelection: () => {
      State.isDefaultSelection = true;
      State.selectedTickers = new Set(DEFAULT_SUBSET.filter((s) => State.allUniverseTickers.includes(s)));
      State.focusedNode = null;
      State.focusedPair = null;
      window.TradeSenseMarketRel.setSelectionMode('multi');
      updateVisualizations();
    },

    onGroupSelected: (idxStr) => {
      if (idxStr === '') return;
      const idx = parseInt(idxStr);
      const cluster = State.stockClusters[idx];
      if (cluster && cluster.members) {
        State.isDefaultSelection = false;
        State.selectedTickers = new Set(cluster.members);
        updateVisualizations();
      }
    },

    onSectorSelected: (sectorName) => {
      if (!sectorName) return;
      const sectorObj = State.allSectors.find((s) => s.sector === sectorName);
      if (sectorObj && sectorObj.stock_symbols) {
        State.isDefaultSelection = false;
        State.selectedTickers = new Set(sectorObj.stock_symbols);
        updateVisualizations();
      }
    },

    onSingleStockSelected: (ticker) => {
      if (!ticker) return;
      State.isDefaultSelection = false;
      // In single stock mode: select that stock + its direct co-moving partners in universe
      const partners = [ticker];
      if (State.rawCorrMatrix[ticker]) {
        Object.keys(State.rawCorrMatrix[ticker]).forEach((o) => {
          if (o !== ticker && Math.abs(Number(State.rawCorrMatrix[ticker][o])) >= State.threshold) {
            partners.push(o);
          }
        });
      }
      State.selectedTickers = new Set(partners);
      State.focusedNode = ticker;
      updateVisualizations();
      updateStockInspector(ticker);
    },

    clearFocus: () => {
      State.focusedNode = null;
      State.focusedPair = null;
      updateVisualizations();
      updateStockInspector(null);
    },

    fitNetworkScreen: () => {
      const el = document.getElementById('market-network-plot');
      if (el) Plotly.relayout(el, { 'xaxis.autorange': true, 'yaxis.autorange': true });
    },

    resetNetworkLayout: () => {
      window.TradeSenseMarketRel.clearFocus();
      window.TradeSenseMarketRel.fitNetworkScreen();
    },
  };
})();
