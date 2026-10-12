/**
 * market_relationships.js
 * -----------------------
 * Interactive force-directed market network and diverging correlation heatmap
 * for TradeSense V2.1.
 *
 * Implements:
 * - Fruchterman-Reingold force-directed 2D layout calculation
 * - Blue-to-Red diverging correlation color scale (fixed [-1, 1])
 * - Cross-visualization linking (Network <-> Heatmap)
 * - Node degree scaling, neighbor highlighting, and dimming
 * - Search, sector filtering, and subset selection
 * - Simple View vs Technical View toggles
 */

(function () {
  'use strict';

  // State management
  const State = {
    threshold: 0.70,
    selectedStock: null,
    selectedPair: null,
    viewMode: 'simple', // 'simple' | 'technical'
    activeSubset: 'all', // 'all' | 'top20' | sectorName
    searchQuery: '',
    graphData: null,
    corrData: null,
    analysisContext: null,
  };

  // Diverging color scale: Blue (-1.0) -> Light Neutral (0.0) -> Red (+1.0)
  const DIVERGING_CORR_SCALE = [
    [0.0, '#1d4ed8'],  // Deep blue (-1.0)
    [0.25, '#60a5fa'], // Soft blue (-0.5)
    [0.48, '#f1f5f9'], // Off-white/slate near zero (-0.04)
    [0.50, '#f8fafc'], // Neutral zero (0.00)
    [0.52, '#f1f5f9'], // Off-white/slate near zero (+0.04)
    [0.75, '#fb7185'], // Soft rose/red (+0.5)
    [1.0, '#dc2626'],  // Deep red (+1.0)
  ];

  // Distinct qualitative colors for graph-coloring independent sets
  const COLORING_PALETTE = [
    '#4f46e5', '#0ea5e9', '#10b981', '#f59e0b', '#ec4899',
    '#8b5cf6', '#06b6d4', '#84cc16', '#f97316', '#64748b'
  ];

  /**
   * Fast 2D force-directed spring-electrical layout (Fruchterman-Reingold variant).
   * Runs in <15ms for 100 nodes, clustering connected components naturally.
   */
  function computeForceDirectedLayout(nodes, edges) {
    const N = nodes.length;
    if (N === 0) return {};

    const pos = {};
    // Initial arrangement in spiral with slight jitter to break symmetry
    nodes.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / N;
      const radius = 0.6 + (i % 4) * 0.08;
      pos[n] = {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
      };
    });

    const k = Math.sqrt(4.0 / Math.max(N, 1));
    const iterations = 55;
    let temp = 0.12;

    for (let it = 0; it < iterations; it++) {
      // 1. Repulsion between all node pairs
      for (let i = 0; i < N; i++) {
        const u = nodes[i];
        for (let j = i + 1; j < N; j++) {
          const v = nodes[j];
          const dx = pos[u].x - pos[v].x;
          const dy = pos[u].y - pos[v].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
          const f = (k * k) / dist;
          const fx = (dx / dist) * f;
          const fy = (dy / dist) * f;
          pos[u].vx += fx;
          pos[u].vy += fy;
          pos[v].vx -= fx;
          pos[v].vy -= fy;
        }
      }

      // 2. Attraction along qualifying edges
      edges.forEach((e) => {
        const u = e.source;
        const v = e.target;
        if (!pos[u] || !pos[v]) return;
        const dx = pos[v].x - pos[u].x;
        const dy = pos[v].y - pos[u].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const weight = Math.abs(e.weight || 0.7);
        const f = ((dist * dist) / k) * (weight * 1.6);
        const fx = (dx / dist) * f;
        const fy = (dy / dist) * f;
        pos[u].vx += fx;
        pos[u].vy += fy;
        pos[v].vx -= fx;
        pos[v].vy -= fy;
      });

      // 3. Central gravitational anchor
      nodes.forEach((n) => {
        const p = pos[n];
        const distCenter = Math.sqrt(p.x * p.x + p.y * p.y) || 0.01;
        p.vx -= (p.x / distCenter) * 0.06 * distCenter;
        p.vy -= (p.y / distCenter) * 0.06 * distCenter;

        const disp = Math.sqrt(p.vx * p.vx + p.vy * p.vy) || 0.01;
        const lim = Math.min(disp, temp);
        p.x += (p.vx / disp) * lim;
        p.y += (p.vy / disp) * lim;
        p.vx = 0;
        p.vy = 0;
      });

      temp *= 0.94; // Exponential temperature cooling
    }

    // Normalize coordinates strictly to [-0.95, 0.95]
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
   * Render the professional force-directed market network.
   */
  function renderMarketNetwork(containerId, graphData, selectedNode = null) {
    const el = document.getElementById(containerId);
    if (!el || !graphData) return;

    const tickers = graphData.tickers || [];
    const edges = graphData.edges || [];
    const stats = graphData.stats || {};
    const coloring = graphData.coloring_dict || {};
    const degrees = stats.degrees || {};

    if (tickers.length === 0) {
      el.innerHTML = `
        <div class="flex items-center justify-center h-full text-slate-400 text-xs">
          No eligible tickers available in the active analysis context.
        </div>`;
      return;
    }

    // Adjacency map for fast neighbor queries
    const adjMap = {};
    tickers.forEach((t) => { adjMap[t] = new Set(); });
    edges.forEach((e) => {
      if (adjMap[e.source]) adjMap[e.source].add(e.target);
      if (adjMap[e.target]) adjMap[e.target].add(e.source);
    });

    // Compute or retrieve node layout positions
    if (!graphData._layoutPositions) {
      graphData._layoutPositions = computeForceDirectedLayout(tickers, edges);
    }
    const pos = graphData._layoutPositions;

    // Identify active neighbors if a stock is selected
    const isSelected = Boolean(selectedNode);
    const activeNeighbors = isSelected ? (adjMap[selectedNode] || new Set()) : null;

    // Build edge line coordinates and hover tooltips
    const edgeX = [], edgeY = [];
    const edgeMidX = [], edgeMidY = [], edgeMidText = [];

    edges.forEach((e) => {
      const s = pos[e.source], t = pos[e.target];
      if (s && t) {
        edgeX.push(s.x, t.x, null);
        edgeY.push(s.y, t.y, null);

        // Edge midpoint hover marker
        edgeMidX.push((s.x + t.x) / 2);
        edgeMidY.push((s.y + t.y) / 2);
        const wStr = (e.weight !== undefined) ? e.weight.toFixed(3) : State.threshold.toFixed(2);
        edgeMidText.push(`${e.source.replace('.NS', '')} ↔ ${e.target.replace('.NS', '')}<br><b>Pearson r = ${wStr}</b>`);
      }
    });

    // Edge lines trace
    const edgeTrace = {
      type: 'scatter',
      x: edgeX,
      y: edgeY,
      mode: 'lines',
      line: {
        width: isSelected ? 1.0 : 1.4,
        color: isSelected ? '#e2e8f0' : '#cbd5e1',
      },
      hoverinfo: 'none',
      showlegend: false,
    };

    // Selected edge highlighted overlay
    const selEdgeX = [], selEdgeY = [];
    if (isSelected) {
      edges.forEach((e) => {
        if (e.source === selectedNode || e.target === selectedNode) {
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
        width: 3.0,
        color: '#4f46e5',
      },
      hoverinfo: 'none',
      showlegend: false,
    };

    // Edge midpoints for hover interaction
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

    // Node markers and labels
    // Rule: Hide most labels by default to prevent clutter; show only selected + neighbors + top hubs
    const topHub = stats.most_connected_stock;
    const nodeX = [], nodeY = [];
    const nodeText = [], nodeHover = [];
    const nodeColors = [], nodeSizes = [], nodeBorders = [], nodeBorderWidths = [];

    tickers.forEach((t) => {
      const p = pos[t] || { x: 0, y: 0 };
      nodeX.push(p.x);
      nodeY.push(p.y);

      const cleanSym = t.replace('.NS', '');
      const deg = degrees[t] || 0;
      const colorGroup = coloring[t] !== undefined ? coloring[t] : 0;
      const baseColor = COLORING_PALETTE[colorGroup % COLORING_PALETTE.length];

      // Degree-based node sizing: 12px min, up to 26px max
      const baseSize = Math.min(12 + deg * 2.2, 26);

      let isDimmed = false;
      let isTarget = false;
      let isNeighbor = false;

      if (isSelected) {
        if (t === selectedNode) {
          isTarget = true;
        } else if (activeNeighbors && activeNeighbors.has(t)) {
          isNeighbor = true;
        } else {
          isDimmed = true;
        }
      }

      // Selective labeling
      let label = '';
      if (isTarget || isNeighbor || (!isSelected && (deg >= 4 || t === topHub))) {
        label = cleanSym;
      }
      nodeText.push(label);

      // Tooltip content
      nodeHover.push(
        `<b>${cleanSym}</b> (${t})<br>` +
        `Degree (Connections): <b>${deg}</b><br>` +
        `Independent Set: <b>Group #${colorGroup + 1}</b><br>` +
        `<i>Click to highlight correlations</i>`
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
        nodeColors.push('rgba(203, 213, 225, 0.45)');
        nodeSizes.push(Math.max(baseSize - 4, 8));
        nodeBorders.push('rgba(148, 163, 184, 0.3)');
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
        line: {
          color: nodeBorders,
          width: nodeBorderWidths,
        },
      },
      hoverinfo: 'text',
      hovertext: nodeHover,
      customdata: tickers,
      showlegend: false,
    };

    const traces = [edgeTrace];
    if (isSelected) traces.push(selEdgeTrace);
    traces.push(edgeHoverTrace, nodeTrace);

    const layout = {
      margin: { t: 15, l: 15, r: 15, b: 15 },
      hovermode: 'closest',
      dragmode: 'pan',
      paper_bgcolor: 'white',
      plot_bgcolor: '#f8fafc',
      xaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        range: [-1.15, 1.15],
      },
      yaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        range: [-1.15, 1.15],
      },
      annotations: edges.length === 0 ? [{
        text: `No connections qualify at r ≥ ${State.threshold.toFixed(2)}.<br><span style="font-size:11px;color:#64748b;">Stocks exhibit statistically independent co-movement under this threshold.</span>`,
        showarrow: false,
        font: { family: 'Inter', size: 13, color: '#334155' },
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
      // Wire up interactive click selection
      el.removeAllListeners && el.removeAllListeners('plotly_click');
      el.on('plotly_click', (data) => {
        if (!data || !data.points || data.points.length === 0) return;
        const pt = data.points[0];
        if (pt.customdata) {
          const clickedTicker = pt.customdata;
          handleStockSelection(clickedTicker === State.selectedStock ? null : clickedTicker);
        }
      });
    });
  }

  /**
   * Render the professional diverging correlation heatmap.
   */
  function renderCorrelationHeatmap(containerId, matrix, allLabels, selectedPair = null) {
    const el = document.getElementById(containerId);
    if (!el || !matrix || !allLabels) return;

    // Filter labels based on active search or subset selection
    let filteredLabels = allLabels.slice();
    if (State.searchQuery) {
      const q = State.searchQuery.toUpperCase();
      filteredLabels = filteredLabels.filter((s) => s.toUpperCase().includes(q));
    } else if (State.activeSubset === 'top20' && State.graphData?.stats?.degrees) {
      const degs = State.graphData.stats.degrees;
      filteredLabels = allLabels
        .slice()
        .sort((a, b) => (degs[b] || 0) - (degs[a] || 0))
        .slice(0, 20);
    }

    if (filteredLabels.length === 0) {
      el.innerHTML = `
        <div class="flex items-center justify-center h-full text-slate-400 text-xs">
          No stocks match the search filter. Clear search to view matrix.
        </div>`;
      return;
    }

    const cleanDisplayLabels = filteredLabels.map((s) => s.replace('.NS', ''));
    const zValues = filteredLabels.map((rowSym) =>
      filteredLabels.map((colSym) => {
        if (matrix[rowSym] && matrix[rowSym][colSym] !== undefined) {
          return matrix[rowSym][colSym];
        }
        return rowSym === colSym ? 1.0 : 0.0;
      })
    );

    const showTextValues = filteredLabels.length <= 16;
    const textTemplates = showTextValues
      ? zValues.map((row) => row.map((v) => v.toFixed(2)))
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
      customdata: filteredLabels.map((r) => filteredLabels.map((c) => [r, c])),
    };

    const layout = {
      margin: { t: 30, l: 60, r: 25, b: 60 },
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      xaxis: {
        tickangle: -45,
        tickfont: { family: 'Inter', size: filteredLabels.length > 25 ? 8 : 10 },
      },
      yaxis: {
        tickfont: { family: 'Inter', size: filteredLabels.length > 25 ? 8 : 10 },
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
        const stockA = filteredLabels[pt.yIndex || pt.pointIndex?.[0] || pt.y];
        const stockB = filteredLabels[pt.xIndex || pt.pointIndex?.[1] || pt.x];
        const rVal = pt.z;
        if (stockA && stockB) {
          handleHeatmapCellSelection(stockA, stockB, rVal);
        }
      });
    });
  }

  /**
   * Handle stock selection from Network.
   */
  function handleStockSelection(ticker) {
    State.selectedStock = ticker;
    State.selectedPair = null;

    // Re-render network with focused node & neighbors highlighted
    renderMarketNetwork('market-network-plot', State.graphData, ticker);

    // Update inspector details panel
    updateStockInspector(ticker);
  }

  /**
   * Handle cell selection from Heatmap.
   */
  function handleHeatmapCellSelection(stockA, stockB, rVal) {
    State.selectedPair = { stockA, stockB, r: rVal };
    State.selectedStock = stockA;

    // Highlight stockA in network
    renderMarketNetwork('market-network-plot', State.graphData, stockA);

    // Update pair callout badge
    updatePairCallout(stockA, stockB, rVal);
    updateStockInspector(stockA);
  }

  /**
   * Update the Pair Callout detail card.
   */
  function updatePairCallout(stockA, stockB, rVal) {
    const card = document.getElementById('selected-pair-card');
    if (!card) return;

    card.classList.remove('hidden');
    const titleEl = document.getElementById('pair-title');
    const valEl = document.getElementById('pair-val');
    const statusEl = document.getElementById('pair-status');
    const descEl = document.getElementById('pair-desc');

    const cleanA = stockA.replace('.NS', '');
    const cleanB = stockB.replace('.NS', '');
    const qualifies = Math.abs(rVal) >= State.threshold;

    if (titleEl) titleEl.textContent = `${cleanA} ↔ ${cleanB}`;
    if (valEl) {
      valEl.textContent = `r = ${rVal.toFixed(3)}`;
      valEl.className = rVal >= 0 ? 'text-rose-600 font-mono font-bold' : 'text-blue-600 font-mono font-bold';
    }
    if (statusEl) {
      if (qualifies) {
        statusEl.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200';
        statusEl.textContent = 'Active Connection (Edge Qualified)';
      } else {
        statusEl.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-slate-100 text-slate-700 border border-slate-200';
        statusEl.textContent = 'Disconnected (Sub-threshold)';
      }
    }
    if (descEl) {
      if (stockA === stockB) {
        descEl.textContent = 'Identity correlation: A stock with itself always has r = 1.0.';
      } else if (qualifies) {
        descEl.textContent = `Pearson correlation ${rVal.toFixed(3)} exceeds active threshold ${State.threshold.toFixed(2)}. An edge connects both stocks in the network.`;
      } else {
        descEl.textContent = `Correlation ${rVal.toFixed(3)} is below threshold ${State.threshold.toFixed(2)}. These stocks do not form a network edge under current sensitivity.`;
      }
    }
  }

  /**
   * Update the Stock Inspector details panel below the charts.
   */
  function updateStockInspector(ticker) {
    const inspectorEl = document.getElementById('stock-inspector-panel');
    if (!inspectorEl) return;

    if (!ticker) {
      inspectorEl.innerHTML = `
        <div class="p-6 text-center text-slate-400 text-xs">
          Select any stock in the network or heatmap to inspect its connections and correlation ranking.
        </div>`;
      return;
    }

    const cleanSym = ticker.replace('.NS', '');
    const matrix = State.corrData?.matrix || {};
    const degrees = State.graphData?.stats?.degrees || {};
    const deg = degrees[ticker] || 0;
    const coloring = State.graphData?.coloring_dict || {};
    const groupNum = (coloring[ticker] !== undefined ? coloring[ticker] : 0) + 1;

    // Get sorted correlations for this ticker
    const stockRow = matrix[ticker] || {};
    const sortedPairs = Object.entries(stockRow)
      .filter(([s]) => s !== ticker)
      .map(([s, r]) => ({ ticker: s, r: Number(r) }))
      .sort((a, b) => b.r - a.r);

    const topPos = sortedPairs.slice(0, 4);
    const topNeg = sortedPairs.slice(-3).reverse();

    inspectorEl.innerHTML = `
      <div class="p-5 bg-white rounded-2xl border border-slate-200 shadow-xs space-y-4">
        <div class="flex items-center justify-between flex-wrap gap-2 border-b border-slate-100 pb-3">
          <div class="flex items-center gap-2.5">
            <span class="w-8 h-8 rounded-xl bg-indigo-600 text-white font-mono font-bold text-xs flex items-center justify-center">
              ${cleanSym.substring(0, 3)}
            </span>
            <div>
              <h4 class="font-bold text-slate-900 text-sm">${cleanSym} <span class="text-xs text-slate-400 font-normal">(${ticker})</span></h4>
              <p class="text-[11px] text-slate-500 font-medium">Network Degree: <strong class="text-indigo-600">${deg} connections</strong> · Independent Set: <strong class="text-slate-700">Group #${groupNum}</strong></p>
            </div>
          </div>
          <button onclick="TradeSenseMarketRel.clearSelection()" class="text-xs text-slate-500 hover:text-indigo-600 underline font-medium">
            Clear Selection ✕
          </button>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-sans">
          <!-- Strongest Co-Moving Partners -->
          <div class="space-y-2">
            <span class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Strongest Co-Moving Partners</span>
            <div class="space-y-1.5">
              ${topPos.map(p => `
                <div class="p-2 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-between">
                  <span class="font-bold text-slate-800">${p.ticker.replace('.NS', '')}</span>
                  <div class="flex items-center gap-2">
                    <div class="w-16 bg-slate-200 h-1.5 rounded-full overflow-hidden">
                      <div class="bg-rose-500 h-full rounded-full" style="width: ${Math.max(p.r * 100, 0)}%"></div>
                    </div>
                    <span class="font-mono font-bold text-rose-600 text-[11px]">+${p.r.toFixed(3)}</span>
                  </div>
                </div>
              `).join('')}
            </div>
          </div>

          <!-- Lowest or Inverse Correlation (Diversifiers) -->
          <div class="space-y-2">
            <span class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Top Natural Diversifiers (Lowest r)</span>
            <div class="space-y-1.5">
              ${topNeg.map(p => `
                <div class="p-2 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-between">
                  <span class="font-bold text-slate-800">${p.ticker.replace('.NS', '')}</span>
                  <div class="flex items-center gap-2">
                    <span class="font-mono font-bold text-blue-600 text-[11px]">${p.r >= 0 ? '+' : ''}${p.r.toFixed(3)}</span>
                  </div>
                </div>
              `).join('')}
            </div>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Main initializer for Market Relationships page.
   */
  async function initMarketRelationships(threshold = 0.70) {
    State.threshold = threshold;
    const thresholdValEl = document.getElementById('threshold-display-val');
    if (thresholdValEl) thresholdValEl.textContent = threshold.toFixed(2);

    try {
      // Fetch analysis graph and correlation matrix concurrently
      const [graphRes, corrRes, ctxRes] = await Promise.all([
        TradeSense.apiCall(`/api/graph?threshold=${threshold}`),
        TradeSense.apiCall('/api/correlation'),
        TradeSense.apiCall('/api/context').catch(() => null),
      ]);

      State.graphData = graphRes;
      State.corrData = corrRes;
      State.analysisContext = ctxRes;

      // Update 4 Summary KPI Cards
      updateSummaryCards(graphRes, corrRes, ctxRes, threshold);

      // Render Visualizations
      renderMarketNetwork('market-network-plot', graphRes, State.selectedStock);
      renderCorrelationHeatmap('correlation-heatmap-plot', corrRes.matrix, corrRes.tickers || []);

      // Update Compact Network Insights (Phase 6)
      updateNetworkInsights(graphRes, ctxRes);
    } catch (err) {
      console.error('Market relationships load error:', err);
    }
  }

  /**
   * Update the 4 Summary KPI Cards.
   */
  function updateSummaryCards(graphRes, corrRes, ctxRes, threshold) {
    const stats = graphRes?.stats || {};
    const vertCount = ctxRes?.validated_count || graphRes?.tickers?.length || 93;
    const edgeCount = stats.num_edges || graphRes?.edges?.length || 0;
    const densityPct = ((stats.density || 0) * 100).toFixed(1);

    const elV = document.getElementById('kpi-rel-stocks');
    const elE = document.getElementById('kpi-rel-edges');
    const elD = document.getElementById('kpi-rel-density');
    const elT = document.getElementById('kpi-rel-threshold');

    if (elV) elV.textContent = vertCount;
    if (elE) elE.textContent = edgeCount;
    if (elD) elD.textContent = `${densityPct}%`;
    if (elT) elT.textContent = `r ≥ ${threshold.toFixed(2)}`;
  }

  /**
   * Update compact Network Insights section (Phase 6).
   */
  function updateNetworkInsights(graphRes, ctxRes) {
    const stats = graphRes?.stats || {};
    const vertCount = ctxRes?.validated_count || graphRes?.tickers?.length || 93;
    const edgeCount = stats.num_edges || 0;
    const densityPct = ((stats.density || 0) * 100).toFixed(1);
    const hub = stats.most_connected_stock || 'None';
    const hubDeg = stats.degrees?.[hub] || 0;
    const components = graphRes?.connected_components || [];

    const insV = document.getElementById('insight-val-stocks');
    const insE = document.getElementById('insight-val-edges');
    const insD = document.getElementById('insight-val-density');
    const insH = document.getElementById('insight-val-hub');
    const insC = document.getElementById('insight-val-components');

    if (insV) insV.textContent = vertCount;
    if (insE) insE.textContent = edgeCount;
    if (insD) insD.textContent = `${densityPct}%`;
    if (insH) insH.textContent = `${hub.replace('.NS', '')} (${hubDeg} links)`;
    if (insC) insC.textContent = `${components.length} Components`;

    // Dynamic explanatory text strictly respecting mathematical definitions
    const explEl = document.getElementById('network-insight-explanation');
    if (explEl) {
      if (edgeCount === 0) {
        explEl.innerHTML = `
          Under threshold <strong>r ≥ ${State.threshold.toFixed(2)}</strong>, no pair of stocks shares sufficient linear co-movement to form an edge. 
          The graph decomposes into <strong>${vertCount} isolated single-vertex components</strong>.`;
      } else {
        explEl.innerHTML = `
          The network contains <strong>${edgeCount} active edges</strong> across <strong>${vertCount} stocks</strong>, creating <strong>${components.length} connected components</strong>.
          <strong>${hub.replace('.NS', '')}</strong> has the highest node degree (${hubDeg} connections), making it the primary hub.`;
      }
    }
  }

  // Public API exposure
  window.TradeSenseMarketRel = {
    init: initMarketRelationships,
    setThreshold: (val) => {
      initMarketRelationships(parseFloat(val));
    },
    setSearchQuery: (query) => {
      State.searchQuery = query;
      if (State.corrData) {
        renderCorrelationHeatmap('correlation-heatmap-plot', State.corrData.matrix, State.corrData.tickers || []);
      }
    },
    setSubset: (subset) => {
      State.activeSubset = subset;
      if (State.corrData) {
        renderCorrelationHeatmap('correlation-heatmap-plot', State.corrData.matrix, State.corrData.tickers || []);
      }
    },
    setViewMode: (mode) => {
      State.viewMode = mode;
      const simpleBtn = document.getElementById('btn-view-simple');
      const techBtn = document.getElementById('btn-view-technical');
      const techPanel = document.getElementById('technical-details-panel');

      if (mode === 'technical') {
        if (techPanel) techPanel.classList.remove('hidden');
        if (techBtn) techBtn.className = 'px-3 py-1 text-xs font-bold rounded-lg bg-indigo-600 text-white shadow-xs';
        if (simpleBtn) simpleBtn.className = 'px-3 py-1 text-xs font-bold rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200';
      } else {
        if (techPanel) techPanel.classList.add('hidden');
        if (simpleBtn) simpleBtn.className = 'px-3 py-1 text-xs font-bold rounded-lg bg-indigo-600 text-white shadow-xs';
        if (techBtn) techBtn.className = 'px-3 py-1 text-xs font-bold rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200';
      }
    },
    clearSelection: () => {
      State.selectedStock = null;
      State.selectedPair = null;
      renderMarketNetwork('market-network-plot', State.graphData, null);
      const pairCard = document.getElementById('selected-pair-card');
      if (pairCard) pairCard.classList.add('hidden');
      updateStockInspector(null);
    },
  };
})();
