# TradeSense V2 — Intelligent Market Insights

> **Subtitle:** *Discrete Mathematics Powered Stock Market Analysis & Portfolio Intelligence*

TradeSense V2 is a modern computational finance and discrete mathematics application that transforms stock market data into formal discrete mathematical structures (Pearson correlation matrices, undirected graphs, partially ordered sets, Hasse diagrams, graph coloring, set operations, and Boolean screening logic) to automatically generate, score, and rank candidate portfolios.

---

## 🏛️ System Architecture

- **Backend:** FastAPI (Python 3.12+), SQLAlchemy ORM with SQLite, Pydantic v2 schemas
- **Mathematical Engine:** All algorithms implemented **from scratch**:
  - **Graph Theory:** Custom BFS, DFS, Connected Components, and graph density metrics
  - **Graph Coloring:** Custom Welsh-Powell greedy algorithm, Independent Set verification, Chromatic Number search
  - **Relations & Posets:** Dominance relation $A \succeq B \iff \text{Return}(A) \ge \text{Return}(B) \land \text{Risk}(A) \le \text{Risk}(B)$, Reflexivity, Antisymmetry, Transitivity validation
  - **Hasse Diagram:** Custom Transitive Reduction without external graph reduction libraries
  - **Set Theory & Boolean Logic:** Python `frozenset` operations ($A \cap B$, $A \cup C$, $A - B$), 3-condition Boolean truth table
  - **Combinatorics:** Combinatorial generation $C(n,k) = \frac{n!}{k!(n-k)!}$
  - **Portfolio Risk:** Annualized portfolio covariance matrix $\sigma_p = \sqrt{w^T \Sigma w}$
  - **DM Scoring Engine:** 5-component weighted scoring formula (30% Return, 25% Risk, 25% Diversification, 10% Color Groups, 10% Dominance)
- **Frontend:** Responsive Jinja2 HTML templates + Tailwind CSS CDN + Plotly.js CDN (Interactive Charts)
- **Cross-Device Ready:** Mobile slide-over drawer navigation, touch-friendly tables, and live QR code sharing
- **Testing:** 147 passing unit and integration tests

---

## 🚀 Quick Start & Running

### 1. Installation
```powershell
cd c:\TradeSense
pip install -r backend_requirements.txt
```

### 2. Run the Application
```powershell
python run.py
```
The server binds to `0.0.0.0:8000`, making it accessible locally and over your local Wi-Fi network.

### 3. Run the Test Suite
```powershell
pytest tests/ backend/tests/ -v
```
*(All 147 test cases pass)*

---

## 📱 Cross-Device Access Links

TradeSense is accessible on smartphones (iOS/Android), tablets, and secondary laptops:

| Connection Type | Link | Details |
|---|---|---|
| **Public Worldwide Link** | **`https://ready-jars-stop.loca.lt`** | Accessible anywhere worldwide over 4G/5G mobile data or remote Wi-Fi |
| **Local Wi-Fi Network** | **`http://10.20.8.4:8000`** | Direct connection for any device connected to the same Wi-Fi router |
| **Local PC Loopback** | **`http://127.0.0.1:8000`** | Direct desktop access |

> **Scan QR Code with Phone:** Click the **"📱 Connect Devices"** button in the header to view and scan the live QR code with your phone camera.

---

## 🌐 Application Pages & Navigation (11 Sections)

| Route | Icon & Page | Description |
|---|---|---|
| `/` | 🏠 **Dashboard** | Overview KPI cards, Network Graph preview, Pearson Correlation Heatmap, Top 3 Portfolios, and Market Pulse |
| `/live-market` | 📡 **Live Market** | Universe scanner with data freshness indicator, return/risk heatbars, and Data Source Transparency |
| `/stock-explorer` | 🔎 **Stock Explorer** | 360° In-Real-Life inspector: metrics, correlated peers in $G=(V,E)$, dominance ranking, color groups, and news |
| `/relationships` | 🔗 **Relationships** | Full Pearson Correlation Matrix heatmap, mathematical matrix properties, and top co-moving pairs |
| `/market-network` | 🕸️ **Market Network** | Graph $G=(V,E)$ visualizer, dynamic threshold slider, BFS & DFS step tracers, and connected components |
| `/hasse-ranking` | 📐 **Hasse & Ranking** | Dominance relation matrix, Poset mathematical verification (reflexive, antisymmetric, transitive), and Hasse DAG |
| `/stock-groups` | 🎨 **Stock Groups** | Welsh-Powell graph coloring visualizer with step trace, chromatic number $\chi(G)$, and verified independent sets |
| `/portfolio-lab` | 💼 **Portfolio Lab** | Combinatorics $C(n,k)$ space, 5-pillar DM Score breakdown, Top 3 awards, and radar chart comparisons |
| `/how-tradesense-thinks` | 🧠 **How TradeSense Thinks** | 10-step story walkthrough with live computational artifacts and dual-mode **`[Explain Simply]`** vs **`[Show the Math]`** |
| `/market-news` | 📰 **Market News** | Contextual financial headlines with sentiment badges (Bullish, Neutral, Bearish), source attribution, and filters |
| `/reports` | 📄 **Reports** | Executive audit summary, 10-stage mathematical methodology reference table, and JSON/PDF export |

---

## 🔌 API Endpoints Reference

### Market & Stocks
- `GET /api/health` — API health check and version
- `GET /api/market/status` — Live NSE market open/closed status
- `GET /api/market/overview` — High-level KPI summary of the latest analysis
- `GET /api/market/news` — Curated market news mapped to universe stocks
- `GET /api/market/network-info` — Dynamic host IP address and network URL resolution
- `GET /api/stocks` — List all stocks enriched with computed metrics
- `GET /api/stocks/sectors` — List unique market sectors
- `GET /api/stocks/{symbol}` — Detailed metrics for a single stock
- `GET /api/stocks/{symbol}/history` — Historical price time series

### Analysis & Discrete Mathematics
- `POST /api/analyze` — Run the full 15-stage discrete math analysis pipeline
- `POST /api/refresh` — Clear cache and re-run analysis
- `GET /api/correlation` — Correlation matrix and statistical properties
- `GET /api/graph?threshold={float}` — Graph adjacency, statistics, BFS/DFS traversal
- `GET /api/coloring` — Welsh-Powell coloring, independent sets, chromatic info
- `GET /api/relation` — Dominance relation matrix and non-dominated stock set
- `GET /api/poset` — Partial order property verification (reflexive, antisymmetric, transitive)
- `GET /api/hasse` — Hasse cover relation DAG with hierarchical Plotly layout coordinates
- `GET /api/sets` — Set operations ($A, B, C$, unions, intersections, differences)
- `GET /api/boolean` — Stock screening Boolean truth table
- `GET /api/portfolios` — All evaluated candidate combinations ranked by DM Score
- `GET /api/portfolios/top` — Top 3 portfolios automatically scored by DM Score formula

---

## 🧮 Discrete Mathematics Formula Reference

1. **Pearson Correlation Coefficient:**
   $$r_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

2. **Graph Construction Rule:**
   $$G = (V, E), \quad E = \{(u, v) \mid |\text{corr}(u, v)| \ge \theta, u \neq v\}$$

3. **Dominance Relation (Poset):**
   $$A \succeq B \iff \text{Return}(A) \ge \text{Return}(B) \land \text{Risk}(A) \le \text{Risk}(B)$$

4. **Hasse Diagram Transitive Reduction:**
   $$\text{Cover}(A, B) \iff A \succ B \land \nexists C \text{ s.t. } A \succ C \succ B$$

5. **Independent Sets (Welsh-Powell Graph Coloring):**
   $$c: V \to \{1, \dots, k\} \text{ s.t. } (u,v) \in E \implies c(u) \neq c(v)$$

6. **Combinatorics:**
   $$C(n, k) = \frac{n!}{k!(n-k)!}$$

7. **DM Portfolio Score (Multi-Criteria):**
   $$\text{DM Score} = 0.30 S_{\text{return}} + 0.25 S_{\text{risk}} + 0.25 S_{\text{div}} + 0.10 S_{\text{group}} + 0.10 S_{\text{dom}}$$
