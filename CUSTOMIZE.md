# Customizing the Campus Energy Monitoring Console

This console is an engineer-facing presentation layer for **Campus Energy Consumption Intelligence & Anomaly Detection** (`Ashira-senthal/campus-energy`).
All visual tokens, color palettes, fonts, spacing, and chart styling are centralized in a single CSS stylesheet and Python theme dictionary, enabling complete UI restyling without altering any application logic or data pipelines.

---

## 1. Where Design Decisions Live

- **`app/style.css`**: The single source of truth for all CSS design tokens, typography, layout rules, BEM semantic classes (`ce-*`), and Streamlit widget chrome overrides.
- **`app/theme.py`**: Holds `CHART_THEME` (a Python dictionary directly mirroring the CSS tokens) and `chart_layout()`, which is used by every Plotly figure in the application.
- **`app/components.py`**: Clean semantic HTML helpers returning BEM classes prefixed with `ce-`. Contains zero inline hex colors or hard-coded font families.
- **`.streamlit/config.toml`**: Streamlit runtime theme configuration, aligned with the CSS tokens.

---

## 2. Design Tokens (`app/style.css`)

All tokens are defined in a single `:root` block at the top of `app/style.css`.
Every component throughout the stylesheet references these tokens via `var(--token-name)`.

### Token Groups

| Group | Tokens | Description |
|---|---|---|
| **Colors** | `--bg`, `--surface`, `--surface-alt`, `--border`, `--border-strong`, `--text`, `--text-muted`, `--text-faint`, `--accent`, `--accent-contrast`, `--positive`, `--negative`, `--warning`, `--info` | Semantic UI surface and text colors. |
| **Series** | `--series-actual`, `--series-expected`, `--series-flag`, `--series-compare-1` .. `--series-compare-4` | Time-series lines and comparison colors. |
| **Type** | `--font-body`, `--font-heading`, `--font-mono`, `--fs-xs` .. `--fs-xl`, `--fs-kpi`, `--fw-regular`, `--fw-medium`, `--fw-bold` | System font stack, fluid scale, and tabular numeral support. |
| **Space** | `--space-1` (4px), `--space-2` (8px), `--space-3` (12px), `--space-4` (16px), `--space-5` (24px), `--space-6` (32px) | Consistent 4px grid spacing. |
| **Shape** | `--radius-sm` (4px), `--radius-md` (6px), `--radius-lg` (8px), `--shadow-sm` | Card borders and elevation. |
| **Layout** | `--sidebar-width` (220px), `--sidebar-width-collapsed` (56px), `--content-max-width` (1360px) | Navigation and viewport widths. |
| **Motion** | `--dur-fast` (120ms), `--dur-base` (150ms), `--ease` | Restrained, non-distracting UI transitions. |

---

## 3. How to Change the Color Palette

To apply your own branding or color scheme:
1. Open `app/style.css`.
2. Edit the hex values inside the `:root` block. Keep token names identical.
3. Mirror the series colors in `app/theme.py` under `CHART_THEME`:
   ```python
   CHART_THEME = {
       "paper_bgcolor": "#ffffff",
       "plot_bgcolor": "#ffffff",
       "actual": "#1f4d8a",       # Mirror --series-actual
       "expected": "#b45309",     # Mirror --series-expected
       "flag": "#b91c1c",         # Mirror --series-flag
       "compare": [ ... ],        # Mirror --series-compare-1..4
   }
   ```
4. If you want Streamlit's initial loading screen to match, update `.streamlit/config.toml`:
   ```toml
   [theme]
   primaryColor = "#1f4d8a"
   backgroundColor = "#f8f9fa"
   secondaryBackgroundColor = "#ffffff"
   textColor = "#1a202c"
   ```

---

## 4. How to Change Typography

1. Open `app/style.css`.
2. Update `--font-body`, `--font-heading`, or `--font-mono` in `:root`:
   ```css
   :root {
     --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
     --font-heading: 'Inter', sans-serif;
     --font-mono: 'JetBrains Mono', monospace;
   }
   ```
3. Update `font_family` in `app/theme.py` to match:
   ```python
   CHART_THEME["font_family"] = "Inter, sans-serif"
   ```

---

## 5. Dark Mode Configuration

The application includes a built-in dark mode switch in the sidebar.
The dark theme overrides the exact same tokens in `app/style.css`:
```css
:root[data-theme="dark"],
[data-theme="dark"] {
  --bg: #0d1117;
  --surface: #161b22;
  --surface-alt: #21262d;
  --border: #30363d;
  --text: #e6edf3;
  --text-muted: #8b949e;
  --accent: #58a6ff;
  --positive: #3fb950;
  --negative: #f85149;
  --warning: #d29922;
}
```
When toggled, `app/app.py` automatically updates `CHART_THEME` background and grid colors to match `#161b22`, ensuring Plotly figures render seamlessly in both modes.

---

## 6. How to Restyle Charts via `CHART_THEME`

Plotly figures do not inherit CSS stylesheets directly. All figures in this project are constructed using `chart_layout()` in `app/theme.py`, which pulls values exclusively from `CHART_THEME`:
- **Backgrounds:** `CHART_THEME["paper_bgcolor"]` and `plot_bgcolor`.
- **Text & Grids:** `CHART_THEME["text_color"]`, `text_muted`, and `grid_color`.
- **Line Widths & Shapes:** Defined uniformly in `app/theme.py`.
- **Anomaly Marker Standard:** Shapes are standardized per failure type:
  - Spike: `triangle-up` (▲)
  - After-Hours: `square` (■)
  - Flatline: `diamond` (◆)
  - Drift: `circle-open-dot` (●)
  - Marker sizes: low (8px), medium (11px), high (15px).

Modifying `CHART_THEME` immediately restyles all charts across all 11 pages simultaneously.

---

## 7. Adding New Components or Pages

- **New HTML Elements:** Add functions to `app/components.py` returning HTML with `ce-` classes.
- **New Pages:** Add a new `page_*.py` file in `app/pages/`, define a `render(ctx)` function, and register it in `PAGES` in `app/app.py`.
- **Data Traceability:** Always add a source note using `ce_source_note("path/to/report.csv")` under any new metric table or chart.
