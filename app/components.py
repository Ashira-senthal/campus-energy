"""
Semantic HTML component builders for the Campus Energy Monitoring Console.

Returns clean HTML with BEM classes prefixed 'ce-'.
No inline styles or hard-coded colors.
"""

from __future__ import annotations
import base64
import html
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

@lru_cache(maxsize=1)
def _logo_data_uri() -> str:
    """Inline the transparent brand mark so it renders inside sidebar HTML."""
    path = Path(__file__).resolve().parent / "assets" / "logo.png"
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")

def ce_page_header(title: str, purpose: str) -> str:
    safe_title = html.escape(title)
    safe_purpose = html.escape(purpose)
    return f"""
<div class="ce-page-header">
  <h1 class="ce-page-header__title">{safe_title}</h1>
  <p class="ce-page-header__purpose">{safe_purpose}</p>
</div>
"""

def ce_context_bar(
    buildings: str,
    date_range: str,
    model: str,
    detector: str,
    current_page: str = "overview",
) -> str:
    caret = '<span class="ce-context-bar__caret" aria-hidden="true">▾</span>'
    return f"""
<div class="ce-context-bar" role="region" aria-label="Global filter context">
  <div class="ce-context-bar__items">
    <div class="ce-context-bar__item">
      <span class="ce-context-bar__key">Buildings</span>
      <span class="ce-context-bar__val">{html.escape(buildings)}</span>
      {caret}
    </div>
    <div class="ce-context-bar__item">
      <span class="ce-context-bar__key">Range</span>
      <span class="ce-context-bar__val">{html.escape(date_range)}</span>
      {caret}
    </div>
    <div class="ce-context-bar__item">
      <span class="ce-context-bar__key">Model</span>
      <span class="ce-context-bar__val">{html.escape(model)}</span>
      {caret}
    </div>
    <div class="ce-context-bar__item">
      <span class="ce-context-bar__key">Detector</span>
      <span class="ce-context-bar__val">{html.escape(detector)}</span>
      {caret}
    </div>
  </div>
  <a href="?page={current_page}&reset_filters=1" target="_self" class="ce-context-bar__reset" title="Reset all global filters to defaults">
    Reset filters
  </a>
</div>
"""

def ce_filter_bar(inner_html: str) -> str:
    return f"""
<div class="ce-filter-bar">
  {inner_html}
</div>
"""

def ce_kpi_strip(kpi_cards_html: str) -> str:
    return f"""
<div class="ce-kpi-strip">
  {kpi_cards_html}
</div>
"""

def ce_metric_row(
    label: str,
    value: str | int | float,
    unit: str | None = None,
    delta: str | None = None,
    delta_kind: str = "neutral",
    source: str | None = None,
) -> str:
    unit_html = f'<span class="ce-kpi__unit">{html.escape(unit)}</span>' if unit else ""
    delta_html = ""
    if delta is not None and str(delta).strip() != "":
        kind_cls = f"ce-kpi__delta--{delta_kind}"
        delta_html = f'<span class="ce-kpi__delta {kind_cls}">{html.escape(str(delta))}</span>'

    source_html = f'<div class="ce-kpi__source" title="{html.escape(source)}">Source: {html.escape(source)}</div>' if source else ""

    # Keep the markup on a single line: Streamlit's markdown parser ends an
    # HTML block at the first blank line, which would turn indented tags into
    # literal code.
    return (
        '<div class="ce-kpi">'
        f'<div class="ce-kpi__label">{html.escape(label)}</div>'
        '<div class="ce-kpi__value-wrap">'
        f'<span class="ce-kpi__value">{html.escape(str(value))}</span>'
        f"{unit_html}{delta_html}"
        "</div>"
        f"{source_html}"
        "</div>"
    )

def ce_card(
    title: str,
    body_html: str,
    badge: str | None = None,
    source: str | None = None,
) -> str:
    badge_html = f'<span class="ce-card__badge">{html.escape(badge)}</span>' if badge else ""
    source_html = f'<div class="ce-card__source">Source: {html.escape(source)}</div>' if source else ""

    return f"""
<div class="ce-card">
  <div class="ce-card__head">
    <h3 class="ce-card__title">{html.escape(title)}</h3>
    {badge_html}
  </div>
  <div class="ce-card__body">
    {body_html}
  </div>
  {source_html}
</div>
"""

def ce_tag(text: str, kind: str) -> str:
    """Render status tag with text and marker shape (never color alone)."""
    kind_safe = kind.lower().replace(" ", "-").replace("_", "-")
    return f'<span class="ce-tag ce-tag--{kind_safe}">{html.escape(text)}</span>'

def ce_table(
    headers: list[str],
    rows: list[list[Any]],
    sort_keys: list[str] | None = None,
    empty_msg: str = "No data available in this range.",
    source: str | None = None,
) -> str:
    if not rows:
        return f"""
<div class="ce-table-wrap">
  <div class="ce-empty-message">{html.escape(empty_msg)}</div>
</div>
"""
    th_cells = []
    num_keywords = ("kwh", "cost", "inr", "co2", "kg", "load", "rmse", "mae", "nmbe", "smape", "pct", "%", "z", "hours", "ratio", "score", "f1", "precision", "recall")

    for i, h in enumerate(headers):
        is_num = False
        if sort_keys and i < len(sort_keys):
            key_lower = sort_keys[i].lower()
            if any(k in key_lower for k in num_keywords):
                is_num = True
        else:
            h_lower = h.lower()
            if any(k in h_lower for k in num_keywords):
                is_num = True

        cls = ' class="ce-table__th--num"' if is_num else ""
        th_cells.append(f"<th{cls}>{html.escape(h)}</th>")

    thead = "".join(th_cells)

    tbody_rows = []
    for row in rows:
        td_cells = []
        for i, val in enumerate(row):
            is_num = False
            if sort_keys and i < len(sort_keys):
                key_lower = sort_keys[i].lower()
                if any(k in key_lower for k in num_keywords):
                    is_num = True
            else:
                h_lower = headers[i].lower() if i < len(headers) else ""
                if any(k in h_lower for k in num_keywords):
                    is_num = True

            val_str = str(val) if val is not None else "—"
            # If the cell already contains HTML markup (links, tags, bold), do not escape it
            if re.search(r"<[a-zA-Z/][^>]*>", val_str):
                cell_content = val_str
            else:
                cell_content = html.escape(val_str)

            cls = ' class="ce-table__td--num"' if is_num else ""
            td_cells.append(f"<td{cls}>{cell_content}</td>")

        tbody_rows.append(f"<tr>{''.join(td_cells)}</tr>")

    tbody = "".join(tbody_rows)
    source_html = f'<div class="ce-source-note">Source: {html.escape(source)}</div>' if source else ""

    return f"""
<div class="ce-table-wrap">
  <table class="ce-table">
    <thead><tr>{thead}</tr></thead>
    <tbody>{tbody}</tbody>
  </table>
</div>
{source_html}
"""

def ce_sidebar_html(
    pages: list[dict[str, Any]],
    active_page: str,
    last_run_text: str,
    data_window_text: str,
) -> str:
    """Render grouped navigation sidebar."""
    groups: dict[str, list[dict[str, Any]]] = {}

    for p in pages:
        groups.setdefault(p.get("group", "Projects"), []).append(p)

    nav_items_html = []
    for grp_key, grp_pages in groups.items():
        nav_items_html.append(f'<div class="ce-sidebar__group-label">{grp_key}</div>')
        for p in grp_pages:
            slug = p["slug"]
            is_active = (slug == active_page)
            active_cls = " ce-nav__item--active" if is_active else ""
            icon = p.get("icon", "•")
            label = p["label"]
            url = f"?page={slug}"

            nav_items_html.append(f"""
<a class="ce-nav__item{active_cls}" href="{url}" target="_self" title="{html.escape(label)}">
  <span class="ce-nav__icon" aria-hidden="true">{icon}</span>
  <span class="ce-nav__label">{html.escape(label)}</span>
</a>
""")

    nav_content = "".join(nav_items_html)

    return f"""
<div class="ce-sidebar" role="navigation" aria-label="Main Navigation">
  <div class="ce-sidebar__brand" onclick="try{{const b=window.parent.document.querySelector('button[data-testid=\'stExpandSidebarButton\']')||document.querySelector('button[data-testid=\'stExpandSidebarButton\']');if(b)b.click();}}catch(e){{}}" title="Campus Energy Console">
    <div class="ce-sidebar__brandrow">
      <img class="ce-sidebar__logo" src="{_logo_data_uri()}" alt="" aria-hidden="true">
      <h2 class="ce-sidebar__title">CAMPUS ENERGY</h2>
    </div>
    <span class="ce-sidebar__subtitle">Monitoring console</span>
    <div class="ce-sidebar__status">
      <div>Run: {html.escape(last_run_text)}</div>
      <div>Data: {html.escape(data_window_text)}</div>
    </div>
  </div>
  <nav class="ce-sidebar__nav">
    {nav_content}
  </nav>
</div>
"""

def ce_source_note(source_filepath: str) -> str:
    return f'<div class="ce-source-note">Source: {html.escape(source_filepath)}</div>'

def ce_empty_message(message: str) -> str:
    return f'<div class="ce-empty-message">{html.escape(message)}</div>'
