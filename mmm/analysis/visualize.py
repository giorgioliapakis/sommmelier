"""
Visualization module for Sommmelier.

Generates charts and visualizations that laypeople can understand.
"""

import json
from pathlib import Path
from typing import Any


def normalize_results(results: dict) -> dict:
    """Normalize results from different formats (simple vs full)."""
    normalized = dict(results)

    # Handle simple format (channel_roi -> roi)
    if "channel_roi" in results and "roi" not in results:
        normalized["roi"] = {
            ch: {"mean": val, "ci_lower": val * 0.7, "ci_upper": val * 1.3}
            for ch, val in results["channel_roi"].items()
        }

    # Handle simple format (channel_contributions -> contributions)
    if "channel_contributions" in results and "contributions" not in results:
        normalized["contributions"] = {
            ch: {"percentage": val * 100, "absolute": val}
            for ch, val in results["channel_contributions"].items()
        }

    # Handle metadata
    if "metadata" not in normalized:
        normalized["metadata"] = {
            "n_time_periods": results.get("n_time_periods", 0),
            "n_geos": results.get("n_geos", 0),
            "channels": results.get("channels", []),
            "total_spend": {},
            "total_kpi": 0,
        }

    return normalized


def generate_roi_chart_svg(results: dict) -> str:
    """Generate an SVG bar chart showing ROI by channel."""
    results = normalize_results(results)
    roi_data = results.get("roi", {})
    if not roi_data:
        return "<p>No ROI data available</p>"

    # Sort by ROI
    sorted_channels = sorted(roi_data.items(), key=lambda x: -x[1].get("mean", 0))

    # Chart dimensions
    width = 500
    bar_height = 40
    gap = 10
    label_width = 100
    chart_height = len(sorted_channels) * (bar_height + gap) + 20

    # Find max ROI for scaling
    max_roi = max(d.get("mean", 0) for _, d in sorted_channels) or 1
    scale = (width - label_width - 80) / max_roi

    # Color gradient (green for high ROI, orange for low)
    def get_color(roi: float) -> str:
        if roi >= 1.0:
            return "#22c55e"  # green
        elif roi >= 0.5:
            return "#eab308"  # yellow
        else:
            return "#f97316"  # orange

    svg_bars = []
    for i, (ch, data) in enumerate(sorted_channels):
        y = i * (bar_height + gap) + 10
        roi = data.get("mean", 0)
        bar_width = max(roi * scale, 2)
        color = get_color(roi)

        # Error bars for confidence interval
        ci_lo = data.get("ci_lower", roi)
        ci_hi = data.get("ci_upper", roi)

        svg_bars.append(f'''
        <g transform="translate(0, {y})">
            <text x="{label_width - 10}" y="{bar_height/2 + 5}" text-anchor="end" font-size="14" fill="#374151">{ch}</text>
            <rect x="{label_width}" y="5" width="{bar_width}" height="{bar_height - 10}" fill="{color}" rx="4"/>
            <text x="{label_width + bar_width + 10}" y="{bar_height/2 + 5}" font-size="14" font-weight="bold" fill="#1f2937">{roi:.2f}x</text>
            <line x1="{label_width + ci_lo * scale}" y1="{bar_height/2}" x2="{label_width + ci_hi * scale}" y2="{bar_height/2}" stroke="#6b7280" stroke-width="2"/>
        </g>
        ''')

    return f'''
    <svg width="{width}" height="{chart_height}" viewBox="0 0 {width} {chart_height}">
        <style>
            text {{ font-family: system-ui, -apple-system, sans-serif; }}
        </style>
        {''.join(svg_bars)}
    </svg>
    '''


def generate_contribution_chart_svg(results: dict) -> str:
    """Generate an SVG pie/donut chart showing contribution by channel."""
    results = normalize_results(results)
    contrib_data = results.get("contributions", {})
    if not contrib_data:
        return "<p>No contribution data available</p>"

    # Sort by contribution
    sorted_channels = sorted(contrib_data.items(), key=lambda x: -x[1].get("percentage", 0))

    # Colors for channels
    colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]

    # Chart dimensions
    size = 300
    cx, cy = size / 2, size / 2
    outer_r = 120
    inner_r = 70

    # Generate arcs
    svg_arcs = []
    svg_labels = []
    start_angle = -90  # Start from top

    for i, (ch, data) in enumerate(sorted_channels):
        pct = data.get("percentage", 0)
        angle = pct / 100 * 360
        end_angle = start_angle + angle

        # Convert to radians
        import math
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)

        # Calculate arc points
        x1 = cx + outer_r * math.cos(start_rad)
        y1 = cy + outer_r * math.sin(start_rad)
        x2 = cx + outer_r * math.cos(end_rad)
        y2 = cy + outer_r * math.sin(end_rad)
        x3 = cx + inner_r * math.cos(end_rad)
        y3 = cy + inner_r * math.sin(end_rad)
        x4 = cx + inner_r * math.cos(start_rad)
        y4 = cy + inner_r * math.sin(start_rad)

        large_arc = 1 if angle > 180 else 0
        color = colors[i % len(colors)]

        # Create arc path
        path = f"M {x1} {y1} A {outer_r} {outer_r} 0 {large_arc} 1 {x2} {y2} L {x3} {y3} A {inner_r} {inner_r} 0 {large_arc} 0 {x4} {y4} Z"
        svg_arcs.append(f'<path d="{path}" fill="{color}" stroke="white" stroke-width="2"/>')

        # Add label
        mid_angle = math.radians((start_angle + end_angle) / 2)
        label_r = (outer_r + inner_r) / 2
        lx = cx + label_r * math.cos(mid_angle)
        ly = cy + label_r * math.sin(mid_angle)

        if pct >= 8:  # Only show label if segment is big enough
            svg_labels.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="12" font-weight="bold">{pct:.0f}%</text>')

        start_angle = end_angle

    # Legend
    legend_items = []
    for i, (ch, data) in enumerate(sorted_channels):
        pct = data.get("percentage", 0)
        color = colors[i % len(colors)]
        y = 20 + i * 25
        legend_items.append(f'''
        <g transform="translate({size + 20}, {y})">
            <rect width="16" height="16" fill="{color}" rx="2"/>
            <text x="24" y="13" font-size="13" fill="#374151">{ch}: {pct:.1f}%</text>
        </g>
        ''')

    return f'''
    <svg width="{size + 180}" height="{max(size, len(sorted_channels) * 25 + 40)}" viewBox="0 0 {size + 180} {max(size, len(sorted_channels) * 25 + 40)}">
        <style>
            text {{ font-family: system-ui, -apple-system, sans-serif; }}
        </style>
        {''.join(svg_arcs)}
        {''.join(svg_labels)}
        {''.join(legend_items)}
    </svg>
    '''


def generate_marginal_roi_chart_svg(results: dict) -> str:
    """Generate chart comparing average ROI vs marginal ROI."""
    results = normalize_results(results)
    roi_data = results.get("roi", {})
    mroi_data = results.get("marginal_roi", {})

    if not roi_data or not mroi_data:
        return "<p>No marginal ROI data available</p>"

    channels = list(roi_data.keys())

    # Chart dimensions
    width = 500
    bar_height = 30
    gap = 40
    label_width = 100
    chart_height = len(channels) * gap + 40

    # Find max for scaling
    all_values = [roi_data.get(ch, {}).get("mean", 0) for ch in channels] + [mroi_data.get(ch, 0) for ch in channels]
    max_val = max(all_values) if all_values else 1
    scale = (width - label_width - 100) / max_val

    svg_bars = []
    for i, ch in enumerate(sorted(channels, key=lambda c: -roi_data.get(c, {}).get("mean", 0))):
        y = i * gap + 20
        avg_roi = roi_data.get(ch, {}).get("mean", 0)
        m_roi = mroi_data.get(ch, 0)

        svg_bars.append(f'''
        <g transform="translate(0, {y})">
            <text x="{label_width - 10}" y="20" text-anchor="end" font-size="13" fill="#374151">{ch}</text>

            <!-- Average ROI bar -->
            <rect x="{label_width}" y="0" width="{max(avg_roi * scale, 2)}" height="12" fill="#3b82f6" rx="2"/>
            <text x="{label_width + avg_roi * scale + 5}" y="10" font-size="11" fill="#3b82f6">{avg_roi:.2f}x avg</text>

            <!-- Marginal ROI bar -->
            <rect x="{label_width}" y="16" width="{max(m_roi * scale, 2)}" height="12" fill="#22c55e" rx="2"/>
            <text x="{label_width + m_roi * scale + 5}" y="26" font-size="11" fill="#22c55e">{m_roi:.2f}x marginal</text>
        </g>
        ''')

    legend = f'''
    <g transform="translate({label_width}, {chart_height - 15})">
        <rect width="12" height="12" fill="#3b82f6" rx="2"/>
        <text x="18" y="10" font-size="11" fill="#374151">Average ROI</text>
        <rect x="120" width="12" height="12" fill="#22c55e" rx="2"/>
        <text x="138" y="10" font-size="11" fill="#374151">Marginal ROI (at current spend)</text>
    </g>
    '''

    return f'''
    <svg width="{width}" height="{chart_height + 20}" viewBox="0 0 {width} {chart_height + 20}">
        <style>
            text {{ font-family: system-ui, -apple-system, sans-serif; }}
        </style>
        {''.join(svg_bars)}
        {legend}
    </svg>
    '''


def interpret_roi(roi: float) -> str:
    """Return plain English interpretation of ROI."""
    if roi >= 2.0:
        return "Excellent - every $1 spent returns $" + f"{roi:.2f}"
    elif roi >= 1.0:
        return "Good - profitable investment"
    elif roi >= 0.5:
        return "Moderate - some return but below breakeven"
    else:
        return "Low - consider reducing spend"


def interpret_marginal_roi(avg_roi: float, mroi: float) -> str:
    """Interpret marginal vs average ROI."""
    if mroi > avg_roi * 1.1:
        return "Room to grow - additional spend would be efficient"
    elif mroi < avg_roi * 0.5:
        return "Saturated - hitting diminishing returns"
    else:
        return "Balanced - near optimal spend level"


def generate_insights(results: dict) -> list[dict]:
    """Generate actionable insights from results."""
    results = normalize_results(results)
    insights = []

    roi_data = results.get("roi", {})
    mroi_data = results.get("marginal_roi", {})
    contrib_data = results.get("contributions", {})
    spend_data = results.get("metadata", {}).get("total_spend", {})

    if not roi_data:
        return insights

    # Find best and worst ROI channels
    sorted_roi = sorted(roi_data.items(), key=lambda x: -x[1].get("mean", 0))
    best_ch, best_data = sorted_roi[0]
    worst_ch, worst_data = sorted_roi[-1]

    # Insight 1: Best performing channel
    insights.append({
        "type": "success",
        "title": f"{best_ch.title()} is your best performing channel",
        "detail": f"With an ROI of {best_data.get('mean', 0):.2f}x, every $1 spent on {best_ch} generates ${best_data.get('mean', 0):.2f} in conversions.",
        "action": "Consider increasing budget allocation to this channel."
    })

    # Insight 2: Worst performing (if below 1.0)
    if worst_data.get("mean", 0) < 1.0:
        insights.append({
            "type": "warning",
            "title": f"{worst_ch.title()} has ROI below breakeven",
            "detail": f"ROI of {worst_data.get('mean', 0):.2f}x means you're losing money on this channel.",
            "action": "Consider reducing spend or improving targeting/creative."
        })

    # Insight 3: Saturation check via marginal ROI
    if mroi_data:
        for ch, mroi in mroi_data.items():
            avg_roi = roi_data.get(ch, {}).get("mean", 0)
            if mroi < avg_roi * 0.5 and avg_roi > 0:
                insights.append({
                    "type": "info",
                    "title": f"{ch.title()} is showing saturation",
                    "detail": f"Marginal ROI ({mroi:.2f}x) is much lower than average ROI ({avg_roi:.2f}x).",
                    "action": "You may be over-investing in this channel."
                })

    # Insight 4: Underinvested channels
    if mroi_data and spend_data:
        for ch, mroi in mroi_data.items():
            avg_roi = roi_data.get(ch, {}).get("mean", 0)
            if mroi > avg_roi * 1.2 and avg_roi > 0.5:
                insights.append({
                    "type": "opportunity",
                    "title": f"{ch.title()} has room to scale",
                    "detail": f"Marginal ROI ({mroi:.2f}x) exceeds average ROI ({avg_roi:.2f}x).",
                    "action": "Additional spend on this channel would likely be efficient."
                })

    return insights


def _embed_png_chart(chart_path: str | None) -> str | None:
    """Read a PNG file and return a base64-encoded <img> tag, or None."""
    if not chart_path:
        return None
    try:
        import base64
        chart_file = Path(chart_path)
        if chart_file.exists():
            data = chart_file.read_bytes()
            b64 = base64.b64encode(data).decode('ascii')
            return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;" />'
    except Exception:
        pass
    return None


def generate_html_report(results: dict, output_path: Path | str) -> str:
    """
    Generate a comprehensive HTML report for laypeople.

    Uses native Meridian PNG charts if available (from GPU run),
    falls back to inline SVG charts generated from the JSON data.

    Args:
        results: Dictionary from fit_mmm_full() or fit_mmm()
        output_path: Path to save the HTML report

    Returns:
        Path to the generated report
    """
    output_path = Path(output_path)
    results = normalize_results(results)

    metadata = results.get("metadata", {})
    insights = generate_insights(results)
    charts = results.get("charts", {})

    # Use native PNG charts if available, fall back to SVG generation
    roi_chart = _embed_png_chart(charts.get("roi_chart")) or generate_roi_chart_svg(results)
    contrib_chart = _embed_png_chart(charts.get("contribution_chart")) or generate_contribution_chart_svg(results)
    mroi_chart = _embed_png_chart(charts.get("response_curves")) or generate_marginal_roi_chart_svg(results)

    # Insight cards HTML
    insight_cards = ""
    type_colors = {
        "success": "#22c55e",
        "warning": "#f59e0b",
        "info": "#3b82f6",
        "opportunity": "#8b5cf6",
    }
    type_icons = {
        "success": "✓",
        "warning": "⚠",
        "info": "ℹ",
        "opportunity": "💡",
    }

    for insight in insights:
        color = type_colors.get(insight["type"], "#6b7280")
        icon = type_icons.get(insight["type"], "•")
        insight_cards += f'''
        <div class="insight-card" style="border-left: 4px solid {color};">
            <div class="insight-header">
                <span class="insight-icon" style="color: {color};">{icon}</span>
                <strong>{insight["title"]}</strong>
            </div>
            <p class="insight-detail">{insight["detail"]}</p>
            <p class="insight-action"><strong>Recommendation:</strong> {insight["action"]}</p>
        </div>
        '''

    # ROI interpretation table
    roi_rows = ""
    roi_data = results.get("roi", {})
    for ch, data in sorted(roi_data.items(), key=lambda x: -x[1].get("mean", 0)):
        roi = data.get("mean", 0)
        interpretation = interpret_roi(roi)
        roi_rows += f'''
        <tr>
            <td><strong>{ch.title()}</strong></td>
            <td>{roi:.2f}x</td>
            <td>{interpretation}</td>
        </tr>
        '''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Marketing Mix Model Report</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            background: #f9fafb;
            padding: 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 40px;
        }}
        h1 {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #111827;
        }}
        .subtitle {{
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 32px;
        }}
        h2 {{
            font-size: 20px;
            font-weight: 600;
            margin-top: 32px;
            margin-bottom: 16px;
            color: #374151;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 8px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .summary-card {{
            background: #f3f4f6;
            padding: 20px;
            border-radius: 8px;
        }}
        .summary-card .label {{
            font-size: 12px;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 4px;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: 700;
            color: #111827;
        }}
        .chart-container {{
            margin: 24px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
            overflow-x: auto;
        }}
        .insight-card {{
            background: #fafafa;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }}
        .insight-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        .insight-icon {{
            font-size: 18px;
        }}
        .insight-detail {{
            color: #4b5563;
            margin-bottom: 8px;
        }}
        .insight-action {{
            color: #374151;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f9fafb;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            color: #6b7280;
        }}
        .explainer {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 16px;
            margin: 16px 0;
            border-radius: 0 8px 8px 0;
        }}
        .explainer-title {{
            font-weight: 600;
            color: #1e40af;
            margin-bottom: 4px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
            font-size: 12px;
            color: #9ca3af;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Marketing Mix Model Report</h1>
        <p class="subtitle">Generated on {results.get("timestamp", "Unknown")[:10]} • {metadata.get("n_time_periods", 0)} time periods • {metadata.get("n_geos", 0)} regions</p>

        <div class="summary-grid">
            <div class="summary-card">
                <div class="label">Channels Analyzed</div>
                <div class="value">{len(metadata.get("channels", []))}</div>
            </div>
            <div class="summary-card">
                <div class="label">Total Spend</div>
                <div class="value">${sum(metadata.get("total_spend", {}).values()):,.0f}</div>
            </div>
            <div class="summary-card">
                <div class="label">Total Conversions</div>
                <div class="value">{metadata.get("total_kpi", 0):,.0f}</div>
            </div>
        </div>

        <h2>Key Insights</h2>
        <div class="explainer">
            <div class="explainer-title">What are these insights?</div>
            These are AI-generated recommendations based on your marketing data. They highlight which channels are performing well, which need attention, and where you might be able to improve ROI.
        </div>
        {insight_cards if insight_cards else "<p>No significant insights detected.</p>"}

        <h2>Return on Investment by Channel</h2>
        <div class="explainer">
            <div class="explainer-title">What is ROI?</div>
            ROI (Return on Investment) tells you how much revenue you get back for every dollar spent. An ROI of 1.5x means you get $1.50 back for every $1 spent. Above 1.0x is profitable.
        </div>
        <div class="chart-container">
            {roi_chart}
        </div>
        <table>
            <thead>
                <tr>
                    <th>Channel</th>
                    <th>ROI</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>
                {roi_rows}
            </tbody>
        </table>

        <h2>Contribution to Conversions</h2>
        <div class="explainer">
            <div class="explainer-title">What does this show?</div>
            This shows what percentage of your total conversions can be attributed to each marketing channel. It answers "where are my conversions coming from?"
        </div>
        <div class="chart-container">
            {contrib_chart}
        </div>

        <h2>Marginal ROI Analysis</h2>
        <div class="explainer">
            <div class="explainer-title">What is Marginal ROI?</div>
            Marginal ROI is the return you'd get from spending the <em>next</em> dollar. If marginal ROI is lower than average ROI, you're hitting diminishing returns (saturation). If it's higher, there's room to scale.
        </div>
        <div class="chart-container">
            {mroi_chart}
        </div>

        <div class="footer">
            Generated by Sommmelier powered by Google Meridian<br>
            This report is based on statistical modeling and should be used alongside business judgment.
        </div>
    </div>
</body>
</html>'''

    output_path.write_text(html)
    return str(output_path)
