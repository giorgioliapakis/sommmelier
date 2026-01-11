"""Analysis and reporting for Sommmelier."""

from mmm.analysis.insights import generate_insights, Insight
from mmm.analysis.reports import generate_report
from mmm.analysis.visualize import generate_html_report

__all__ = [
    "generate_insights",
    "generate_report",
    "generate_html_report",
    "Insight",
]
