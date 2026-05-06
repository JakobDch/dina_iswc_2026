#!/usr/bin/env python3
"""
Adaptive Evaluation Dashboard.

Displays adaptive evaluation metrics with GT vs LLM result comparison.

Usage:
    python scripts/tiered_dashboard.py EXPERIMENT_NAME
    python scripts/tiered_dashboard.py deepseek_test_2
"""

import argparse
import http.server
import json
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import RESULTS_DIR

DEFAULT_PORT = 8051


def get_tiered_dashboard_html(experiment_name: str) -> str:
    """Generate the adaptive dashboard HTML."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adaptive Evaluation - {experiment_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f23;
            color: #ccc;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1800px; margin: 0 auto; }}

        header {{
            text-align: center;
            padding: 20px;
            margin-bottom: 30px;
            background: linear-gradient(135deg, #1a1a3e 0%, #0f0f23 100%);
            border-radius: 15px;
            border: 1px solid #333;
        }}
        h1 {{
            font-size: 1.8rem;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        h2 {{ color: #00d4ff; margin: 20px 0 15px; font-size: 1.3rem; }}
        h3 {{ color: #888; margin: 15px 0 10px; font-size: 1.1rem; }}
        h4 {{ color: #666; margin: 10px 0 8px; font-size: 0.95rem; }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid #333;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{ color: #888; font-size: 0.9rem; }}
        .recall {{ color: #4d96ff; }}
        .precision {{ color: #9d4edd; }}
        .f1 {{ color: #00d4ff; }}
        .correct {{ color: #6bcb77; }}
        .noise {{ color: #ff6b6b; }}
        /* Schema metrics colors */
        .schema-recall {{ color: #f0a500; }}
        .schema-precision {{ color: #e85d04; }}
        .schema-f1 {{ color: #ff6b35; }}

        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .tab {{
            padding: 10px 20px;
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 8px;
            cursor: pointer;
            color: #888;
            transition: all 0.2s;
        }}
        .tab:hover {{ background: #252545; }}
        .tab.active {{
            background: #00d4ff;
            color: #000;
            border-color: #00d4ff;
        }}

        .query-list {{
            display: grid;
            gap: 15px;
        }}
        .query-card {{
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #333;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .query-card:hover {{
            border-color: #00d4ff;
            transform: translateX(5px);
        }}
        .query-card.expanded {{
            border-color: #00d4ff;
        }}
        .query-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .query-id {{
            font-weight: bold;
            color: #00d4ff;
            font-size: 1.1rem;
        }}
        .query-metrics {{
            display: flex;
            gap: 15px;
            font-size: 0.85rem;
        }}
        .query-text {{
            color: #aaa;
            font-style: italic;
            margin-bottom: 10px;
        }}

        .query-details {{
            display: none;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #333;
        }}
        .query-card.expanded .query-details {{
            display: block;
        }}

        .sparql-box {{
            background: #0d0d1a;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.8rem;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
            margin: 10px 0;
            max-height: 200px;
            overflow-y: auto;
        }}

        .comparison-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }}
        .comparison-box {{
            background: #0d0d1a;
            border-radius: 8px;
            padding: 15px;
            overflow: hidden;
        }}
        .comparison-box h4 {{
            margin-bottom: 10px;
            font-size: 0.9rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.75rem;
        }}
        .data-table th {{
            padding: 8px 6px;
            text-align: left;
            background: #252545;
            position: sticky;
            top: 0;
            border-bottom: 1px solid #333;
            font-weight: bold;
        }}
        .data-table td {{
            padding: 6px;
            border-bottom: 1px solid #222;
            font-family: monospace;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .data-table tr:hover td {{
            background: #1a1a3e;
        }}
        .table-scroll {{
            max-height: 400px;
            overflow: auto;
        }}

        /* Column level colors */
        .col-essential {{ color: #ffd93d; }}
        .col-preferred {{ color: #6bcb77; }}
        .col-acceptable {{ color: #888; }}

        /* Value match highlighting (legacy - individual cells) */
        .match-essential {{ background: rgba(255, 217, 61, 0.2); }}
        .match-preferred {{ background: rgba(107, 203, 119, 0.2); }}
        .match-acceptable {{ background: rgba(136, 136, 136, 0.15); }}
        .no-match {{ color: #ff6b6b; }}

        /* Cell match highlighting (value-level) */
        .cell-match {{ background: rgba(107, 203, 119, 0.2); color: #6bcb77; }}
        .cell-no-match {{ background: rgba(255, 107, 107, 0.1); color: #ff6b6b; }}

        /* Tuple match highlighting (row-level) - legacy */
        .tuple-match {{ background: rgba(107, 203, 119, 0.25); }}
        .tuple-match td {{ color: #6bcb77; }}
        .tuple-no-match {{ background: rgba(255, 107, 107, 0.1); }}
        .tuple-no-match td {{ color: #ff6b6b; }}

        .filter-bar {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .filter-bar select, .filter-bar input {{
            padding: 8px 15px;
            border-radius: 8px;
            border: 1px solid #333;
            background: #1a1a2e;
            color: #ccc;
        }}

        .approach-badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
        }}
        .approach-grep {{ background: #2d5a27; color: #90EE90; }}
        .approach-semantic {{ background: #4a2d82; color: #DDA0DD; }}

        .level-badge {{
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.65rem;
            font-weight: bold;
        }}
        .level-essential {{ background: rgba(255, 217, 61, 0.3); color: #ffd93d; }}
        .level-preferred {{ background: rgba(107, 203, 119, 0.3); color: #6bcb77; }}
        .level-acceptable {{ background: rgba(136, 136, 136, 0.3); color: #aaa; }}

        .loading {{
            text-align: center;
            padding: 50px;
            color: #888;
        }}

        .info-box {{
            background: #1a1a2e;
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 0.85rem;
            color: #888;
            border-left: 3px solid #00d4ff;
        }}

        .columns-info {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0;
        }}
        .column-tag {{
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-family: monospace;
        }}
        .column-active {{ border: 1px solid #6bcb77; background: rgba(107, 203, 119, 0.1); }}
        .column-inactive {{ border: 1px solid #666; background: rgba(100, 100, 100, 0.1); opacity: 0.5; }}

        /* Paired Query Groups */
        .paired-group-container {{
            margin-bottom: 25px;
            background: rgba(255,255,255,0.02);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
        }}
        .paired-group-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: rgba(0,0,0,0.2);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .paired-group {{
            display: grid;
            grid-template-columns: 50px 1fr 1fr 1fr;
            gap: 15px;
            padding: 15px;
            align-items: start;
        }}
        .approach-label {{
            writing-mode: vertical-rl;
            text-orientation: mixed;
            padding: 10px 5px;
            font-size: 0.7rem;
            font-weight: 600;
            color: #666;
            text-align: center;
            border-radius: 4px;
        }}
        .approach-grep .approach-label {{ background: rgba(45, 90, 39, 0.3); color: #90EE90; }}
        .approach-semantic .approach-label {{ background: rgba(74, 45, 130, 0.3); color: #DDA0DD; }}

        .variant-card {{
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.08);
            min-height: 120px;
            transition: border-color 0.2s;
        }}
        .variant-card:hover {{
            border-color: #00d4ff;
        }}
        .variant-card.expanded {{
            border-color: #00d4ff;
            grid-column: 1 / -1;
        }}
        .variant-card.expanded .query-details {{
            display: block;
        }}
        .variant-card.empty {{
            opacity: 0.4;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .variant-label {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .variant-BASE .variant-label {{ background: rgba(0,255,136,0.2); color: #00ff88; }}
        .variant-SYN .variant-label {{ background: rgba(0,212,255,0.2); color: #00d4ff; }}
        .variant-TYPO .variant-label {{ background: rgba(255,170,0,0.2); color: #ffaa00; }}

        .variant-query-text {{
            font-size: 0.8rem;
            color: #aaa;
            font-style: italic;
            margin-bottom: 12px;
            line-height: 1.4;
            max-height: 40px;
            overflow: hidden;
        }}
        .variant-metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
        }}
        .variant-metric {{
            text-align: center;
            padding: 6px;
            background: rgba(255,255,255,0.03);
            border-radius: 4px;
        }}
        .variant-metric-value {{
            font-size: 1rem;
            font-weight: 600;
        }}
        .variant-metric-label {{
            font-size: 0.6rem;
            color: #666;
            margin-top: 2px;
        }}

        /* Standalone queries sections */
        .standalone-section {{
            margin-top: 30px;
        }}
        .standalone-section-title {{
            font-size: 1.1rem;
            color: #888;
            margin-bottom: 15px;
            padding: 10px 15px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            border-left: 3px solid #00d4ff;
        }}

        @media (max-width: 1200px) {{
            .comparison-container {{ grid-template-columns: 1fr; }}
            .paired-group {{ grid-template-columns: 40px 1fr; }}
            .paired-group .variant-card:nth-child(n+3) {{ grid-column: 2; }}
        }}
        @media (max-width: 800px) {{
            .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>Adaptive Evaluation Dashboard</h1>
            <div style="color: #888; margin-bottom: 10px;">{experiment_name}</div>
            <button onclick="refreshData()" style="background: #00d4ff; color: #000; border: none; padding: 8px 20px; border-radius: 6px; cursor: pointer; font-weight: bold;">
                Refresh Data
            </button>
        </header>

        <div id="content">
            <div class="loading">Loading evaluation data...</div>
        </div>
    </div>

    <script>
        let evaluationData = null;
        let currentFilter = {{ approach: 'all', querySet: 'all', search: '' }};

        async function loadData() {{
            try {{
                const response = await fetch('/api/tiered_evaluation?t=' + Date.now());
                evaluationData = await response.json();
                renderDashboard();
            }} catch (error) {{
                document.getElementById('content').innerHTML =
                    '<div class="loading">Error loading data: ' + error.message + '</div>';
            }}
        }}

        function renderDashboard() {{
            if (!evaluationData) return;

            const summary = evaluationData.summary?.overall || {{}};
            const traces = Object.entries(evaluationData.traces || {{}});

            if (!summary.count) {{
                document.getElementById('content').innerHTML =
                    '<div class="loading">No evaluation data yet. Run: python scripts/reevaluate_tiered.py {experiment_name}</div>';
                return;
            }}

            const fmt = (val) => val != null ? (val * 100).toFixed(1) : '0.0';
            const fmtInt = (val) => val != null ? val.toFixed(1) : '0';

            const expectedTotal = evaluationData.expected_total || Object.keys(evaluationData.traces || {{}}).length || 1;
            const completed = summary.count || Object.keys(evaluationData.traces || {{}}).length || 0;
            const progressPct = expectedTotal > 0 ? Math.min(100, (completed / expectedTotal) * 100) : 0;

            let html = `
                <!-- Progress Bar -->
                <div style="background: #1a1a2e; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: #00d4ff; font-weight: bold;">Experiment Progress</span>
                        <span style="color: #888;">${{completed}} / ${{expectedTotal}} runs (${{progressPct.toFixed(0)}}%)</span>
                    </div>
                    <div style="background: #0d0d1a; border-radius: 4px; height: 12px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #00d4ff, #6bcb77); height: 100%; width: ${{progressPct}}%; transition: width 0.3s;"></div>
                    </div>
                </div>

                <!-- Results Metrics (WHERE clause) -->
                <h2>Results Metrics <span style="color: #888; font-size: 0.8rem; font-weight: normal;">(WHERE clause - data completeness)</span></h2>
                <div class="info-box">
                    <strong>Results Metrics:</strong> Measures whether the LLM returned the correct DATA rows.
                    Higher recall = more GT rows found. Higher precision = fewer irrelevant rows returned.
                </div>
                <div class="summary-grid">
                    <div class="metric-card">
                        <div class="metric-value" style="color: #4d96ff;">${{fmt(summary.best_recall_mean)}}%</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #9d4edd;">${{fmt(summary.best_precision_mean)}}%</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #00d4ff;">${{fmt(summary.best_f1_mean)}}%</div>
                        <div class="metric-label">F1</div>
                    </div>
                </div>

                <!-- Schema Metrics (SELECT clause) -->
                <h2>Schema Metrics <span style="color: #888; font-size: 0.8rem; font-weight: normal;">(SELECT clause - column coverage)</span></h2>
                <div class="info-box">
                    <strong>Schema Metrics:</strong> Measures whether the LLM addressed the right COLUMNS.
                    Higher recall = more critical GT columns addressed. Higher precision = fewer irrelevant columns returned.
                </div>
                <div class="summary-grid">
                    <div class="metric-card">
                        <div class="metric-value" style="color: #f0a500;">${{fmt(summary.schema_recall_mean)}}%</div>
                        <div class="metric-label">Schema Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #e85d04;">${{fmt(summary.schema_precision_mean)}}%</div>
                        <div class="metric-label">Schema Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #ff6b35;">${{fmt(summary.schema_f1_mean)}}%</div>
                        <div class="metric-label">Schema F1</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #ffd93d;">${{fmt(summary.essential_active_rate)}}%</div>
                        <div class="metric-label">PREFERRED Active</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #6bcb77;">${{fmt(summary.preferred_active_rate)}}%</div>
                        <div class="metric-label">PREFERRED Active</div>
                    </div>
                </div>

                <!-- Retrieval Metrics (schema element coverage) -->
                <h2>Retrieval Metrics <span style="color: #888; font-size: 0.8rem; font-weight: normal;">(schema-aware triple matching)</span></h2>
                <div class="info-box">
                    <strong>Retrieval Metrics:</strong> Schema-aware evaluation of retrieved triples against ground truth using subclass-aware matching.
                    Path Coherence = fraction of GT class-property edges matched. Triple F1 = schema triple overlap.
                </div>
                <div class="summary-grid">
                    <div class="metric-card">
                        <div class="metric-value" style="color: #c4b5fd;">${{fmt(summary.retrieval_path_coherence_mean)}}%</div>
                        <div class="metric-label">Path Coherence</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #a78bfa;">${{fmt(summary.retrieval_triple_recall_mean)}}%</div>
                        <div class="metric-label">Triple Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #7c3aed;">${{fmt(summary.retrieval_triple_precision_mean)}}%</div>
                        <div class="metric-label">Triple Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #8b5cf6;">${{fmt(summary.retrieval_triple_f1_mean)}}%</div>
                        <div class="metric-label">Triple F1</div>
                    </div>
                </div>

                <h2>By Approach</h2>
                <div class="summary-grid">
            `;

            for (const [approach, stats] of Object.entries(evaluationData.summary.by_approach || {{}})) {{
                const label = approach.replace('agentic_', '').toUpperCase();
                html += `
                    <div class="metric-card" style="text-align: left; padding: 15px;">
                        <div style="font-size: 1.2rem; margin-bottom: 10px; text-align: center;">${{label}} <span style="color: #888; font-size: 0.8rem;">(${{stats.count || 0}} traces)</span></div>
                        <div style="margin-bottom: 8px; font-size: 0.75rem; color: #888; text-align: center;">Results (WHERE)</div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center; margin-bottom: 12px;">
                            <div>
                                <div style="color: #4d96ff; font-size: 1.2rem; font-weight: bold;">${{Math.round((stats.best_recall_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.65rem;">Recall</div>
                            </div>
                            <div>
                                <div style="color: #9d4edd; font-size: 1.2rem; font-weight: bold;">${{Math.round((stats.best_precision_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.65rem;">Precision</div>
                            </div>
                            <div>
                                <div style="color: #00d4ff; font-size: 1.2rem; font-weight: bold;">${{Math.round((stats.best_f1_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.65rem;">F1</div>
                            </div>
                        </div>
                        <div style="margin-bottom: 8px; font-size: 0.75rem; color: #888; text-align: center; border-top: 1px solid #333; padding-top: 10px;">Schema (SELECT)</div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;">
                            <div>
                                <div style="color: #f0a500; font-size: 1.2rem; font-weight: bold;">${{Math.round((stats.schema_recall_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.65rem;">Recall</div>
                            </div>
                            <div>
                                <div style="color: #e85d04; font-size: 1.2rem; font-weight: bold;">${{Math.round((stats.schema_precision_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.65rem;">Precision</div>
                            </div>
                            <div>
                                <div style="color: #ff6b35; font-size: 1.2rem; font-weight: bold;">${{Math.round((stats.schema_f1_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.65rem;">F1</div>
                            </div>
                        </div>
                        <div style="margin-bottom: 8px; font-size: 0.75rem; color: #888; text-align: center; border-top: 1px solid #333; padding-top: 10px;">Retrieval</div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
                            <div>
                                <div style="color: #c4b5fd; font-size: 1.2rem; font-weight: bold;">${{Math.round((stats.retrieval_path_coherence_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.65rem;">PathCoh</div>
                            </div>
                            <div>
                                <div style="color: #a78bfa; font-size: 1.2rem; font-weight: bold;">${{Math.round((stats.retrieval_triple_recall_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.65rem;">Recall</div>
                            </div>
                            <div>
                                <div style="color: #7c3aed; font-size: 1.2rem; font-weight: bold;">${{Math.round((stats.retrieval_triple_precision_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.65rem;">Precision</div>
                            </div>
                            <div>
                                <div style="color: #8b5cf6; font-size: 1.2rem; font-weight: bold;">${{Math.round((stats.retrieval_triple_f1_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.65rem;">F1</div>
                            </div>
                        </div>
                    </div>
                `;
            }}
            html += '</div>';

            // By Query Set section
            const querySetOrder = ['BASE', 'SYN', 'TYPO', 'LARGE', 'UNDER', 'CROSS'];
            const querySetLabels = {{
                'BASE': 'BASE (Baseline)',
                'SYN': 'SYN (Synonyms)',
                'TYPO': 'TYPO (Typos)',
                'LARGE': 'LARGE (Large Dataset)',
                'UNDER': 'UNDER (Underspec.)',
                'CROSS': 'CROSS (Cross-Dataset)'
            }};
            const querySetColors = {{
                'BASE': '#00ff88',
                'SYN': '#00d4ff',
                'TYPO': '#ffaa00',
                'LARGE': '#ff6b9d',
                'UNDER': '#9d4edd',
                'CROSS': '#4d96ff'
            }};

            html += `<h2>By Query Set</h2><div class="summary-grid">`;
            for (const qset of querySetOrder) {{
                const stats = (evaluationData.summary.by_query_set || {{}})[qset];
                if (!stats || !stats.count) continue;

                const color = querySetColors[qset] || '#888';
                const label = querySetLabels[qset] || qset;
                html += `
                    <div class="metric-card" style="text-align: left; padding: 15px; border-left: 3px solid ${{color}};">
                        <div style="font-size: 1rem; margin-bottom: 10px; text-align: center; color: ${{color}};">${{label}} <span style="color: #888; font-size: 0.8rem;">(${{stats.count}} traces)</span></div>
                        <div style="margin-bottom: 8px; font-size: 0.75rem; color: #888; text-align: center;">Results (WHERE)</div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center; margin-bottom: 12px;">
                            <div>
                                <div style="color: #4d96ff; font-size: 1.1rem; font-weight: bold;">${{Math.round((stats.best_recall_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.6rem;">Recall</div>
                            </div>
                            <div>
                                <div style="color: #9d4edd; font-size: 1.1rem; font-weight: bold;">${{Math.round((stats.best_precision_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.6rem;">Precision</div>
                            </div>
                            <div>
                                <div style="color: #00d4ff; font-size: 1.1rem; font-weight: bold;">${{Math.round((stats.best_f1_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.6rem;">F1</div>
                            </div>
                        </div>
                        <div style="margin-bottom: 8px; font-size: 0.75rem; color: #888; text-align: center; border-top: 1px solid #333; padding-top: 10px;">Schema (SELECT)</div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;">
                            <div>
                                <div style="color: #f0a500; font-size: 1.1rem; font-weight: bold;">${{Math.round((stats.schema_recall_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.6rem;">Recall</div>
                            </div>
                            <div>
                                <div style="color: #e85d04; font-size: 1.1rem; font-weight: bold;">${{Math.round((stats.schema_precision_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.6rem;">Precision</div>
                            </div>
                            <div>
                                <div style="color: #ff6b35; font-size: 1.1rem; font-weight: bold;">${{Math.round((stats.schema_f1_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.6rem;">F1</div>
                            </div>
                        </div>
                        <div style="margin-bottom: 8px; font-size: 0.75rem; color: #888; text-align: center; border-top: 1px solid #333; padding-top: 10px;">Retrieval</div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
                            <div>
                                <div style="color: #c4b5fd; font-size: 1.1rem; font-weight: bold;">${{Math.round((stats.retrieval_path_coherence_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.6rem;">PathCoh</div>
                            </div>
                            <div>
                                <div style="color: #a78bfa; font-size: 1.1rem; font-weight: bold;">${{Math.round((stats.retrieval_triple_recall_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.6rem;">Recall</div>
                            </div>
                            <div>
                                <div style="color: #7c3aed; font-size: 1.1rem; font-weight: bold;">${{Math.round((stats.retrieval_triple_precision_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.6rem;">Precision</div>
                            </div>
                            <div>
                                <div style="color: #8b5cf6; font-size: 1.1rem; font-weight: bold;">${{Math.round((stats.retrieval_triple_f1_mean || 0) * 100)}}%</div>
                                <div style="color: #666; font-size: 0.6rem;">F1</div>
                            </div>
                        </div>
                    </div>
                `;
            }}
            html += '</div>';

            // -------- Approach Comparison per Query Set --------
            const compareData = computeApproachComparison(traces);
            html += renderApproachComparisonHTML(compareData, querySetOrder, querySetLabels, querySetColors);

            html += `
                <h2>Query Results</h2>
                <div class="filter-bar">
                    <select id="approachFilter" onchange="updateFilter()">
                        <option value="all">All Approaches</option>
                        <option value="agentic_grep">Grep</option>
                        <option value="agentic_semantic">Semantic</option>
                    </select>
                    <select id="querySetFilter" onchange="updateFilter()">
                        <option value="all">All Query Sets</option>
                        <option value="paired">Paired Only (BASE/SYN/TYPO)</option>
                        <option value="BASE">BASE (Baseline)</option>
                        <option value="SYN">SYN (Synonyms)</option>
                        <option value="TYPO">TYPO (Typos)</option>
                        <option value="LARGE">LARGE (Large Dataset)</option>
                        <option value="UNDER">UNDER (Underspecified)</option>
                        <option value="CROSS">CROSS (Cross-Dataset)</option>
                    </select>
                    <input type="text" id="searchFilter" placeholder="Search query..." oninput="updateFilter()">
                </div>
                <div id="queryList" class="query-list">
            `;

            html += renderQueryList(traces);
            html += '</div>';

            document.getElementById('content').innerHTML = html;

            // Build Chart.js plots AFTER HTML injection so canvases exist
            try {{
                buildApproachCompareCharts(compareData, querySetOrder, querySetColors);
            }} catch (e) {{
                console.error('Error building approach comparison charts:', e);
            }}
        }}

        // ========== Approach Comparison per Query Set helpers ==========

        const CMP_METRIC_ROWS = [
            {{ key: 'best_recall',            group_key: 'adaptive_metrics',  label: 'Results Recall',     group: 'Results (WHERE)' }},
            {{ key: 'best_precision',         group_key: 'adaptive_metrics',  label: 'Results Precision',  group: 'Results (WHERE)' }},
            {{ key: 'best_f1',                group_key: 'adaptive_metrics',  label: 'Results F1',         group: 'Results (WHERE)' }},
            {{ key: 'schema_recall',          group_key: 'adaptive_metrics',  label: 'Schema Recall',      group: 'Schema (SELECT)' }},
            {{ key: 'schema_precision',       group_key: 'adaptive_metrics',  label: 'Schema Precision',   group: 'Schema (SELECT)' }},
            {{ key: 'schema_f1',              group_key: 'adaptive_metrics',  label: 'Schema F1',          group: 'Schema (SELECT)' }},
            {{ key: 'schema_path_coherence',  group_key: 'retrieval_metrics', label: 'Path Coherence',     group: 'Retrieval' }},
            {{ key: 'schema_triple_recall',   group_key: 'retrieval_metrics', label: 'Triple Recall',      group: 'Retrieval' }},
            {{ key: 'schema_triple_precision',group_key: 'retrieval_metrics', label: 'Triple Precision',   group: 'Retrieval' }},
            {{ key: 'schema_triple_f1',       group_key: 'retrieval_metrics', label: 'Triple F1',          group: 'Retrieval' }},
        ];

        function computeApproachComparison(tracesEntries) {{
            const buckets = {{}};
            for (const [key, trace] of tracesEntries) {{
                const qs = trace.query_set;
                if (!qs) continue;
                const aStr = (trace.approach || '').toLowerCase();
                const ap = aStr.includes('grep') ? 'grep' : aStr.includes('semantic') ? 'semantic' : null;
                if (!ap) continue;
                if (!buckets[qs]) {{
                    buckets[qs] = {{
                        grep: {{ count: 0, sums: {{}}, counts: {{}} }},
                        semantic: {{ count: 0, sums: {{}}, counts: {{}} }},
                    }};
                }}
                buckets[qs][ap].count += 1;
                for (const row of CMP_METRIC_ROWS) {{
                    const g = trace[row.group_key] || {{}};
                    const v = g[row.key];
                    if (v == null || isNaN(v)) continue;
                    buckets[qs][ap].sums[row.key] = (buckets[qs][ap].sums[row.key] || 0) + v;
                    buckets[qs][ap].counts[row.key] = (buckets[qs][ap].counts[row.key] || 0) + 1;
                }}
            }}
            const out = {{}};
            for (const qs of Object.keys(buckets)) {{
                out[qs] = {{}};
                for (const ap of ['grep', 'semantic']) {{
                    const b = buckets[qs][ap];
                    const means = {{ count: b.count }};
                    for (const row of CMP_METRIC_ROWS) {{
                        means[row.key] = b.counts[row.key] ? b.sums[row.key] / b.counts[row.key] : null;
                    }}
                    out[qs][ap] = means;
                }}
            }}
            return out;
        }}

        function renderApproachComparisonHTML(compareData, querySetOrder, querySetLabels, querySetColors) {{
            let html = '<h2>Approach Comparison per Query Set <span style="color:#888;font-size:0.8rem;font-weight:normal;">(Grep vs Semantic)</span></h2>';
            html += '<div class="info-box" style="margin-bottom:15px;">Per-query-set breakdown for both approaches — the key analytical view for BASE / SYN / TYPO / UNDER / CROSS. Plots below, detailed tables further down.</div>';

            // Delta heatmap
            html += '<div style="background:#1a1a2e; padding:15px; border-radius:8px; margin-bottom:20px;">';
            html += '<div style="color:#00d4ff; font-weight:bold; margin-bottom:10px;">Δ heatmap (Semantic − Grep, percentage points)</div>';
            html += '<div style="height:640px;"><canvas id="cmpDeltaChart"></canvas></div>';
            html += '</div>';

            // Two line-trend plots side by side
            html += '<div style="display:grid; grid-template-columns:1fr 1fr; gap:15px; margin-bottom:20px;">';
            html += '<div style="background:#1a1a2e; padding:15px; border-radius:8px;"><div style="color:#00d4ff; font-weight:bold; margin-bottom:10px;">Retrieval Combined F1 across query sets</div><div style="height:280px;"><canvas id="cmpRetrievalTrend"></canvas></div></div>';
            html += '<div style="background:#1a1a2e; padding:15px; border-radius:8px;"><div style="color:#00d4ff; font-weight:bold; margin-bottom:10px;">Schema F1 across query sets</div><div style="height:280px;"><canvas id="cmpResultsTrend"></canvas></div></div>';
            html += '</div>';

            // Two grouped bar charts (stacked vertically)
            html += '<div style="background:#1a1a2e; padding:15px; border-radius:8px; margin-bottom:20px;">';
            html += '<div style="color:#00d4ff; font-weight:bold; margin-bottom:10px;">Grouped bars — all metrics side-by-side (Grep vs Semantic)</div>';
            html += '<div style="height:320px; margin-bottom:15px;"><canvas id="cmpBarsSchema"></canvas></div>';
            html += '<div style="height:320px;"><canvas id="cmpBarsRetrieval"></canvas></div>';
            html += '</div>';

            // Per-query-set comparison tables
            html += '<div class="summary-grid" style="grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));">';
            for (const qs of querySetOrder) {{
                const data = compareData[qs];
                if (!data || (data.grep.count === 0 && data.semantic.count === 0)) continue;
                const color = querySetColors[qs] || '#888';
                const label = querySetLabels[qs] || qs;
                html += `<div class="metric-card" style="text-align:left; padding:15px; border-left:3px solid ${{color}};">`;
                html += `<div style="font-size:1rem; margin-bottom:10px; text-align:center; color:${{color}};">${{label}} <span style="color:#888;font-size:0.8rem;">(grep: ${{data.grep.count}}, semantic: ${{data.semantic.count}})</span></div>`;
                html += '<table style="width:100%; border-collapse:collapse; font-size:0.85rem;">';
                html += '<thead><tr style="border-bottom:1px solid #333;"><th style="text-align:left; padding:4px;">Metric</th><th style="text-align:right; padding:4px; color:#90EE90;">Grep</th><th style="text-align:right; padding:4px; color:#DDA0DD;">Semantic</th><th style="text-align:right; padding:4px;">Δ</th></tr></thead><tbody>';
                let lastGroup = '';
                for (const row of CMP_METRIC_ROWS) {{
                    if (row.group !== lastGroup) {{
                        html += `<tr><td colspan="4" style="padding:7px 4px 2px 0; color:#888; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.5px;">${{row.group}}</td></tr>`;
                        lastGroup = row.group;
                    }}
                    const g = data.grep[row.key];
                    const s = data.semantic[row.key];
                    const gStr = g == null ? '<span style="color:#555;">n/a</span>' : ((g * 100).toFixed(1) + '%');
                    const sStr = s == null ? '<span style="color:#555;">n/a</span>' : ((s * 100).toFixed(1) + '%');
                    let deltaHtml = '<span style="color:#555;">–</span>';
                    if (g != null && s != null) {{
                        const d = (s - g) * 100;
                        let col = '#888';
                        if (d > 1) col = '#00ff88';
                        else if (d < -1) col = '#ff6b6b';
                        const sign = d > 0 ? '+' : '';
                        deltaHtml = `<span style="color:${{col}}; font-weight:bold;">${{sign}}${{d.toFixed(1)}}pp</span>`;
                    }}
                    html += `<tr><td style="padding:3px 4px; color:#ccc;">${{row.label}}</td><td style="text-align:right; padding:3px 4px;">${{gStr}}</td><td style="text-align:right; padding:3px 4px;">${{sStr}}</td><td style="text-align:right; padding:3px 4px;">${{deltaHtml}}</td></tr>`;
                }}
                html += '</tbody></table></div>';
            }}
            html += '</div>';
            return html;
        }}

        function buildApproachCompareCharts(compareData, querySetOrder, querySetColors) {{
            if (typeof Chart === 'undefined') {{
                console.warn('Chart.js not loaded; skipping approach comparison plots.');
                return;
            }}
            const present = querySetOrder.filter(qs => compareData[qs] && (compareData[qs].grep.count || compareData[qs].semantic.count));
            if (present.length === 0) return;

            const GREP_COLOR = 'rgba(144, 238, 144, 0.85)';
            const SEM_COLOR  = 'rgba(221, 160, 221, 0.85)';
            const GREP_BORDER = '#90EE90';
            const SEM_BORDER  = '#DDA0DD';

            const darkAxes = {{
                x: {{ ticks: {{ color: '#ccc' }}, grid: {{ color: '#333' }} }},
                y: {{ ticks: {{ color: '#ccc' }}, grid: {{ color: '#333' }}, min: 0, max: 1 }}
            }};
            const darkLegend = {{ labels: {{ color: '#ccc' }} }};

            // --- Plot A: 3 grouped bar charts (Results / Schema / Retrieval) ---
            const grepShades = ['rgba(144,238,144,0.95)','rgba(144,238,144,0.70)','rgba(144,238,144,0.45)'];
            const semShades  = ['rgba(221,160,221,0.95)','rgba(221,160,221,0.70)','rgba(221,160,221,0.45)'];
            const makeBars = (canvasId, metrics, title) => {{
                const el = document.getElementById(canvasId);
                if (!el) return;
                const datasets = [];
                metrics.forEach((m, i) => {{
                    datasets.push({{
                        label: 'Grep ' + m.label,
                        backgroundColor: grepShades[i % 3],
                        borderColor: GREP_BORDER,
                        borderWidth: 1,
                        data: present.map(qs => compareData[qs].grep[m.key] ?? 0),
                    }});
                    datasets.push({{
                        label: 'Semantic ' + m.label,
                        backgroundColor: semShades[i % 3],
                        borderColor: SEM_BORDER,
                        borderWidth: 1,
                        data: present.map(qs => compareData[qs].semantic[m.key] ?? 0),
                    }});
                }});
                new Chart(el, {{
                    type: 'bar',
                    data: {{ labels: present, datasets }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: darkLegend,
                            title: {{ display: true, text: title, color: '#00d4ff' }},
                            tooltip: {{ callbacks: {{ label: (c) => c.dataset.label + ': ' + (c.parsed.y * 100).toFixed(1) + '%' }} }}
                        }},
                        scales: darkAxes,
                    }}
                }});
            }};
            makeBars('cmpBarsSchema', [
                {{ key: 'schema_recall',    label: 'Recall' }},
                {{ key: 'schema_precision', label: 'Precision' }},
                {{ key: 'schema_f1',        label: 'F1' }},
            ], 'Schema (SELECT)');
            makeBars('cmpBarsRetrieval', [
                {{ key: 'schema_path_coherence',  label: 'PathCoh' }},
                {{ key: 'schema_triple_recall',    label: 'Recall' }},
                {{ key: 'schema_triple_precision', label: 'Precision' }},
                {{ key: 'schema_triple_f1',        label: 'F1' }},
            ], 'Retrieval (Schema-aware)');

            // --- Plot B: Δ heatmap-style horizontal bar chart ---
            const deltaLabels = [];
            const deltaValues = [];
            const deltaColors = [];
            for (const qs of present) {{
                for (const row of CMP_METRIC_ROWS) {{
                    const g = compareData[qs].grep[row.key];
                    const s = compareData[qs].semantic[row.key];
                    if (g == null || s == null) continue;
                    const d = (s - g) * 100;
                    deltaLabels.push(qs + ' · ' + row.label);
                    deltaValues.push(Number(d.toFixed(2)));
                    if (d > 1) deltaColors.push('rgba(0, 255, 136, 0.85)');
                    else if (d < -1) deltaColors.push('rgba(255, 107, 107, 0.85)');
                    else deltaColors.push('rgba(120, 120, 120, 0.55)');
                }}
            }}
            const deltaEl = document.getElementById('cmpDeltaChart');
            if (deltaEl) {{
                new Chart(deltaEl, {{
                    type: 'bar',
                    data: {{
                        labels: deltaLabels,
                        datasets: [{{ label: 'Δ (Semantic − Grep, pp)', data: deltaValues, backgroundColor: deltaColors, borderWidth: 0 }}],
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        indexAxis: 'y',
                        plugins: {{
                            legend: {{ display: false }},
                            title: {{ display: true, text: 'Green = semantic wins · Red = grep wins · Grey = tied (|Δ|<1pp)', color: '#00d4ff' }},
                            tooltip: {{ callbacks: {{ label: (c) => (c.parsed.x >= 0 ? '+' : '') + c.parsed.x.toFixed(2) + ' pp' }} }}
                        }},
                        scales: {{
                            x: {{ ticks: {{ color: '#ccc' }}, grid: {{ color: '#333' }}, title: {{ display: true, text: 'Δ (percentage points)', color: '#ccc' }} }},
                            y: {{ ticks: {{ color: '#ccc', font: {{ size: 10 }} }}, grid: {{ color: '#222' }} }}
                        }}
                    }}
                }});
            }}

            // --- Plot C1: Retrieval Triple F1 trend ---
            const retrCtx = document.getElementById('cmpRetrievalTrend');
            if (retrCtx) {{
                new Chart(retrCtx, {{
                    type: 'line',
                    data: {{
                        labels: present,
                        datasets: [
                            {{ label: 'Grep',     data: present.map(qs => compareData[qs].grep.schema_triple_f1),     borderColor: GREP_BORDER, backgroundColor: GREP_COLOR, pointRadius: 5, tension: 0.2 }},
                            {{ label: 'Semantic', data: present.map(qs => compareData[qs].semantic.schema_triple_f1), borderColor: SEM_BORDER,  backgroundColor: SEM_COLOR,  pointRadius: 5, tension: 0.2 }},
                        ]
                    }},
                    options: {{
                        responsive: true, maintainAspectRatio: false,
                        plugins: {{ legend: darkLegend }},
                        scales: darkAxes,
                    }}
                }});
            }}

            // --- Plot C2: Schema F1 trend ---
            const resCtx = document.getElementById('cmpResultsTrend');
            if (resCtx) {{
                new Chart(resCtx, {{
                    type: 'line',
                    data: {{
                        labels: present,
                        datasets: [
                            {{ label: 'Grep',     data: present.map(qs => compareData[qs].grep.schema_f1),     borderColor: GREP_BORDER, backgroundColor: GREP_COLOR, pointRadius: 5, tension: 0.2 }},
                            {{ label: 'Semantic', data: present.map(qs => compareData[qs].semantic.schema_f1), borderColor: SEM_BORDER,  backgroundColor: SEM_COLOR,  pointRadius: 5, tension: 0.2 }},
                        ]
                    }},
                    options: {{
                        responsive: true, maintainAspectRatio: false,
                        plugins: {{ legend: darkLegend }},
                        scales: darkAxes,
                    }}
                }});
            }}
        }}

        // ================================================================

        function renderQueryList(traces) {{
            let html = '';

            const pct = (v) => v != null ? ((v || 0) * 100).toFixed(1) : '0.0';
            const pctRound = (v) => Math.round((v || 0) * 100);
            const num = (v) => v != null ? v : 0;
            const getColorClass = (v) => v >= 0.7 ? 'correct' : v >= 0.4 ? 'recall' : 'noise';

            // Group traces by variant type
            const pairedGroups = {{}};  // baseline_id -> {{ approach -> {{BASE, SYN, TYPO}} }}
            const standaloneTraces = [];

            for (const [key, trace] of traces) {{
                const querySet = trace.query_set || '';
                const approach = (trace.approach || '').includes('grep') ? 'grep' : 'semantic';

                // Apply filters
                if (currentFilter.approach !== 'all' && trace.approach !== currentFilter.approach) continue;
                if (currentFilter.search && !(trace.query_text || '').toLowerCase().includes(currentFilter.search.toLowerCase())
                    && !(trace.query_id || '').toLowerCase().includes(currentFilter.search.toLowerCase())) continue;

                // Query set filter
                if (currentFilter.querySet === 'paired') {{
                    if (!['BASE', 'SYN', 'TYPO'].includes(querySet)) continue;
                }} else if (currentFilter.querySet !== 'all' && querySet !== currentFilter.querySet) {{
                    continue;
                }}

                if (['BASE', 'SYN', 'TYPO'].includes(querySet)) {{
                    // Paired query - group by baseline_id
                    const baselineId = trace.baseline_id || trace.query_id;
                    if (!pairedGroups[baselineId]) pairedGroups[baselineId] = {{}};
                    if (!pairedGroups[baselineId][approach]) pairedGroups[baselineId][approach] = {{}};
                    pairedGroups[baselineId][approach][querySet] = {{ key, ...trace }};
                }} else {{
                    // Standalone query
                    standaloneTraces.push({{ key, ...trace }});
                }}
            }}

            // Render paired groups
            const sortedBaselineIds = Object.keys(pairedGroups).sort();
            if (sortedBaselineIds.length > 0) {{
                html += '<h3 style="color: #00d4ff; margin: 20px 0 15px;">Paired Query Comparison (BASE / SYN / TYPO)</h3>';

                for (const baselineId of sortedBaselineIds) {{
                    html += renderPairedGroup(baselineId, pairedGroups[baselineId], pctRound, getColorClass);
                }}
            }}

            // Render standalone queries grouped by set
            const bySet = {{}};
            for (const trace of standaloneTraces) {{
                const set = trace.query_set || 'OTHER';
                if (!bySet[set]) bySet[set] = [];
                bySet[set].push(trace);
            }}

            const setOrder = ['LARGE', 'UNDER', 'CROSS'];
            for (const setName of setOrder) {{
                const setTraces = bySet[setName];
                if (!setTraces || setTraces.length === 0) continue;

                const setLabels = {{
                    'LARGE': 'Large Dataset',
                    'UNDER': 'Underspecified',
                    'CROSS': 'Cross-Dataset'
                }};

                html += `
                    <div class="standalone-section">
                        <div class="standalone-section-title">${{setLabels[setName] || setName}} (${{setTraces.length}} queries)</div>
                `;
                setTraces.sort((a, b) => (a.query_id || '').localeCompare(b.query_id || ''));
                for (const trace of setTraces) {{
                    html += renderStandaloneCard(trace, pct, pctRound, num);
                }}
                html += '</div>';
            }}

            return html || '<div class="loading">No matching queries found</div>';
        }}

        function renderPairedGroup(baselineId, variantsByApproach, pctRound, getColorClass) {{
            const anyVariant = variantsByApproach.grep?.BASE || variantsByApproach.semantic?.BASE ||
                               variantsByApproach.grep?.SYN || variantsByApproach.semantic?.SYN ||
                               variantsByApproach.grep?.TYPO || variantsByApproach.semantic?.TYPO;
            const dataset = anyVariant?.dataset || 'Unknown';

            let html = `
            <div class="paired-group-container">
                <div class="paired-group-header">
                    <div>
                        <span style="font-weight:600;color:#00d4ff;">${{baselineId}}</span>
                        <span style="color:#666;margin-left:15px;">${{dataset}}</span>
                    </div>
                    <div style="color:#888;font-size:0.8rem;">Paired Comparison</div>
                </div>`;

            // One row per approach
            for (const approach of ['grep', 'semantic']) {{
                const variants = variantsByApproach[approach];
                if (!variants) continue;

                const approachClass = approach === 'grep' ? 'approach-grep' : 'approach-semantic';
                html += `
                <div class="paired-group ${{approachClass}}">
                    <div class="approach-label">${{approach.toUpperCase()}}</div>
                    ${{renderVariantCard('BASE', variants.BASE, pctRound, getColorClass)}}
                    ${{renderVariantCard('SYN', variants.SYN, pctRound, getColorClass)}}
                    ${{renderVariantCard('TYPO', variants.TYPO, pctRound, getColorClass)}}
                </div>`;
            }}

            html += '</div>';
            return html;
        }}

        function renderVariantCard(variantType, trace, pctRound, getColorClass) {{
            if (!trace) {{
                return `
                <div class="variant-card variant-${{variantType}} empty">
                    <div class="variant-label">${{variantType}}</div>
                    <div style="color:#666;font-style:italic;font-size:0.8rem;">Not executed</div>
                </div>`;
            }}

            const m = trace.adaptive_metrics || {{}};
            const f1 = m.best_f1 || 0;
            const recall = m.best_recall || 0;
            const precision = m.best_precision || 0;
            const sf1 = m.schema_f1 || 0;
            const srecall = m.schema_recall || 0;
            const sprecision = m.schema_precision || 0;
            const rm = trace.retrieval_metrics || {{}};
            const rf1 = rm.schema_triple_f1 || 0;
            const rrecall = rm.schema_triple_recall || 0;
            const rprecision = rm.schema_triple_precision || 0;
            const rPathCoh = rm.schema_path_coherence || 0;
            const key = trace.key || trace.query_id;

            return `
            <div class="variant-card variant-${{variantType}}" onclick="toggleVariantDetail(this, event)" style="cursor:pointer;">
                <div class="variant-label">${{variantType}}</div>
                <div class="variant-query-text">"${{trace.query_text || ''}}"</div>
                <div style="font-size: 0.65rem; color: #888; margin-bottom: 2px;">Results</div>
                <div class="variant-metrics">
                    <div class="variant-metric">
                        <div class="variant-metric-value ${{getColorClass(f1)}}">${{pctRound(f1)}}%</div>
                        <div class="variant-metric-label">F1</div>
                    </div>
                    <div class="variant-metric">
                        <div class="variant-metric-value">${{pctRound(precision)}}%</div>
                        <div class="variant-metric-label">P</div>
                    </div>
                    <div class="variant-metric">
                        <div class="variant-metric-value">${{pctRound(recall)}}%</div>
                        <div class="variant-metric-label">R</div>
                    </div>
                </div>
                <div style="font-size: 0.65rem; color: #888; margin-bottom: 2px; margin-top: 6px;">Schema</div>
                <div class="variant-metrics">
                    <div class="variant-metric">
                        <div class="variant-metric-value" style="color: #f0a500;">${{pctRound(sf1)}}%</div>
                        <div class="variant-metric-label">F1</div>
                    </div>
                    <div class="variant-metric">
                        <div class="variant-metric-value" style="color: #e85d04;">${{pctRound(sprecision)}}%</div>
                        <div class="variant-metric-label">P</div>
                    </div>
                    <div class="variant-metric">
                        <div class="variant-metric-value" style="color: #f0a500;">${{pctRound(srecall)}}%</div>
                        <div class="variant-metric-label">R</div>
                    </div>
                </div>
                <div style="font-size: 0.65rem; color: #888; margin-bottom: 2px; margin-top: 6px;">Retrieval</div>
                <div class="variant-metrics">
                    <div class="variant-metric">
                        <div class="variant-metric-value" style="color: #8b5cf6;">${{pctRound(rf1)}}%</div>
                        <div class="variant-metric-label">F1</div>
                    </div>
                    <div class="variant-metric">
                        <div class="variant-metric-value" style="color: #7c3aed;">${{pctRound(rprecision)}}%</div>
                        <div class="variant-metric-label">P</div>
                    </div>
                    <div class="variant-metric">
                        <div class="variant-metric-value" style="color: #a78bfa;">${{pctRound(rrecall)}}%</div>
                        <div class="variant-metric-label">R</div>
                    </div>
                </div>
                <div class="query-details">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                        <div>
                            <h3 style="color: #00d4ff;">Generated SPARQL</h3>
                            <div class="sparql-box" style="height: 200px; overflow: auto;">${{escapeHtml(trace.final_sparql || 'No SPARQL generated')}}</div>
                        </div>
                        <div>
                            <h3 style="color: #6bcb77;">Ground Truth SPARQL</h3>
                            <div class="sparql-box" style="height: 200px; overflow: auto;">${{escapeHtml(trace.gt_sparql || 'No GT SPARQL')}}</div>
                        </div>
                    </div>

                    <h3>Result Metrics</h3>
                    <div class="metrics-grid" style="margin-bottom: 20px;">
                        <div class="metric-box recall">
                            <div class="metric-value">${{pctRound(m.best_recall)}}%</div>
                            <div class="metric-label">Recall</div>
                        </div>
                        <div class="metric-box precision">
                            <div class="metric-value">${{pctRound(m.best_precision)}}%</div>
                            <div class="metric-label">Precision</div>
                        </div>
                        <div class="metric-box f1">
                            <div class="metric-value">${{pctRound(m.best_f1)}}%</div>
                            <div class="metric-label">F1</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">${{m.llm_row_count || 0}}</div>
                            <div class="metric-label">LLM Rows</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">${{m.best_recall_gt_size || 0}}</div>
                            <div class="metric-label">GT Rows</div>
                        </div>
                    </div>

                    ${{renderRetrievalMetrics(trace)}}

                    ${{renderColumnInfo(trace)}}

                    <h3>Data Comparison</h3>
                    ${{renderDataComparison(trace, key)}}
                </div>
            </div>`;
        }}

        function renderStandaloneCard(trace, pct, pctRound, num) {{
            const m = trace.adaptive_metrics || {{}};
            const approachClass = (trace.approach || '').includes('grep') ? 'approach-grep' : 'approach-semantic';
            const approachLabel = (trace.approach || '').replace('agentic_', '').toUpperCase();
            const key = trace.key || trace.query_id;

            return `
                <div class="query-card" onclick="toggleCard(this)">
                    <div class="query-header">
                        <div>
                            <span class="query-id">${{trace.query_id}}</span>
                            <span class="approach-badge ${{approachClass}}">${{approachLabel}}</span>
                            <span style="color: #666; font-size: 0.8rem; margin-left: 10px;">${{trace.dataset || ''}}</span>
                        </div>
                        <div class="query-metrics">
                            <span style="color: #888; margin-right: 5px;">Results:</span>
                            <span class="recall" title="Results Recall">R:${{pctRound(m.best_recall)}}%</span>
                            <span class="precision" title="Results Precision">P:${{pctRound(m.best_precision)}}%</span>
                            <span class="f1" style="font-weight: bold;" title="Results F1">F1:${{pctRound(m.best_f1)}}%</span>
                            <span style="color: #555; margin: 0 8px;">|</span>
                            <span style="color: #888; margin-right: 5px;">Schema:</span>
                            <span style="color: #f0a500;" title="Schema Recall">R:${{pctRound(m.schema_recall)}}%</span>
                            <span style="color: #e85d04;" title="Schema Precision">P:${{pctRound(m.schema_precision)}}%</span>
                            <span style="color: #ff6b35; font-weight: bold;" title="Schema F1">F1:${{pctRound(m.schema_f1)}}%</span>
                            ${{(() => {{
                                const rm = trace.retrieval_metrics || {{}};
                                if (rm.schema_triple_f1 != null) {{
                                    return `<span style="color: #555; margin: 0 8px;">|</span>
                                        <span style="color: #888; margin-right: 5px;">Retrieval:</span>
                                        <span style="color: #c4b5fd;" title="Path Coherence">PC:${{pctRound(rm.schema_path_coherence)}}%</span>
                                        <span style="color: #a78bfa;" title="Triple Recall">R:${{pctRound(rm.schema_triple_recall)}}%</span>
                                        <span style="color: #7c3aed;" title="Triple Precision">P:${{pctRound(rm.schema_triple_precision)}}%</span>
                                        <span style="color: #8b5cf6; font-weight: bold;" title="Triple F1">F1:${{pctRound(rm.schema_triple_f1)}}%</span>`;
                                }}
                                return '';
                            }})()}}
                        </div>
                    </div>
                    <div class="query-text">"${{trace.query_text}}"</div>

                    <div class="query-details">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                            <div>
                                <h3 style="color: #00d4ff;">Generated SPARQL</h3>
                                <div class="sparql-box" style="height: 200px; overflow: auto;">${{escapeHtml(trace.final_sparql || 'No SPARQL generated')}}</div>
                            </div>
                            <div>
                                <h3 style="color: #6bcb77;">Ground Truth SPARQL</h3>
                                <div class="sparql-box" style="height: 200px; overflow: auto;">${{escapeHtml(trace.gt_sparql || 'No GT SPARQL')}}</div>
                            </div>
                        </div>

                        <h3>Results Metrics <span style="color: #666; font-size: 0.8rem; font-weight: normal;">(WHERE clause - data completeness)</span></h3>
                        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 10px 0;">
                            <div class="metric-card" style="padding: 10px;">
                                <div class="recall" style="font-size: 1.3rem;">${{pct(m.best_recall)}}%</div>
                                <div class="metric-label">Results Recall</div>
                            </div>
                            <div class="metric-card" style="padding: 10px;">
                                <div class="precision" style="font-size: 1.3rem;">${{pct(m.best_precision)}}%</div>
                                <div class="metric-label">Results Precision</div>
                            </div>
                            <div class="metric-card" style="padding: 10px;">
                                <div class="f1" style="font-size: 1.3rem;">${{pct(m.best_f1)}}%</div>
                                <div class="metric-label">Results F1</div>
                            </div>
                            <div class="metric-card" style="padding: 10px;">
                                <div style="color: #ccc; font-size: 1.2rem;">${{num(trace.generated_total_rows)}}</div>
                                <div class="metric-label">LLM Rows</div>
                            </div>
                            <div class="metric-card" style="padding: 10px;">
                                <div style="color: #ccc; font-size: 1.2rem;">${{num(trace.gt_total_rows)}}</div>
                                <div class="metric-label">GT Rows (projected)</div>
                            </div>
                        </div>

                        <h3>Schema Metrics <span style="color: #666; font-size: 0.8rem; font-weight: normal;">(SELECT clause - column coverage)</span></h3>
                        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 10px 0;">
                            <div class="metric-card" style="padding: 10px;">
                                <div style="color: #f0a500; font-size: 1.3rem;">${{pct(m.schema_recall)}}%</div>
                                <div class="metric-label">Schema Recall</div>
                            </div>
                            <div class="metric-card" style="padding: 10px;">
                                <div style="color: #e85d04; font-size: 1.3rem;">${{pct(m.schema_precision)}}%</div>
                                <div class="metric-label">Schema Precision</div>
                            </div>
                            <div class="metric-card" style="padding: 10px;">
                                <div style="color: #ff6b35; font-size: 1.3rem;">${{pct(m.schema_f1)}}%</div>
                                <div class="metric-label">Schema F1</div>
                            </div>
                            <div class="metric-card" style="padding: 10px;">
                                <div style="color: #ccc; font-size: 1.2rem;">${{num(m.schema_matched_count)}} / ${{num(m.schema_expected_count)}}</div>
                                <div class="metric-label">GT Cols Active</div>
                            </div>
                            <div class="metric-card" style="padding: 10px;">
                                <div style="color: #ccc; font-size: 1.2rem;">${{num(m.schema_llm_matched)}} / ${{num(m.schema_llm_columns)}}</div>
                                <div class="metric-label">LLM Cols Matched</div>
                            </div>
                        </div>

                        ${{renderRetrievalMetrics(trace)}}

                        <!-- Level Activity -->
                        <h3>Active Levels</h3>
                        <div style="display: flex; gap: 20px; margin: 10px 0;">
                            <span style="color: ${{trace.essential_active ? '#6bcb77' : '#ff6b6b'}};">
                                ${{trace.essential_active ? '✓' : '✗'}} PREFERRED (${{trace.essential_matches || 0}} matches)
                            </span>
                            <span style="color: ${{trace.preferred_active ? '#6bcb77' : '#ff6b6b'}};">
                                ${{trace.preferred_active ? '✓' : '✗'}} PREFERRED (${{trace.preferred_matches || 0}} matches)
                            </span>
                            <span style="color: ${{trace.acceptable_active ? '#6bcb77' : '#666'}};">
                                ${{trace.acceptable_active ? '✓' : '-'}} ACCEPTABLE (${{trace.acceptable_matches || 0}} matches)
                            </span>
                        </div>

                        <!-- Column Projection Info -->
                        ${{renderColumnInfo(trace)}}

                        <!-- GT vs LLM Comparison -->
                        <h3>Data Comparison</h3>
                        ${{renderDataComparison(trace, key)}}
                    </div>
                </div>
            `;
        }}

        function renderRetrievalMetrics(trace) {{
            const rm = trace.retrieval_metrics || {{}};
            if (rm.schema_triple_f1 == null) return '';

            const pctR = v => v != null ? (v * 100).toFixed(1) : 'N/A';

            let html = `<h3>Retrieval Metrics <span style="color: #666; font-size: 0.8rem; font-weight: normal;">(schema-aware triple matching)</span></h3>`;
            html += `<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 10px 0;">
                <div class="metric-card" style="padding: 10px;">
                    <div style="color: #c4b5fd; font-size: 1.3rem;">${{pctR(rm.schema_path_coherence)}}%</div>
                    <div class="metric-label">Path Coherence</div>
                </div>
                <div class="metric-card" style="padding: 10px;">
                    <div style="color: #a78bfa; font-size: 1.3rem;">${{pctR(rm.schema_triple_recall)}}%</div>
                    <div class="metric-label">Triple Recall</div>
                </div>
                <div class="metric-card" style="padding: 10px;">
                    <div style="color: #7c3aed; font-size: 1.3rem;">${{pctR(rm.schema_triple_precision)}}%</div>
                    <div class="metric-label">Triple Precision</div>
                </div>
                <div class="metric-card" style="padding: 10px;">
                    <div style="color: #8b5cf6; font-size: 1.3rem;">${{pctR(rm.schema_triple_f1)}}%</div>
                    <div class="metric-label">Triple F1</div>
                </div>
            </div>`;

            // Path details
            const pathDetails = rm.schema_paths_details || [];
            if (pathDetails.length > 0) {{
                const pathItems = pathDetails.map(pd => {{
                    const color = pd.matched ? '#6bcb77' : (pd.necessity === 'acceptable' ? '#888' : '#ff6b6b');
                    const icon = pd.matched ? '✓' : (pd.necessity === 'acceptable' ? '-' : '✗');
                    const suffix = pd.necessity === 'acceptable' ? ' <span style="color:#888;font-size:0.7rem;">(ACC)</span>' : '';
                    const domain = pd.domain ? pd.domain.split('#').pop() || pd.domain.split('/').pop() : '?';
                    const prop = pd.property.split('#').pop() || pd.property.split('/').pop();
                    return `<span style="color: ${{color}};">${{icon}} ${{domain}}.${{prop}}${{suffix}}</span>`;
                }});
                html += `<div style="background: #1a1a3a; padding: 10px; border-radius: 6px; margin: 10px 0;">
                    <div style="font-weight: bold; color: #a78bfa; margin-bottom: 6px;">Path Details (${{rm.schema_paths_matched || 0}}/${{rm.schema_paths_total || 0}} matched)</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 6px; font-size: 0.8rem;">
                        ${{pathItems.join('')}}
                    </div>
                </div>`;
            }}

            // GT Turtle
            const gtTurtle = rm.gt_schema_turtle || '';
            if (gtTurtle) {{
                html += `<div style="background: #1a1a3a; padding: 10px; border-radius: 6px; margin: 10px 0;">
                    <div style="font-weight: bold; color: #a78bfa; margin-bottom: 6px;">GT Schema Triples</div>
                    <pre style="font-size: 0.75rem; color: #ccc; margin: 0; white-space: pre-wrap; max-height: 200px; overflow-y: auto;">${{escapeHtml(gtTurtle)}}</pre>
                </div>`;
            }}

            return html;
        }}

        function renderColumnInfo(trace) {{
            const gtCols = trace.gt_columns || [];
            const colMapping = trace.column_mapping || [];
            const originalCols = trace.gt_original_columns || [];
            const projectedCols = trace.projected_columns || [];

            if (!gtCols.length) return '';

            // Build lookup: gt_col_name -> matched llm_col_name
            const gtToLlm = {{}};
            for (const m of colMapping) {{
                gtToLlm[m.gt_col_name] = m.llm_col_name;
            }}

            let html = `<h3>Column Mapping (GT &rarr; LLM)</h3>
                <div class="info-box" style="font-size: 0.8rem;">
                    <strong>GT columns:</strong> ${{originalCols.length}} |
                    <strong>Matched:</strong> ${{colMapping.length}}
                    ${{originalCols.length !== colMapping.length ?
                        `<span style="color: #ffd93d;"> (${{originalCols.length - colMapping.length}} column(s) excluded from result metrics)</span>` : ''}}
                </div>
                <div class="columns-info">`;

            for (const col of gtCols) {{
                const llmMatch = gtToLlm[col.var_name];
                const isMatched = !!llmMatch;
                const levelClass = col.level === 'PREFERRED' ? 'level-essential' : 'level-acceptable';
                html += `
                    <div class="column-tag ${{isMatched ? 'column-active' : 'column-inactive'}}">
                        <span class="level-badge ${{levelClass}}">${{col.level.charAt(0)}}</span>
                        ?${{col.var_name}}
                        ${{isMatched
                            ? `<span style="color: #6bcb77;"> &rarr; ?${{llmMatch}}</span>`
                            : '<span style="color: #ff6b6b;"> (no match)</span>'}}
                    </div>
                `;
            }}

            html += '</div>';
            return html;
        }}

        function renderDataComparison(trace, traceKey) {{
            const gtRows = trace.gt_sample_rows || [];
            const llmRows = trace.generated_sample_rows || [];
            const gtAllCols = trace.gt_original_columns || trace.projected_columns || [];
            const llmCols = trace.generated_columns || [];
            const gtTotal = trace.gt_total_rows || gtRows.length;
            const llmTotal = trace.generated_total_rows || llmRows.length;

            // Column mapping from evaluation: which GT cols matched which LLM cols
            const colMapping = trace.column_mapping || [];
            const matchedGtCols = new Set(colMapping.map(m => m.gt_col_name));
            const matchedLlmCols = new Set(colMapping.map(m => m.llm_col_name));

            // Count TP/FP/FN from variant_results (best variant)
            const adaptiveMetrics = trace.adaptive_metrics || {{}};
            const bestRecall = adaptiveMetrics.best_recall || 0;
            const bestPrecision = adaptiveMetrics.best_precision || 0;

            let html = `<div class="comparison-container">`;

            // --- GT Table: ALL columns, unmatched columns in red, TP rows in green ---
            html += `
                <div class="comparison-box">
                    <h4>
                        <span style="color: #6bcb77;">Ground Truth (Solution Set)</span>
                        <span style="color: #666; font-weight: normal;">${{gtTotal}} rows</span>
                    </h4>
                    <div class="table-scroll">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    ${{gtAllCols.map(col => {{
                                        const isMatched = matchedGtCols.has(col);
                                        const style = isMatched
                                            ? 'color: #6bcb77;'
                                            : 'color: #ff6b6b; text-decoration: line-through; opacity: 0.5;';
                                        const title = isMatched ? 'Matched to LLM column' : 'NOT matched - excluded from result metrics';
                                        return `<th style="${{style}}" title="${{title}}">${{escapeHtml(col)}}</th>`;
                                    }}).join('')}}
                                </tr>
                            </thead>
                            <tbody>
            `;

            for (const row of gtRows.slice(0, 50)) {{
                const isTP = row._is_tp === true;
                const rowStyle = isTP
                    ? 'background: rgba(107, 203, 119, 0.12);'
                    : '';
                html += `<tr style="${{rowStyle}}">`;
                for (const col of gtAllCols) {{
                    const val = row[col] || '';
                    const isMatched = matchedGtCols.has(col);
                    let cellStyle = '';
                    if (!isMatched) {{
                        cellStyle = 'color: #ff6b6b; opacity: 0.35;';
                    }} else if (isTP) {{
                        cellStyle = 'color: #6bcb77;';
                    }}
                    html += `<td style="${{cellStyle}}" title="${{escapeHtml(val)}}">${{escapeHtml(truncate(val, 40))}}</td>`;
                }}
                html += '</tr>';
            }}

            if (gtTotal > 50) {{
                html += `<tr><td colspan="${{gtAllCols.length}}" style="text-align: center; color: #666;">... and ${{gtTotal - 50}} more rows</td></tr>`;
            }}

            html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            `;

            // --- LLM Table: ALL columns, unmatched columns in red, TP rows in green ---
            html += `
                <div class="comparison-box">
                    <h4>
                        <span style="color: #00d4ff;">LLM Generated Results</span>
                        <span style="color: #666; font-weight: normal;">${{llmTotal}} rows</span>
                    </h4>
                    <div class="table-scroll">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    ${{llmCols.map(col => {{
                                        const isMatched = matchedLlmCols.has(col);
                                        const style = isMatched
                                            ? 'color: #6bcb77;'
                                            : 'color: #ff6b6b; opacity: 0.5;';
                                        const mapping = colMapping.find(m => m.llm_col_name === col);
                                        const title = isMatched
                                            ? `Matched to GT column: ${{mapping ? mapping.gt_col_name : '?'}}`
                                            : 'NOT matched - excluded from result metrics';
                                        return `<th style="${{style}}" title="${{title}}">${{escapeHtml(col)}}</th>`;
                                    }}).join('')}}
                                </tr>
                            </thead>
                            <tbody>
            `;

            for (const row of llmRows.slice(0, 50)) {{
                const isTP = row._is_tp === true;
                const rowStyle = isTP
                    ? 'background: rgba(107, 203, 119, 0.12);'
                    : '';
                html += `<tr style="${{rowStyle}}">`;
                for (const col of llmCols) {{
                    const val = row[col] || '';
                    const isMatched = matchedLlmCols.has(col);
                    let cellStyle = '';
                    if (!isMatched) {{
                        cellStyle = 'color: #ff6b6b; opacity: 0.35;';
                    }} else if (isTP) {{
                        cellStyle = 'color: #6bcb77;';
                    }}
                    html += `<td style="${{cellStyle}}" title="${{escapeHtml(val)}}">${{escapeHtml(truncate(val, 40))}}</td>`;
                }}
                html += '</tr>';
            }}

            if (llmTotal > 50) {{
                html += `<tr><td colspan="${{llmCols.length}}" style="text-align: center; color: #666;">... and ${{llmTotal - 50}} more rows</td></tr>`;
            }}

            html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            `;

            html += '</div>';

            // Legend
            html += `
                <div style="margin-top: 10px; display: flex; gap: 20px; font-size: 0.75rem; color: #888;">
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: rgba(107, 203, 119, 0.15); border: 1px solid #6bcb77; margin-right: 5px;"></span>TP row (matched for Result Metrics)</span>
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: transparent; border: 1px solid #666; margin-right: 5px;"></span>FN/FP row (not matched)</span>
                    <span><span style="color: #ff6b6b; text-decoration: line-through;">column</span> = excluded from Result Metrics</span>
                </div>
            `;

            return html;
        }}

        function truncate(str, len) {{
            if (!str) return '';
            str = String(str);
            return str.length > len ? str.substring(0, len - 3) + '...' : str;
        }}

        function escapeHtml(text) {{
            if (text == null) return '';
            const div = document.createElement('div');
            div.textContent = String(text);
            return div.innerHTML;
        }}

        function toggleCard(card) {{
            card.classList.toggle('expanded');
        }}

        function toggleVariantDetail(card, event) {{
            event.stopPropagation();
            card.classList.toggle('expanded');
        }}

        function updateFilter() {{
            currentFilter.approach = document.getElementById('approachFilter').value;
            currentFilter.querySet = document.getElementById('querySetFilter').value;
            currentFilter.search = document.getElementById('searchFilter').value;

            const traces = Object.entries(evaluationData.traces);
            document.getElementById('queryList').innerHTML = renderQueryList(traces);
        }}

        function refreshData() {{
            document.getElementById('content').innerHTML = '<div class="loading">Refreshing data...</div>';
            loadData();
        }}

        loadData();
    </script>
</body>
</html>'''


def run_tiered_dashboard(experiment_name: str, port: int = DEFAULT_PORT):
    """Run the adaptive evaluation dashboard."""

    experiment_dir = RESULTS_DIR / "experiments" / experiment_name
    adaptive_eval_file = experiment_dir / "adaptive_evaluation.json"

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    class DashboardHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path == "/" or parsed.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(get_tiered_dashboard_html(experiment_name).encode("utf-8"))

            elif parsed.path == "/api/tiered_evaluation":
                # Try adaptive first, then fall back to tiered
                eval_file = adaptive_eval_file
                if not eval_file.exists():
                    eval_file = experiment_dir / "tiered_evaluation.json"

                if eval_file.exists():
                    with open(eval_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                else:
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    empty = {"experiment_name": experiment_name, "traces": {}, "summary": {"overall": {}, "by_approach": {}, "by_query_set": {}}}
                    self.wfile.write(json.dumps(empty).encode("utf-8"))

            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass

    for try_port in range(port, port + 10):
        try:
            with socketserver.TCPServer(("", try_port), DashboardHandler) as httpd:
                url = f"http://localhost:{try_port}"
                print(f"\n[*] Adaptive Evaluation Dashboard: {experiment_name}")
                print(f"[>] Dashboard URL: {url}")
                print(f"\n[i] Press F5 in browser to refresh data")
                print(f"[i] Run 'python scripts/reevaluate_tiered.py {experiment_name}' to update metrics")
                print(f"\nPress Ctrl+C to stop the dashboard\n")

                threading.Timer(0.5, lambda: webbrowser.open(url)).start()
                httpd.serve_forever()
        except OSError:
            continue
        break
    else:
        print(f"Error: Could not find available port in range {port}-{port + 9}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Evaluation Dashboard"
    )
    parser.add_argument(
        "experiment",
        help="Name of the experiment"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run on (default: {DEFAULT_PORT})"
    )

    args = parser.parse_args()
    run_tiered_dashboard(args.experiment, args.port)


if __name__ == "__main__":
    main()
