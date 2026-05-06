#!/usr/bin/env python3
"""
Agent Context Visualization Dashboard with Live Query Execution.

Displays the full context of each agent (Retrieval, SPARQL, MultiStep, MappingOptimizer)
in a synchronized 4-column timeline view. All columns scroll together, and when one
agent is active, others show empty space at the same vertical position.

Usage:
    python scripts/agent_context_dashboard.py --live
    python scripts/agent_context_dashboard.py --live --port 8052
    python scripts/agent_context_dashboard.py EXPERIMENT_NAME  # Load saved traces
"""

import argparse
import asyncio
import http.server
import json
import queue
import socketserver
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import RESULTS_DIR

DEFAULT_PORT = 8052

# Global queue for live events (thread-safe)
live_events_queue: queue.Queue = queue.Queue()
current_run_state: dict = {}
run_lock = threading.Lock()

# Global stop signal for runs
current_run_id: str | None = None
stop_requested: threading.Event = threading.Event()


def get_dashboard_html(mode: str = "live", experiment_name: str = "") -> str:
    """Generate the dashboard HTML with embedded JavaScript."""
    title = "Live Agent Visualization" if mode == "live" else f"Agent Context - {experiment_name}"
    show_live_controls = "true" if mode == "live" else "false"

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f0f23;
            min-height: 100vh;
            color: #ccc;
        }}
        .header {{
            padding: 15px 20px;
            background: #1a1a2e;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .header h1 {{
            font-size: 1.2rem;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header-controls {{
            display: flex;
            gap: 15px;
            align-items: center;
        }}
        .run-selector {{
            padding: 8px 12px;
            background: #252545;
            border: 1px solid #444;
            border-radius: 6px;
            color: #ccc;
            font-size: 0.9rem;
            min-width: 300px;
        }}
        .live-controls {{
            padding: 15px 20px;
            background: #1a1a2e;
            border-bottom: 1px solid #333;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .query-input {{
            flex: 1;
            min-width: 300px;
            padding: 12px 15px;
            background: #252545;
            border: 1px solid #444;
            border-radius: 6px;
            color: #eee;
            font-size: 1rem;
        }}
        .query-input:focus {{
            outline: none;
            border-color: #00d4ff;
        }}
        .selector {{
            padding: 12px 15px;
            background: #252545;
            border: 1px solid #444;
            border-radius: 6px;
            color: #ccc;
            font-size: 0.9rem;
        }}
        .execute-btn {{
            padding: 12px 25px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            border: none;
            border-radius: 6px;
            color: #fff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }}
        .execute-btn:hover {{
            opacity: 0.9;
        }}
        .execute-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        .stop-btn {{
            padding: 12px 25px;
            background: #dc3545;
            border: none;
            border-radius: 6px;
            color: #fff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
        }}
        .clear-btn {{
            padding: 12px 15px;
            background: #444;
            border: none;
            border-radius: 6px;
            color: #ccc;
            font-size: 0.9rem;
            cursor: pointer;
        }}
        .query-info {{
            padding: 10px 20px;
            background: rgba(0, 212, 255, 0.1);
            border-bottom: 1px solid #333;
        }}
        .query-text {{
            font-style: italic;
            color: #00d4ff;
        }}
        .info-badge {{
            display: inline-block;
            padding: 3px 8px;
            background: #7b2cbf;
            border-radius: 4px;
            font-size: 0.75rem;
            margin-left: 10px;
        }}
        .info-badge.dataset {{
            background: #2d6a4f;
        }}
        .status-bar {{
            padding: 8px 20px;
            background: #252545;
            border-bottom: 1px solid #333;
            font-size: 0.85rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .status-text {{
            color: #888;
        }}
        .status-text.running {{
            color: #6bcb77;
        }}
        .status-text.error {{
            color: #ff6b6b;
        }}

        /* Column headers - sticky */
        .column-headers {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            background: #252545;
            border-bottom: 2px solid #444;
            position: sticky;
            top: 0;
            z-index: 50;
        }}
        .column-header {{
            padding: 12px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-right: 1px solid #333;
        }}
        .column-header:last-child {{
            border-right: none;
        }}
        .column-header h2 {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #eee;
        }}
        .status-indicator {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #444;
        }}
        .status-indicator.active {{
            background: #6bcb77;
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(107, 203, 119, 0.4); }}
            50% {{ opacity: 0.8; box-shadow: 0 0 0 8px rgba(107, 203, 119, 0); }}
        }}

        /* Main timeline container - synchronized scroll */
        .timeline-wrapper {{
            height: calc(100vh - 200px);
            overflow-y: auto;
        }}

        /* Each row spans all 4 columns */
        .timeline-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            border-bottom: 1px solid #222;
            min-height: 20px;
        }}
        .timeline-row:hover {{
            background: rgba(255,255,255,0.02);
        }}

        /* Cell in each column */
        .timeline-cell {{
            padding: 8px 10px;
            border-right: 1px solid #333;
            min-height: 40px;
        }}
        .timeline-cell:last-child {{
            border-right: none;
        }}
        .timeline-cell.empty {{
            /* Empty cell - just takes up space */
        }}

        /* Message styling */
        .message-event {{
            padding: 10px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            line-height: 1.5;
            word-break: break-word;
            animation: fadeIn 0.3s ease;
            margin: 2px 0;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-5px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .message-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            font-size: 0.7rem;
        }}
        .message-type {{
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .timestamp {{
            color: #666;
        }}
        .message-content {{
            white-space: pre-wrap;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.75rem;
            max-height: 200px;
            overflow-y: auto;
            background: rgba(0,0,0,0.2);
            padding: 8px;
            border-radius: 4px;
            margin-top: 6px;
        }}
        .message-system {{
            background: rgba(139, 92, 246, 0.15);
            border-left: 3px solid #8b5cf6;
        }}
        .message-system .message-type {{ color: #8b5cf6; }}
        .message-human {{
            background: rgba(0, 212, 255, 0.15);
            border-left: 3px solid #00d4ff;
        }}
        .message-human .message-type {{ color: #00d4ff; }}
        .message-ai {{
            background: rgba(107, 203, 119, 0.15);
            border-left: 3px solid #6bcb77;
        }}
        .message-ai .message-type {{ color: #6bcb77; }}
        .message-tool {{
            background: rgba(255, 170, 0, 0.15);
            border-left: 3px solid #ffaa00;
        }}
        .message-tool .message-type {{ color: #ffaa00; }}
        .tool-call {{
            background: rgba(255, 107, 107, 0.1);
            border: 1px solid rgba(255, 107, 107, 0.3);
            border-radius: 4px;
            padding: 6px 8px;
            margin-top: 8px;
            font-family: monospace;
            font-size: 0.7rem;
        }}
        .tool-call-name {{
            color: #ff6b6b;
            font-weight: 600;
        }}

        /* Delegation marker - spans all columns */
        .delegation-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            background: linear-gradient(90deg, rgba(0, 212, 255, 0.1), rgba(123, 44, 191, 0.1));
            border-top: 2px solid #7b2cbf;
            border-bottom: 2px solid #00d4ff;
        }}
        .delegation-cell {{
            padding: 10px 15px;
            border-right: 1px solid rgba(123, 44, 191, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .delegation-cell:last-child {{
            border-right: none;
        }}
        .delegation-from {{
            color: #ff6b6b;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        .delegation-to {{
            color: #6bcb77;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        .delegation-arrow {{
            color: #7b2cbf;
            font-size: 1.5rem;
            padding: 0 10px;
        }}
        .delegation-reason {{
            color: #888;
            font-size: 0.75rem;
            margin-left: 10px;
        }}

        .no-data {{
            text-align: center;
            color: #666;
            padding: 50px 20px;
            grid-column: span 4;
        }}
        .loading {{
            text-align: center;
            color: #888;
            padding: 50px 20px;
            grid-column: span 4;
        }}

        /* Final result */
        .final-result {{
            background: rgba(107, 203, 119, 0.2);
            border: 2px solid #6bcb77;
            border-radius: 8px;
            padding: 15px;
            margin: 20px;
        }}
        .final-result h3 {{
            color: #6bcb77;
            margin-bottom: 10px;
        }}
        .final-result pre {{
            background: #1a1a2e;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.8rem;
            max-height: 400px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Agent Context Visualization</h1>
        <div class="header-controls" id="savedRunControls" style="display: none;">
            <select class="run-selector" id="runSelector" onchange="loadRun(this.value)">
                <option value="">Select a run...</option>
            </select>
        </div>
    </div>

    <div class="live-controls" id="liveControls" style="display: none;">
        <input type="text" class="query-input" id="queryInput"
               placeholder="Enter your natural language query..."
               onkeypress="if(event.key==='Enter') executeQuery()">
        <select class="selector" id="approachSelector">
            <option value="agentic_grep">Grep Retrieval</option>
            <option value="agentic_semantic">Semantic Retrieval</option>
        </select>
        <select class="selector" id="datasetSizeSelector">
            <option value="small">Small Datasets</option>
            <option value="large">Large Datasets</option>
        </select>
        <button class="execute-btn" id="executeBtn" onclick="executeQuery()">Execute</button>
        <button class="stop-btn" id="stopBtn" onclick="stopExecution()" style="display: none;">Stop</button>
        <button class="clear-btn" onclick="clearTimeline()">Clear</button>
    </div>

    <div class="status-bar" id="statusBar" style="display: none;">
        <span class="status-text" id="statusText">Ready</span>
        <span id="elapsedTime"></span>
    </div>

    <div class="query-info" id="queryInfo" style="display: none;">
        <span class="query-text" id="queryText"></span>
        <span class="info-badge" id="approachBadge"></span>
        <span class="info-badge dataset" id="datasetBadge"></span>
    </div>

    <div class="column-headers">
        <div class="column-header">
            <h2>RetrievalAgent</h2>
            <span class="status-indicator" id="status-retrieval"></span>
        </div>
        <div class="column-header">
            <h2>SPARQLGenerationAgent</h2>
            <span class="status-indicator" id="status-sparql_generation"></span>
        </div>
        <div class="column-header">
            <h2>MultiStepAgent</h2>
            <span class="status-indicator" id="status-multi_step"></span>
        </div>
        <div class="column-header">
            <h2>MappingOptimizerAgent</h2>
            <span class="status-indicator" id="status-mapping_optimizer"></span>
        </div>
    </div>

    <div class="timeline-wrapper" id="timelineWrapper">
        <div id="timelineContent">
            <div class="timeline-row">
                <div class="no-data">Enter a query to start</div>
            </div>
        </div>
    </div>

    <div id="finalResult" class="final-result" style="display: none;">
        <h3>Final Result</h3>
        <pre id="finalResultContent"></pre>
    </div>

    <script>
        const SHOW_LIVE_CONTROLS = {show_live_controls};
        const EXPERIMENT_NAME = '{experiment_name}';
        const AGENTS = ['retrieval', 'sparql_generation', 'multi_step', 'mapping_optimizer'];
        const AGENT_INDEX = {{ 'retrieval': 0, 'sparql_generation': 1, 'multi_step': 2, 'mapping_optimizer': 3 }};

        let currentData = null;
        let eventSource = null;
        let isRunning = false;
        let startTime = null;
        let timerInterval = null;
        let currentActiveAgent = null;

        // Initialize based on mode
        document.addEventListener('DOMContentLoaded', () => {{
            if (SHOW_LIVE_CONTROLS) {{
                document.getElementById('liveControls').style.display = 'flex';
                document.getElementById('statusBar').style.display = 'flex';
                document.getElementById('savedRunControls').style.display = 'none';
            }} else {{
                document.getElementById('liveControls').style.display = 'none';
                document.getElementById('statusBar').style.display = 'none';
                document.getElementById('savedRunControls').style.display = 'flex';
                loadRuns();
            }}
        }});

        function truncate(text, maxLen) {{
            if (!text) return '';
            if (text.length <= maxLen) return text;
            return text.substring(0, maxLen) + '...';
        }}

        function formatTime(isoString) {{
            if (!isoString) return '';
            const date = new Date(isoString);
            return date.toLocaleTimeString();
        }}

        function getMessageTypeClass(messageType) {{
            if (!messageType) return '';
            const type = messageType.toLowerCase().replace('message', '');
            if (type.includes('system')) return 'message-system';
            if (type.includes('human')) return 'message-human';
            if (type.includes('ai')) return 'message-ai';
            if (type.includes('tool')) return 'message-tool';
            return '';
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        function clearTimeline() {{
            document.getElementById('timelineContent').innerHTML = `
                <div class="timeline-row">
                    <div class="no-data">Enter a query to start</div>
                </div>
            `;
            for (const agent of AGENTS) {{
                document.getElementById('status-' + agent).classList.remove('active');
            }}
            document.getElementById('queryInfo').style.display = 'none';
            document.getElementById('finalResult').style.display = 'none';
            currentActiveAgent = null;
            updateStatus('Ready', '');
        }}

        function updateStatus(text, className) {{
            const statusEl = document.getElementById('statusText');
            statusEl.textContent = text;
            statusEl.className = 'status-text ' + (className || '');
        }}

        function startTimer() {{
            startTime = Date.now();
            timerInterval = setInterval(() => {{
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const mins = Math.floor(elapsed / 60);
                const secs = elapsed % 60;
                document.getElementById('elapsedTime').textContent =
                    mins > 0 ? `${{mins}}m ${{secs}}s` : `${{secs}}s`;
            }}, 1000);
        }}

        function stopTimer() {{
            if (timerInterval) {{
                clearInterval(timerInterval);
                timerInterval = null;
            }}
        }}

        function createMessageElement(event) {{
            const typeClass = getMessageTypeClass(event.message_type);
            const eventEl = document.createElement('div');
            eventEl.className = 'message-event ' + typeClass;

            const typeLabel = event.message_type || 'Unknown';
            const content = event.content || '';

            let html = `
                <div class="message-header">
                    <span class="message-type">${{typeLabel}}</span>
                    <span class="timestamp">${{formatTime(event.timestamp)}}</span>
                </div>
            `;

            if (content) {{
                html += `<div class="message-content">${{escapeHtml(truncate(content, 1500))}}</div>`;
            }}

            // Show tool calls
            if (event.tool_calls && event.tool_calls.length > 0) {{
                for (const call of event.tool_calls) {{
                    html += `
                        <div class="tool-call">
                            <span class="tool-call-name">${{call.name || 'tool'}}</span>
                            <pre>${{escapeHtml(JSON.stringify(call.args || {{}}, null, 2).substring(0, 300))}}</pre>
                        </div>
                    `;
                }}
            }}

            // Show tool name for tool results
            if (event.tool_name && event.message_type && event.message_type.toLowerCase().includes('tool')) {{
                eventEl.innerHTML = `
                    <div class="message-header">
                        <span class="message-type">TOOL: ${{event.tool_name}}</span>
                        <span class="timestamp">${{formatTime(event.timestamp)}}</span>
                    </div>
                    <div class="message-content">${{escapeHtml(truncate(content, 1500))}}</div>
                `;
            }} else {{
                eventEl.innerHTML = html;
            }}

            return eventEl;
        }}

        function addEventRow(agentKey, event) {{
            const content = document.getElementById('timelineContent');

            // Remove "no data" placeholder if present
            const noData = content.querySelector('.no-data');
            if (noData) {{
                noData.parentElement.remove();
            }}

            // Create a new row with 4 cells
            const row = document.createElement('div');
            row.className = 'timeline-row';

            for (let i = 0; i < 4; i++) {{
                const cell = document.createElement('div');
                cell.className = 'timeline-cell';

                if (AGENTS[i] === agentKey) {{
                    // This is the active agent's cell - add the message
                    const msgEl = createMessageElement(event);
                    cell.appendChild(msgEl);
                }} else {{
                    // Empty cell
                    cell.classList.add('empty');
                }}

                row.appendChild(cell);
            }}

            content.appendChild(row);

            // Auto-scroll to bottom
            const wrapper = document.getElementById('timelineWrapper');
            wrapper.scrollTop = wrapper.scrollHeight;
        }}

        function addDelegationRow(fromAgent, toAgent, reason) {{
            const content = document.getElementById('timelineContent');

            // Remove "no data" placeholder if present
            const noData = content.querySelector('.no-data');
            if (noData) {{
                noData.parentElement.remove();
            }}

            const row = document.createElement('div');
            row.className = 'delegation-row';

            const fromIndex = AGENT_INDEX[fromAgent];
            const toIndex = AGENT_INDEX[toAgent];

            for (let i = 0; i < 4; i++) {{
                const cell = document.createElement('div');
                cell.className = 'delegation-cell';

                if (i === fromIndex) {{
                    cell.innerHTML = `<span class="delegation-from">→ DELEGATED</span>`;
                }} else if (i === toIndex) {{
                    cell.innerHTML = `<span class="delegation-to">← RECEIVED</span>`;
                    if (reason) {{
                        cell.innerHTML += `<span class="delegation-reason">${{escapeHtml(reason)}}</span>`;
                    }}
                }}

                row.appendChild(cell);
            }}

            content.appendChild(row);

            // Update active status
            setAgentActive(fromAgent, false);
            setAgentActive(toAgent, true);
            currentActiveAgent = toAgent;

            // Auto-scroll
            const wrapper = document.getElementById('timelineWrapper');
            wrapper.scrollTop = wrapper.scrollHeight;
        }}

        function setAgentActive(agentKey, active) {{
            const indicator = document.getElementById('status-' + agentKey);
            if (indicator) {{
                if (active) {{
                    indicator.classList.add('active');
                }} else {{
                    indicator.classList.remove('active');
                }}
            }}
        }}

        function showFinalResult(result) {{
            const container = document.getElementById('finalResult');
            const content = document.getElementById('finalResultContent');

            let resultText = '';
            if (result.final_query) {{
                resultText += 'SPARQL Query:\\n' + result.final_query + '\\n\\n';
            }}
            if (result.final_results) {{
                if (result.final_results.success) {{
                    const bindings = result.final_results.results?.bindings || [];
                    resultText += `Results: ${{bindings.length}} rows\\n`;
                    resultText += JSON.stringify(bindings.slice(0, 5), null, 2);
                    if (bindings.length > 5) {{
                        resultText += '\\n... and ' + (bindings.length - 5) + ' more rows';
                    }}
                }} else {{
                    resultText += 'Error: ' + (result.final_results.error || 'Unknown error');
                }}
            }}
            if (result.is_unanswerable) {{
                resultText = 'Query Unanswerable\\n\\nReason: ' + (result.unanswerable_reason || 'Unknown');
            }}

            content.textContent = resultText || 'No result';
            container.style.display = 'block';
        }}

        async function executeQuery() {{
            const queryInput = document.getElementById('queryInput');
            const approachSelector = document.getElementById('approachSelector');
            const datasetSizeSelector = document.getElementById('datasetSizeSelector');
            const executeBtn = document.getElementById('executeBtn');
            const stopBtn = document.getElementById('stopBtn');

            const query = queryInput.value.trim();
            if (!query) {{
                alert('Please enter a query');
                return;
            }}

            // Clear previous results
            clearTimeline();

            // Show query info
            document.getElementById('queryText').textContent = '"' + query + '"';
            document.getElementById('approachBadge').textContent = approachSelector.value;
            document.getElementById('datasetBadge').textContent = datasetSizeSelector.value + ' datasets';
            document.getElementById('queryInfo').style.display = 'block';

            // Update UI state
            isRunning = true;
            executeBtn.style.display = 'none';
            stopBtn.style.display = 'inline-block';
            queryInput.disabled = true;
            approachSelector.disabled = true;
            datasetSizeSelector.disabled = true;
            updateStatus('Connecting...', 'running');
            startTimer();

            // Start SSE connection
            const url = `/api/execute?query=${{encodeURIComponent(query)}}&approach=${{approachSelector.value}}&dataset_size=${{datasetSizeSelector.value}}`;
            eventSource = new EventSource(url);

            eventSource.onopen = () => {{
                updateStatus('Running...', 'running');
            }};

            eventSource.onmessage = (e) => {{
                try {{
                    const data = JSON.parse(e.data);
                    handleEvent(data);
                }} catch (err) {{
                    console.error('Failed to parse event:', e.data, err);
                }}
            }};

            eventSource.onerror = (e) => {{
                console.error('SSE error:', e);
                stopExecution();
                updateStatus('Connection error', 'error');
            }};
        }}

        function handleEvent(event) {{
            console.log('Event:', event.type, event);

            switch (event.type) {{
                case 'agent_message':
                    addEventRow(event.agent, {{
                        event_type: 'message',
                        message_type: event.message_type,
                        content: event.content,
                        tool_calls: event.tool_calls,
                        tool_name: event.tool_name,
                        timestamp: event.timestamp
                    }});
                    break;

                case 'agent_start':
                    setAgentActive(event.agent, true);
                    currentActiveAgent = event.agent;
                    updateStatus('Running: ' + event.agent, 'running');
                    break;

                case 'agent_end':
                    setAgentActive(event.agent, false);
                    break;

                case 'delegation':
                    addDelegationRow(event.from_agent, event.to_agent, event.reason);
                    break;

                case 'phase_start':
                    updateStatus('Phase: ' + event.phase, 'running');
                    break;

                case 'complete':
                    stopExecution();
                    showFinalResult(event.result);
                    updateStatus('Complete', '');
                    break;

                case 'error':
                    stopExecution();
                    updateStatus('Error: ' + (event.message || 'Unknown'), 'error');
                    break;

                case 'stopped':
                    // Don't call stopExecution again - it triggers this event
                    if (eventSource) {{
                        eventSource.close();
                        eventSource = null;
                    }}
                    isRunning = false;
                    stopTimer();
                    document.getElementById('executeBtn').style.display = 'inline-block';
                    document.getElementById('stopBtn').style.display = 'none';
                    document.getElementById('queryInput').disabled = false;
                    document.getElementById('approachSelector').disabled = false;
                    document.getElementById('datasetSizeSelector').disabled = false;
                    for (const agent of AGENTS) {{
                        setAgentActive(agent, false);
                    }}
                    currentActiveAgent = null;
                    updateStatus('Stopped by user', '');
                    break;
            }}
        }}

        async function stopExecution() {{
            // Send stop signal to server
            try {{
                await fetch('/api/stop');
                console.log('Stop signal sent to server');
            }} catch (e) {{
                console.error('Failed to send stop signal:', e);
            }}

            if (eventSource) {{
                eventSource.close();
                eventSource = null;
            }}

            isRunning = false;
            stopTimer();

            document.getElementById('executeBtn').style.display = 'inline-block';
            document.getElementById('stopBtn').style.display = 'none';
            document.getElementById('queryInput').disabled = false;
            document.getElementById('approachSelector').disabled = false;
            document.getElementById('datasetSizeSelector').disabled = false;

            // Clear all active indicators
            for (const agent of AGENTS) {{
                setAgentActive(agent, false);
            }}
            currentActiveAgent = null;

            updateStatus('Stopped by user', '');
        }}

        // ========== SAVED RUNS MODE ==========

        async function loadRuns() {{
            try {{
                const response = await fetch('/api/runs');
                const runs = await response.json();

                const selector = document.getElementById('runSelector');
                selector.innerHTML = '<option value="">Select a run...</option>';

                for (const run of runs) {{
                    const option = document.createElement('option');
                    option.value = run.run_id;
                    option.textContent = `${{run.query_id || run.run_id}} - ${{run.approach || 'unknown'}} (${{run.run_id.substring(0, 8)}})`;
                    selector.appendChild(option);
                }}
            }} catch (e) {{
                console.error('Failed to load runs:', e);
            }}
        }}

        async function loadRun(runId) {{
            if (!runId) return;

            const content = document.getElementById('timelineContent');
            content.innerHTML = '<div class="timeline-row"><div class="loading">Loading...</div></div>';

            try {{
                const response = await fetch('/api/context?run_id=' + encodeURIComponent(runId));
                const data = await response.json();

                if (data.error) {{
                    content.innerHTML = `<div class="timeline-row"><div class="no-data">${{data.error}}</div></div>`;
                    return;
                }}

                currentData = data;

                // Update query info
                const queryInfo = document.getElementById('queryInfo');
                const queryText = document.getElementById('queryText');
                const approachBadge = document.getElementById('approachBadge');

                if (data.query_text) {{
                    queryText.textContent = '"' + data.query_text + '"';
                    approachBadge.textContent = data.approach || 'unknown';
                    queryInfo.style.display = 'block';
                }} else {{
                    queryInfo.style.display = 'none';
                }}

                // Clear and rebuild timeline
                content.innerHTML = '';

                // Collect all events with timestamps and sort
                const allEvents = [];

                // Get timelines (handle both formats)
                const timelines = data.timelines || {{
                    retrieval: data.retrieval_timeline || [],
                    sparql_generation: data.sparql_generation_timeline || [],
                    multi_step: data.multi_step_timeline || [],
                    mapping_optimizer: data.mapping_optimizer_timeline || []
                }};

                for (const agent of AGENTS) {{
                    const timeline = timelines[agent] || [];
                    for (const event of timeline) {{
                        if (event.event_type === 'message' || event.message_type) {{
                            allEvents.push({{
                                agent: agent,
                                event: event,
                                timestamp: new Date(event.timestamp).getTime()
                            }});
                        }}
                    }}
                }}

                // Add delegations
                for (const delegation of (data.delegations || [])) {{
                    allEvents.push({{
                        type: 'delegation',
                        from_agent: delegation.from_agent,
                        to_agent: delegation.to_agent,
                        reason: delegation.reason,
                        timestamp: new Date(delegation.timestamp).getTime()
                    }});
                }}

                // Sort by timestamp
                allEvents.sort((a, b) => a.timestamp - b.timestamp);

                // Render in order
                for (const item of allEvents) {{
                    if (item.type === 'delegation') {{
                        addDelegationRow(item.from_agent, item.to_agent, item.reason);
                    }} else {{
                        addEventRow(item.agent, item.event);
                    }}
                }}

                if (allEvents.length === 0) {{
                    content.innerHTML = '<div class="timeline-row"><div class="no-data">No events in this run</div></div>';
                }}

            }} catch (e) {{
                console.error('Failed to load run:', e);
                content.innerHTML = '<div class="timeline-row"><div class="no-data">Error loading data</div></div>';
            }}
        }}
    </script>
</body>
</html>'''


def load_context_traces(experiment_dir: Path) -> list[dict]:
    """Load all context traces from the experiment directory."""
    context_dir = experiment_dir / "context_traces"
    if not context_dir.exists():
        return []

    traces = []
    for trace_file in context_dir.glob("*_context.json"):
        try:
            with open(trace_file, encoding="utf-8") as f:
                trace = json.load(f)
                run_id = trace_file.stem.replace("_context", "")
                trace["run_id"] = run_id

                traces.append({
                    "run_id": run_id,
                    "query_id": trace.get("query_id", run_id),
                    "approach": trace.get("approach", ""),
                    "file": str(trace_file),
                })
        except Exception as e:
            print(f"Warning: Could not load {trace_file}: {e}")

    return sorted(traces, key=lambda x: x.get("run_id", ""), reverse=True)


def load_single_context(experiment_dir: Path, run_id: str) -> dict:
    """Load a single context trace by run_id."""
    context_dir = experiment_dir / "context_traces"
    trace_file = context_dir / f"{run_id}_context.json"

    if not trace_file.exists():
        for f in context_dir.glob(f"*{run_id}*_context.json"):
            trace_file = f
            break

    if not trace_file.exists():
        return {"error": f"Context trace not found for run_id: {run_id}"}

    try:
        with open(trace_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to load context: {e}"}


class LiveExecutionHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for live query execution with SSE."""

    mode: str = "live"
    experiment_dir: Path | None = None
    experiment_name: str = ""

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_html()
        elif parsed.path == "/api/runs":
            self._serve_runs()
        elif parsed.path == "/api/context":
            params = parse_qs(parsed.query)
            run_id = params.get("run_id", [None])[0]
            self._serve_context(run_id)
        elif parsed.path == "/api/execute":
            params = parse_qs(parsed.query)
            query = params.get("query", [None])[0]
            approach = params.get("approach", ["agentic_grep"])[0]
            dataset_size = params.get("dataset_size", ["small"])[0]
            self._serve_execute_sse(query, approach, dataset_size)
        elif parsed.path == "/api/stop":
            self._serve_stop()
        else:
            self.send_error(404, "Not Found")

    def _serve_stop(self):
        """Signal the current run to stop."""
        global stop_requested, current_run_id
        stop_requested.set()
        print(f"[Dashboard] Stop requested for run: {current_run_id}")
        response = json.dumps({"success": True, "message": "Stop signal sent"})
        self.send_response(200)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(response.encode("utf-8")))
        self.end_headers()
        self.wfile.write(response.encode("utf-8"))

    def _serve_html(self):
        html = get_dashboard_html(self.mode, self.experiment_name)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html.encode("utf-8")))
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_runs(self):
        if self.experiment_dir:
            runs = load_context_traces(self.experiment_dir)
        else:
            runs = []
        data = json.dumps(runs)
        self.send_response(200)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(data.encode("utf-8")))
        self.end_headers()
        self.wfile.write(data.encode("utf-8"))

    def _serve_context(self, run_id: str):
        if not run_id:
            data = json.dumps({"error": "run_id is required"})
        elif self.experiment_dir:
            context = load_single_context(self.experiment_dir, run_id)
            data = json.dumps(context)
        else:
            data = json.dumps({"error": "No experiment directory configured"})

        self.send_response(200)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(data.encode("utf-8")))
        self.end_headers()
        self.wfile.write(data.encode("utf-8"))

    def _serve_execute_sse(self, query: str, approach: str, dataset_size: str):
        """Execute query and stream results via Server-Sent Events."""
        global current_run_id, stop_requested

        if not query:
            self.send_error(400, "query parameter is required")
            return

        self.send_response(200)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        run_id = str(uuid.uuid4())
        current_run_id = run_id
        stop_requested.clear()  # Reset stop flag for new run

        def send_event(data: dict):
            """Send an SSE event."""
            try:
                event_data = json.dumps(data, ensure_ascii=False)
                self.wfile.write(f"data: {event_data}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception as e:
                print(f"Error sending SSE event: {e}")

        result_container = {"result": None, "error": None}

        def run_async():
            """Run the async orchestrator in a new event loop."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Set dataset size using the proper config function
                from src.config import set_dataset_size, get_dataset_size, get_ontop_endpoint
                set_dataset_size(dataset_size)
                print(f"[Dashboard] Dataset size set to: {dataset_size}")
                print(f"[Dashboard] Verified get_dataset_size(): {get_dataset_size()}")
                # Test endpoint resolution
                print(f"[Dashboard] edu endpoint: {get_ontop_endpoint('edu')}")
                print(f"[Dashboard] trn endpoint: {get_ontop_endpoint('trn')}")

                from src.agents.orchestrator import run_agentic_pipeline
                from src.tracing import (
                    AgentContextCollector,
                    get_tracer,
                    trace_run_context,
                )

                collector = AgentContextCollector(
                    run_id=run_id,
                    query_text=query,
                    approach=approach,
                )

                tracer = get_tracer()
                tracer.add_listener(collector)

                class SSEListener:
                    def __init__(self, send_fn, agent_name_map):
                        self.send = send_fn
                        self.agent_name_map = agent_name_map

                    def _get_agent_key(self, agent_name):
                        if agent_name is None:
                            return None
                        return self.agent_name_map.get(agent_name, agent_name)

                    def on_event(self, event):
                        from src.tracing.models import TraceEventType

                        if event.run_id is not None and event.run_id != run_id:
                            return

                        agent_key = self._get_agent_key(event.agent)

                        if event.event_type == TraceEventType.AGENT_MESSAGE:
                            self.send({
                                "type": "agent_message",
                                "agent": agent_key,
                                "message_type": event.data.get("message_type"),
                                "content": event.data.get("content", "")[:2000],
                                "tool_calls": event.data.get("tool_calls"),
                                "tool_name": event.data.get("tool_name"),
                                "timestamp": event.timestamp.isoformat(),
                            })

                        elif event.event_type == TraceEventType.AGENT_START:
                            if agent_key:
                                self.send({
                                    "type": "agent_start",
                                    "agent": agent_key,
                                    "timestamp": event.timestamp.isoformat(),
                                })

                        elif event.event_type == TraceEventType.AGENT_END:
                            if agent_key:
                                self.send({
                                    "type": "agent_end",
                                    "agent": agent_key,
                                    "timestamp": event.timestamp.isoformat(),
                                })

                        elif event.event_type == TraceEventType.DELEGATION_START:
                            from_agent = self._get_agent_key(event.data.get("from_agent"))
                            to_agent = self._get_agent_key(event.data.get("to_agent"))
                            if from_agent and to_agent:
                                self.send({
                                    "type": "delegation",
                                    "from_agent": from_agent,
                                    "to_agent": to_agent,
                                    "reason": event.data.get("reason"),
                                    "timestamp": event.timestamp.isoformat(),
                                })

                        elif event.event_type == TraceEventType.PHASE_START:
                            self.send({
                                "type": "phase_start",
                                "phase": event.phase,
                                "timestamp": event.timestamp.isoformat(),
                            })

                from src.tracing.context_collector import AGENT_NAME_MAP
                sse_listener = SSEListener(send_event, AGENT_NAME_MAP)
                tracer.add_listener(sse_listener)

                try:
                    with trace_run_context(run_id):
                        result = loop.run_until_complete(
                            run_agentic_pipeline(
                                user_query=query,
                                approach=approach,
                            )
                        )
                        result_container["result"] = result
                finally:
                    tracer.remove_listener(collector)
                    tracer.remove_listener(sse_listener)
                    loop.close()

            except Exception as e:
                import traceback
                traceback.print_exc()
                result_container["error"] = str(e)

        exec_thread = threading.Thread(target=run_async, daemon=True)
        exec_thread.start()

        was_stopped = False
        while exec_thread.is_alive():
            exec_thread.join(timeout=0.5)
            # Check if stop was requested
            if stop_requested.is_set():
                print(f"[Dashboard] Stop detected, terminating run {run_id}")
                was_stopped = True
                send_event({
                    "type": "stopped",
                    "message": "Run stopped by user",
                })
                break

        if was_stopped:
            # Don't send complete/error - already sent stopped event
            current_run_id = None
        elif result_container["error"]:
            send_event({
                "type": "error",
                "message": result_container["error"],
            })
            current_run_id = None
        else:
            result = result_container["result"]
            send_event({
                "type": "complete",
                "result": {
                    "final_query": result.get("final_query"),
                    "final_results": result.get("final_results"),
                    "is_unanswerable": result.get("is_unanswerable", False),
                    "unanswerable_reason": result.get("unanswerable_reason", ""),
                },
            })
            current_run_id = None

    def log_message(self, format, *args):
        pass


def run_dashboard(mode: str = "live", experiment_name: str = "", port: int = DEFAULT_PORT):
    """Run the dashboard server."""
    experiment_dir = None

    if mode == "saved" and experiment_name:
        experiment_dir = RESULTS_DIR / "experiments" / experiment_name
        if not experiment_dir.exists():
            print(f"Error: Experiment directory not found: {experiment_dir}")
            sys.exit(1)

        context_dir = experiment_dir / "context_traces"
        if not context_dir.exists():
            print(f"Warning: No context_traces directory found at {context_dir}")

    LiveExecutionHandler.mode = mode
    LiveExecutionHandler.experiment_dir = experiment_dir
    LiveExecutionHandler.experiment_name = experiment_name

    # Use ThreadingTCPServer to handle concurrent requests (needed for /api/stop while SSE is running)
    with socketserver.ThreadingTCPServer(("", port), LiveExecutionHandler) as httpd:
        url = f"http://localhost:{port}"
        print(f"Agent Context Dashboard: {url}")
        if mode == "live":
            print("Mode: LIVE - Enter queries to execute in real-time")
        else:
            print(f"Mode: SAVED - Loading traces from {experiment_name}")
        print("Press Ctrl+C to stop.")

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping dashboard...")


def main():
    parser = argparse.ArgumentParser(description="Agent Context Visualization Dashboard")
    parser.add_argument("experiment_name", nargs="?", help="Name of the experiment to visualize (for saved mode)")
    parser.add_argument("--live", action="store_true", help="Run in live mode for executing new queries")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to run on (default: {DEFAULT_PORT})")

    args = parser.parse_args()

    if args.live:
        run_dashboard(mode="live", port=args.port)
    elif args.experiment_name:
        run_dashboard(mode="saved", experiment_name=args.experiment_name, port=args.port)
    else:
        run_dashboard(mode="live", port=args.port)


if __name__ == "__main__":
    main()
