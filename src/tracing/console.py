"""Rich-based console rendering for trace events."""

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from src.tracing.models import TraceEvent, TraceEventType


class RichTraceConsole:
    """Rich-based trace visualization."""

    # Phase colors
    PHASE_COLORS = {
        "retrieval": "cyan",
        "generation": "yellow",
        "complete": "green",
    }

    # Event icons
    EVENT_ICONS = {
        TraceEventType.PHASE_START: "▶",
        TraceEventType.PHASE_END: "■",
        TraceEventType.TOOL_CALL: "🔧",
        TraceEventType.TOOL_RESULT: "📋",
        TraceEventType.ORCHESTRATOR_DECISION: "🧠",
        TraceEventType.SPARQL_AGENT_DECISION: "🎯",
        TraceEventType.SPARQL_EXECUTION: "⚡",
        TraceEventType.SPARQL_RESULT: "📊",
        TraceEventType.ERROR: "❌",
        TraceEventType.INFO: "ℹ️",
        TraceEventType.QUERY_START: "🔍",
        TraceEventType.QUERY_COMPLETE: "✅",
        TraceEventType.SCHEMA_CONTEXT: "📑",
        TraceEventType.SPARQL_QUERY_TEST: "🧪",
        TraceEventType.MULTI_STEP_INPUT: "📥",
        TraceEventType.MULTI_STEP_STEP: "📌",
        TraceEventType.MULTI_STEP_TRANSFORM: "⚙️",
    }

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._tool_call_count = 0

    def on_event(self, event: TraceEvent) -> None:
        """Handle a trace event."""
        handler_name = f"_render_{event.event_type.value}"
        handler = getattr(self, handler_name, None)
        if handler:
            handler(event)
        else:
            self._render_generic(event)

    def _render_query_start(self, event: TraceEvent) -> None:
        """Render query start."""
        query = event.data.get("query", "")
        self.console.print()
        self.console.rule("[bold blue]New Query[/bold blue]", style="blue")
        self.console.print(
            Panel(
                query,
                title="[bold]User Query[/bold]",
                border_style="blue",
                padding=(0, 2),
            )
        )
        self._tool_call_count = 0

    def _render_phase_start(self, event: TraceEvent) -> None:
        """Render phase start."""
        phase = event.phase or "unknown"
        color = self.PHASE_COLORS.get(phase, "white")
        icon = self.EVENT_ICONS[TraceEventType.PHASE_START]

        self.console.print()
        self.console.print(f"[bold {color}]{icon} Phase: {phase.upper()}[/bold {color}]")
        self.console.print("─" * 50, style=f"dim {color}")

    def _render_phase_end(self, event: TraceEvent) -> None:
        """Render phase end with duration."""
        phase = event.phase or "unknown"
        color = self.PHASE_COLORS.get(phase, "white")
        duration = event.duration_ms or 0
        icon = self.EVENT_ICONS[TraceEventType.PHASE_END]

        self.console.print(f"[{color}]{icon} {phase} completed in {duration:.0f}ms[/{color}]")

    def _render_tool_call(self, event: TraceEvent) -> None:
        """Render a tool call."""
        self._tool_call_count += 1
        tool_name = event.data.get("tool_name", "unknown")
        tool_args = event.data.get("tool_args", {})

        # Format arguments
        args_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        args_table.add_column("Param", style="dim")
        args_table.add_column("Value")

        for key, value in tool_args.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 60:
                str_value = str_value[:57] + "..."
            args_table.add_row(key, str_value)

        self.console.print(
            Panel(
                args_table,
                title=f"[bold]🔧 Tool #{self._tool_call_count}: {tool_name}[/bold]",
                border_style="magenta",
                padding=(0, 1),
            )
        )

    def _render_tool_result(self, event: TraceEvent) -> None:
        """Render a tool result with detailed data."""
        success = event.data.get("success", True)
        tool_name = event.data.get("tool_name", "")

        if not success:
            error = event.data.get("error_message", "Unknown error")
            self.console.print(f"   [red]✗ {tool_name} Error: {error}[/red]")
            return

        result_summary = event.data.get("result_summary", event.data.get("result", {}))
        result_data = event.data.get("result_data")

        # Show summary line
        if isinstance(result_summary, dict):
            result_type = result_summary.get("type", "unknown")
            if result_type == "list":
                count = result_summary.get("count", 0)
                self.console.print(f"   [green]✓ {count} results[/green]")
            elif result_type == "string":
                self.console.print(f"   [green]✓ {result_summary.get('length', 0)} chars[/green]")
            else:
                self.console.print(f"   [green]✓ Done[/green]")
        else:
            self.console.print(f"   [green]✓ Done[/green]")

        # Show detailed results if available
        if result_data:
            self._render_result_details(tool_name, result_data)

    def _render_result_details(self, tool_name: str, result_data: Any) -> None:
        """Render detailed tool result data."""
        if isinstance(result_data, list) and result_data:
            # Create a table for list results
            table = Table(
                box=box.SIMPLE,
                show_header=True,
                padding=(0, 1),
                expand=False,
            )

            # Determine columns from first item
            first_item = result_data[0]
            if isinstance(first_item, dict):
                # Prefix declarations - show prominently
                if first_item.get("type") == "prefixes":
                    prefixes_content = first_item.get("content", "")
                    self.console.print(
                        Panel(
                            Syntax(prefixes_content, "sparql", theme="monokai"),
                            title="[bold yellow]SPARQL Prefixes (use these!)[/bold yellow]",
                            border_style="yellow",
                            padding=(0, 1),
                        )
                    )
                    # Continue with remaining results
                    result_data = result_data[1:]
                    if not result_data:
                        return
                    first_item = result_data[0]

                # Schema search results - show key fields
                if "content" in first_item:
                    table.add_column("Content", style="cyan", max_width=80)
                    table.add_column("File", style="dim", max_width=30)
                    table.add_column("Endpoint", style="yellow", max_width=40)

                    for item in result_data[:10]:
                        content = str(item.get("content", ""))[:80]
                        file = str(item.get("file", ""))[-30:]
                        endpoint = str(item.get("sparql_endpoint", ""))[:40]
                        table.add_row(content, file, endpoint)

                elif "matches" in first_item:
                    # Grep results
                    table.add_column("File", style="dim", max_width=30)
                    table.add_column("Matches", style="cyan")
                    table.add_column("Endpoint", style="yellow", max_width=40)

                    for item in result_data[:10]:
                        file = str(item.get("file", ""))[-30:]
                        matches = len(item.get("matches", []))
                        endpoint = str(item.get("sparql_endpoint", ""))[:40]
                        table.add_row(file, str(matches), endpoint)

                elif "class_name" in first_item or "name" in first_item:
                    # Class list results
                    table.add_column("Class", style="cyan", max_width=40)
                    table.add_column("Dataset", style="yellow", max_width=20)
                    table.add_column("Endpoint", style="dim", max_width=40)

                    for item in result_data[:10]:
                        name = str(item.get("class_name", item.get("name", "")))
                        dataset = str(item.get("dataset", ""))
                        endpoint = str(item.get("sparql_endpoint", ""))[:40]
                        table.add_row(name, dataset, endpoint)

                else:
                    # Generic dict results
                    keys = list(first_item.keys())[:4]
                    for key in keys:
                        table.add_column(key, max_width=40)

                    for item in result_data[:10]:
                        row = [str(item.get(k, ""))[:40] for k in keys]
                        table.add_row(*row)

                if len(result_data) > 10:
                    self.console.print(f"   [dim]... and {len(result_data) - 10} more[/dim]")

                self.console.print(table)

            elif isinstance(first_item, str):
                # List of strings
                for item in result_data[:5]:
                    truncated = item[:100] + "..." if len(item) > 100 else item
                    self.console.print(f"   [dim]• {truncated}[/dim]")
                if len(result_data) > 5:
                    self.console.print(f"   [dim]... and {len(result_data) - 5} more[/dim]")

        elif isinstance(result_data, dict):
            # Single dict result
            if "schema" in result_data:
                schema = result_data["schema"]
                if len(schema) > 500:
                    schema = schema[:500] + "\n..."
                self.console.print(
                    Panel(
                        Syntax(schema, "turtle", theme="monokai"),
                        title="[dim]Schema[/dim]",
                        border_style="dim",
                        padding=(0, 1),
                    )
                )
            else:
                for key, value in list(result_data.items())[:5]:
                    val_str = str(value)[:60]
                    self.console.print(f"   [dim]{key}: {val_str}[/dim]")

        elif isinstance(result_data, str) and result_data:
            # String result (like full schema)
            if len(result_data) > 300:
                result_data = result_data[:300] + "\n..."
            self.console.print(
                Panel(
                    result_data,
                    border_style="dim",
                    padding=(0, 1),
                )
            )

    def _render_orchestrator_decision(self, event: TraceEvent) -> None:
        """Render orchestrator decision."""
        next_phase = event.data.get("next_phase", "")
        reasoning = event.data.get("reasoning", "")
        feedback = event.data.get("feedback", "")

        # Decision panel
        content = Text()
        content.append("Next Phase: ", style="bold")
        content.append(f"{next_phase}\n", style="yellow")

        if reasoning:
            content.append("\nReasoning: ", style="bold")
            reasoning_text = reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
            content.append(reasoning_text)

        if feedback:
            content.append("\n\nFeedback: ", style="bold italic")
            content.append(feedback, style="italic")

        self.console.print(
            Panel(
                content,
                title="[bold]🧠 Orchestrator Decision[/bold]",
                border_style="green",
                padding=(0, 2),
            )
        )

    def _render_sparql_agent_decision(self, event: TraceEvent) -> None:
        """Render SPARQL agent decision with detailed reasoning."""
        decision = event.data.get("sparql_agent_decision", "unknown")
        decision_reasoning = event.data.get("decision_reasoning", "")
        reasoning = event.data.get("reasoning", "")
        next_phase = event.data.get("next_phase", "")
        needed_triple = event.data.get("needed_triple", "")
        query_success = event.data.get("query_success")
        multi_step_success = event.data.get("multi_step_success")
        steps_count = event.data.get("steps_count")

        # Color based on decision
        decision_colors = {
            "DONE": "green",
            "NEED_TRIPLE": "yellow",
            "MULTI_STEP_RESULT": "cyan",
        }
        color = decision_colors.get(decision, "white")

        content = Text()
        content.append("Decision: ", style="bold")
        content.append(f"{decision}\n", style=f"bold {color}")

        if next_phase:
            content.append("Next Phase: ", style="bold")
            content.append(f"{next_phase}\n", style="dim")

        # Show query success/failure for DONE
        if decision == "DONE" and query_success is not None:
            status_icon = "✓" if query_success else "✗"
            status_color = "green" if query_success else "red"
            content.append("Query Execution: ", style="bold")
            content.append(f"{status_icon} {'Success' if query_success else 'Failed'}\n", style=status_color)

        # Show multi-step info
        if decision == "MULTI_STEP_RESULT":
            if multi_step_success is not None:
                status_icon = "✓" if multi_step_success else "✗"
                status_color = "green" if multi_step_success else "red"
                content.append("Multi-Step Result: ", style="bold")
                content.append(f"{status_icon} {'Success' if multi_step_success else 'Failed'}", style=status_color)
                if steps_count:
                    content.append(f" ({steps_count} steps)\n", style="dim")
                else:
                    content.append("\n")

        # Show needed triple for NEED_TRIPLE
        if needed_triple:
            content.append("\nNeeded Schema: ", style="bold yellow")
            content.append(f"{needed_triple}\n", style="yellow")

        # Show reasoning
        if reasoning:
            content.append("\nReasoning: ", style="bold")
            reasoning_text = reasoning[:400] + "..." if len(reasoning) > 400 else reasoning
            content.append(f"{reasoning_text}\n", style="dim")

        # Show decision reasoning (most important!)
        if decision_reasoning:
            content.append("\n")
            decision_text = decision_reasoning[:600] + "..." if len(decision_reasoning) > 600 else decision_reasoning
            self.console.print(
                Panel(
                    content,
                    title=f"[bold]🎯 SPARQL Agent Decision: {decision}[/bold]",
                    border_style=color,
                    padding=(0, 2),
                )
            )
            self.console.print(
                Panel(
                    decision_text,
                    title="[bold italic]Decision Reasoning[/bold italic]",
                    border_style="blue",
                    padding=(0, 2),
                )
            )
        else:
            content.append("\n[dim italic]No decision_reasoning provided[/dim italic]")
            self.console.print(
                Panel(
                    content,
                    title=f"[bold]🎯 SPARQL Agent Decision: {decision}[/bold]",
                    border_style=color,
                    padding=(0, 2),
                )
            )

    def _render_sparql_result(self, event: TraceEvent) -> None:
        """Render SPARQL query results as a table."""
        query = event.data.get("query", "")
        results = event.data.get("results", {})
        bindings = results.get("bindings", [])
        success = event.data.get("success", False)
        index = event.data.get("index", 0)

        # Show query with syntax highlighting
        self.console.print(
            Panel(
                Syntax(query, "sparql", theme="monokai", line_numbers=True),
                title=f"[bold]SPARQL Query #{index + 1}[/bold]",
                border_style="cyan",
            )
        )

        if not success:
            error = event.data.get("error_message", "Unknown error")
            self.console.print(f"  [red]✗ Error: {error}[/red]")
            return

        if not bindings:
            self.console.print("  [yellow]⚠ No results[/yellow]")
            return

        # Create result table
        table = Table(
            title=f"Results ({len(bindings)} rows)",
            box=box.ROUNDED,
            show_lines=True,
        )

        # Extract columns from first binding (without internal fields)
        columns = [k for k in bindings[0].keys() if not k.startswith("_")]
        for col in columns:
            table.add_column(col, style="cyan")

        # Add rows (max 10 for clarity)
        for binding in bindings[:10]:
            row = []
            for col in columns:
                val = binding.get(col, {})
                # Extract RDF value
                if isinstance(val, dict):
                    cell_value = val.get("value", str(val))
                else:
                    cell_value = str(val)
                # Truncate long values
                if len(cell_value) > 50:
                    cell_value = cell_value[:47] + "..."
                row.append(cell_value)
            table.add_row(*row)

        if len(bindings) > 10:
            table.add_row(*[f"... ({len(bindings) - 10} more)" if i == 0 else "" for i in range(len(columns))])

        self.console.print(table)

    def _render_sparql_execution(self, event: TraceEvent) -> None:
        """Render SPARQL execution start."""
        query_count = event.data.get("query_count", 0)
        self.console.print(f"\n[cyan]⚡ Executing {query_count} SPARQL queries...[/cyan]")

    def _render_query_complete(self, event: TraceEvent) -> None:
        """Render query completion."""
        final_query = event.data.get("final_query", "")
        result_count = event.data.get("result_count", 0)
        success = event.data.get("success", False)
        final_results = event.data.get("final_results", {})
        is_multi_step = final_results.get("multi_step_execution", False) if isinstance(final_results, dict) else False

        self.console.print()
        self.console.rule(
            f"[bold {'green' if success else 'red'}]Query Complete[/bold {'green' if success else 'red'}]",
            style="green" if success else "red",
        )

        # Handle Multi-Step execution results
        if is_multi_step:
            steps = final_results.get("steps", [])
            transform_script = final_results.get("transform_script", "")
            multi_step_final_results = final_results.get("final_results", [])  # Direct list from transform

            self.console.print(
                Panel(
                    f"[cyan]Multi-Step Execution[/cyan]\n"
                    f"Steps executed: {len(steps)}\n"
                    f"Transform script: {'Yes' if transform_script else 'No'}\n"
                    f"Final results: {len(multi_step_final_results)} rows",
                    title="[bold cyan]Multi-Step Agent Results[/bold cyan]",
                    border_style="cyan",
                )
            )

            # Show step summaries
            if steps:
                step_table = Table(title="Execution Steps", box=box.ROUNDED)
                step_table.add_column("Step", style="cyan")
                step_table.add_column("Dataset", style="yellow")
                step_table.add_column("Results", style="green")
                for step in steps[:10]:
                    step_name = step.get("name", "unknown")
                    step_dataset = step.get("dataset", "")
                    # Use result_count if available, otherwise count results list
                    step_count = step.get("result_count", len(step.get("results", [])))
                    step_table.add_row(step_name, step_dataset, str(step_count))
                self.console.print(step_table)

            # Show transformation script
            if transform_script:
                self.console.print(
                    Panel(
                        Syntax(transform_script, "python", theme="monokai", line_numbers=True),
                        title="[bold magenta]Transformation Script[/bold magenta]",
                        border_style="magenta",
                    )
                )

            # Show final transformed results as table
            if multi_step_final_results and isinstance(multi_step_final_results, list) and len(multi_step_final_results) > 0:
                table = Table(
                    title=f"[bold green]Final Transformed Results ({len(multi_step_final_results)} rows)[/bold green]",
                    box=box.ROUNDED,
                    show_lines=True,
                )

                # Get columns from first result
                first_result = multi_step_final_results[0]
                if isinstance(first_result, dict):
                    columns = list(first_result.keys())[:10]  # Max 10 columns
                    for col in columns:
                        table.add_column(str(col), style="green", max_width=50)

                    for row_data in multi_step_final_results[:20]:  # Show up to 20 rows
                        row = []
                        for col in columns:
                            val = row_data.get(col, "")
                            val_str = str(val)
                            if len(val_str) > 50:
                                val_str = val_str[:47] + "..."
                            row.append(val_str)
                        table.add_row(*row)

                    if len(multi_step_final_results) > 20:
                        table.add_row(*[f"... ({len(multi_step_final_results) - 20} more)" if i == 0 else "" for i in range(len(columns))])

                    self.console.print(table)

            # Also check for SPARQL-format bindings (fallback)
            bindings = final_results.get("results", {}).get("bindings", [])
            if bindings and not multi_step_final_results:
                result_count = len(bindings)
            else:
                bindings = []  # Already showed multi_step_final_results

        elif final_query:
            self.console.print(
                Panel(
                    Syntax(final_query, "sparql", theme="monokai", line_numbers=True),
                    title="[bold]Final SPARQL Query[/bold]",
                    border_style="green" if success else "red",
                )
            )
            # Show results table
            bindings = final_results.get("results", {}).get("bindings", [])

        else:
            self.console.print("[red]No valid query generated[/red]")
            bindings = []

        # Show results table for single-query (SPARQL bindings format)
        if bindings:
            table = Table(
                title=f"Final Results ({len(bindings)} rows)",
                box=box.ROUNDED,
                show_lines=True,
            )

            columns = [k for k in bindings[0].keys() if not k.startswith("_")]
            for col in columns:
                table.add_column(col, style="green", max_width=50)

            for binding in bindings[:20]:  # Show up to 20 rows
                row = []
                for col in columns:
                    val = binding.get(col, {})
                    if isinstance(val, dict):
                        cell_value = val.get("value", str(val))
                    else:
                        cell_value = str(val)
                    if len(cell_value) > 50:
                        cell_value = cell_value[:47] + "..."
                    row.append(cell_value)
                table.add_row(*row)

            if len(bindings) > 20:
                table.add_row(*[f"... ({len(bindings) - 20} more)" if i == 0 else "" for i in range(len(columns))])

            self.console.print(table)
            result_count = len(bindings)

        self.console.print(f"\nTotal results: [bold]{result_count}[/bold] rows")

    def _render_error(self, event: TraceEvent) -> None:
        """Render an error."""
        error = event.data.get("error", "Unknown error")
        phase = event.phase or ""
        self.console.print(
            Panel(
                f"[bold red]{error}[/bold red]",
                title=f"[bold red]❌ Error{' in ' + phase if phase else ''}[/bold red]",
                border_style="red",
            )
        )

    def _render_info(self, event: TraceEvent) -> None:
        """Render informational message (timeouts, warnings, progress)."""
        message = event.data.get("message", "")
        agent = event.agent or event.data.get("agent", "")

        # Check for specific info types
        if "timeout" in message.lower():
            self.console.print(f"   [yellow]⚠ {message}[/yellow]")
        elif "auto-delegation" in message.lower() or "multi-step" in message.lower():
            self.console.print(f"   [cyan]→ {message}[/cyan]")
        else:
            self.console.print(f"   [dim]ℹ️ {message}[/dim]")

    def _render_schema_context(self, event: TraceEvent) -> None:
        """Render schema context (triples passed to SPARQL agent)."""
        triples = event.data.get("triples", [])
        triples_count = event.data.get("triples_count", 0)

        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]{triples_count} schema triples[/bold cyan] passed to SPARQL Agent",
                title="[bold]Schema Context[/bold]",
                border_style="cyan",
                padding=(0, 1),
            )
        )

        # Show triples in a compact format
        if triples:
            triples_text = "\n".join(triples[:30])  # Show first 30 triples
            if len(triples) > 30:
                triples_text += f"\n... and {len(triples) - 30} more triples"

            self.console.print(
                Panel(
                    Syntax(triples_text, "turtle", theme="monokai", word_wrap=True),
                    title="[dim]Triples Sample[/dim]",
                    border_style="dim cyan",
                    padding=(0, 1),
                )
            )

    def _render_sparql_query_test(self, event: TraceEvent) -> None:
        """Render SPARQL query test with full query and data sample."""
        query = event.data.get("query", "")
        dataset = event.data.get("dataset", "")
        success = event.data.get("success", False)
        result_count = event.data.get("result_count", 0)
        variables = event.data.get("variables", [])
        sample_results = event.data.get("sample_results", [])
        execution_time = event.data.get("execution_time_ms", 0)
        is_timeout = event.data.get("is_timeout", False)
        error_message = event.data.get("error_message", "")

        # Color based on success
        border_color = "green" if success else ("yellow" if is_timeout else "red")
        status_icon = "✓" if success else ("⏱" if is_timeout else "✗")

        self.console.print()
        self.console.print(
            Panel(
                Syntax(query, "sparql", theme="monokai", line_numbers=True, word_wrap=True),
                title=f"[bold]SPARQL Query Test [{dataset}][/bold]",
                subtitle=f"[dim]{execution_time:.0f}ms[/dim]",
                border_style=border_color,
            )
        )

        if success:
            self.console.print(f"   [{border_color}]{status_icon} {result_count} results[/{border_color}] | Variables: {', '.join(variables)}")

            # Show data sample table
            if sample_results:
                table = Table(
                    title="Data Sample",
                    box=box.SIMPLE,
                    show_lines=True,
                    expand=False,
                )

                # Add columns based on variables or first result keys
                columns = variables if variables else list(sample_results[0].keys()) if sample_results else []
                for col in columns[:8]:  # Max 8 columns
                    table.add_column(col, style="cyan", max_width=40)

                # Add rows
                for row in sample_results[:10]:
                    row_values = []
                    for col in columns[:8]:
                        val = row.get(col, "")
                        if isinstance(val, dict):
                            val = val.get("value", str(val))
                        val_str = str(val)[:40]
                        row_values.append(val_str)
                    table.add_row(*row_values)

                if len(sample_results) > 10:
                    table.add_row(*[f"... ({len(sample_results) - 10} more)" if i == 0 else "" for i in range(min(8, len(columns)))])

                self.console.print(table)
        else:
            self.console.print(f"   [{border_color}]{status_icon} {'TIMEOUT' if is_timeout else 'FAILED'}[/{border_color}]")
            if error_message:
                error_preview = error_message[:200] + "..." if len(error_message) > 200 else error_message
                self.console.print(f"   [dim]{error_preview}[/dim]")

    def _render_multi_step_input(self, event: TraceEvent) -> None:
        """Render Multi-Step Agent input."""
        failed_query = event.data.get("failed_query", "")
        user_query = event.data.get("user_query", "")
        schema_context = event.data.get("schema_context", "")
        error_message = event.data.get("error_message", "")

        self.console.print()
        self.console.rule("[bold cyan]Multi-Step Agent Started[/bold cyan]", style="cyan")

        # Show user query
        self.console.print(
            Panel(
                user_query,
                title="[bold]User Query[/bold]",
                border_style="blue",
                padding=(0, 1),
            )
        )

        # Show failed query
        self.console.print(
            Panel(
                Syntax(failed_query, "sparql", theme="monokai", line_numbers=True),
                title="[bold yellow]Failed Query (timed out)[/bold yellow]",
                border_style="yellow",
            )
        )

        # Show schema context summary
        schema_lines = schema_context.split("\n") if schema_context else []
        self.console.print(f"   [dim]Schema context: {len(schema_lines)} lines[/dim]")
        self.console.print(f"   [dim]Error: {error_message}[/dim]")

    def _render_multi_step_step(self, event: TraceEvent) -> None:
        """Render individual Multi-Step step execution."""
        step_name = event.data.get("step_name", "")
        query = event.data.get("query", "")
        dataset = event.data.get("dataset", "")
        result_count = event.data.get("result_count", 0)
        variables = event.data.get("variables", [])
        sample_results = event.data.get("sample_results", [])

        self.console.print()
        self.console.print(
            Panel(
                Syntax(query, "sparql", theme="monokai", line_numbers=True, word_wrap=True),
                title=f"[bold cyan]Step: {step_name}[/bold cyan] [{dataset}]",
                subtitle=f"[green]{result_count} results[/green]",
                border_style="cyan",
            )
        )

        self.console.print(f"   Variables: {', '.join(variables)}")

        # Show data sample
        if sample_results:
            table = Table(
                title=f"{step_name} Data Sample",
                box=box.SIMPLE,
                show_lines=True,
                expand=False,
            )

            # Determine columns from first result
            first_result = sample_results[0] if sample_results else {}
            columns = list(first_result.keys())[:6]  # Max 6 columns
            for col in columns:
                table.add_column(col, style="cyan", max_width=35)

            for row in sample_results[:8]:
                row_values = []
                for col in columns:
                    val = row.get(col, {})
                    if isinstance(val, dict):
                        val = val.get("value", str(val))
                    val_str = str(val)[:35]
                    row_values.append(val_str)
                table.add_row(*row_values)

            if len(sample_results) > 8:
                table.add_row(*[f"... ({len(sample_results) - 8} more)" if i == 0 else "" for i in range(len(columns))])

            self.console.print(table)

    def _render_multi_step_transform(self, event: TraceEvent) -> None:
        """Render Multi-Step transformation script and results."""
        transform_script = event.data.get("transform_script", "")
        input_steps = event.data.get("input_steps", [])
        input_counts = event.data.get("input_counts", {})
        output_count = event.data.get("output_count", 0)
        sample_output = event.data.get("sample_output", [])
        error = event.data.get("error")
        success = event.data.get("success", True) if "success" in event.data else error is None

        self.console.print()

        # Show transformation script
        self.console.print(
            Panel(
                Syntax(transform_script, "python", theme="monokai", line_numbers=True),
                title="[bold magenta]Transformation Script[/bold magenta]",
                border_style="magenta",
            )
        )

        # Show input/output summary
        input_summary = ", ".join([f"{k}: {v} rows" for k, v in input_counts.items()])
        self.console.print(f"   [dim]Input:[/dim] {input_summary}")

        if success:
            self.console.print(f"   [green]Output: {output_count} rows[/green]")

            # Show output sample
            if sample_output:
                table = Table(
                    title="Transformation Output Sample",
                    box=box.ROUNDED,
                    show_lines=True,
                    expand=False,
                )

                # Determine columns from first result
                first_result = sample_output[0] if sample_output else {}
                columns = list(first_result.keys())[:8]  # Max 8 columns
                for col in columns:
                    table.add_column(col, style="green", max_width=40)

                for row in sample_output[:15]:
                    row_values = []
                    for col in columns:
                        val = row.get(col, "")
                        val_str = str(val)[:40]
                        row_values.append(val_str)
                    table.add_row(*row_values)

                if len(sample_output) > 15:
                    table.add_row(*[f"... ({len(sample_output) - 15} more)" if i == 0 else "" for i in range(len(columns))])

                self.console.print(table)
        else:
            self.console.print(f"   [red]Transform Error: {error}[/red]")

    def _render_generic(self, event: TraceEvent) -> None:
        """Render generic event."""
        self.console.print(f"[dim]{event.event_type.value}: {event.data}[/dim]")
