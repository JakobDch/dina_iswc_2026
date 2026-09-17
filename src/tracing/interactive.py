"""Interactive debugging mode for the DINA agent."""

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.agents.orchestrator import OrchestratorAgent
from src.tracing.console import RichTraceConsole
from src.tracing.models import TraceEvent, TraceEventType
from src.tracing.tracer import get_tracer


class InteractiveDebugger:
    """Interactive debugger for testing queries with live tracing.

    The agent automatically searches all datasets and determines
    which endpoints to query based on the schema elements found.
    """

    def __init__(
        self,
        approach: str = "agentic_semantic",
        llm_model: str = "deepseek-chat",
        verbosity: int = 3,
    ) -> None:
        self.approach = approach
        self.llm_model = llm_model

        self.console = Console()
        self.tracer = get_tracer()
        self.tracer.set_verbosity(verbosity)
        self.tracer.set_enabled(True)

        # Add rich console as listener
        self.renderer = RichTraceConsole(self.console)
        self.tracer.add_listener(self.renderer)

        self.agent = OrchestratorAgent()

    async def run_query(self, query: str) -> dict:
        """Run a single query with tracing."""
        # Emit query start
        self.tracer.emit(
            TraceEvent(
                event_type=TraceEventType.QUERY_START,
                data={"query": query},
            )
        )

        try:
            state = await self.agent.run(
                user_query=query,
                approach=self.approach,
                llm_model=self.llm_model,
            )

            # Emit query complete
            final_results = state.get("final_results", {}) or {}

            # Determine success and result count based on execution type
            is_multi_step = final_results.get("multi_step_execution", False)
            if is_multi_step:
                success = final_results.get("success", False)
                # For multi-step: check both results.bindings and direct final_results
                bindings = final_results.get("results", {}).get("bindings", [])
                direct_results = final_results.get("final_results", [])
                result_count = len(bindings) if bindings else len(direct_results)
            else:
                success = state.get("final_query") is not None
                bindings = final_results.get("results", {}).get("bindings", [])
                result_count = len(bindings)

            self.tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.QUERY_COMPLETE,
                    data={
                        "final_query": state.get("final_query", ""),
                        "final_results": final_results,
                        "result_count": result_count,
                        "success": success,
                        "iterations": state.get("iteration_count", 0),
                    },
                )
            )

            return state

        except Exception as e:
            self.tracer.emit(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    data={"error": str(e)},
                )
            )
            raise

    def run_interactive(self) -> None:
        """Run interactive REPL loop."""
        self.console.print(
            Panel(
                "[bold]DINA Agent Interactive Debugger[/bold]\n\n"
                "[cyan]Dataspace Mode[/cyan] - Agent searches all datasets automatically\n\n"
                f"Approach: [cyan]{self.approach}[/cyan]\n"
                f"LLM: [cyan]{self.llm_model}[/cyan]\n"
                f"Verbosity: [cyan]{self.tracer.get_verbosity()}[/cyan]\n\n"
                "Commands:\n"
                "  [dim]/verbosity <0-3>[/dim] - Set verbosity level\n"
                "  [dim]/approach <name>[/dim] - Change approach (agentic_grep, agentic_semantic)\n"
                "  [dim]/model <name>[/dim]    - Change LLM model\n"
                "  [dim]/quit[/dim]            - Exit\n",
                title="Welcome",
                border_style="blue",
            )
        )

        while True:
            try:
                query = Prompt.ask("\n[bold blue]Query[/bold blue]")

                if not query.strip():
                    continue

                if query.startswith("/"):
                    self._handle_command(query)
                    continue

                asyncio.run(self.run_query(query))

            except KeyboardInterrupt:
                self.console.print("\n[dim]Interrupted[/dim]")
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"\n[red]Error: {e}[/red]")

        self.console.print("[dim]Goodbye![/dim]")

    def _handle_command(self, cmd: str) -> None:
        """Handle a command."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/quit" or command == "/exit":
            raise EOFError
        elif command == "/verbosity":
            try:
                level = int(arg)
                self.tracer.set_verbosity(level)
                self.console.print(f"Verbosity set to {level}")
            except ValueError:
                self.console.print("[red]Invalid verbosity level. Use 0-3.[/red]")
        elif command == "/approach":
            if arg in ["agentic_grep", "agentic_semantic"]:
                self.approach = arg
                self.console.print(f"Approach changed to [cyan]{arg}[/cyan]")
            else:
                self.console.print("[red]Invalid approach. Use agentic_grep or agentic_semantic.[/red]")
        elif command == "/model":
            if arg:
                self.llm_model = arg
                self.console.print(f"Model changed to [cyan]{arg}[/cyan]")
            else:
                self.console.print("[red]Please specify a model name.[/red]")
        elif command == "/help":
            self.console.print(
                "Commands:\n"
                "  /verbosity <0-3> - Set verbosity level\n"
                "  /approach <name> - Change approach\n"
                "  /model <name>    - Change LLM model\n"
                "  /quit            - Exit"
            )
        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("Type /help for available commands.")
