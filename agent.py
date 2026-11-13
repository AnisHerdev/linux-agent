"""
agent.py - AI Infrastructure Monitoring Agent
Part of OpsPilot AI: Agentic Linux & Cloud Infrastructure Monitoring System.

Connects to mcp_server.py via MCP stdio transport.
Features:
- OpenAI reasoning
- LangChain Tool-Calling Agent with modern Middleware
- @wrap_model_call Guardrail Middleware (Read-Only Safety & Correctness)
- SummarizationMiddleware for Context Management
- Pydantic Data Validation
- Human-in-the-Loop (HITL) approval gate
- Rich terminal formatting and interactive Q&A
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file if present
load_dotenv()

# ==========================================
# OpenAI & Model Configuration
# ==========================================

# OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
MODEL_NAME = "gpt-4.1-mini"

# Rich CLI formatting
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text
from rich.markdown import Markdown

console = Console()

# MCP Client Imports
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, SummarizationMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage


# ==========================================
# Pydantic Schemas for Validation
# ==========================================

class TelemetrySnapshot(BaseModel):
    """Pydantic model validating complete telemetry retrieved for a VM."""
    vm_id: str
    status: str
    hostname: Optional[str] = None
    os: Optional[str] = None
    uptime: Optional[str] = None
    cpu_percent: Optional[float] = None
    cpu_threshold: Optional[str] = None
    memory_percent: Optional[float] = None
    memory_used_mb: Optional[int] = None
    memory_total_mb: Optional[int] = None
    disk_percent: Optional[float] = None
    disk_used_gb: Optional[float] = None
    disk_total_gb: Optional[float] = None
    large_directories: List[Dict[str, Any]] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    """Pydantic model validating structured AI health report."""
    vm_id: str
    overall_health: str = Field(..., description="Healthy, Warning, Critical, or Offline")
    summary: str = Field(..., description="High-level assessment summary")
    critical_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    advisory_notice: str = Field(
        default="Advisory only. No remediation actions are executed automatically.",
        description="Mandatory guardrail safety statement"
    )


# ==========================================
# LangChain Middleware: Guardrails & Safety
# ==========================================

DISALLOWED_PATTERNS = [
    "rm -rf", "delete", "format", "drop table", "drop database",
    "mkfs", "dd if=", "reboot", "shutdown", "truncate", "kill -9"
]

@wrap_model_call
def guardrail_middleware(request: ModelRequest, handler) -> ModelResponse:
    """Wrap-style middleware hook to enforce guardrails and functional correctness.
    
    1. Pre-invocation: Inspects user messages to ensure advisory-only constraints.
    2. Invocation: Calls the underlying model handler.
    3. Post-invocation: Verifies response correctness, advisory tone, and disclaimer.
    """
    # 1. Pre-call inspection: Check for disallowed destructive command requests
    for msg in request.messages:
        content = str(getattr(msg, "content", ""))
        for pattern in DISALLOWED_PATTERNS:
            if pattern in content.lower() and "do not execute" not in content.lower():
                # Block command execution attempts and enforce advisory framing
                request.system_prompt = (
                    f"{request.system_prompt or ''}\n\n"
                    "GUARDRAIL ALERT: The user may have referenced system commands. "
                    "Remember that OpsPilot AI is strictly read-only and advisory. "
                    "Never attempt to execute, simulate execution, or claim execution of destructive commands."
                )
                break

    # 2. Invoke the model
    response = handler(request)

    # 3. Post-call verification: Ensure the model did not claim changes were applied
    raw_content = ""
    if hasattr(response, "content") and response.content:
        raw_content = str(response.content)
    elif hasattr(response, "message") and hasattr(response.message, "content"):
        raw_content = str(response.message.content)

    if any(phrase in raw_content.lower() for phrase in ["i have deleted", "i executed", "i wiped", "i removed the files"]):
        disclaimer = (
            "\n\n[Guardrail Notice: OpsPilot AI is an advisory-only assistant. "
            "No commands or file modifications have been executed on the virtual machine.]"
        )
        if hasattr(response, "content") and isinstance(response.content, str):
            response.content += disclaimer
        elif hasattr(response, "message") and hasattr(response.message, "content"):
            response.message.content += disclaimer

    return response


# ==========================================
# Telemetry Retrieval & UI Presentation
# ==========================================

async def fetch_vm_telemetry(tools_map: Dict[str, Any], vm_id: str) -> TelemetrySnapshot:
    """Interacts with MCP tools to gather and validate telemetry snapshot via Pydantic."""
    norm_id = vm_id.strip().upper()

    # Step 1: Check VM availability
    status_tool = tools_map.get("get_vm_status")
    status_res = await status_tool.ainvoke({"vm_id": norm_id})
    status_data = _extract_tool_payload(status_res)

    if status_data.get("status") == "dead" or "error" in status_data:
        return TelemetrySnapshot(
            vm_id=norm_id,
            status=status_data.get("status", "dead"),
            hostname=status_data.get("hostname"),
            os=status_data.get("os"),
            uptime=None
        )

    # Step 2: Retrieve resource metrics for alive VM
    cpu_res = await tools_map["get_cpu_usage"].ainvoke({"vm_id": norm_id})
    mem_res = await tools_map["get_memory_usage"].ainvoke({"vm_id": norm_id})
    disk_res = await tools_map["get_disk_usage"].ainvoke({"vm_id": norm_id})
    dirs_res = await tools_map["get_large_directories"].ainvoke({"vm_id": norm_id})

    cpu_data = _extract_tool_payload(cpu_res)
    mem_data = _extract_tool_payload(mem_res)
    disk_data = _extract_tool_payload(disk_res)
    dirs_data = _extract_tool_payload(dirs_res)

    return TelemetrySnapshot(
        vm_id=norm_id,
        status="alive",
        hostname=status_data.get("hostname"),
        os=status_data.get("os"),
        uptime=status_data.get("uptime"),
        cpu_percent=cpu_data.get("usage_percent"),
        cpu_threshold=cpu_data.get("threshold_level"),
        memory_percent=mem_data.get("usage_percent"),
        memory_used_mb=mem_data.get("used_mb"),
        memory_total_mb=mem_data.get("total_mb"),
        disk_percent=disk_data.get("usage_percent"),
        disk_used_gb=disk_data.get("used_gb"),
        disk_total_gb=disk_data.get("total_gb"),
        large_directories=dirs_data.get("directories", [])
    )


def _extract_tool_payload(tool_output: Any) -> Dict[str, Any]:
    """Helper to parse MCP tool output content blocks into Python dicts."""
    if isinstance(tool_output, list) and len(tool_output) > 0:
        first = tool_output[0]
        if isinstance(first, dict) and "text" in first:
            try:
                return json.loads(first["text"])
            except Exception:
                return {"raw": first["text"]}
    elif isinstance(tool_output, dict):
        return tool_output
    elif isinstance(tool_output, str):
        try:
            return json.loads(tool_output)
        except Exception:
            return {"raw": tool_output}
    return {}


def display_telemetry_tables(telemetry: TelemetrySnapshot):
    """Renders Rich formatted telemetry tables with color-coded health indicators."""
    if telemetry.status == "dead":
        console.print(Panel(
            f"[bold red]✖ VM Status: OFFLINE / DEAD[/bold red]\n"
            f"[yellow]The machine '{telemetry.vm_id}' is unreachable or shut down.\n"
            "Resource metrics (CPU, RAM, Disk) cannot be collected.[/yellow]",
            title=f"Infrastructure Telemetry — {telemetry.vm_id}",
            border_style="red"
        ))
        return

    # Metrics Table
    table = Table(title=f"Telemetry Metrics — {telemetry.vm_id} ({telemetry.hostname})", expand=True)
    table.add_column("Resource", style="cyan bold", width=18)
    table.add_column("Utilization", style="white", justify="right")
    table.add_column("Details", style="magenta")
    table.add_column("Health Status", justify="center")

    # Status row
    table.add_row("Availability", "Alive", f"OS: {telemetry.os} | Uptime: {telemetry.uptime}", "[green]✓ Healthy[/green]")

    # CPU row
    cpu_val = f"{telemetry.cpu_percent:.1f}%" if telemetry.cpu_percent is not None else "N/A"
    cpu_style = _get_health_badge(telemetry.cpu_percent)
    table.add_row("CPU Utilization", cpu_val, f"Threshold: {telemetry.cpu_threshold}", cpu_style)

    # Memory row
    mem_val = f"{telemetry.memory_percent:.1f}%" if telemetry.memory_percent is not None else "N/A"
    mem_det = f"{telemetry.memory_used_mb} MB / {telemetry.memory_total_mb} MB" if telemetry.memory_used_mb else "N/A"
    mem_style = _get_health_badge(telemetry.memory_percent)
    table.add_row("Memory (RAM)", mem_val, mem_det, mem_style)

    # Disk row
    disk_val = f"{telemetry.disk_percent:.1f}%" if telemetry.disk_percent is not None else "N/A"
    disk_det = f"{telemetry.disk_used_gb} GB / {telemetry.disk_total_gb} GB" if telemetry.disk_used_gb else "N/A"
    disk_style = _get_health_badge(telemetry.disk_percent)
    table.add_row("Disk Storage", disk_val, disk_det, disk_style)

    console.print(table)

    # Directories Table
    if telemetry.large_directories:
        dir_table = Table(title=f"Storage Breakdown by Directory — {telemetry.vm_id}", expand=True)
        dir_table.add_column("Directory Path", style="yellow bold", width=22)
        dir_table.add_column("Size", justify="right", style="bold red", width=12)
        dir_table.add_column("Description / Workload", style="white")

        for d in telemetry.large_directories:
            dir_table.add_row(d["path"], f"{d['size_gb']} GB", d["description"])

        console.print(dir_table)


def _get_health_badge(percent: Optional[float]) -> str:
    if percent is None:
        return "[dim]Unknown[/dim]"
    if percent >= 85.0:
        return "[bold red]✖ Critical[/bold red]"
    if percent >= 70.0:
        return "[bold yellow]⚠ Warning[/bold yellow]"
    return "[bold green]✓ Normal[/bold green]"


# ==========================================
# AI Analysis & Health Reporting
# ==========================================

SYSTEM_PROMPT = """You are OpsPilot AI, an elite Linux & Cloud Infrastructure Monitoring and Analysis Assistant.
You specialize in inspecting virtual machines, evaluating resource bottlenecks (CPU, RAM, Disk), pinpointing root causes, and providing actionable, practical recommendations.

STRICT SAFETY AND OPERATIONAL CONSTRAINTS:
1. You are strictly ADVISORY ONLY.
2. You must NEVER execute or claim to have executed any cleanup, file deletion, process killing, or remediation commands.
3. Your recommendations must guide human operators on investigative commands (e.g., 'journalctl --disk-usage', 'du -sh /var/log/*') rather than assuming automated execution.
4. When inspecting offline VMs, clearly explain that no metrics can be collected while offline.
"""

async def run_ai_investigation(
    model: ChatOpenAI,
    telemetry: TelemetrySnapshot
) -> AnalysisReport:
    """Uses the OpenAI model to produce a structured infrastructure health report."""
    if telemetry.status == "dead":
        return AnalysisReport(
            vm_id=telemetry.vm_id,
            overall_health="Offline",
            summary=f"Virtual machine '{telemetry.vm_id}' is unreachable (Offline / Dead).",
            critical_issues=[
                "Host ping failure: No response from guest OS or hypervisor.",
                "Service interruption: All hosted workloads on this VM are inactive."
            ],
            recommendations=[
                "Verify power and hypervisor status from the cloud/virtualization console.",
                "Inspect host virtualization logs for hardware or hypervisor faults.",
                "Check network security group and DHCP lease status for the instance."
            ]
        )

    # Prepare analysis prompt for OpenAI
    telemetry_summary = (
        f"Virtual Machine: {telemetry.vm_id} ({telemetry.hostname})\n"
        f"Operating System: {telemetry.os}\n"
        f"Uptime: {telemetry.uptime}\n"
        f"CPU Utilization: {telemetry.cpu_percent}%\n"
        f"Memory Utilization: {telemetry.memory_percent}% ({telemetry.memory_used_mb}MB / {telemetry.memory_total_mb}MB)\n"
        f"Disk Utilization: {telemetry.disk_percent}% ({telemetry.disk_used_gb}GB / {telemetry.disk_total_gb}GB)\n"
        f"High Storage Directories:\n"
        + "\n".join([f" - {d['path']}: {d['size_gb']} GB ({d['description']})" for d in telemetry.large_directories])
    )

    prompt = (
        "Analyze the following Linux infrastructure telemetry metrics:\n\n"
        f"{telemetry_summary}\n\n"
        "Provide your analysis formatted with the following sections:\n"
        "1. OVERALL HEALTH: [Healthy / Elevated / Critical]\n"
        "2. ROOT CAUSE FINDINGS: (Key bottlenecks and what is causing them)\n"
        "3. ACTIONABLE RECOMMENDATIONS: (Numbered list of 3-4 specific advisory steps for the system administrator)"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response = await model.ainvoke(messages)
    content = response.content

    # Determine health classification
    overall_health = "Healthy"
    if telemetry.disk_percent and telemetry.disk_percent >= 85 or telemetry.cpu_percent and telemetry.cpu_percent >= 85:
        overall_health = "Critical"
    elif telemetry.disk_percent and telemetry.disk_percent >= 70 or telemetry.cpu_percent and telemetry.cpu_percent >= 70:
        overall_health = "Elevated"

    # Extract bullet points/recommendations from response
    critical_issues = []
    recommendations = []
    lines = content.splitlines()
    rec_mode = False
    for line in lines:
        stripped = line.strip()
        if "recommendation" in stripped.lower():
            rec_mode = True
            continue
        if rec_mode and (stripped.startswith("-") or stripped.startswith("*") or (len(stripped) > 2 and stripped[:2].isdigit())):
            recommendations.append(stripped.lstrip("-*0123456789. "))
        elif not rec_mode and (stripped.startswith("-") or stripped.startswith("*")):
            critical_issues.append(stripped.lstrip("-* "))

    if not recommendations:
        recommendations = [
            "Inspect and configure log rotation for high-volume logs in /var/log.",
            "Clean up abandoned runtime artifacts and core dumps in /tmp.",
            "Review and purge obsolete package manager caches in /var/cache."
        ]

    return AnalysisReport(
        vm_id=telemetry.vm_id,
        overall_health=overall_health,
        summary=content,
        critical_issues=critical_issues,
        recommendations=recommendations
    )


# ==========================================
# Human-in-the-Loop Approval Gate
# ==========================================

def prompt_human_approval(report: AnalysisReport) -> str:
    """Prompts the operator to approve or reject the AI recommendations.
    
    Strictly records decision; never executes destructive changes.
    """
    console.print()
    console.rule("[bold yellow]HUMAN APPROVAL REQUIRED[/bold yellow]")

    console.print(Panel(
        "\n".join([f"{idx+1}. {rec}" for idx, rec in enumerate(report.recommendations)]),
        title="[bold cyan]AI ADVISORY RECOMMENDATIONS[/bold cyan]",
        border_style="cyan"
    ))

    console.print("[bold]Do you approve these recommendations for operator consideration?[/bold]")
    console.print("[1] Approve")
    console.print("[2] Reject\n")

    choice = Prompt.ask("Your choice", choices=["1", "2"], default="1")
    status = "APPROVED" if choice == "1" else "REJECTED"

    status_color = "green" if status == "APPROVED" else "red"
    console.print(Panel(
        f"[bold {status_color}]Approval Status: {status}[/bold {status_color}]\n\n"
        f"[italic]Note: Recommendations {status.lower()} for consideration.\n"
        "No changes or remediation commands have been executed on the virtual machine.[/italic]",
        border_style=status_color
    ))
    return status


# ==========================================
# Interactive Follow-up Session
# ==========================================

async def run_interactive_session(
    agent_graph: Any,
    initial_telemetry: TelemetrySnapshot,
    tools_map: Dict[str, Any],
    model: ChatOpenAI
):
    """Interactive CLI conversation supporting tool calling, context summarization, and VM switching."""
    current_vm = initial_telemetry.vm_id
    current_telemetry = initial_telemetry

    console.print()
    console.rule(f"[bold cyan]Interactive Troubleshooting Session — {current_vm}[/bold cyan]")
    console.print("[dim]You can ask technical questions about this VM, request deeper analysis, or switch to another VM (e.g. 'Investigate VM001'). Type 'exit' to quit.[/dim]\n")

    conversation_messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"I am actively monitoring virtual machine {current_vm}. Availability is {current_telemetry.status}."),
        AIMessage(content=f"OpsPilot monitoring session ready for {current_vm}. How can I assist you with this host?")
    ]

    while True:
        try:
            user_input = Prompt.ask(f"[bold green]OpsPilot ({current_vm})[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session interrupted. Exiting OpsPilot AI...[/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            console.print("[bold cyan]Exiting OpsPilot AI. Have a productive day![/bold cyan]")
            break

        # Check for VM switching intent
        upper_input = user_input.upper()
        switch_target = None
        for target_id in ["VM001", "VM002", "VM003"]:
            if target_id in upper_input:
                # User mentioned a VM
                if "INVESTIGATE" in upper_input or "SWITCH" in upper_input or upper_input.strip() == target_id:
                    switch_target = target_id
                    break

        if switch_target:
            console.print(f"\n[bold yellow]Switching active target to {switch_target}...[/bold yellow]")
            current_vm = switch_target
            with console.status(f"[cyan]Collecting telemetry for {current_vm}...[/cyan]", spinner="dots"):
                current_telemetry = await fetch_vm_telemetry(tools_map, current_vm)
            display_telemetry_tables(current_telemetry)

            # Analyze newly selected VM
            with console.status("[cyan]Analyzing telemetry with OpenAI...[/cyan]", spinner="dots"):
                report = await run_ai_investigation(model, current_telemetry)
            console.print(Panel(report.summary, title=f"AI Infrastructure Health Report — {current_vm}", border_style="cyan"))
            prompt_human_approval(report)

            # Update context messages
            conversation_messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Switched monitoring to {current_vm}. Status: {current_telemetry.status}."),
                AIMessage(content=f"Now monitoring {current_vm}. What questions do you have regarding {current_vm}?")
            ]
            continue

        # Standard conversational query with agent
        conversation_messages.append(HumanMessage(content=user_input))

        with console.status("[cyan]OpsPilot AI reasoning and invoking tools...[/cyan]", spinner="dots"):
            try:
                result = await agent_graph.ainvoke({"messages": conversation_messages})
                conversation_messages = result["messages"]
                latest_reply = conversation_messages[-1].content
            except Exception as e:
                latest_reply = f"[red]Error during agent reasoning: {str(e)}[/red]"

        console.print()
        console.print(Panel(Markdown(latest_reply), title=f"[bold cyan]OpsPilot ({current_vm})[/bold cyan]", border_style="blue"))
        console.print()


# ==========================================
# Main Application Flow
# ==========================================

async def main():
    """Main CLI entrypoint for OpsPilot AI."""
    console.print(Rule(style="cyan"))
    console.print(Panel.fit(
        "[bold cyan]OpsPilot AI[/bold cyan] — [bold white]Agentic Linux & Cloud Infrastructure Monitoring[/bold white]\n"
        "[dim]LangChain Tool-Calling Agent • MCP Stdio Transport • FastMCP • OpenAI Reasoning[/dim]",
        border_style="cyan"
    ))
    console.print(Rule(style="cyan"))

    # Resolve OpenAI API Key
    effective_api_key = os.getenv("OPENAI_API_KEY")
    if not effective_api_key or effective_api_key == "YOUR_OPENAI_API_KEY":
        if OPENAI_API_KEY != "YOUR_OPENAI_API_KEY":
            effective_api_key = OPENAI_API_KEY

    if not effective_api_key or effective_api_key == "YOUR_OPENAI_API_KEY":
        console.print(Panel(
            "[bold yellow]OpenAI API Key Not Found[/bold yellow]\n\n"
            "Please provide an OpenAI API key via your [cyan].env[/cyan] file or set the [cyan]OPENAI_API_KEY[/cyan] environment variable.\n"
            "Format: [green]OPENAI_API_KEY=sk-...[/green]",
            border_style="yellow"
        ))
        entered_key = Prompt.ask("[bold]Enter your OpenAI API key (or press Enter to exit)[/bold]", password=True).strip()
        if entered_key:
            effective_api_key = entered_key
        else:
            console.print("[red]No API key provided. Exiting OpsPilot AI.[/red]")
            return

    # Initialize OpenAI Chat Model
    try:
        model = ChatOpenAI(
            model=MODEL_NAME,
            api_key=effective_api_key,
            temperature=0.2
        )
    except Exception as e:
        console.print(f"[bold red]Failed to initialize OpenAI model: {e}[/bold red]")
        return

    # MCP Server Parameters (launches mcp_server.py via stdio)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"]
    )

    console.print("[cyan]Connecting to OpsPilot MCP server via stdio transport...[/cyan]")

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as mcp_session:
                await mcp_session.initialize()
                tools = await load_mcp_tools(mcp_session)
                tools_map = {t.name: t for t in tools}

                console.print(f"[green]✓ Connected to MCP Server! Discovered {len(tools)} read-only tools:[/green] "
                              f"[dim]{', '.join([t.name for t in tools])}[/dim]\n")

                # Context management middleware: Summarization
                summarizer = SummarizationMiddleware(
                    model=model,
                    trigger=("messages", 8),
                    keep=("messages", 4)
                )

                # Build LangChain agent with guardrails and context management middleware
                agent_graph = create_agent(
                    model=model,
                    tools=tools,
                    system_prompt=SYSTEM_PROMPT,
                    middleware=[guardrail_middleware, summarizer]
                )

                # ==========================================
                # Stage 1: Automatic Investigation
                # ==========================================
                console.print("[bold]Available Simulated Virtual Machines:[/bold]")
                console.print(" • [bold cyan]VM001[/bold cyan] — Production Web Server (Healthy)")
                console.print(" • [bold cyan]VM002[/bold cyan] — Production Analytics Database (Critical Resource Exhaustion)")
                console.print(" • [bold cyan]VM003[/bold cyan] — Staging Worker (Offline / Host Unreachable)\n")

                vm_input = Prompt.ask("Enter VM ID to investigate", default="VM002").strip()
                if not vm_input:
                    vm_input = "VM002"

                console.print(f"\n[cyan]Starting automated investigation for [bold]{vm_input}[/bold]...[/cyan]")

                with console.status("[cyan]Retrieving infrastructure telemetry via MCP tools...[/cyan]", spinner="dots"):
                    telemetry = await fetch_vm_telemetry(tools_map, vm_input)

                # Render Telemetry Tables
                display_telemetry_tables(telemetry)

                # Run AI Analysis
                with console.status("[cyan]OpenAI analyzing metrics and formulating recommendations...[/cyan]", spinner="dots"):
                    report = await run_ai_investigation(model, telemetry)

                # Render Analysis Report
                console.print()
                console.print(Panel(
                    Markdown(report.summary),
                    title=f"[bold cyan]AI Infrastructure Health Report — {telemetry.vm_id} ({report.overall_health})[/bold cyan]",
                    border_style="cyan"
                ))

                # ==========================================
                # Stage 2: Human-in-the-Loop Approval
                # ==========================================
                prompt_human_approval(report)

                # ==========================================
                # Stage 3: Interactive Follow-up Session
                # ==========================================
                await run_interactive_session(agent_graph, telemetry, tools_map, model)

    except Exception as exc:
        console.print(f"[bold red]OpsPilot runtime error: {exc}[/bold red]")
    finally:
        console.print("[dim]OpsPilot AI session completed.[/dim]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]OpsPilot shutdown requested by user.[/yellow]")
