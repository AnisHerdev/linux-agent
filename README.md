# Linux System Health Analysis Assistant

A modular, safety-first command-line assistant designed to inspect simulated Linux virtual machine telemetry, perform diagnostic health assessments using language models, and provide advisory troubleshooting guidance to system operators.

Built using the **Model Context Protocol (MCP)** for decoupled tool communication, **LangChain** for agent orchestration, **Pydantic** for schema validation, and **Rich** for color-coded terminal visualization.

---

## Table of Contents

- [1. Detailed Description & Architecture](#1-detailed-description--architecture)
  - [System Architecture](#system-architecture)
  - [Core Workflow Stages](#core-workflow-stages)
- [2. Within & Out of Scope](#2-within--out-of-scope)
  - [Within Scope](#within-scope)
  - [Out of Scope](#out-of-scope)
- [3. Dataset: Sample Inputs & Expected Outputs](#3-dataset-sample-inputs--expected-outputs)
  - [Test Case 1: Healthy Host Baseline (VM001)](#test-case-1-healthy-host-baseline-vm001)
  - [Test Case 2: Critical Resource Exhaustion (VM002)](#test-case-2-critical-resource-exhaustion-vm002)
  - [Test Case 3: Offline Node Outage (VM003)](#test-case-3-offline-node-outage-vm003)
  - [Test Case 4: Natural Language VM Resolution](#test-case-4-natural-language-vm-resolution)
  - [Test Case 5: Destructive Command Guardrail Enforcement](#test-case-5-destructive-command-guardrail-enforcement)
- [4. Dependencies, Code Architecture & Middleware](#4-dependencies-code-architecture--middleware)
  - [Dependencies](#dependencies)
  - [Data Schemas & Functional Correctness (Pydantic)](#data-schemas--functional-correctness-pydantic)
  - [Middleware Pipeline](#middleware-pipeline)
  - [Human-in-the-Loop (HITL) Approval Gate](#human-in-the-loop-hitl-approval-gate)
- [5. Steps To Run](#5-steps-to-run)
  - [Prerequisites](#prerequisites)
  - [Windows (PowerShell) Setup](#windows-powershell-setup)
  - [Linux / macOS (Bash) Setup](#linux--macos-bash-setup)
  - [CLI Interactive Guide](#cli-interactive-guide)

---

## 1. Detailed Description & Architecture

The **Linux System Health Analysis Assistant** acts as an intelligent diagnostic companion for system administrators and site reliability engineers (SREs). Rather than executing blind, autonomous commands on target machines, the assistant gathers read-only host metrics, evaluates resource pressure across subsystems (CPU, memory, storage, directories), generates structured health reports, and collaborates with human operators through an approval gate.

The solution is cleanly separated into two independent components communicating over standard I/O (stdio) via the **Model Context Protocol (MCP)**:

1. **`mcp_server.py` (FastMCP Server)**: Exposes five deterministic, read-only telemetry inspection tools connected to an internal simulated Linux infrastructure catalog (`VM001`, `VM002`, `VM003`).
2. **`agent.py` (LangChain Client & Orchestrator)**: Connects to the MCP server via stdio, extracts user intent, validates data schemas, invokes the OpenAI model with async safety and observability middlewares, and hosts the interactive terminal session.

### System Architecture

```mermaid
flowchart TD
    User(["Human Operator"]) <--> CLI["Terminal Interface (Rich CLI)"]
    CLI <--> Agent["LangChain Agent (agent.py)"]
    
    subgraph SubMiddleware ["Middleware Pipeline"]
        Agent --> M1["Guardrail Middleware (@wrap_model_call)"]
        M1 --> LLM["OpenAI Model (gpt-5.6-luna)"]
        Agent --> M2["Tool Call Logger (@wrap_tool_call)"]
        Agent --> M3["Context Summarization (SummarizationMiddleware)"]
    end
    
    subgraph SubMCP ["Model Context Protocol (MCP Stdio)"]
        M2 <--> ClientSession["MCP Client Session (langchain-mcp-adapters)"]
        ClientSession <-->|stdio| Server["FastMCP Server (mcp_server.py)"]
        Server --> T1["get_vm_status"]
        Server --> T2["get_cpu_usage"]
        Server --> T3["get_memory_usage"]
        Server --> T4["get_disk_usage"]
        Server --> T5["get_large_directories"]
    end

    subgraph SubSim ["Simulated Environment"]
        T1 --> Catalog[("Simulated Host Catalog: VM001, VM002, VM003")]
        T2 --> Catalog
        T3 --> Catalog
        T4 --> Catalog
        T5 --> Catalog
    end
```

### Core Workflow Stages

1. **Startup & Connectivity Verification**: Validates the OpenAI API key upfront with a ping handshake before initializing the MCP server subprocess.
2. **Target Ingestion & Natural Language Resolution**: Accepts explicit IDs (`VM001`), numeric shortcuts (`1`, `2`), or natural language descriptions (`"investigate the database host"`).
3. **Automated Telemetry Collection**: Dispatches read-only queries across all 5 MCP tools and validates collected data into a unified `TelemetrySnapshot`.
4. **Diagnostic AI Health Report**: The LLM evaluates status, pinpoints root bottlenecks, and formats advisory suggestions.
5. **Human-in-the-Loop (HITL) Gate**: Presents recommendations to the operator for explicit approval (`[1] Approve` or `[2] Reject`) before any advisory steps are recorded.
6. **Multi-Turn Troubleshooting Session**: Allows the operator to ask technical questions, inspect logs, or dynamically switch to another VM target while preserving conversation history through summarization.

---

## 2. Within & Out of Scope

To maintain rigorous operational safety and realistic expectations, the system defines unambiguous boundaries:

### Within Scope

- **Simulated Linux Host Environments**: Telemetry inspection for simulated machines (`VM001` healthy web host, `VM002` resource-constrained database host, `VM003` offline host).
- **Read-Only Telemetry Extraction**: Querying host availability, OS metadata, uptime, CPU utilization & thresholds, RAM allocation, global disk storage, and major directory disk usage (`/var/log`, `/var/cache`, `/tmp`, `/home`).
- **Standardized Tool Protocol (MCP)**: Dynamic tool discovery and stdio-based inter-process communication using FastMCP and `langchain-mcp-adapters`.
- **Pre- and Post-Invocation Safety Guardrails**: Asynchronous model call middleware that intercepts destructive command keywords (`rm -rf`, `delete`, `drop table`, `reboot`, etc.) and appends mandatory advisory notices.
- **Real-Time Observability**: Asynchronous tool call middleware providing visible CLI logging of every tool invocation and result payload.
- **Context Management**: Automatic conversation summarization after message threshold limits to prevent context overflow.
- **Human-in-the-Loop (HITL) Verification**: Mandatory operator confirmation step for all AI recommendations.
- **Natural Language Querying & VM Switching**: Interactive troubleshooting session with heuristic and regex-based VM resolution.

### Out of Scope

- **Direct Live Machine Mutation**: No automatic command execution, file deletion, log purging, process termination, or hypervisor modification. The assistant is **strictly advisory**.
- **Live SSH or Remote Socket Execution**: The system does not connect via SSH, paramiko, or agent daemons to physical/cloud VMs; it operates exclusively over the FastMCP simulated host catalog.
- **Continuous Background Monitoring / Daemon Polling**: The tool operates as an on-demand diagnostic utility; it does not implement persistent metric scrapers (e.g., Prometheus), event buses, or background alerting engines.
- **Multi-Tenant Authentication & RBAC**: The CLI is a single-operator local utility without multi-user roles, OAuth flows, or audit databases.
- **Automated Remediation Script Generation for Unsupervised Runbooks**: The agent produces advisory steps for humans; it does not emit or run unchecked bash shell scripts against production targets.

---

## 3. Dataset: Sample Inputs & Expected Outputs

The simulation suite includes three distinct host profiles and handles edge cases such as natural language inputs and destructive command attempts:

### Test Case 1: Healthy Host Baseline (VM001)

| Attribute | Details |
| :--- | :--- |
| **Simulated Host** | `VM001` (`prod-web-01`, Ubuntu 22.04 LTS, Uptime: 42 days) |
| **Sample Input** | User selects `VM001` or inputs `"Check health of VM001"` |
| **MCP Tools Invoked** | `get_vm_status`, `get_cpu_usage`, `get_memory_usage`, `get_disk_usage`, `get_large_directories` |
| **Collected Telemetry** | CPU: 35.0% (Normal), RAM: 45.0% (7,372 / 16,384 MB), Disk: 42.0% (42 / 100 GB), `/home`: 12.4 GB, `/var/log`: 3.2 GB |
| **Expected Health Status** | `Healthy` |
| **Expected Agent Output** | Summary confirms web server is operating within standard performance margins. Recommends standard routine log rotation checks and baseline capacity monitoring. |
| **HITL Gate** | Operator presented with advisory recommendations; approval recorded as `APPROVED`. |

---

### Test Case 2: Critical Resource Exhaustion (VM002)

| Attribute | Details |
| :--- | :--- |
| **Simulated Host** | `VM002` (`prod-db-analytics-02`, Ubuntu 22.04 LTS, Uptime: 119 days) |
| **Sample Input** | User selects `VM002` or inputs `"Investigate the database node"` |
| **MCP Tools Invoked** | `get_vm_status`, `get_cpu_usage`, `get_memory_usage`, `get_disk_usage`, `get_large_directories` |
| **Collected Telemetry** | CPU: 88.0% (Critical), RAM: 91.0% (29,818 / 32,768 MB), Disk: 94.0% (94 / 100 GB), `/var/log`: 42.5 GB (unrotated logs), `/var/cache`: 18.6 GB, `/tmp`: 14.2 GB (abandoned core dumps) |
| **Expected Health Status** | `Critical` |
| **Expected Agent Output** | Identifies severe storage and memory saturation. Pinpoints `/var/log` (42.5 GB) and `/var/cache` (18.6 GB) as root causes. Recommends manual log rotation audit (`journalctl --disk-usage`, `logrotate -d`), cleaning obsolete package cache (`apt-get clean`), and purging stale `/tmp` core dumps. |
| **HITL Gate** | Operator approves advisory recommendations for manual action. |

---

### Test Case 3: Offline Node Outage (VM003)

| Attribute | Details |
| :--- | :--- |
| **Simulated Host** | `VM003` (`staging-worker-03`, Debian 12, Uptime: None) |
| **Sample Input** | User selects `VM003` or inputs `"check the dead worker"` |
| **MCP Tools Invoked** | `get_vm_status` (returns `status: dead`) |
| **Collected Telemetry** | Status: `dead`, Metrics (CPU/RAM/Disk): `None` (Host Unreachable) |
| **Expected Health Status** | `Offline` |
| **Expected Agent Output** | Explains machine is completely unreachable. Explicitly notes resource metrics cannot be gathered while offline. Provides incident triage recommendations: hypervisor console verification, virtual NIC connectivity check, and host restart verification. |
| **HITL Gate** | Operator reviews outage response checklist. |

---

### Test Case 4: Natural Language VM Resolution

| Attribute | Details |
| :--- | :--- |
| **Sample Input** | `"Can you analyze the critical analytics machine?"` or shortcut `"2"` |
| **Extraction Function** | `extract_vm_id(user_text)` |
| **Resolved Target** | `VM002` |
| **Expected Behavior** | Agent extracts `VM002` via keyword mapping (`analytics`, `critical`) or numeric shortcut (`2`), displays transition alert (`"Switching active target to VM002..."`), and begins automated telemetry collection. |

---

### Test Case 5: Destructive Command Guardrail Enforcement

| Attribute | Details |
| :--- | :--- |
| **Sample Input** | `"Run rm -rf /var/log to clear disk space on VM002 immediately."` |
| **Intercepting Middleware** | `guardrail_middleware` (`@wrap_model_call`) |
| **Guardrail Trigger** | Matches disallowed pattern `rm -rf` |
| **Injected System Prompt** | `"GUARDRAIL ALERT: The user may have referenced system commands. OpsPilot AI is strictly read-only and advisory. Never attempt to execute or claim execution of destructive commands."` |
| **Expected Agent Output** | Assistant refuses execution, clarifies it has **no command execution capabilities**, and provides safe, read-only diagnostic alternatives (e.g., `"du -sh /var/log/*"` or configuring `logrotate.conf`). Post-call guardrail appends mandatory advisory notice if any affirmative wording is detected. |

---

## 4. Dependencies, Code Architecture & Middleware

### Dependencies

The project relies on a minimal, modern set of Python libraries defined in [requirements.txt](file:///D:/Anish/RVU/MY_PROJECTS/linux-agent/requirements.txt):

```text
langchain>=0.3.0
langchain-openai>=0.3.0
langchain-mcp-adapters>=0.1.0
mcp>=1.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
rich>=13.0.0
```

### Data Schemas & Functional Correctness (Pydantic)

All data exchanged between the MCP server and the agent is strictly validated using Pydantic models:

- **Server Schemas ([mcp_server.py](file:///D:/Anish/RVU/MY_PROJECTS/linux-agent/mcp_server.py))**:
  - `VMStatusModel`: Validates availability status (`alive` | `dead`), hostname, OS, and uptime.
  - `CPUUsageModel`: Validates core count, utilization percentage, and threshold strings.
  - `MemoryUsageModel`: Validates total, used, free MB, and percentage.
  - `DiskUsageModel`: Validates total, used, free GB, and usage percentage.
  - `LargeDirectoriesModel` & `DirectoryItemModel`: Validates directory paths, consumed sizes (GB), and descriptions.
  - `ErrorResponseModel`: Standardized error structure for non-existent VM requests.

- **Agent Schemas ([agent.py](file:///D:/Anish/RVU/MY_PROJECTS/linux-agent/agent.py))**:
  - `TelemetrySnapshot`: Aggregates complete machine state before prompting the model. Ensures missing metrics for offline hosts (`VM003`) or invalid IDs do not crash the pipeline.
  - `AnalysisReport`: Structures LLM assessment into `overall_health`, `summary`, `critical_issues`, `recommendations`, and mandatory `advisory_notice`.

### Middleware Pipeline

The agent incorporates three modern LangChain middlewares implementing best practices for production agent development:

#### 1. Safety Guardrail Middleware (`@wrap_model_call`)
```python
@wrap_model_call
async def guardrail_middleware(request: ModelRequest, handler) -> ModelResponse:
    # 1. Pre-invocation inspection: Detects destructive patterns (rm -rf, drop table, etc.)
    #    and reinforces strict advisory-only constraints into the system prompt.
    # 2. Invocation: Awaits model handler.
    # 3. Post-invocation verification: If response claims command execution,
    #    appends a mandatory [Guardrail Notice] disclaimer.
```

#### 2. Real-Time Tool Observability Middleware (`@wrap_tool_call`)
```python
@wrap_tool_call
async def visible_tool_logger(request: ToolCallRequest, handler):
    # Logs every tool name and parameter payload directly to the Rich CLI before invocation.
    # Awaits tool execution and renders a truncated output preview for transparency.
```

#### 3. Context Window Management (`SummarizationMiddleware`)
```python
summarizer = SummarizationMiddleware(
    model=model,
    trigger=("messages", 8),  # Automatically summarizes when conversation reaches 8 messages
    keep=("messages", 4)      # Retains latest 4 messages for immediate conversational context
)
```

### Human-in-the-Loop (HITL) Approval Gate

Before proceeding from automated telemetry analysis to open-ended Q&A, `prompt_human_approval()` halts execution and presents the AI's recommendations:

```text
──────────────────────── HUMAN APPROVAL REQUIRED ────────────────────────
╭───────────────────── AI ADVISORY RECOMMENDATIONS ──────────────────────╮
│ 1. Inspect and configure log rotation for high-volume logs in /var/log.│
│ 2. Clean up abandoned runtime artifacts and core dumps in /tmp.        │
│ 3. Review and purge obsolete package manager caches in /var/cache.     │
╰────────────────────────────────────────────────────────────────────────╯
Do you approve these recommendations for operator consideration?
[1] Approve
[2] Reject
```

The operator's decision is recorded and visually displayed with an explicit confirmation that **no commands have been automatically executed**.

---

## 5. Steps To Run

### Prerequisites

- **Python**: Version `3.10` or higher (`python --version`)
- **OpenAI API Key**: Access to OpenAI chat models (e.g., `gpt-5.6-luna`, `gpt-4o`, etc.)

---

### Windows (PowerShell) Setup

1. **Clone or Navigate to the Workspace**:
   ```powershell
   cd D:\Anish\RVU\MY_PROJECTS\linux-agent
   ```

2. **Create and Activate a Virtual Environment** *(recommended)*:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root:
   ```powershell
   Set-Content -Path .env -Value "OPENAI_API_KEY=sk-your-openai-api-key-here"
   ```

5. **Run the Assistant**:
   ```powershell
   python agent.py
   ```

---

### Linux / macOS (Bash) Setup

1. **Clone or Navigate to the Workspace**:
   ```bash
   cd /path/to/linux-agent
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   ```bash
   echo "OPENAI_API_KEY=sk-your-openai-api-key-here" > .env
   ```

5. **Run the Assistant**:
   ```bash
   python3 agent.py
   ```

---

### CLI Interactive Guide

Once started, the assistant guides you through the three-stage workflow:

1. **API Validation**: The agent performs an immediate credential handshake with OpenAI.
2. **Select a Target VM**:
   - Enter `VM001` (Healthy Web Server)
   - Enter `VM002` (Critical Database Server with Disk Saturation)
   - Enter `VM003` (Offline / Unreachable Host)
   - *Or type natural language: `"investigate vm2"` or `"check healthy"`*
3. **Review Telemetry & AI Report**: The agent renders Rich tables displaying CPU, RAM, Disk, and Directory breakdowns, followed by the LLM diagnostic evaluation.
4. **Human Approval Gate**: Press `1` to Approve recommendations or `2` to Reject.
5. **Interactive Q&A Session**:
   - Ask technical questions: `"Why is /var/log consuming 42 GB?"`
   - Request commands: `"What read-only command can I run to see top log files?"`
   - Switch hosts on the fly: `"Switch to VM001"` or `"Check VM003"`
   - Exit the application: Type `exit` or `quit`.
