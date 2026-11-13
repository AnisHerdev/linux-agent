"""
mcp_server.py - Infrastructure Simulation & MCP Server
Part of OpsPilot AI: Agentic Linux & Cloud Infrastructure Monitoring System.

Exposes five read-only monitoring tools via MCP stdio transport.
Uses Pydantic for data validation and schema integrity.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("OpsPilot-Infrastructure-Server")


# ==========================================
# Pydantic Schemas for Data Validation
# ==========================================

class VMStatusModel(BaseModel):
    vm_id: str = Field(..., description="Virtual Machine Identifier (e.g., VM001)")
    status: str = Field(..., description="Availability status: 'alive' or 'dead'")
    hostname: str = Field(..., description="Hostname of the VM")
    os: str = Field(..., description="Operating system distribution")
    uptime: Optional[str] = Field(None, description="Current uptime if available")


class CPUUsageModel(BaseModel):
    vm_id: str
    status: str
    usage_percent: Optional[float] = Field(None, description="CPU utilization percentage")
    core_count: Optional[int] = Field(None, description="Number of virtual CPU cores")
    threshold_level: str = Field(..., description="Health status: Normal, Elevated, Critical, or Offline")


class MemoryUsageModel(BaseModel):
    vm_id: str
    status: str
    total_mb: Optional[int] = Field(None, description="Total system RAM in MB")
    used_mb: Optional[int] = Field(None, description="Allocated RAM in MB")
    free_mb: Optional[int] = Field(None, description="Available free RAM in MB")
    usage_percent: Optional[float] = Field(None, description="RAM utilization percentage")


class DiskUsageModel(BaseModel):
    vm_id: str
    status: str
    total_gb: float = Field(..., description="Total simulated disk capacity in GB")
    used_gb: Optional[float] = Field(None, description="Used disk storage in GB")
    available_gb: Optional[float] = Field(None, description="Free disk storage in GB")
    usage_percent: Optional[float] = Field(None, description="Disk utilization percentage")


class DirectoryItemModel(BaseModel):
    path: str = Field(..., description="Filesystem directory path")
    size_gb: float = Field(..., description="Space consumed in gigabytes")
    description: str = Field(..., description="Contextual description of directory contents")


class LargeDirectoriesModel(BaseModel):
    vm_id: str
    status: str
    directories: List[DirectoryItemModel] = Field(default_factory=list)


class ErrorResponseModel(BaseModel):
    error: str = Field(..., description="Detailed error description")
    vm_id: str


# ==========================================
# Simulated Infrastructure Data
# ==========================================

SIMULATED_VMS: Dict[str, Dict[str, Any]] = {
    "VM001": {
        "status": "alive",
        "hostname": "prod-web-01",
        "os": "Ubuntu 22.04 LTS (Jammy Jellyfish)",
        "uptime": "42 days, 14 hours",
        "cpu": {
            "usage_percent": 35.0,
            "core_count": 4,
            "threshold_level": "Normal"
        },
        "memory": {
            "total_mb": 16384,
            "used_mb": 7372,
            "free_mb": 9012,
            "usage_percent": 45.0
        },
        "disk": {
            "total_gb": 100.0,
            "used_gb": 42.0,
            "available_gb": 58.0,
            "usage_percent": 42.0
        },
        "large_directories": [
            {"path": "/home", "size_gb": 12.4, "description": "User files and application deployments"},
            {"path": "/var/log", "size_gb": 3.2, "description": "Standard rotated service and system logs"},
            {"path": "/var/cache", "size_gb": 2.0, "description": "APT package manager cache"},
            {"path": "/tmp", "size_gb": 1.1, "description": "Active session runtime buffers"}
        ]
    },
    "VM002": {
        "status": "alive",
        "hostname": "prod-db-analytics-02",
        "os": "Ubuntu 22.04 LTS (Jammy Jellyfish)",
        "uptime": "119 days, 03 hours",
        "cpu": {
            "usage_percent": 88.0,
            "core_count": 8,
            "threshold_level": "Critical"
        },
        "memory": {
            "total_mb": 32768,
            "used_mb": 29818,
            "free_mb": 2950,
            "usage_percent": 91.0
        },
        "disk": {
            "total_gb": 100.0,
            "used_gb": 94.0,
            "available_gb": 6.0,
            "usage_percent": 94.0
        },
        "large_directories": [
            {"path": "/var/log", "size_gb": 42.5, "description": "Unrotated high-volume debug logs & stack traces"},
            {"path": "/var/cache", "size_gb": 18.6, "description": "Accumulated package caches and query cache dumps"},
            {"path": "/tmp", "size_gb": 14.2, "description": "Abandoned core dumps, temp tables & scratch buffers"},
            {"path": "/home", "size_gb": 8.1, "description": "Developer scripts and archived container snapshots"}
        ]
    },
    "VM003": {
        "status": "dead",
        "hostname": "staging-worker-03",
        "os": "Debian 12 (Bookworm)",
        "uptime": None,
        "cpu": None,
        "memory": None,
        "disk": {
            "total_gb": 100.0,
            "used_gb": None,
            "available_gb": None,
            "usage_percent": None
        },
        "large_directories": []
    }
}


def _validate_vm_exists(vm_id: str) -> Optional[Dict[str, Any]]:
    """Helper to retrieve and validate VM from predefined store."""
    norm_id = vm_id.strip().upper()
    return SIMULATED_VMS.get(norm_id)


# ==========================================
# MCP Tools (5 Read-Only Monitoring Tools)
# ==========================================

@mcp.tool()
def get_vm_status(vm_id: str) -> dict:
    """Checks whether the specified virtual machine is alive or dead and returns system metadata.
    
    Args:
        vm_id: Unique virtual machine identifier (e.g. VM001, VM002, VM003).
    """
    vm = _validate_vm_exists(vm_id)
    if not vm:
        return ErrorResponseModel(
            error=f"Virtual machine '{vm_id}' not found in infrastructure catalog. Valid IDs: VM001, VM002, VM003.",
            vm_id=vm_id
        ).model_dump()

    return VMStatusModel(
        vm_id=vm_id.strip().upper(),
        status=vm["status"],
        hostname=vm["hostname"],
        os=vm["os"],
        uptime=vm["uptime"]
    ).model_dump()


@mcp.tool()
def get_cpu_usage(vm_id: str) -> dict:
    """Returns CPU utilization percentage and threshold status for the VM.
    
    Args:
        vm_id: Unique virtual machine identifier (e.g. VM001, VM002, VM003).
    """
    vm = _validate_vm_exists(vm_id)
    if not vm:
        return ErrorResponseModel(
            error=f"Virtual machine '{vm_id}' not found.",
            vm_id=vm_id
        ).model_dump()

    if vm["status"] == "dead":
        return CPUUsageModel(
            vm_id=vm_id.strip().upper(),
            status="dead",
            usage_percent=None,
            core_count=None,
            threshold_level="Offline - Host Unreachable"
        ).model_dump()

    cpu_info = vm["cpu"]
    return CPUUsageModel(
        vm_id=vm_id.strip().upper(),
        status="alive",
        usage_percent=cpu_info["usage_percent"],
        core_count=cpu_info["core_count"],
        threshold_level=cpu_info["threshold_level"]
    ).model_dump()


@mcp.tool()
def get_memory_usage(vm_id: str) -> dict:
    """Returns RAM utilization, total, used, and free memory in MB.
    
    Args:
        vm_id: Unique virtual machine identifier (e.g. VM001, VM002, VM003).
    """
    vm = _validate_vm_exists(vm_id)
    if not vm:
        return ErrorResponseModel(
            error=f"Virtual machine '{vm_id}' not found.",
            vm_id=vm_id
        ).model_dump()

    if vm["status"] == "dead":
        return MemoryUsageModel(
            vm_id=vm_id.strip().upper(),
            status="dead",
            total_mb=None,
            used_mb=None,
            free_mb=None,
            usage_percent=None
        ).model_dump()

    mem_info = vm["memory"]
    return MemoryUsageModel(
        vm_id=vm_id.strip().upper(),
        status="alive",
        total_mb=mem_info["total_mb"],
        used_mb=mem_info["used_mb"],
        free_mb=mem_info["free_mb"],
        usage_percent=mem_info["usage_percent"]
    ).model_dump()


@mcp.tool()
def get_disk_usage(vm_id: str) -> dict:
    """Returns disk capacity, used space, available space, and utilization percentage.
    
    Args:
        vm_id: Unique virtual machine identifier (e.g. VM001, VM002, VM003).
    """
    vm = _validate_vm_exists(vm_id)
    if not vm:
        return ErrorResponseModel(
            error=f"Virtual machine '{vm_id}' not found.",
            vm_id=vm_id
        ).model_dump()

    disk_info = vm["disk"]
    if vm["status"] == "dead":
        return DiskUsageModel(
            vm_id=vm_id.strip().upper(),
            status="dead",
            total_gb=disk_info["total_gb"],
            used_gb=None,
            available_gb=None,
            usage_percent=None
        ).model_dump()

    return DiskUsageModel(
        vm_id=vm_id.strip().upper(),
        status="alive",
        total_gb=disk_info["total_gb"],
        used_gb=disk_info["used_gb"],
        available_gb=disk_info["available_gb"],
        usage_percent=disk_info["usage_percent"]
    ).model_dump()


@mcp.tool()
def get_large_directories(vm_id: str) -> dict:
    """Identifies directories consuming the most storage on the virtual machine.
    
    Args:
        vm_id: Unique virtual machine identifier (e.g. VM001, VM002, VM003).
    """
    vm = _validate_vm_exists(vm_id)
    if not vm:
        return ErrorResponseModel(
            error=f"Virtual machine '{vm_id}' not found.",
            vm_id=vm_id
        ).model_dump()

    if vm["status"] == "dead":
        return LargeDirectoriesModel(
            vm_id=vm_id.strip().upper(),
            status="dead",
            directories=[]
        ).model_dump()

    dir_items = [DirectoryItemModel(**d) for d in vm.get("large_directories", [])]
    return LargeDirectoriesModel(
        vm_id=vm_id.strip().upper(),
        status="alive",
        directories=dir_items
    ).model_dump()


# ==========================================
# Server Runner
# ==========================================

if __name__ == "__main__":
    # Runs the MCP server with standard I/O (stdio) transport
    mcp.run(transport="stdio")
