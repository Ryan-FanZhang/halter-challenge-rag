"""
Mock API Data
Simulates real API responses for agent status, usage, and metrics.
"""

from datetime import datetime, timedelta
import random

# Mock Data
MOCK_DATA = {
    "agents": [
        {
            "agent_id": "agent_001",
            "name": "Customer Support Agent",
            "description": "Handles customer inquiries and support tickets",
            "status": "running",
            "created_at": "2024-10-15T09:00:00Z",
            "last_active": "2024-12-12T08:45:00Z",
            "total_tokens_used": 1250000,
            "total_requests": 34200,
            "successful_requests": 33500,
            "failed_requests": 700,
            "avg_response_time_ms": 1250,
            "error_rate": 0.02,
            "model": "gpt-4o-mini",
        },
        {
            "agent_id": "agent_002",
            "name": "Data Analysis Agent",
            "description": "Analyzes data and generates reports",
            "status": "running",
            "created_at": "2024-11-01T14:30:00Z",
            "last_active": "2024-12-12T07:30:00Z",
            "total_tokens_used": 850000,
            "total_requests": 12500,
            "successful_requests": 12400,
            "failed_requests": 100,
            "avg_response_time_ms": 2100,
            "error_rate": 0.008,
            "model": "gpt-4o",
        },
        {
            "agent_id": "agent_003",
            "name": "Code Review Agent",
            "description": "Reviews code and suggests improvements",
            "status": "stopped",
            "created_at": "2024-09-20T11:00:00Z",
            "last_active": "2024-12-10T16:00:00Z",
            "total_tokens_used": 2100000,
            "total_requests": 8900,
            "successful_requests": 8850,
            "failed_requests": 50,
            "avg_response_time_ms": 3500,
            "error_rate": 0.005,
            "model": "gpt-4o",
        },
    ],
    "usage": {
        "current_month_tokens": 4500000,
        "quota_limit": 10000000,
        "quota_used_percent": 45.0,
        "billing_period_start": "2024-12-01",
        "billing_period_end": "2024-12-31",
        "estimated_monthly_cost": 125.50,
        "currency": "USD",
    },
    "recent_errors": [
        {
            "error_id": "err_001",
            "agent_id": "agent_001",
            "timestamp": "2024-12-12T08:30:00Z",
            "error_type": "rate_limit",
            "message": "Rate limit exceeded, request queued",
            "resolved": True,
        },
        {
            "error_id": "err_002",
            "agent_id": "agent_001",
            "timestamp": "2024-12-11T14:20:00Z",
            "error_type": "timeout",
            "message": "Request timed out after 30s",
            "resolved": True,
        },
        {
            "error_id": "err_003",
            "agent_id": "agent_002",
            "timestamp": "2024-12-10T09:15:00Z",
            "error_type": "invalid_input",
            "message": "Invalid JSON in request body",
            "resolved": True,
        },
    ],
    "system_status": {
        "api_status": "operational",
        "latency_ms": 45,
        "uptime_percent": 99.95,
        "last_incident": "2024-11-28T03:00:00Z",
    },
}


def get_agent_list() -> list[dict]:
    """Get list of all agents with summary info."""
    return [
        {
            "agent_id": agent["agent_id"],
            "name": agent["name"],
            "status": agent["status"],
            "total_tokens_used": agent["total_tokens_used"],
            "total_requests": agent["total_requests"],
        }
        for agent in MOCK_DATA["agents"]
    ]


def get_agent_status(agent_id: str | None = None) -> dict | list | None:
    """
    Get agent status by ID or all agents if no ID provided.
    
    Args:
        agent_id: Optional agent ID to filter
        
    Returns:
        Agent details or list of all agents
    """
    if agent_id:
        for agent in MOCK_DATA["agents"]:
            if agent["agent_id"] == agent_id:
                return agent
        return None
    return MOCK_DATA["agents"]


def get_usage_info() -> dict:
    """Get current usage and billing information."""
    usage = MOCK_DATA["usage"].copy()
    usage["remaining_tokens"] = usage["quota_limit"] - usage["current_month_tokens"]
    usage["days_remaining"] = (
        datetime.strptime(usage["billing_period_end"], "%Y-%m-%d") - datetime.now()
    ).days
    return usage


def get_recent_errors(agent_id: str | None = None, limit: int = 10) -> list[dict]:
    """
    Get recent errors, optionally filtered by agent.
    
    Args:
        agent_id: Optional agent ID to filter
        limit: Maximum number of errors to return
        
    Returns:
        List of recent errors
    """
    errors = MOCK_DATA["recent_errors"]
    if agent_id:
        errors = [e for e in errors if e["agent_id"] == agent_id]
    return errors[:limit]


def get_system_status() -> dict:
    """Get system status information."""
    return MOCK_DATA["system_status"]


def query_api(query_type: str, **kwargs) -> dict:
    """
    Main query function for API tool.
    
    Args:
        query_type: Type of query (agents, agent_status, usage, errors, system)
        **kwargs: Additional parameters
        
    Returns:
        Query result
    """
    if query_type == "agents" or query_type == "agent_list":
        return {
            "success": True,
            "data": get_agent_list(),
            "query_type": "agent_list",
        }
    
    elif query_type == "agent_status":
        agent_id = kwargs.get("agent_id")
        result = get_agent_status(agent_id)
        if result is None and agent_id:
            return {
                "success": False,
                "error": f"Agent '{agent_id}' not found",
                "query_type": "agent_status",
            }
        return {
            "success": True,
            "data": result,
            "query_type": "agent_status",
        }
    
    elif query_type == "usage" or query_type == "billing":
        return {
            "success": True,
            "data": get_usage_info(),
            "query_type": "usage",
        }
    
    elif query_type == "errors":
        agent_id = kwargs.get("agent_id")
        limit = kwargs.get("limit", 10)
        return {
            "success": True,
            "data": get_recent_errors(agent_id, limit),
            "query_type": "errors",
        }
    
    elif query_type == "system" or query_type == "system_status":
        return {
            "success": True,
            "data": get_system_status(),
            "query_type": "system_status",
        }
    
    elif query_type == "tokens" or query_type == "token_usage":
        # Aggregate token usage
        total_tokens = sum(a["total_tokens_used"] for a in MOCK_DATA["agents"])
        usage = get_usage_info()
        return {
            "success": True,
            "data": {
                "total_tokens_all_agents": total_tokens,
                "current_month_tokens": usage["current_month_tokens"],
                "quota_limit": usage["quota_limit"],
                "quota_used_percent": usage["quota_used_percent"],
            },
            "query_type": "token_usage",
        }
    
    else:
        return {
            "success": False,
            "error": f"Unknown query type: {query_type}",
            "available_types": ["agents", "agent_status", "usage", "errors", "system", "tokens"],
        }

