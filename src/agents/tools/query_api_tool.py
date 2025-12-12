"""
Query API Tool
Tool for querying agent status, usage, and metrics from API.
"""

from typing import Any, Type, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from src.data.mock_api_data import query_api


class QueryAPIToolInput(BaseModel):
    """Input schema for Query API tool."""
    query_type: str = Field(
        description="Type of query: 'agents' (list all agents), 'agent_status' (specific agent), "
                   "'usage' (billing/quota), 'tokens' (token usage), 'errors' (recent errors), "
                   "'system' (system status)"
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Agent ID for agent-specific queries (e.g., 'agent_001'). "
                   "Only needed for 'agent_status' or filtering 'errors'."
    )


class QueryAPITool(BaseTool):
    """
    Query API Tool for retrieving agent status and metrics.
    
    Use this tool when user asks about:
    - Agent running status
    - Token consumption/usage
    - Billing and quota information
    - Error logs and incidents
    - System status
    """
    
    name: str = "query_api"
    description: str = """Query the API for agent status, metrics, and usage information.
Use this tool for questions about:
- Agent status (running, stopped, errors)
- Token usage and consumption
- Billing and quota information
- Recent errors or incidents
- System status and uptime

Available query types:
- 'agents': List all agents with summary
- 'agent_status': Get detailed status of a specific agent (requires agent_id)
- 'usage' or 'billing': Get billing and quota information
- 'tokens': Get token usage statistics
- 'errors': Get recent error logs (optional agent_id filter)
- 'system': Get system status

Examples:
- To check all agents: query_type='agents'
- To check token usage: query_type='tokens'
- To check specific agent: query_type='agent_status', agent_id='agent_001'"""
    
    args_schema: Type[BaseModel] = QueryAPIToolInput
    
    def _run(self, query_type: str, agent_id: Optional[str] = None) -> dict[str, Any]:
        """Execute API query."""
        try:
            # Query the mock API
            result = query_api(query_type, agent_id=agent_id)
            
            if not result.get("success"):
                return {
                    "success": False,
                    "source": "api",
                    "answer": f"API query failed: {result.get('error', 'Unknown error')}",
                    "should_escalate": True,
                    "escalate_reason": "api_error",
                    "data": None,
                }
            
            # Format the response based on query type
            data = result.get("data")
            formatted_answer = self._format_response(query_type, data, agent_id)
            
            return {
                "success": True,
                "source": "api",
                "answer": formatted_answer,
                "confidence": "high",
                "data": data,
                "query_type": result.get("query_type"),
                "should_escalate": False,
            }
            
        except Exception as e:
            return {
                "success": False,
                "source": "api",
                "answer": f"Error querying API: {str(e)}",
                "confidence": "low",
                "should_escalate": True,
                "escalate_reason": "error",
                "error": str(e),
            }
    
    def _format_response(self, query_type: str, data: Any, agent_id: Optional[str]) -> str:
        """Format API data into human-readable response."""
        
        if query_type in ["agents", "agent_list"]:
            if not data:
                return "No agents found."
            
            lines = ["Here are your agents:\n"]
            for agent in data:
                status_emoji = "🟢" if agent["status"] == "running" else "🔴"
                lines.append(
                    f"{status_emoji} **{agent['name']}** ({agent['agent_id']})\n"
                    f"   Status: {agent['status']} | "
                    f"Tokens: {agent['total_tokens_used']:,} | "
                    f"Requests: {agent['total_requests']:,}"
                )
            return "\n".join(lines)
        
        elif query_type == "agent_status":
            if isinstance(data, dict):
                agent = data
                status_emoji = "🟢" if agent["status"] == "running" else "🔴"
                return (
                    f"{status_emoji} **{agent['name']}** ({agent['agent_id']})\n\n"
                    f"**Status:** {agent['status']}\n"
                    f"**Model:** {agent['model']}\n"
                    f"**Total Tokens Used:** {agent['total_tokens_used']:,}\n"
                    f"**Total Requests:** {agent['total_requests']:,}\n"
                    f"**Successful:** {agent['successful_requests']:,} | "
                    f"**Failed:** {agent['failed_requests']:,}\n"
                    f"**Error Rate:** {agent['error_rate']*100:.1f}%\n"
                    f"**Avg Response Time:** {agent['avg_response_time_ms']}ms\n"
                    f"**Last Active:** {agent['last_active']}"
                )
            return "Agent information retrieved."
        
        elif query_type in ["usage", "billing"]:
            return (
                f"📊 **Usage & Billing Information**\n\n"
                f"**Current Month Tokens:** {data['current_month_tokens']:,}\n"
                f"**Quota Limit:** {data['quota_limit']:,}\n"
                f"**Usage:** {data['quota_used_percent']:.1f}%\n"
                f"**Remaining Tokens:** {data['remaining_tokens']:,}\n"
                f"**Billing Period:** {data['billing_period_start']} to {data['billing_period_end']}\n"
                f"**Days Remaining:** {data['days_remaining']}\n"
                f"**Estimated Cost:** ${data['estimated_monthly_cost']:.2f} {data['currency']}"
            )
        
        elif query_type == "token_usage":
            return (
                f"📈 **Token Usage Summary**\n\n"
                f"**Total Tokens (All Agents):** {data['total_tokens_all_agents']:,}\n"
                f"**Current Month:** {data['current_month_tokens']:,}\n"
                f"**Quota Limit:** {data['quota_limit']:,}\n"
                f"**Usage:** {data['quota_used_percent']:.1f}%"
            )
        
        elif query_type == "errors":
            if not data:
                return "✅ No recent errors found."
            
            lines = ["⚠️ **Recent Errors:**\n"]
            for err in data:
                resolved = "✅" if err["resolved"] else "❌"
                lines.append(
                    f"{resolved} [{err['timestamp']}] {err['error_type']}\n"
                    f"   Agent: {err['agent_id']} | {err['message']}"
                )
            return "\n".join(lines)
        
        elif query_type == "system_status":
            status_emoji = "🟢" if data["api_status"] == "operational" else "🔴"
            return (
                f"{status_emoji} **System Status**\n\n"
                f"**API Status:** {data['api_status']}\n"
                f"**Latency:** {data['latency_ms']}ms\n"
                f"**Uptime:** {data['uptime_percent']}%\n"
                f"**Last Incident:** {data['last_incident']}"
            )
        
        return f"Query completed. Data: {data}"
    
    async def _arun(self, query_type: str, agent_id: Optional[str] = None) -> dict[str, Any]:
        """Async version - just calls sync for now."""
        return self._run(query_type, agent_id)


# Create singleton instance
query_api_tool = QueryAPITool()

