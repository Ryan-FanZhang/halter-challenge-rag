"""
Ticket Escalate Tool
Tool for creating support tickets and escalating to human agents.
"""

import os
import json
from datetime import datetime
from typing import Any, Type, Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class TicketToolInput(BaseModel):
    """Input schema for Ticket tool."""
    reason: str = Field(
        description="Reason for creating ticket: 'user_request' (user asked for human), "
                   "'rag_no_answer' (knowledge base couldn't answer), "
                   "'api_no_data' (API had no data), "
                   "'low_confidence' (answer confidence too low), "
                   "'error' (system error occurred)"
    )
    user_question: str = Field(
        description="The original question from the user"
    )
    priority: Optional[str] = Field(
        default="medium",
        description="Ticket priority: 'low', 'medium', 'high'"
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Any additional context or conversation history"
    )


class TicketTool(BaseTool):
    """
    Ticket Escalation Tool for creating support tickets.
    
    Use this tool when:
    - User explicitly asks for human support
    - RAG and API tools cannot provide an answer
    - Answer confidence is below threshold
    - An error occurred during processing
    """
    
    name: str = "escalate_ticket"
    description: str = """Create a support ticket and escalate to human agent.
Use this tool when:
- User explicitly requests human support (says "human", "agent", "talk to someone", etc.)
- The knowledge base (RAG) cannot answer the question
- The API query returns no data
- Previous answer had low confidence
- An error occurred that requires human intervention

Always provide:
- reason: Why the ticket is being created
- user_question: The original question
- priority: 'low', 'medium', or 'high'
- additional_context: Any relevant conversation history or error details"""
    
    args_schema: Type[BaseModel] = TicketToolInput
    
    # Configuration
    logs_dir: Path = Path("logs")
    ticket_counter: int = 0
    
    def __init__(self, logs_dir: str = "logs", **kwargs):
        super().__init__(**kwargs)
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize counter based on existing tickets
        self.ticket_counter = self._get_next_ticket_number()
    
    def _get_next_ticket_number(self) -> int:
        """Get the next ticket number based on existing files."""
        existing_tickets = list(self.logs_dir.glob("ticket_*.json"))
        if not existing_tickets:
            return 1
        
        numbers = []
        for ticket_file in existing_tickets:
            try:
                # Extract number from filename like ticket_20241212_001.json
                parts = ticket_file.stem.split("_")
                if len(parts) >= 3:
                    numbers.append(int(parts[-1]))
            except (ValueError, IndexError):
                continue
        
        return max(numbers, default=0) + 1
    
    def _run(
        self,
        reason: str,
        user_question: str,
        priority: str = "medium",
        additional_context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a support ticket."""
        try:
            # Generate ticket ID
            now = datetime.now()
            date_str = now.strftime("%Y%m%d")
            ticket_id = f"TKT-{date_str}-{self.ticket_counter:03d}"
            
            # Map reason to human-readable
            reason_map = {
                "user_request": "User requested human support",
                "rag_no_answer": "Knowledge base could not answer",
                "api_no_data": "API returned no data",
                "low_confidence": "Answer confidence below threshold",
                "error": "System error occurred",
            }
            
            # Create ticket data
            ticket = {
                "ticket_id": ticket_id,
                "created_at": now.isoformat(),
                "status": "open",
                "priority": priority,
                "reason": reason,
                "reason_description": reason_map.get(reason, reason),
                "user_question": user_question,
                "additional_context": additional_context,
                "assigned_to": None,
                "resolution": None,
                "resolved_at": None,
            }
            
            # Save to file
            filename = f"ticket_{date_str}_{self.ticket_counter:03d}.json"
            filepath = self.logs_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(ticket, f, indent=2, ensure_ascii=False)
            
            # Increment counter
            self.ticket_counter += 1
            
            # Format response
            priority_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(priority, "⚪")
            
            return {
                "success": True,
                "source": "ticket",
                "ticket_id": ticket_id,
                "answer": (
                    f"📋 **Support Ticket Created**\n\n"
                    f"**Ticket ID:** {ticket_id}\n"
                    f"**Priority:** {priority_emoji} {priority.upper()}\n"
                    f"**Reason:** {reason_map.get(reason, reason)}\n"
                    f"**Status:** Open\n\n"
                    f"A human agent will review your request and get back to you shortly. "
                    f"Please save your ticket ID for reference."
                ),
                "ticket_data": ticket,
                "filepath": str(filepath),
                "should_escalate": False,  # Already escalated
            }
            
        except Exception as e:
            return {
                "success": False,
                "source": "ticket",
                "answer": f"Failed to create ticket: {str(e)}",
                "error": str(e),
                "should_escalate": False,
            }
    
    async def _arun(
        self,
        reason: str,
        user_question: str,
        priority: str = "medium",
        additional_context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Async version - just calls sync for now."""
        return self._run(reason, user_question, priority, additional_context)


# Create singleton instance
ticket_tool = TicketTool()

