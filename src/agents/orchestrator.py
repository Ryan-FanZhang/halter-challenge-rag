"""
Agent Orchestrator
Main agent that routes queries and manages tool execution using Tool Calling.
"""

import os
from typing import Any, Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from loguru import logger

from .tools import rag_tool, query_api_tool, ticket_tool

load_dotenv()


# System prompt for the orchestrator
SYSTEM_PROMPT = """You are an intelligent AI support assistant that helps users with questions about AI agents and their systems.

## Tool Selection Guidelines:

1. **rag_knowledge_base**: Use this for:
   - Questions about agent concepts, definitions, architectures
   - Technical documentation queries ("What is...", "How does... work", "Explain...")
   - Best practices and recommendations

2. **query_api**: Use this for:
   - Agent status checks ("Is my agent running?", "Show me my agents")
   - Token usage queries ("How many tokens have I used?")
   - Billing and quota information
   - System status and metrics
   - Error logs and recent incidents

3. **escalate_ticket**: Use this when:
   - User explicitly asks for human help
   - You cannot answer the question with available tools
   - User seems frustrated

## Important Rules:

1. Analyze the user's question to determine the best tool
2. For status/metrics questions, use query_api
3. For concept/documentation questions, use rag_knowledge_base
4. Be helpful and provide clear responses
5. If you're unsure, ask the user for clarification

Respond in a friendly and helpful manner."""


@dataclass
class AgentResponse:
    """Response from the agent."""
    answer: str
    source: str  # rag, api, ticket, agent
    success: bool = True
    tool_calls: list[dict] = field(default_factory=list)
    should_escalate: bool = False
    ticket_id: Optional[str] = None


class AgentOrchestrator:
    """
    Main Agent Orchestrator using ReAct pattern.
    
    Routes queries to appropriate tools and manages conversation.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        memory_window: int = 10,
        verbose: bool = True,
    ):
        """
        Initialize Agent Orchestrator.
        
        Args:
            model: LLM model to use
            temperature: Model temperature
            memory_window: Number of conversation turns to remember
            verbose: Whether to show agent reasoning
        """
        self.model_name = model
        self.verbose = verbose
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
        )
        
        # Initialize tools
        self.tools = [rag_tool, query_api_tool, ticket_tool]
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=memory_window,
        )
        
        # Create agent
        self._create_agent()
    
    def _create_agent(self):
        """Create the Tool Calling agent."""
        # Create prompt with chat history
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent using tool calling (more reliable than ReAct)
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        # Create executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True,
        )
    
    def _check_human_request(self, question: str) -> bool:
        """Check if user is explicitly requesting human help."""
        # More specific phrases to avoid false positives
        human_phrases = [
            "talk to human", "talk to a human", "speak to human",
            "human support", "human agent", "real person", "real human",
            "transfer me", "transfer to", "customer service",
            "live chat", "live agent", "support agent",
            "escalate", "i want a human", "need a human", "get me a human",
        ]
        question_lower = question.lower()
        return any(phrase in question_lower for phrase in human_phrases)
    
    def ask(self, question: str) -> AgentResponse:
        """
        Process a user question.
        
        Args:
            question: User's question
            
        Returns:
            AgentResponse with answer and metadata
        """
        logger.info(f"Processing question: {question[:50]}...")
        
        # Check for explicit human request
        if self._check_human_request(question):
            logger.info("🎫 Routing: User explicitly requested human support")
            result = ticket_tool._run(
                reason="user_request",
                user_question=question,
                priority="medium",
                additional_context=self._get_conversation_context(),
            )
            return AgentResponse(
                answer=result["answer"],
                source="ticket",
                success=result["success"],
                ticket_id=result.get("ticket_id"),
            )
        
        try:
            # Run agent
            logger.info("🤖 Running agent to determine best tool...")
            result = self.agent_executor.invoke({"input": question})
            
            # Parse result
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Extract tool calls with transparency logging
            tool_calls = []
            should_escalate = False
            source = "agent"
            
            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    tool_name = getattr(action, "tool", "unknown")
                    tool_input = getattr(action, "tool_input", {})
                    
                    # Log tool selection
                    logger.info(f"🔧 Tool Selected: {tool_name}")
                    logger.info(f"   Input: {tool_input}")
                    
                    tool_calls.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "output": observation if isinstance(observation, dict) else str(observation)[:200],
                    })
                    
                    # Check if any tool suggests escalation
                    if isinstance(observation, dict):
                        if observation.get("should_escalate"):
                            should_escalate = True
                            logger.info(f"   ⚠️ Tool suggests escalation: {observation.get('escalate_reason')}")
                        if observation.get("source"):
                            source = observation["source"]
                        if observation.get("confidence"):
                            logger.info(f"   📊 Confidence: {observation.get('confidence')}")
            
            # Log if no tools were called
            if not tool_calls:
                logger.info("💬 No tools called - agent responded directly")
            
            # If escalation suggested but not yet done, create ticket
            if should_escalate and source != "ticket":
                logger.info("🎫 Escalation triggered - creating ticket...")
                ticket_result = ticket_tool._run(
                    reason="low_confidence",
                    user_question=question,
                    priority="medium",
                    additional_context=f"Agent response: {output}\n\nTool calls: {tool_calls}",
                )
                output += f"\n\n---\n\n{ticket_result['answer']}"
                return AgentResponse(
                    answer=output,
                    source="ticket",
                    success=True,
                    tool_calls=tool_calls,
                    should_escalate=True,
                    ticket_id=ticket_result.get("ticket_id"),
                )
            
            logger.info(f"✅ Response source: {source}")
            
            return AgentResponse(
                answer=output,
                source=source,
                success=True,
                tool_calls=tool_calls,
                should_escalate=should_escalate,
            )
            
        except Exception as e:
            logger.error(f"❌ Agent error: {e}")
            
            # Return error without auto-escalating (let user decide)
            return AgentResponse(
                answer=f"I encountered an error processing your request: {str(e)}\n\nWould you like me to create a support ticket?",
                source="agent",
                success=False,
                should_escalate=True,
            )
    
    def _get_conversation_context(self) -> str:
        """Get recent conversation history as context."""
        try:
            messages = self.memory.chat_memory.messages
            if not messages:
                return "No previous conversation"
            
            context_parts = []
            for msg in messages[-6:]:  # Last 3 exchanges
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                content = msg.content[:500]  # Truncate long messages
                context_parts.append(f"{role}: {content}")
            
            return "\n".join(context_parts)
        except:
            return "Unable to retrieve conversation history"
    
    def reset_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_conversation_history(self) -> list[dict]:
        """Get conversation history as list of dicts."""
        try:
            messages = self.memory.chat_memory.messages
            return [
                {
                    "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content,
                }
                for msg in messages
            ]
        except:
            return []

