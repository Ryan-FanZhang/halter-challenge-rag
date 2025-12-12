"""
AI Agent Interactive Script
Main entry point for the intelligent agent with RAG, API, and Ticket tools.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

from src.agents import AgentOrchestrator


def print_welcome():
    """Print welcome message."""
    print("=" * 70)
    print("🤖 AI Support Agent")
    print("   Powered by LangChain ReAct Agent")
    print("=" * 70)
    print("\n📚 Available Capabilities:")
    print("   • Knowledge Base (RAG) - Technical docs, concepts, best practices")
    print("   • API Queries - Agent status, token usage, billing, errors")
    print("   • Human Escalation - Create support tickets")
    print("\n💡 Example Questions:")
    print("   • What is the difference between workflows and agents?")
    print("   • How many tokens have I used this month?")
    print("   • What is the status of my agents?")
    print("   • I want to talk to a human")
    print("\n⌨️  Commands:")
    print("   • Type your question to get help")
    print("   • 'clear' - Clear conversation history")
    print("   • 'history' - Show conversation history")
    print("   • 'quit' or 'q' - Exit")
    print("=" * 70)


def print_response(response):
    """Print agent response in a formatted way."""
    source_emoji = {
        "rag": "📚",
        "api": "🔌",
        "ticket": "📋",
        "agent": "🤖",
    }
    
    # Show tool calls first (transparency)
    if response.tool_calls and len(response.tool_calls) > 0:
        print("\n🔧 Tool Execution:")
        print("─" * 60)
        for i, tc in enumerate(response.tool_calls, 1):
            print(f"   [{i}] {tc['tool']}")
            if tc.get('input'):
                input_str = str(tc['input'])[:100]
                print(f"       Input: {input_str}{'...' if len(str(tc['input'])) > 100 else ''}")
        print("─" * 60)
    
    emoji = source_emoji.get(response.source, "💬")
    
    print(f"\n{emoji} Response [{response.source.upper()}]")
    print("─" * 60)
    print(response.answer)
    
    if response.ticket_id:
        print(f"\n🎫 Ticket ID: {response.ticket_id}")
    
    print("─" * 60)


def main():
    print_welcome()
    
    # Initialize agent
    print("\n⏳ Initializing agent...")
    agent = AgentOrchestrator(
        model="gpt-4o-mini",
        temperature=0.1,
        memory_window=10,
        verbose=False,  # Set to True to see agent reasoning
    )
    print("✅ Agent ready!\n")
    
    while True:
        try:
            user_input = input("\n❓ You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ['quit', 'q', 'exit']:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            agent.reset_memory()
            print("🗑️  Conversation history cleared")
            continue
        
        if user_input.lower() == 'history':
            history = agent.get_conversation_history()
            if not history:
                print("📭 No conversation history")
            else:
                print("\n📜 Conversation History:")
                print("─" * 40)
                for msg in history:
                    role = "You" if msg["role"] == "user" else "Agent"
                    content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                    print(f"{role}: {content}\n")
            continue
        
        if user_input.lower() == 'verbose':
            agent.verbose = not agent.verbose
            agent.agent_executor.verbose = agent.verbose
            print(f"🔊 Verbose mode: {'ON' if agent.verbose else 'OFF'}")
            continue
        
        # Process question
        print("\n🔄 Processing...")
        response = agent.ask(user_input)
        print_response(response)


if __name__ == "__main__":
    main()

