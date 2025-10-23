from langgraph.graph import END, StateGraph, START
from graphnodes import Nodes
from graphedges import Edge
from mcpnodes import MCPNode
from typing_extensions import TypedDict
import asyncio
from contextlib import AsyncExitStack
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, START, END
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import MemorySaver
from multiagent import RetrieverManager
import os
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv('API_KEY')

class GraphState(TypedDict):
    """
    The state that flows through our LangGraph pipeline.
    
    Each node can read from and update this state.
    """
    # Input
    path: str                    # Path to log file
    question: str                # User's question (can be enriched by nodes)
    metrics_csv_path: str        # Path to metrics CSV
    
    # Processing
    documents: list[str]         # Retrieved documents
    metrics_analysis: str
    iterations: int              # Number of query transformations
    
    # Output
    generation: str              # Final answer from LLM
    
    # Optional metadata
    target_service: str          # Which microservice to analyze
    error_time: str
    endpoints: str  # Comma-separated
    user_ids: str   # Comma-separated
    correlated_findings: dict

# ============================================
# MCP Manager
# ============================================
class MCPManager:
    """
    Manages the lifecycle of an MCP (Model Context Protocol) server connection.
    
    The MCP server runs as a separate process and provides tools
    that the LLM can use through the Model Context Protocol.
    """
    
    def __init__(self):
        self.session = None
        self.exit_stack = None
        self.tools = None
    
    async def setup(self, mcp_config):
        """
        Initialize MCP connection and load tools.
        
        Args:
            mcp_config: Dict with 'command' and 'args' for starting MCP server
        
        Returns:
            List of LangChain tools loaded from the MCP server
        """
        # Configure how to start the MCP server process
        server_params = StdioServerParameters(
            command=mcp_config.get("command", "./.venv/bin/python"),
            args=mcp_config.get("args", ["mcp_server_metrics.py"]),
        )
        
        # Use AsyncExitStack to manage async context managers
        self.exit_stack = AsyncExitStack()
        
        # Start the MCP server and get read/write streams
        read, write = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        
        # Create a session to communicate with the MCP server
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        
        # Initialize the session
        await self.session.initialize()
        
        # Load tools from the MCP server
        # These become LangChain tools that can be used by agents
        self.tools = await load_mcp_tools(self.session)
        
        print(f"✅ Loaded {len(self.tools)} tools from MCP server:")
        for tool in self.tools:
            print(f"   • {tool.name}")
        
        return self.tools
    
    async def cleanup(self):
        """Close MCP connection and cleanup resources."""
        if self.exit_stack:
            await self.exit_stack.aclose()
            print("✅ MCP connection closed")


# ============================================
# Build Graph
# ============================================
async def build_graph(metrics_manager: MCPManager, cross_service_manager: MCPManager, mcp_config: dict, cross_service_config: dict):
    """
    Build the complete LangGraph with MCP integration.
    
    Args:
        mcp_manager: MCPManager instance
        mcp_config: Configuration for MCP server
        
    Returns:
        Compiled LangGraph application
    """
    print("🏗️ Building LangGraph...")
    
    # Step 1: Setup both MCP servers
    print("\n1️⃣ Setting up Metrics MCP...")
    metrics_tools = await metrics_manager.setup(mcp_config)
    print("\n2️⃣ Setting up Cross-Service MCP...")
    cross_service_tools = await cross_service_manager.setup(cross_service_config)
    
    
    
    # Step 2: Create the LLM
    model = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        api_key=os.getenv("API_KEY"),  # Same key you're using
        temperature=0,
        max_tokens=20000
    )
    
    
    # Step 3: Create the MCP agent node
    metrics_node = MCPNode(mcp_tools=metrics_tools, model=model)
    cross_service_node = MCPNode(mcp_tools=cross_service_tools, model=model)

    # Step 4: Create the graph
    graph = StateGraph(GraphState)
    
    # Step 5: Add all nodes
    graph.add_node("retrieve", Nodes.retrieve)
    graph.add_node("rerank", Nodes.rerank)
    graph.add_node("grade_documents", Nodes.grade_documents)
    graph.add_node("generate", Nodes.generate)
    graph.add_node("transform_query", Nodes.transform_query)
    graph.add_node("cross_service_correlation", cross_service_node.cross_service_node)
    graph.add_node("mcp_agent", metrics_node.metrics_node)  # MCP-powered node!
    
    # Step 6: Define the flow
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        Edge.decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "cross_service_correlation",
        },
    )
    
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("cross_service_correlation", "mcp_agent")
    graph.add_edge("mcp_agent", "generate")
    
    # Final quality check
    graph.add_conditional_edges(
        "generate",
        Edge.grade_generation_vs_documents_and_question,
        {
            "not supported": "transform_query",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    
    # Step 7: Compile the graph
    print("✅ Graph built successfully!")
    return graph.compile()

# ============================================
# Initialize Retriever
# ============================================
def initialize_retriever(folder_path):
    """Initialize the retriever once at startup"""
    try:
        retriever = RetrieverManager.initialize(folder_path, api_key)
        return True
    except Exception as e:
        return False
    
# ============================================
# Main Execution
# ============================================
async def main():
    """
    Main execution function that runs the complete pipeline.
    """
    # Step 1: Configure MCP Server
    mcp_config = {
        "command": "./.venv/bin/python",
        "args": ["mcp_server_metrics.py"],
    }
    cross_service_config = {
        "command": "./.venv/bin/python",
        "args": ["mcp_server_cross-services.py"],
    }  

    # Step 2: Create MCP Manager
    metrics_manager = MCPManager()
    cross_service_manager = MCPManager()  
        
    try:

        if initialize_retriever("data/logs/update_password_404_user_not_found"):
            print("✅ Retriever initialized successfully!")
        else:
            print("❌ Retriever initialization failed!")
            return        
        
        # Step 3: Build the graph
        graph = await build_graph(
            metrics_manager=metrics_manager,
            cross_service_manager=cross_service_manager,
            mcp_config=mcp_config,
            cross_service_config=cross_service_config
        )
        
        # Step 4: Prepare initial state
        initial_state = {
            "path": "data/logs/update_password_404_user_not_found",
            "question": "What are the critical errors in the log files? Summarize the metrics analysis, and explain how the metrics correlate with potential issues",
            "metrics_csv_path": "data/metrics/light-oauth2-data-1719771248.csv",
            "documents": [],
            "metrics_analysis": "",
            "iterations": 0,
            "generation": "",
            "target_service": "",
            "error_time": "",
            "endpoints": "",
            "user_ids": "",
            "correlated_findings": {}
        }
        
        print("\n📝 Initial Question:")
        print(f"   {initial_state['question']}")
        
        # Step 5: Run the graph
        print("\n🏃 Running graph pipeline...\n")
        result = await graph.ainvoke(initial_state)
        
        # Step 6: Display results
        print("\n" + "=" * 60)
        print("✅ FINAL RESULT")
        print("=" * 60)
        print(f"\n📄 Generation:\n{result['generation']}\n")
        
        print("=" * 60)
        print("🎉 Pipeline completed successfully!")
        print("=" * 60)
        
    finally:
        # Step 7: Always cleanup
        try:
            await metrics_manager.cleanup()
        except:
            pass
        try:
            await cross_service_manager.cleanup()
        except:
            pass
        

if __name__ == "__main__":
    asyncio.run(main())