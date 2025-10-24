import streamlit as st
import asyncio
import nest_asyncio
import platform
from utils import random_uuid
from datetime import datetime

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Apply nest_asyncio: Allow nested calls within an already running event loop
nest_asyncio.apply()

# Create and reuse global event loop (create once and continue using)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

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
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv
load_dotenv(override=True)
api_key = os.getenv('API_KEY')

# Page config
st.set_page_config(page_title="MicroAI Agent", page_icon="🤖", layout="wide")


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
# Terminal Logger
# ============================================
class TerminalLogger:
    """Logs intermediate steps to session state for display"""
    
    @staticmethod
    def log(message: str, level: str = "INFO"):
        """Add a log entry with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        
        if "terminal_logs" not in st.session_state:
            st.session_state.terminal_logs = []
        
        st.session_state.terminal_logs.append(log_entry)
    
    @staticmethod
    def clear():
        """Clear all logs"""
        st.session_state.terminal_logs = []

# ============================================
# MCP Manager
# ============================================
class MCPManager:
    """
    Manages the lifecycle of an MCP (Model Context Protocol) server connection.
    
    The MCP server runs as a separate process and provides tools
    that the LLM can use through the Model Context Protocol.
    """
    
    def __init__(self, name: str = "MCP"):
        self.session = None
        self.exit_stack = None
        self.tools = None
        self.name = name
    
    async def setup(self, mcp_config):
        """
        Initialize MCP connection and load tools.
        
        Args:
            mcp_config: Dict with 'command' and 'args' for starting MCP server
        
        Returns:
            List of LangChain tools loaded from the MCP server
        """
        TerminalLogger.log(f"🔧 Setting up {self.name} server...", "INFO")
        
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
        
        TerminalLogger.log(f"✅ Loaded {len(self.tools)} tools from {self.name}:", "SUCCESS")
        for tool in self.tools:
            TerminalLogger.log(f"   • {tool.name}: {tool.description}", "INFO")
        
        return self.tools
    
    async def cleanup(self):
        """Close MCP connection and cleanup resources."""
        if self.exit_stack:
            await self.exit_stack.aclose()
            TerminalLogger.log(f"✅ {self.name} connection closed", "INFO")


# ============================================
# Build Graph
# ============================================
async def build_graph(metrics_manager: MCPManager, cross_service_manager: MCPManager, metrics_config: dict, cross_service_config: dict):
    """
    Build the complete LangGraph with MCP integration.
    
    Args:
        metrics_manager: MCPManager instance for metrics
        cross_service_manager: MCPManager instance for cross-service
        mcp_config: Configuration for metrics MCP server
        cross_service_config: Configuration for cross-service MCP server
        
    Returns:
        Compiled LangGraph application
    """
    TerminalLogger.log("🏗️ Building LangGraph...", "INFO")
    
    # Step 1: Setup both MCP servers
    TerminalLogger.log("1️⃣ Setting up Metrics MCP...", "INFO")
    metrics_tools = await metrics_manager.setup(metrics_config)
    
    TerminalLogger.log("2️⃣ Setting up Cross-Service MCP...", "INFO")
    cross_service_tools = await cross_service_manager.setup(cross_service_config)
    
    # Store tools in session state for display
    st.session_state.metrics_tools = metrics_tools
    st.session_state.cross_service_tools = cross_service_tools
    
    # Step 2: Create the LLM
    TerminalLogger.log("3️⃣ Initializing LLM...", "INFO")
    model = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        api_key=os.getenv("API_KEY"),
        temperature=0,
        max_tokens=20000
    )
    
    # Step 3: Create the MCP agent nodes
    TerminalLogger.log("4️⃣ Creating agent nodes...", "INFO")
    metrics_node = MCPNode(mcp_tools=metrics_tools, model=model)
    cross_service_node = MCPNode(mcp_tools=cross_service_tools, model=model)

    # Step 4: Create the graph
    TerminalLogger.log("5️⃣ Building graph structure...", "INFO")
    graph = StateGraph(GraphState)
    
    # Step 5: Add all nodes
    graph.add_node("retrieve", Nodes.retrieve)
    graph.add_node("rerank", Nodes.rerank)
    graph.add_node("grade_documents", Nodes.grade_documents)
    graph.add_node("generate", Nodes.generate)
    graph.add_node("transform_query", Nodes.transform_query)
    graph.add_node("cross_service_correlation", cross_service_node.cross_service_node)
    graph.add_node("metrics_analysis", metrics_node.metrics_node)
    
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
    graph.add_edge("cross_service_correlation", "metrics_analysis")
    graph.add_edge("metrics_analysis", "generate")
    
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
    TerminalLogger.log("✅ Graph built successfully!", "SUCCESS")
    return graph.compile()

# ============================================
# Initialize Retriever
# ============================================
def initialize_retriever(folder_path):
    """Initialize the retriever once at startup"""
    try:
        TerminalLogger.log(f"📂 Initializing retriever with folder: {folder_path}", "INFO")
        retriever = RetrieverManager.initialize(folder_path, api_key)
        TerminalLogger.log("✅ Retriever initialized successfully!", "SUCCESS")
        return True
    except Exception as e:
        TerminalLogger.log(f"❌ Retriever initialization failed: {str(e)}", "ERROR")
        return False

# ============================================
# Process Query with Logging
# ============================================
async def process_query_with_logging(graph: StateGraph, question: str, log_folder: str, metrics_csv: str):
    """Process a user query with detailed logging."""
    TerminalLogger.log("=" * 60, "INFO")
    TerminalLogger.log("🚀 Starting query processing", "INFO")
    TerminalLogger.log("=" * 60, "INFO")
    
    # Prepare initial state
    initial_state = {
        "path": log_folder,
        "question": question,
        "metrics_csv_path": metrics_csv,
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
    
    TerminalLogger.log(f"📝 Question: {question}", "INFO")
    TerminalLogger.log(f"📂 Log folder: {log_folder}", "INFO")
    TerminalLogger.log(f"📊 Metrics CSV: {metrics_csv}", "INFO")
    
    # Run the graph
    TerminalLogger.log("\n🏃 Running graph pipeline...", "INFO")
    result = await graph.ainvoke(initial_state, config=RunnableConfig(recursion_limit=100))
    
    # Log completion
    TerminalLogger.log("\n" + "=" * 60, "INFO")
    TerminalLogger.log("✅ Pipeline completed successfully!", "SUCCESS")
    TerminalLogger.log("=" * 60, "INFO")
    
    return result


# ============================================
# Streamlit UI Components
# ============================================

def render_terminal():
    """Render the live terminal section"""
    st.subheader("🖥️ Live Terminal Output")
    
    terminal_container = st.container()
    with terminal_container:
        if "terminal_logs" in st.session_state and st.session_state.terminal_logs:
            # Create terminal-like display
            terminal_text = ""
            for log in st.session_state.terminal_logs[-50:]:  # Show last 50 logs
                level_color = {
                    "INFO": "🔵",
                    "SUCCESS": "🟢",
                    "WARNING": "🟡",
                    "ERROR": "🔴"
                }.get(log["level"], "⚪")
                
                terminal_text += f"{level_color} [{log['timestamp']}] {log['message']}\n"
            
            st.code(terminal_text, language=None)
        else:
            st.info("Terminal output will appear here during processing...")


def render_mcp_tools():
    """Render the MCP tools list"""
    st.subheader("🔧 Available MCP Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Metrics Tools**")
        if "metrics_tools" in st.session_state and st.session_state.metrics_tools:
            for tool in st.session_state.metrics_tools:
                with st.expander(f"🛠️ {tool.name}", expanded=False):
                    st.markdown(f"**Description:** {tool.description}")
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        st.markdown("**Parameters:**")
                        st.json(tool.args_schema.schema() if hasattr(tool.args_schema, 'schema') else {})
        else:
            st.info("Initialize agent to see tools")
    
    with col2:
        st.markdown("**Cross-Service Tools**")
        if "cross_service_tools" in st.session_state and st.session_state.cross_service_tools:
            for tool in st.session_state.cross_service_tools:
                with st.expander(f"🛠️ {tool.name}", expanded=False):
                    st.markdown(f"**Description:** {tool.description}")
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        st.markdown("**Parameters:**")
                        st.json(tool.args_schema.schema() if hasattr(tool.args_schema, 'schema') else {})
        else:
            st.info("Initialize agent to see tools")


def render_analysis_sections(result):
    """Render the analysis output sections"""
    st.subheader("📊 Analysis Results")
    
    # Create tabs for different outputs
    tab1, tab2, tab3 = st.tabs(["💬 Final Answer", "📈 Metrics Analysis", "🔗 Cross-Service Analysis"])
    
    with tab1:
        st.markdown("### Final Answer")
        if result and "generation" in result:
            st.markdown(result["generation"])
        else:
            st.info("No answer generated yet")
    
    with tab2:
        st.markdown("### Metrics Analysis")
        if result and "metrics_analysis" in result and result["metrics_analysis"]:
            st.markdown(result["metrics_analysis"])
            
            # Show additional metrics metadata
            if "target_service" in result and result["target_service"]:
                st.markdown(f"**Target Service:** `{result['target_service']}`")
            if "error_time" in result and result["error_time"]:
                st.markdown(f"**Error Time:** `{result['error_time']}`")
            if "endpoints" in result and result["endpoints"]:
                st.markdown(f"**Endpoints:** `{result['endpoints']}`")
            if "user_ids" in result and result["user_ids"]:
                st.markdown(f"**User IDs:** `{result['user_ids']}`")
        else:
            st.info("No metrics analysis available")
    
    with tab3:
        st.markdown("### Cross-Service Correlation")
        if result and "correlated_findings" in result and result["correlated_findings"]:
            st.markdown(str(result["correlated_findings"]))
        else:
            st.info("No cross-service analysis available")
        
        # # Show documents used
        # if result and "documents" in result and result["documents"]:
        #     with st.expander("📄 Retrieved Documents", expanded=False):
        #         for i, doc in enumerate(result["documents"], 1):
        #             st.markdown(f"**Document {i}:**")
        #             st.text(doc[:500] + "..." if len(doc) > 500 else doc)
        #             st.divider()


# ============================================
# Session State Initialization
# ============================================
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False
    st.session_state.agent = None
    st.session_state.history = []
    st.session_state.mcp_manager = None
    st.session_state.cross_service_manager = None
    st.session_state.terminal_logs = []
    st.session_state.metrics_tools = []
    st.session_state.cross_service_tools = []
    st.session_state.cross_service_config = {
        "command": "./.venv/bin/python",
        "args": ["mcp_server_cross-services.py"]
    }
    st.session_state.metrics_config = {
        "command": "./.venv/bin/python",
        "args": ["mcp_server_metrics.py"]
    }
    st.session_state.selected_model = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    st.session_state.current_node = ""
    st.session_state.node_status = []
    st.session_state.thread_id = "default_thread"
    st.session_state.current_result = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()

# ============================================
# Sidebar Configuration
# ============================================
st.sidebar.title("⚙️ Settings")

# Model selection
st.session_state.selected_model = st.sidebar.selectbox(
    "🤖 Select Model",
    ["nvidia/llama-3.3-nemotron-super-49b-v1.5"],
    index=0
)

# Initialize button
if st.sidebar.button("🚀 Initialize Agent", type="primary", use_container_width=True):
    with st.spinner("🔄 Initializing agent..."):
        # Clear terminal logs
        TerminalLogger.clear()
        
        # Cleanup old managers
        if st.session_state.mcp_manager:
            st.session_state.event_loop.run_until_complete(
                st.session_state.mcp_manager.cleanup()
            )

        if st.session_state.cross_service_manager:
            st.session_state.event_loop.run_until_complete(
                st.session_state.cross_service_manager.cleanup()
            )
        
        # Create new managers
        st.session_state.mcp_manager = MCPManager("Metrics MCP")
        st.session_state.cross_service_manager = MCPManager("Cross-Service MCP")
        
        try:
            if initialize_retriever("data/logs/update_password_404_user_not_found"):
                TerminalLogger.log("✅ Retriever initialized successfully!", "SUCCESS")
            else:
                TerminalLogger.log("❌ Retriever initialization failed!", "ERROR")
    
            st.session_state.agent = st.session_state.event_loop.run_until_complete(
                build_graph(
                    st.session_state.mcp_manager,
                    st.session_state.cross_service_manager,
                    st.session_state.metrics_config,
                    st.session_state.cross_service_config
                )
            )
            st.session_state.session_initialized = True
            st.sidebar.success("✅ Agent initialized!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Initialization failed: {str(e)}")
            TerminalLogger.log(f"❌ Initialization failed: {str(e)}", "ERROR")

st.sidebar.divider()

# Reset button
if st.sidebar.button("🔄 Reset Conversation", use_container_width=True):
    st.session_state.history = []
    st.session_state.node_status = []
    st.session_state.current_result = None
    TerminalLogger.clear()
    st.success("✅ Conversation reset")
    st.rerun()

# ============================================
# Main UI Layout
# ============================================
st.title("🤖 MicroAIgent")
st.markdown("✨ AI-Powered Microservices Log Analysis")

# Status indicator
if st.session_state.session_initialized:
    st.success("✅ Agent is ready")
else:
    st.info("ℹ️ Click 'Initialize Agent' in the sidebar to start")

# Create main layout with tabs
main_tab1, main_tab2, main_tab3 = st.tabs(["💬 Chat", "🔧 MCP Tools", "🖥️ Terminal"])

with main_tab1:
    # Display chat history
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])
                
                # Show processing steps
                if "steps" in message:
                    with st.expander("🔍 Processing Steps", expanded=False):
                        for step in message["steps"]:
                            st.markdown(f"- {step}")

    # User input
    user_query = st.chat_input("💬 Enter your question")

    if user_query:
        if not st.session_state.session_initialized:
            st.warning("⚠️ Please initialize the agent first")
        else:
            # Display user message
            st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
            
            # Process query
            with st.chat_message("assistant", avatar="🤖"):
                status_placeholder = st.empty()
                answer_placeholder = st.empty()
                
                # Reset node status and terminal logs
                st.session_state.node_status = []
                st.session_state.current_node = ""
                TerminalLogger.clear()
                
                try:
                    # Run the query with logging
                    async def run_query():
                        result = await process_query_with_logging(
                            st.session_state.agent,
                            user_query,
                            "data/logs/update_password_404_user_not_found",
                            "data/metrics/light-oauth2-data-1719771248.csv"
                        )
                        return result
                    
                    # Update status while processing
                    status_placeholder.info("🔄 Processing your query...")
                    result = st.session_state.event_loop.run_until_complete(run_query())
                    status_placeholder.empty()
                    
                    # Store result
                    st.session_state.current_result = result
                    
                    # Display answer
                    answer_placeholder.markdown(result["generation"])
                    
                    # Save to history
                    st.session_state.history.append({
                        "role": "user",
                        "content": user_query
                    })
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": result["generation"],
                        "steps": st.session_state.node_status.copy()
                    })
                    
                    # Force rerun to update other tabs
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    TerminalLogger.log(f"❌ Error: {str(e)}", "ERROR")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Show analysis results if available
    if st.session_state.current_result:
        st.divider()
        render_analysis_sections(st.session_state.current_result)

with main_tab2:
    render_mcp_tools()

with main_tab3:
    render_terminal()
    
    # Auto-refresh option
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()