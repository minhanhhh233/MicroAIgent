from datetime import datetime
import re
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# System prompt for the MCP agent
SYSTEM_MESSAGE_TEMPLATE = """You are an expert log and metrics analyst for OAuth2 microservices.

You have access to:
1. **Log Analysis**: Document retrieval from vector database (logs are pre-loaded)
2. **Metrics Analysis**: Real-time metrics tools from MCP server

WORKFLOW:
1. First, understand WHAT errors occurred from the logs
2. Then, check metrics to understand WHY (root cause)
3. Finally, synthesize findings into actionable insights

METRICS CONTEXT:
- Current metrics CSV: {metrics_csv_path}
- You can check: memory, GC, goroutines, CPU, system resources
- Use diagnose_error_correlation to link errors to performance issues

EXAMPLE WORKFLOW:
User: "Why did update_password fail with 404?"
1. Check metrics: get_go_memory_stats, get_go_gc_stats
2. Correlate: diagnose_error_correlation("404 user not found")
3. Conclude: "Memory pressure → slow queries → timeouts → 404s"

BE PROACTIVE: When you see errors, always check relevant metrics without being asked!

Output a 2-3 sentence summary of your findings.
"""


class MCPNode:
    """
    A LangGraph node that uses a ReAct agent with MCP tools.
    
    This node analyzes metrics and enriches the question with context
    that will be used by downstream nodes.
    """
    
    def __init__(self, mcp_tools, model):
        """
        Initialize the MCP node with tools and model.
        
        Args:
            mcp_tools: List of MCP tools loaded from the server
            model: LangChain LLM (e.g., ChatAnthropic)
        """
        self.mcp_tools = mcp_tools
        self.model = model
        
        # Create a ReAct agent that can use MCP tools
        # ReAct = Reasoning + Acting pattern
        # The agent will: Think → Act (use tool) → Observe → Repeat
        self.mcp_agent = create_react_agent(
            model=self.model,
            tools=self.mcp_tools,
            checkpointer=MemorySaver()  # Remember conversation history
        )
        
    async def metrics_node(self, state) -> dict:
        """
        The actual node function that processes the state.
        
        Args:
            state: Current GraphState
            
        Returns:
            Dict with updated state fields
        """
        print("🔍 --- MCP AGENT NODE ---")
        
        # Extract information from state
        question = state["question"]
        metrics_csv_path = state["metrics_csv_path"]
        documents = state["documents"]
        
        # Format the system message with dynamic context
        system_message_content = SYSTEM_MESSAGE_TEMPLATE.format(
            metrics_csv_path=metrics_csv_path
        )
        
        # Build context from retrieved log documents
        log_context = ""
        if documents:
            # Extract content from Document objects properly
            log_entries = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                else:
                    content = str(doc)
                log_entries.append(content)
            
            log_context = "\n\nRelevant logs:\n" + "\n---\n".join(log_entries)
        
        # Construct the full prompt for the agent
        agent_prompt = f"""{system_message_content}
        User question: {question}{log_context}
        Analyze the metrics and provide a brief summary of key findings."""
        
        # Run the ReAct agent
        # The agent will autonomously decide which tools to call
        result = await self.mcp_agent.ainvoke(
            {"messages": [HumanMessage(content=agent_prompt)]},
            config={"configurable": {"thread_id": "mcp_agent"}}
        )
        
        # Log which tools the agent used
        print("\n🔧 Tools used by agent:")
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"   → {tool_call['name']}")
        
        # Extract the agent's analysis
        metrics_analysis = result["messages"][-1].content
        
        # Return updated state
        # This allows downstream nodes to use the metrics analysis insights
        return {
            "metrics_analysis": metrics_analysis
        }


    def extract_primary_service(self, documents) -> str:
        """Extract primary service from document metadata."""
        # Get from first document's metadata
        if documents and len(documents) > 0:
            if hasattr(documents[0], 'metadata'):
                return documents[0].metadata.get('service', 'unknown')
        return 'unknown'
    
    def extract_error_time(self, documents) -> str:
        """Extract earliest error timestamp from documents."""
        earliest_time = None
        
        for doc in documents:
            if hasattr(doc, 'metadata'):
                # Try to get timestamp from metadata
                timestamp = doc.metadata.get('start_time')
                if timestamp:
                    try:
                        time_obj = datetime.strptime(timestamp, "%H:%M:%S.%f").time()
                        if earliest_time is None or time_obj < earliest_time:
                            earliest_time = time_obj
                    except:
                        pass
        
        if earliest_time:
            return earliest_time.isoformat()
        else:
            return None
    
    def extract_endpoints(self, documents) -> str:
        """Extract unique endpoints from documents."""
        endpoints = set()
        
        for doc in documents:
            if hasattr(doc, 'metadata'):
                doc_endpoints = doc.metadata.get('endpoints', [])
                if isinstance(doc_endpoints, list):
                    endpoints.update(doc_endpoints)
        
        return ','.join(sorted(endpoints)) if endpoints else ''
    
    def extract_user_ids(self, documents) -> str:
        """Extract unique user IDs from documents."""
        user_ids = set()
        
        for doc in documents:
            if hasattr(doc, 'metadata'):
                doc_user_ids = doc.metadata.get('user_ids', [])
                if isinstance(doc_user_ids, list):
                    user_ids.update(doc_user_ids)
        
        return ','.join(sorted(user_ids)[:5]) if user_ids else ''  # Limit to 5 IDs
    
    async def cross_service_node(self, state) -> dict:
        """
        Node that analyzes documents and correlates cross-service errors.
        """
        print("🔗 --- CROSS-SERVICE CORRELATION NODE ---")
        
        documents = state.get("documents", [])
        
        if not documents:
            print("⚠️  No documents to analyze")
            return {
                "target_service": "unknown",
                "error_time": datetime.now().isoformat(),
                "endpoints": "",
                "user_ids": "",
                "correlated_findings": {}
            }
        
        # Step 1: Extract correlation parameters from documents
        print("\n📊 Analyzing documents to extract correlation parameters...")
        
        primary_service = self.extract_primary_service(documents)
        error_time = self.extract_error_time(documents)
        endpoints = self.extract_endpoints(documents)
        user_ids = self.extract_user_ids(documents)
        
        print(f"   ✓ Primary service: {primary_service}")
        print(f"   ✓ Error time: {error_time}")
        print(f"   ✓ Endpoints: {endpoints or '(none found)'}")
        print(f"   ✓ User IDs: {user_ids or '(none found)'}")
        
        # Step 2: Call MCP tool to find correlated errors
        print("\n🔍 Searching for correlated errors in other services...")
        
        agent_prompt = f"""
            You are an expert log and metrics analyst for OAuth2 microservices.
            Find correlated errors across services using these parameters:
            - Primary service: {primary_service}
            - Error time: {error_time}
            - Endpoints: {endpoints}
            - User IDs: {user_ids}
            - Time tolerance: 100ms
            
            Use the find_correlated_errors tool to search for related logs.
            """
        
        # Run the ReAct agent
        # The agent will autonomously decide which tools to call
        result = await self.mcp_agent.ainvoke(
            {"messages": [HumanMessage(content=agent_prompt)]},
            config={"configurable": {"thread_id": "cross_service_agent"}}
        )
        
        # Log which tools the agent used
        print("\n🔧 Tools used by agent:")
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"   → {tool_call['name']}")
        
        # Extract the agent's analysis
        correlated_findings = result["messages"][-1].content
        
        return {
            "target_service": primary_service,
            "error_time": error_time,
            "endpoints": endpoints,
            "user_ids": user_ids,
            "correlated_findings": correlated_findings
        }