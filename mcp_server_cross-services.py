import json
import sys
from multiagent import RetrieverManager
from mcp.server.fastmcp import FastMCP
from datetime import datetime, timedelta
from typing import Dict
from utils import microseconds_to_time_str
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP(
    "Cross-Service",
    instructions="Correlates errors across multiple OAuth2 microservices by analyzing temporal and contextual relationships.",
    host="0.0.0.0",
    port=8006,
)

# ============================================
# INITIALIZE RETRIEVER ON SERVER STARTUP
# ============================================
_retriever_initialized = False

def ensure_retriever_initialized():
    """Initialize retriever once on first use"""
    global _retriever_initialized
    
    if not _retriever_initialized:
        print("🔧 Initializing retriever in Cross-Service MCP server...")
        api_key = os.getenv('API_KEY')
        folder_path = "data/logs/update_password_404_user_not_found"
        
        try:
            RetrieverManager.initialize(folder_path, api_key)
            _retriever_initialized = True
            print("✅ Retriever initialized successfully in MCP server")
        except Exception as e:
            print(f"❌ Retriever initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize retriever: {e}")

@mcp.tool()
def correlate_cross_service_errors(
    primary_service: str,
    error_time: str,  # ISO format: "2024-10-23T18:13:32.142000"
    endpoints: str = "",  # Comma-separated: "/oauth2/user,/oauth2/token"
    user_ids: str = "",   # Comma-separated: "user123,user456"
    time_tolerance_ms: int = 100
) -> Dict[str, any]:
    
    
    # Get the retriever instance
    retriever = RetrieverManager.get_retriever()
    
    # Parse inputs
    time_obj = datetime.strptime(error_time, "%H:%M:%S.%f").time()
    endpoints_list = [e.strip() for e in endpoints.split(',') if e.strip()]
    user_ids_list = [u.strip() for u in user_ids.split(',') if u.strip()]
    
    error_microseconds = (
        time_obj.hour * 3600000000 +  # hours to microseconds
        time_obj.minute * 60000000 +   # minutes to microseconds
        time_obj.second * 1000000 +    # seconds to microseconds
        time_obj.microsecond           # microseconds
    )

    tolerance_microseconds = time_tolerance_ms * 1000  # ms to microseconds

    start_microseconds = error_microseconds - tolerance_microseconds
    end_microseconds = error_microseconds + tolerance_microseconds
    
    time_start = microseconds_to_time_str(start_microseconds)
    time_end = microseconds_to_time_str(end_microseconds)

    # All OAuth2 services
    all_services = [
        "oauth2-client", "oauth2-code", "oauth2-key", 
        "oauth2-refresh-token", "oauth2-service", 
        "oauth2-user", "oauth2-token"
    ]

    # Results grouped by service
    correlated_findings = {
        "primary_service": primary_service,
        "error_time": error_time,
        "time_window": {
            "start": time_start,
            "end": time_end,
            "tolerance_ms": time_tolerance_ms
        },
        "related_services": {},
        "summary": {
            "total_related_logs": 0,
            "services_affected": 0
        }
    }
    
    # Search other services
    for service in all_services:
        if service == primary_service:
            continue
        
        # Search by time range in this service
        service_docs = retriever.search_by_time_range(
            service=service,
            start_time=time_start,
            end_time=time_end
        )
        
        print(f"      • Found {len(service_docs) if service_docs else 0} documents", file=sys.stderr)

        if not service_docs:
            continue
        
        # Filter documents by shared context
        related_docs = []
        for doc in service_docs:
            doc_endpoints = doc.metadata.get('endpoints', [])
            doc_user_ids = doc.metadata.get('user_ids', [])
            doc_has_error = doc.metadata.get('has_error', False)
            
            # Check for context overlap
            shared_endpoints = set(endpoints_list) & set(doc_endpoints)
            shared_users = set(user_ids_list) & set(doc_user_ids)
            
            # Include if: shared context OR has error
            # if shared_endpoints or shared_users or doc_has_error:
            related_docs.append({
                "timestamp": doc.metadata.get('timestamp', 'unknown'),
                "log_level": doc.metadata.get('log_level', 'unknown'),
                "has_error": doc_has_error,
                "shared_endpoints": list(shared_endpoints),
                "shared_users": list(shared_users),
                "content_preview": doc.page_content[:500]
            })
        
        # Add to results if we found related logs
        if related_docs:
            correlated_findings["related_services"][service] = {
                "count": len(related_docs),
                "logs": related_docs
            }
            correlated_findings["summary"]["total_related_logs"] += len(related_docs)
            correlated_findings["summary"]["services_affected"] += 1
    
    # Add analysis
    if correlated_findings["summary"]["services_affected"] > 0:
        correlated_findings["analysis"] = (
            f"Found {correlated_findings['summary']['total_related_logs']} related logs "
            f"across {correlated_findings['summary']['services_affected']} services "
            f"within {time_tolerance_ms}ms of the error."
        )
    else:
        correlated_findings["analysis"] = "No related logs found in other services."
    
    # Final results
    print("\n" + "="*60, file=sys.stderr)
    print("📊 FINAL RESULTS", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"   • Total related logs: {correlated_findings['summary']['total_related_logs']}", file=sys.stderr)
    print(f"   • Services affected: {correlated_findings['summary']['services_affected']}", file=sys.stderr)
    print(f"\n📋 Full JSON:", file=sys.stderr)
    print(json.dumps(correlated_findings, indent=2), file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)

    return correlated_findings

if __name__ == "__main__":
    
    print("=" * 60)
    print("🚀 Starting Cross-Service MCP Server")
    print("=" * 60)
    
    # Initialize retriever at startup
    try:
        ensure_retriever_initialized()
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        exit(1)
    
    print("✅ Server ready - listening on stdio")
    print("=" * 60)
    
    # Run the MCP server
    mcp.run(transport="stdio")