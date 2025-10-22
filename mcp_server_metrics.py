import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP(
    "Lo2Metrics",
    instructions="Provides real-time metrics analysis from Lo2 performance data including Go runtime, system resources, and network statistics.",
    host="0.0.0.0",
    port=8006,
)

# Global metrics cache
_metrics_df = None
_metrics_path = None


def load_metrics(csv_path: str) -> pd.DataFrame:
    """Load metrics CSV file into pandas DataFrame"""
    global _metrics_df, _metrics_path
    
    if _metrics_df is None or _metrics_path != csv_path:
        try:
            _metrics_df = pd.read_csv(csv_path)
            _metrics_path = csv_path
            print(f"✅ Loaded metrics from: {csv_path}")
            print(f"   Rows: {len(_metrics_df)}, Columns: {len(_metrics_df.columns)}")
        except Exception as e:
            raise ValueError(f"Failed to load metrics CSV: {e}")
    
    return _metrics_df


@mcp.tool()
def get_go_memory_stats(csv_path: str) -> Dict[str, any]:
    """
    Get Go memory statistics from metrics.
    
    Shows heap allocation, GC stats, and memory pressure indicators.
    
    Args:
        csv_path: Path to metrics CSV file
        
    Returns:
        Dictionary with memory statistics including heap usage, GC metrics
    """
    df = load_metrics(csv_path)
    
    # Extract Go memory metrics
    memory_stats = {
        "heap_alloc_bytes": int(df['go_memstats_heap_alloc_bytes'].iloc[0]) if 'go_memstats_heap_alloc_bytes' in df.columns else 0,
        "heap_alloc_mb": round(int(df['go_memstats_heap_alloc_bytes'].iloc[0]) / (1024**2), 2) if 'go_memstats_heap_alloc_bytes' in df.columns else 0,
        "heap_sys_bytes": int(df['go_memstats_heap_sys_bytes'].iloc[0]) if 'go_memstats_heap_sys_bytes' in df.columns else 0,
        "heap_idle_bytes": int(df['go_memstats_heap_idle_bytes'].iloc[0]) if 'go_memstats_heap_idle_bytes' in df.columns else 0,
        "heap_inuse_bytes": int(df['go_memstats_heap_inuse_bytes'].iloc[0]) if 'go_memstats_heap_inuse_bytes' in df.columns else 0,
        "heap_objects": int(df['go_memstats_heap_objects'].iloc[0]) if 'go_memstats_heap_objects' in df.columns else 0,
        "next_gc_bytes": int(df['go_memstats_next_gc_bytes'].iloc[0]) if 'go_memstats_next_gc_bytes' in df.columns else 0,
    }
    
    # Calculate heap usage percentage
    if memory_stats["heap_sys_bytes"] > 0:
        memory_stats["heap_usage_percent"] = round(
            (memory_stats["heap_alloc_bytes"] / memory_stats["heap_sys_bytes"]) * 100, 2
        )
    else:
        memory_stats["heap_usage_percent"] = 0
    
    # Memory pressure indicator
    if memory_stats["heap_usage_percent"] > 90:
        memory_stats["status"] = "CRITICAL - Memory pressure detected"
    elif memory_stats["heap_usage_percent"] > 80:
        memory_stats["status"] = "WARNING - High memory usage"
    else:
        memory_stats["status"] = "NORMAL"
    
    return memory_stats


@mcp.tool()
def get_go_gc_stats(csv_path: str) -> Dict[str, any]:
    """
    Get Go garbage collection statistics.
    
    Shows GC frequency, pause times, and performance impact.
    
    Args:
        csv_path: Path to metrics CSV file
        
    Returns:
        Dictionary with GC statistics and performance analysis
    """
    df = load_metrics(csv_path)
    
    gc_stats = {
        "gc_count": int(df['go_gc_duration_seconds_count'].iloc[0]) if 'go_gc_duration_seconds_count' in df.columns else 0,
        "gc_total_time_seconds": float(df['go_gc_duration_seconds_sum'].iloc[0]) if 'go_gc_duration_seconds_sum' in df.columns else 0,
        "gc_min_seconds": float(df['go_gc_duration_seconds&quantile=0'].iloc[0]) if 'go_gc_duration_seconds&quantile=0' in df.columns else 0,
        "gc_p50_seconds": float(df['go_gc_duration_seconds&quantile=0.5'].iloc[0]) if 'go_gc_duration_seconds&quantile=0.5' in df.columns else 0,
        "gc_p75_seconds": float(df['go_gc_duration_seconds&quantile=0.75'].iloc[0]) if 'go_gc_duration_seconds&quantile=0.75' in df.columns else 0,
        "gc_max_seconds": float(df['go_gc_duration_seconds&quantile=1'].iloc[0]) if 'go_gc_duration_seconds&quantile=1' in df.columns else 0,
    }
    
    # Convert to milliseconds for readability
    gc_stats["gc_p50_ms"] = round(gc_stats["gc_p50_seconds"] * 1000, 2)
    gc_stats["gc_p75_ms"] = round(gc_stats["gc_p75_seconds"] * 1000, 2)
    gc_stats["gc_max_ms"] = round(gc_stats["gc_max_seconds"] * 1000, 2)
    
    # Average GC time
    if gc_stats["gc_count"] > 0:
        gc_stats["gc_avg_ms"] = round((gc_stats["gc_total_time_seconds"] / gc_stats["gc_count"]) * 1000, 2)
    else:
        gc_stats["gc_avg_ms"] = 0
    
    # GC pressure analysis
    if gc_stats["gc_max_ms"] > 100:
        gc_stats["status"] = "CRITICAL - GC pauses > 100ms causing latency"
    elif gc_stats["gc_max_ms"] > 50:
        gc_stats["status"] = "WARNING - GC pauses affecting performance"
    else:
        gc_stats["status"] = "NORMAL"
    
    return gc_stats


@mcp.tool()
def get_goroutine_stats(csv_path: str) -> Dict[str, any]:
    """
    Get goroutine statistics.
    
    Detects goroutine leaks and concurrency issues.
    
    Args:
        csv_path: Path to metrics CSV file
        
    Returns:
        Dictionary with goroutine count and leak detection
    """
    df = load_metrics(csv_path)
    
    goroutine_count = int(df['go_goroutines'].iloc[0]) if 'go_goroutines' in df.columns else 0
    
    stats = {
        "goroutine_count": goroutine_count,
        "expected_baseline": 50,  # Typical baseline for OAuth2 services
    }
    
    # Goroutine leak detection
    if goroutine_count > 500:
        stats["status"] = "CRITICAL - Possible goroutine leak (>500)"
        stats["recommendation"] = "Investigate goroutine creation, check for unclosed channels or blocked goroutines"
    elif goroutine_count > 200:
        stats["status"] = "WARNING - High goroutine count (>200)"
        stats["recommendation"] = "Monitor for potential leak"
    else:
        stats["status"] = "NORMAL"
        stats["recommendation"] = "Goroutine count within expected range"
    
    return stats


@mcp.tool()
def get_system_memory_stats(csv_path: str) -> Dict[str, any]:
    """
    Get system-level memory statistics.
    
    Shows total memory, available memory, and memory pressure.
    
    Args:
        csv_path: Path to metrics CSV file
        
    Returns:
        Dictionary with system memory statistics
    """
    df = load_metrics(csv_path)
    
    stats = {
        "total_mb": round(int(df['node_memory_MemTotal_bytes'].iloc[0]) / (1024**2), 2) if 'node_memory_MemTotal_bytes' in df.columns else 0,
        "available_mb": round(int(df['node_memory_MemAvailable_bytes'].iloc[0]) / (1024**2), 2) if 'node_memory_MemAvailable_bytes' in df.columns else 0,
        "free_mb": round(int(df['node_memory_MemFree_bytes'].iloc[0]) / (1024**2), 2) if 'node_memory_MemFree_bytes' in df.columns else 0,
        "cached_mb": round(int(df['node_memory_Cached_bytes'].iloc[0]) / (1024**2), 2) if 'node_memory_Cached_bytes' in df.columns else 0,
        "buffers_mb": round(int(df['node_memory_Buffers_bytes'].iloc[0]) / (1024**2), 2) if 'node_memory_Buffers_bytes' in df.columns else 0,
    }
    
    # Calculate usage
    if stats["total_mb"] > 0:
        stats["used_mb"] = stats["total_mb"] - stats["available_mb"]
        stats["usage_percent"] = round((stats["used_mb"] / stats["total_mb"]) * 100, 2)
    else:
        stats["used_mb"] = 0
        stats["usage_percent"] = 0
    
    # Memory pressure assessment
    if stats["usage_percent"] > 95:
        stats["status"] = "CRITICAL - System memory exhausted"
    elif stats["usage_percent"] > 85:
        stats["status"] = "WARNING - High system memory usage"
    else:
        stats["status"] = "NORMAL"
    
    return stats


@mcp.tool()
def get_cpu_stats(csv_path: str) -> Dict[str, any]:
    """
    Get CPU usage statistics.
    
    Shows system load and CPU utilization.
    
    Args:
        csv_path: Path to metrics CSV file
        
    Returns:
        Dictionary with CPU statistics
    """
    df = load_metrics(csv_path)
    
    stats = {
        "load_1min": float(df['node_load1'].iloc[0]) if 'node_load1' in df.columns else 0,
        "load_5min": float(df['node_load5'].iloc[0]) if 'node_load5' in df.columns else 0,
        "load_15min": float(df['node_load15'].iloc[0]) if 'node_load15' in df.columns else 0,
        "cpu_cores": 16,  # Based on your data (16 CPUs)
    }
    
    # Calculate load per core
    stats["load_per_core_1min"] = round(stats["load_1min"] / stats["cpu_cores"], 2)
    
    # CPU pressure assessment
    if stats["load_per_core_1min"] > 0.8:
        stats["status"] = "WARNING - High CPU load"
    elif stats["load_per_core_1min"] > 0.5:
        stats["status"] = "MODERATE"
    else:
        stats["status"] = "NORMAL"
    
    return stats


@mcp.tool()
def detect_anomalies(csv_path: str) -> Dict[str, List[str]]:
    """
    Detect performance anomalies across all metrics.
    
    Automatically identifies issues like memory pressure, GC problems,
    goroutine leaks, and resource exhaustion.
    
    Args:
        csv_path: Path to metrics CSV file
        
    Returns:
        Dictionary with list of detected anomalies
    """
    anomalies = {
        "critical": [],
        "warnings": [],
        "info": []
    }
    
    # Check memory
    mem_stats = get_go_memory_stats(csv_path)
    if "CRITICAL" in mem_stats["status"]:
        anomalies["critical"].append(f"Go heap at {mem_stats['heap_usage_percent']}% capacity")
    elif "WARNING" in mem_stats["status"]:
        anomalies["warnings"].append(f"Go heap at {mem_stats['heap_usage_percent']}% capacity")
    
    # Check GC
    gc_stats = get_go_gc_stats(csv_path)
    if "CRITICAL" in gc_stats["status"]:
        anomalies["critical"].append(f"GC max pause: {gc_stats['gc_max_ms']}ms")
    elif "WARNING" in gc_stats["status"]:
        anomalies["warnings"].append(f"GC max pause: {gc_stats['gc_max_ms']}ms")
    
    # Check goroutines
    goroutine_stats = get_goroutine_stats(csv_path)
    if "CRITICAL" in goroutine_stats["status"]:
        anomalies["critical"].append(f"Goroutines: {goroutine_stats['goroutine_count']} (possible leak)")
    elif "WARNING" in goroutine_stats["status"]:
        anomalies["warnings"].append(f"Goroutines: {goroutine_stats['goroutine_count']} (elevated)")
    
    # Check system memory
    sys_mem = get_system_memory_stats(csv_path)
    if "CRITICAL" in sys_mem["status"]:
        anomalies["critical"].append(f"System memory at {sys_mem['usage_percent']}%")
    elif "WARNING" in sys_mem["status"]:
        anomalies["warnings"].append(f"System memory at {sys_mem['usage_percent']}%")
    
    # Check CPU
    cpu_stats = get_cpu_stats(csv_path)
    if "WARNING" in cpu_stats["status"]:
        anomalies["warnings"].append(f"CPU load: {cpu_stats['load_1min']} (high)")
    
    if not anomalies["critical"] and not anomalies["warnings"]:
        anomalies["info"].append("All metrics within normal ranges")
    
    return anomalies


@mcp.tool()
def diagnose_error_correlation(csv_path: str, error_description: str) -> Dict[str, any]:
    """
    Correlate error patterns with metrics data.
    
    Analyzes metrics to find likely causes of errors (e.g., 404, 500, timeouts).
    
    Args:
        csv_path: Path to metrics CSV file
        error_description: Description of the error (e.g., "404 user not found")
        
    Returns:
        Diagnosis with likely root causes based on metrics
    """
    diagnosis = {
        "error": error_description,
        "likely_causes": [],
        "supporting_metrics": {}
    }
    
    # Get all metrics
    mem_stats = get_go_memory_stats(csv_path)
    gc_stats = get_go_gc_stats(csv_path)
    goroutine_stats = get_goroutine_stats(csv_path)
    sys_mem = get_system_memory_stats(csv_path)
    
    # Analyze for different error patterns
    error_lower = error_description.lower()
    
    if "404" in error_lower or "not found" in error_lower:
        # Database/lookup failures often correlate with memory/performance issues
        if mem_stats["heap_usage_percent"] > 80:
            diagnosis["likely_causes"].append(
                f"High memory pressure ({mem_stats['heap_usage_percent']}%) may be slowing database queries, causing timeouts and 404 responses"
            )
            diagnosis["supporting_metrics"]["heap_usage"] = f"{mem_stats['heap_usage_percent']}%"
        
        if gc_stats["gc_max_ms"] > 50:
            diagnosis["likely_causes"].append(
                f"Long GC pauses ({gc_stats['gc_max_ms']}ms) may be blocking request processing, causing lookup failures"
            )
            diagnosis["supporting_metrics"]["gc_max_pause"] = f"{gc_stats['gc_max_ms']}ms"
        
        if goroutine_stats["goroutine_count"] > 200:
            diagnosis["likely_causes"].append(
                f"High goroutine count ({goroutine_stats['goroutine_count']}) suggests resource contention, slowing database connections"
            )
            diagnosis["supporting_metrics"]["goroutines"] = goroutine_stats["goroutine_count"]
    
    elif "timeout" in error_lower or "slow" in error_lower:
        if gc_stats["gc_max_ms"] > 100:
            diagnosis["likely_causes"].append(
                f"Critical GC pauses ({gc_stats['gc_max_ms']}ms) are blocking request handlers"
            )
        
        if sys_mem["usage_percent"] > 90:
            diagnosis["likely_causes"].append(
                f"System memory exhaustion ({sys_mem['usage_percent']}%) causing swap thrashing"
            )
    
    elif "500" in error_lower or "panic" in error_lower:
        if mem_stats["heap_usage_percent"] > 95:
            diagnosis["likely_causes"].append(
                "Memory exhaustion likely causing OOM or allocation failures"
            )
        
        if goroutine_stats["goroutine_count"] > 500:
            diagnosis["likely_causes"].append(
                "Goroutine leak may be exhausting system resources"
            )
    
    if not diagnosis["likely_causes"]:
        diagnosis["likely_causes"].append("Metrics appear normal - error may be application logic related")
    
    return diagnosis


@mcp.tool()
def get_metrics_summary(csv_path: str) -> Dict[str, any]:
    """
    Get comprehensive metrics summary.
    
    One-stop overview of all key metrics and health status.
    
    Args:
        csv_path: Path to metrics CSV file
        
    Returns:
        Complete summary with all key metrics
    """
    summary = {
        "timestamp": load_metrics(csv_path)['timestamp'].iloc[0] if 'timestamp' in load_metrics(csv_path).columns else "unknown",
        "go_memory": get_go_memory_stats(csv_path),
        "go_gc": get_go_gc_stats(csv_path),
        "goroutines": get_goroutine_stats(csv_path),
        "system_memory": get_system_memory_stats(csv_path),
        "cpu": get_cpu_stats(csv_path),
        "anomalies": detect_anomalies(csv_path)
    }
    
    # Overall health assessment
    critical_count = len(summary["anomalies"]["critical"])
    warning_count = len(summary["anomalies"]["warnings"])
    
    if critical_count > 0:
        summary["overall_status"] = f"CRITICAL - {critical_count} critical issue(s) detected"
    elif warning_count > 0:
        summary["overall_status"] = f"WARNING - {warning_count} warning(s) detected"
    else:
        summary["overall_status"] = "HEALTHY"
    
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Lo2 Metrics MCP Server")
    print("=" * 60)
    print("\nAvailable tools:")
    print("  1. get_go_memory_stats - Go heap and memory metrics")
    print("  2. get_go_gc_stats - Garbage collection statistics")
    print("  3. get_goroutine_stats - Goroutine count and leak detection")
    print("  4. get_system_memory_stats - System memory usage")
    print("  5. get_cpu_stats - CPU load and usage")
    print("  6. detect_anomalies - Automatic anomaly detection")
    print("  7. diagnose_error_correlation - Correlate errors with metrics")
    print("  8. get_metrics_summary - Complete metrics overview")
    print("\n" + "=" * 60)
    print("Server running on stdio transport")
    print("=" * 60 + "\n")
    
    # Run the MCP server
    mcp.run(transport="stdio")