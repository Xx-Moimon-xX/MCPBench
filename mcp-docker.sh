#!/bin/bash
# 
# MCP Server Docker Management Script
# Simple wrapper for managing MCP servers via Docker
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.mcp.yml"

show_help() {
    echo "MCP Server Docker Management"
    echo ""
    echo "Usage: $0 <command> [config_file]"
    echo ""
    echo "Commands:"
    echo "  start <config>    - Start MCP servers from config file"
    echo "  stop              - Stop all MCP servers"
    echo "  restart <config>  - Restart MCP servers"
    echo "  status            - Show running MCP servers"
    echo "  logs [service]    - Show logs (optionally for specific service)"
    echo "  clean             - Stop and remove all MCP containers"
    echo ""
    echo "Examples:"
    echo "  $0 start brave.json"
    echo "  $0 start external/MCPBench/configs/slack.json"
    echo "  $0 stop"
    echo "  $0 logs mcp-brave-search"
}

start_servers() {
    local config_file="$1"
    
    if [ -z "$config_file" ]; then
        echo "Error: Config file required for start command"
        echo ""
        show_help
        exit 1
    fi
    
    echo "Starting MCP servers with config: $config_file"
    python3 "$SCRIPT_DIR/launch_mcp_servers_docker.py" "$config_file"
}

stop_servers() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo "No docker-compose file found. Servers may not be running."
        exit 0
    fi
    
    echo "Stopping MCP servers..."
    docker-compose -f "$COMPOSE_FILE" down
}

restart_servers() {
    local config_file="$1"
    
    if [ -z "$config_file" ]; then
        echo "Error: Config file required for restart command"
        show_help
        exit 1
    fi
    
    echo "Restarting MCP servers..."
    stop_servers
    start_servers "$config_file"
}

show_status() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo "No docker-compose file found. No servers running."
        exit 0
    fi
    
    echo "MCP Server Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    echo "Available endpoints:"
    docker-compose -f "$COMPOSE_FILE" ps --format "table {{.Service}}\t{{.Ports}}" | grep -v "Service" | while IFS=$'\t' read -r service ports; do
        if [[ -n "$ports" ]]; then
            port=$(echo "$ports" | grep -o '[0-9]*->' | cut -d'-' -f1)
            if [[ -n "$port" ]]; then
                echo "  $service: http://localhost:$port (SSE: /sse, Message: /message)"
            fi
        fi
    done
}

show_logs() {
    local service="$1"
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo "No docker-compose file found. No servers running."
        exit 0
    fi
    
    if [ -n "$service" ]; then
        docker-compose -f "$COMPOSE_FILE" logs -f "$service"
    else
        docker-compose -f "$COMPOSE_FILE" logs -f
    fi
}

clean_servers() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo "No docker-compose file found."
        exit 0
    fi
    
    echo "Stopping and removing all MCP containers..."
    docker-compose -f "$COMPOSE_FILE" down --volumes --remove-orphans
    
    echo "Cleaning up dangling MCP images..."
    docker image prune -f --filter label=com.docker.compose.project=mcpservers 2>/dev/null || true
    
    echo "Removing docker-compose file..."
    rm -f "$COMPOSE_FILE"
    
    echo "Cleanup complete!"
}

# Main command handling
case "${1:-}" in
    "start")
        start_servers "$2"
        ;;
    "stop")
        stop_servers
        ;;
    "restart")
        restart_servers "$2"
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2"
        ;;
    "clean")
        clean_servers
        ;;
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    *)
        echo "Error: Unknown command '$1'"
        echo ""
        show_help
        exit 1
        ;;
esac