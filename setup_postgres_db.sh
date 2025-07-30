#!/bin/bash

# PostgreSQL Database Setup Script for MCPBench
# This script sets up the car business intelligence database for PostgreSQL MCP server evaluation

set -e

# Configuration
DB_NAME="mcpbench_car_bi"
DB_USER="mcpbench_user"
DB_PASSWORD="mcpbench_pass"
DB_HOST="localhost"
DB_PORT="5432"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up PostgreSQL database for MCPBench...${NC}"

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo -e "${RED}PostgreSQL is not installed. Please install it first.${NC}"
    echo "On macOS: brew install postgresql"
    echo "On Ubuntu: sudo apt-get install postgresql postgresql-contrib"
    exit 1
fi

# Check if PostgreSQL service is running
if ! pg_isready -h $DB_HOST -p $DB_PORT &> /dev/null; then
    echo -e "${RED}PostgreSQL service is not running. Please start it first.${NC}"
    echo "On macOS: brew services start postgresql"
    echo "On Ubuntu: sudo systemctl start postgresql"
    exit 1
fi

echo -e "${YELLOW}Creating database and user...${NC}"

# Create database and user (assuming you have superuser access)
psql -h $DB_HOST -p $DB_PORT -U postgres -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || echo "Database $DB_NAME already exists"
psql -h $DB_HOST -p $DB_PORT -U postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" 2>/dev/null || echo "User $DB_USER already exists"
psql -h $DB_HOST -p $DB_PORT -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
psql -h $DB_HOST -p $DB_PORT -U postgres -d $DB_NAME -c "GRANT ALL ON SCHEMA public TO $DB_USER;"

echo -e "${YELLOW}Setting up database schema...${NC}"

# Run the schema creation script
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "langProBe/DB/DB_utils/postgres_schema.sql"

echo -e "${YELLOW}Inserting sample data...${NC}"

# Insert sample data
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "langProBe/DB/DB_utils/sample_data.sql"

echo -e "${GREEN}Database setup completed successfully!${NC}"
echo -e "${YELLOW}Connection details:${NC}"
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo "Password: $DB_PASSWORD"
echo "Host: $DB_HOST"
echo "Port: $DB_PORT"
echo ""
echo -e "${YELLOW}Connection string for postgres.json:${NC}"
echo "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
echo ""
echo -e "${YELLOW}To test the connection:${NC}"
echo "psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"