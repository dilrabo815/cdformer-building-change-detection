#!/bin/bash

# Uzcosmos Change Detection - One-Command Launcher

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Uzcosmos Change Detection System...${NC}"

# 1. Start Python Backend (FastAPI)
echo -e "${GREEN}Starting AI Backend (FastAPI on port 8000)...${NC}"
source venv/bin/activate
cd api
uvicorn main:app --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# 2. Wait a moment for backend to initialize
sleep 2

# 3. Start Frontend (Next.js)
echo -e "${GREEN}Starting Dashboard UI (Next.js on port 3000)...${NC}"
cd web
npm run dev &
FRONTEND_PID=$!

echo -e "${BLUE}--------------------------------------------------${NC}"
echo -e "${BLUE}SYSTEM READY!${NC}"
echo -e "Backend: http://localhost:8000"
echo -e "Frontend: http://localhost:3000"
echo -e "${BLUE}--------------------------------------------------${NC}"
echo -e "Press [CTRL+C] to stop both services."

# Function to kill both processes on exit
cleanup() {
    echo -e "\n${BLUE}Shutting down services...${NC}"
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit
}

trap cleanup SIGINT

# Keep the script running
wait
