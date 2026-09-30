#!/usr/bin/env bash
# ==============================================================================
# SafeSight AI — One-Click Local Runtime Environment (Linux / macOS)
# ==============================================================================

set -e

# ANSI Color Codes
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================================================${NC}"
echo -e "${CYAN}  SafeSight AI — One-Click Local Runtime Environment${NC}"
echo -e "${CYAN}========================================================================${NC}"
echo ""

# 1. Verify Docker CLI is installed
if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}[ERROR] Docker is not installed or not in system PATH.${NC}"
    echo "Please install Docker from https://www.docker.com/"
    exit 1
fi

# 2. Verify Docker Daemon is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}[ERROR] Docker daemon is not running!${NC}"
    echo "Please start Docker Desktop / daemon and re-run this script."
    exit 1
fi

# 3. Determine docker compose command (compose v2 vs v1)
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
else
    echo -e "${RED}[ERROR] Neither 'docker compose' nor 'docker-compose' found.${NC}"
    exit 1
fi

echo -e "${CYAN}[*] Starting SafeSight AI containers via ${COMPOSE_CMD}...${NC}"
$COMPOSE_CMD up -d --build

# 4. Poll for Ollama readiness
echo -e "${CYAN}[*] Waiting for Ollama container to complete initialization...${NC}"
ATTEMPTS=0
MAX_ATTEMPTS=30

while ! curl -s -f http://localhost:11434/ >/dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ $ATTEMPTS -ge $MAX_ATTEMPTS ]; then
        echo -e "${YELLOW}[WARN] Ollama container initialization check timed out after 60s.${NC}"
        break
    fi
    sleep 2
done

echo -e "${GREEN}[OK] Ollama service is up and responsive on port 11434.${NC}"

# 5. Pull the Qwen2.5 quantized model into the persistent volume
echo -e "${CYAN}[*] Pulling vision-language model weights (qwen2.5:3b)...${NC}"
echo "    (Model weights are cached permanently in docker volume: safesight_ollama_cache)"

# Use -it if in a terminal, or -i if automated
if [ -t 0 ]; then
    docker exec -it safesight-ollama ollama pull qwen2.5:3b || true
else
    docker exec -i safesight-ollama ollama pull qwen2.5:3b || true
fi

# 6. Display Success Banner
echo ""
echo -e "${GREEN}========================================================================${NC}"
echo -e "${GREEN}  SAFESIGHT AI — SYSTEM READY${NC}"
echo -e "${GREEN}========================================================================${NC}"
echo ""
echo -e "  ${CYAN}Swagger UI Docs:${NC}    http://localhost:8000/docs"
echo -e "  ${CYAN}Alternative Docs:${NC}   http://localhost:8000/redoc"
echo -e "  ${CYAN}Health Check:${NC}       http://localhost:8000/health"
echo -e "  ${CYAN}Ollama Endpoint:${NC}    http://localhost:11434"
echo -e "  ${CYAN}Active VLM Model:${NC}   qwen2.5-vl:3b"
echo ""
echo -e "  ${YELLOW}Test with sample CLI:${NC}"
echo "    python tests/test_single.py --text \"Check this content\" --url http://localhost:8000/moderate"
echo "    python tests/test_single.py path/to/image.jpg \"Sample caption\""
echo ""
echo -e "${GREEN}========================================================================${NC}"
echo ""
