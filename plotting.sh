#!/bin/bash
set -e

VENV_NAME="ml-predictor"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

BOX_H="━"
BOX_V="┃"
BOX_TL="┏"
BOX_TR="┓"
BOX_BL="┗"
BOX_BR="┛"

print_header() {
    local text="$1"
    echo -e "${CYAN}${BOX_TL}$(printf "${BOX_H}%.0s" {1..68})${BOX_TR}${NC}"
    printf "${CYAN}${BOX_V}${NC} ${BOLD}%-66s${NC} ${CYAN}${BOX_V}${NC}\n" "$text"
    echo -e "${CYAN}${BOX_BL}$(printf "${BOX_H}%.0s" {1..68})${BOX_BR}${NC}"
    echo ""
}

echo ""
print_header "Model Performance Visualization"

if [ ! -d "$VENV_NAME" ]; then
    echo -e "${RED}Error: Virtual environment '${VENV_NAME}' not found.${NC}"
    echo -e "${YELLOW}Please create the virtual environment first.${NC}"
    exit 1
fi

if [ ! -d "models/trained" ]; then
    echo -e "${RED}Error: No trained models found in models/trained/${NC}"
    echo -e "${YELLOW}Please train at least one model first.${NC}"
    exit 1
fi

source $VENV_NAME/bin/activate

echo -e "${BLUE}Checking for trained models...${NC}"
model_count=$(find models/trained -name "*_metadata.json" | wc -l | tr -d ' ')
echo -e "${GREEN}✓${NC} Found ${BOLD}${model_count}${NC} trained model(s)"
echo ""

START_TIME=$(date +%s)

python scripts/plot_model_comparison.py \
    --models-dir models/trained \
    --output-dir plots

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${CYAN}${BOX_TL}$(printf "${BOX_H}%.0s" {1..68})${BOX_TR}${NC}"
printf "${CYAN}${BOX_V}${NC} ${BOLD}${GREEN}Visualization Complete${NC}%-43s ${CYAN}${BOX_V}${NC}\n" ""
echo -e "${CYAN}${BOX_V}${NC}                                                                    ${CYAN}${BOX_V}${NC}"
printf "${CYAN}${BOX_V}${NC}   Output directory: ${BOLD}plots/${NC}%-40s ${CYAN}${BOX_V}${NC}\n" ""
printf "${CYAN}${BOX_V}${NC}   Time elapsed: ${BOLD}${ELAPSED}s${NC}%-47s ${CYAN}${BOX_V}${NC}\n" ""
echo -e "${CYAN}${BOX_BL}$(printf "${BOX_H}%.0s" {1..68})${BOX_BR}${NC}"
echo ""
echo -e "${GREEN}${BOLD}View plots:${NC}"
echo -e "  • Open plots/ folder to view all visualizations"
echo -e "  • Read plots/comparison_report.txt for detailed analysis"
echo ""
