#!/bin/bash

# SAM Experiment Runner
# Usage: ./run.sh [fig34|fig5|fig9|all]
#
# This script runs experiments to reproduce specific figures from the paper:
#   fig34 - Figure 3&4: Performance comparison under hotspot shifts
#   fig5  - Figure 5: Robustness against adversarial attacks
#   fig9  - Figure 9: Scalability analysis
#   all   - Run all experiments (default when no argument provided)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to format time
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# Function to run Figure 3&4 experiment (Type 1)
run_fig34() {
    echo ""
    echo "======================================"
    echo "FIGURE 3&4: Performance Comparison"
    echo "======================================"
    print_info "Reproducing Figure 3&4 - Performance under hotspot shifts"
    print_info "Comparing SAM with baselines B2, B7, B12"
    print_warning "Estimated time: ~45 minutes"
    echo ""
    
    START_TIME=$(date +%s)
    python scripts/run_cache_strategy_comparison.py --experiment-type 1
    END_TIME=$(date +%s)
    
    print_success "Figure 3&4 completed in $(format_time $((END_TIME - START_TIME)))"
    print_info "Results saved to: results/comparison/experiment_*"
    print_info "Figures generated: figures/paper/figure1_performance_comparison.pdf"
    print_info "                  figures/paper/figure2_timeseries.pdf"
}

# Function to run Figure 5 experiment (Type 2)
run_fig5() {
    echo ""
    echo "======================================"
    echo "FIGURE 5: Robustness Test"
    echo "======================================"
    print_info "Reproducing Figure 5 - Robustness against adversarial attacks"
    print_info "Testing SAM and B7 against cache pollution attacks"
    print_warning "Estimated time: ~20 minutes"
    echo ""
    
    START_TIME=$(date +%s)
    python scripts/run_cache_strategy_comparison.py --experiment-type 2
    END_TIME=$(date +%s)
    
    print_success "Figure 5 completed in $(format_time $((END_TIME - START_TIME)))"
    print_info "Results saved to: results/comparison/experiment_*"
    print_info "Figure generated: figures/paper/figure7_dual_combat_robustness.pdf"
}

# Function to run Figure 9 experiment (Type 3)
run_fig9() {
    echo ""
    echo "======================================"
    echo "FIGURE 9: Scalability Analysis"
    echo "======================================"
    print_info "Reproducing Figure 9 - Scalability with varying database counts"
    print_info "Testing SAM with 20, 40, 80, 120 databases"
    print_warning "Estimated time: ~30 minutes"
    echo ""
    
    START_TIME=$(date +%s)
    python scripts/run_cache_strategy_comparison.py --experiment-type 3
    END_TIME=$(date +%s)
    
    print_success "Figure 9 completed in $(format_time $((END_TIME - START_TIME)))"
    print_info "Results saved to: results/scalability/scale_*"
    print_info "Figures generated: figures/cpu_performance/cpu_performance_comparison.pdf"
    print_info "                  figures/cpu_performance/cpu_decision_time_distribution.pdf"
}

# Function to run all experiments
run_all() {
    echo ""
    echo "======================================"
    echo "SAM: Complete Experiment Suite"
    echo "======================================"
    print_info "This will reproduce all paper figures:"
    echo "  • Figure 3&4: Performance comparison (~45 minutes)"
    echo "  • Figure 5: Robustness test (~20 minutes)"
    echo "  • Figure 9: Scalability analysis (~30 minutes)"
    echo ""
    print_warning "Total estimated time: ~95 minutes"
    echo ""
    
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Experiment cancelled."
        exit 0
    fi
    
    TOTAL_START=$(date +%s)
    
    # Run all three experiments
    run_fig34
    run_fig5
    run_fig9
    
    TOTAL_END=$(date +%s)
    
    echo ""
    echo "======================================"
    echo "ALL EXPERIMENTS COMPLETED"
    echo "======================================"
    print_success "Total execution time: $(format_time $((TOTAL_END - TOTAL_START)))"
    echo ""
    print_info "All figures have been generated:"
    echo "  • Figure 3&4: figures/paper/figure[1-2]_*.pdf"
    echo "  • Figure 5: figures/paper/figure7_dual_combat_robustness.pdf"
    echo "  • Figure 9: figures/cpu_performance/*.pdf"
}

# Function to show usage
show_usage() {
    echo "SAM Experiment Runner"
    echo "====================="
    echo ""
    echo "Usage: $0 [fig34|fig5|fig9|all]"
    echo ""
    echo "Arguments:"
    echo "  fig34  - Reproduce Figure 3&4: Performance comparison (~45 min)"
    echo "  fig5   - Reproduce Figure 5: Robustness test (~20 min)"
    echo "  fig9   - Reproduce Figure 9: Scalability analysis (~30 min)"
    echo "  all    - Run all experiments (default, ~95 min total)"
    echo ""
    echo "Examples:"
    echo "  $0          # Run all experiments"
    echo "  $0 all      # Run all experiments (explicit)"
    echo "  $0 fig34    # Run only Figure 3&4 experiment"
    echo "  $0 fig5     # Run only Figure 5 experiment"
    echo "  $0 fig9     # Run only Figure 9 experiment"
}

# Main script logic
case "${1:-all}" in
    fig34)
        run_fig34
        ;;
    fig5)
        run_fig5
        ;;
    fig9)
        run_fig9
        ;;
    all)
        run_all
        ;;
    -h|--help|help)
        show_usage
        exit 0
        ;;
    *)
        print_error "Invalid argument: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac