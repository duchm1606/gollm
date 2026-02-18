#!/bin/bash
# run_tests.sh - Script to generate Python reference data and run Go tests
#
# This script:
# 1. Generates binary test data from Python PyTorch implementation
# 2. Runs Go unit tests against the generated data
#
# Usage:
#   ./scripts/run_tests.sh
#   ./scripts/run_tests.sh -v          # Verbose mode
#   ./scripts/run_tests.sh attention   # Run only attention tests

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "  gollm Test Runner"
echo "=========================================="
echo ""

# Parse arguments
VERBOSE=""
TEST_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [TEST_PATTERN]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Run tests with verbose output"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Test Patterns:"
            echo "  attention        Run only attention tests"
            echo "  tensor           Run only tensor tests"
            echo "  tokenizer        Run only tokenizer tests"
            echo ""
            exit 0
            ;;
        *)
            TEST_FILTER="$1"
            shift
            ;;
    esac
done

# Step 1: Generate Python reference data
echo "=== Step 1: Generating Python reference data ==="
echo ""

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    exit 1
fi

if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}Warning: PyTorch not installed. Installing...${NC}"
    pip3 install torch numpy
fi

echo "Running: scripts/generate_test_weights.py"
python3 "$SCRIPT_DIR/generate_test_weights.py"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to generate test data${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Test data generated successfully${NC}"
echo ""

# Step 2: Run Go tests
echo "=== Step 2: Running Go tests ==="
echo ""

# Determine which tests to run
if [ -z "$TEST_FILTER" ]; then
    echo "Running all tests..."
    TEST_PATTERN="./pkg/..."
else
    echo "Running tests matching: $TEST_FILTER"
    case $TEST_FILTER in
        attention)
            TEST_PATTERN="./pkg/model/attention/..."
            ;;
        tensor)
            TEST_PATTERN="./pkg/tensor/..."
            ;;
        tokenizer)
            TEST_PATTERN="./pkg/tokenizer/..."
            ;;
        *)
            echo -e "${RED}Unknown test pattern: $TEST_FILTER${NC}"
            echo "Available patterns: attention, tensor, tokenizer"
            exit 1
            ;;
    esac
fi

echo ""
echo "Running: go test $VERBOSE $TEST_PATTERN"
echo ""

cd "$PROJECT_DIR"
if ! go test $VERBOSE $TEST_PATTERN; then
    echo ""
    echo -e "${RED}✗ Tests failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All tests passed!${NC}"
echo ""
echo "=========================================="
