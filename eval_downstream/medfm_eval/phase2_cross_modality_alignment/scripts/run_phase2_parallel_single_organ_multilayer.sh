#!/bin/bash
# Phase 2 Parallel Single-Organ Alignment - MULTILAYER feature family.
#
# Thin wrapper over run_phase2_parallel_single_organ.sh that keeps the existing
# checkpoint scheduling and cache reuse behavior while selecting multilayer.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PHASE2_FEATURE_TYPE=multilayer

bash "$ROOT_DIR/scripts/run_phase2_parallel_single_organ.sh" "$@"