#!/usr/bin/env bash
# Shared pod shutdown helpers — source from phase scripts.

_SHUTDOWN_DONE=0

resolve_runpod_pod_id() {
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    return 0
  fi
  for path in /etc/runpod/pod_id /runpod/pod_id; do
    if [[ -f "$path" ]]; then
      export RUNPOD_POD_ID="$(tr -d '[:space:]' < "$path")"
      return 0
    fi
  done
  return 1
}

shutdown_pod() {
  local reason="${1:-unknown}"
  if [[ "${AUTO_TERMINATE_POD:-1}" != "1" ]]; then
    echo "[shutdown] AUTO_TERMINATE_POD disabled; skipping ($reason)"
    return 0
  fi
  if [[ "${_SHUTDOWN_DONE}" == "1" ]]; then
    return 0
  fi
  _SHUTDOWN_DONE=1
  resolve_runpod_pod_id || true
  echo "[shutdown] requesting terminate ($reason) pod_id=${RUNPOD_POD_ID:-UNKNOWN}..."
  python scripts/runpod_shutdown.py --reason "$reason" || true
}

phase1_exit_trap() {
  local ec=$?
  if [[ $ec -eq 0 ]]; then
    if [[ "${AUTO_TERMINATE_ON_SUCCESS:-1}" == "1" ]]; then
      export AUTO_TERMINATE_POD=1
      shutdown_pod "phase1_complete"
    else
      echo "[phase1] success — pod left running (AUTO_TERMINATE_ON_SUCCESS=0)"
    fi
  elif [[ "${AUTO_TERMINATE_ON_FAILURE:-0}" == "1" ]]; then
    export AUTO_TERMINATE_POD=1
    shutdown_pod "phase1_failed_exit_${ec}"
  else
    echo "[phase1] FAILED exit=$ec — pod STAYS RUNNING (no billing stop on error)."
    echo "[phase1] fix then rerun:  bash scripts/train_phase1_now.sh"
    echo "[phase1] logs: /workspace/train_phase1_now.log  /workspace/outputs/train_phase1.log"
  fi
}
