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
    shutdown_pod "phase1_complete"
  elif [[ "${AUTO_TERMINATE_ON_FAILURE:-0}" == "1" ]]; then
    shutdown_pod "phase1_failed_exit_${ec}"
  else
    echo "[phase1] exited with code $ec — pod LEFT RUNNING for debug/resume."
    echo "[phase1] logs: /workspace/run.log /workspace/outputs/train_phase1.log"
    echo "[phase1] resume: tmux attach -t qlora  OR  bash scripts/resume_train_only.sh"
  fi
}
