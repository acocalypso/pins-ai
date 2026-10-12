# Connection chat correction — 2026-10-12

The diagnostic checkpoint 700 produced unrelated Dutch prose for the German
request `Verbinde die Kamera und die Montierung`. The previous chat diagnostics
also inherited informational-only MCP capability defaults.

The plugin now recognizes complete, explicit English/German device connection
instructions as bounded proposals. Each device still requires independent exact
operator approval. The next preview waits for confirmed job completion. Profile
changes, failures, rejection and cancellation stop the plan. Other requests use
the model with the latest user language prioritized and locale used as fallback.
This does not establish general multilingual model accuracy or promote a model.

Diagnostics now probe MCP discovery. The frontend shows queued arguments,
appends fresh continuation approvals, displays a translated stopped-plan warning
without losing completed results, and disables superseded approval controls.
Stale approval attempts cannot revoke a newer server-side plan.

## Measured checks

- 197 simulated gateway checks passed, including two independently approved
  operations, job polling, failed jobs, rejection, cancellation, stale approvals,
  replay prevention and profile changes between operations.
- 28 runtime checks passed.
- All 867 frontend tests passed, including four new reply/approval tests.
  The four focused tests passed again after the final display correction.
- Frontend lint, production build, translation integrity and formatting passed.
  Whole-project type checking passed with an 8 GiB Node heap; the initial default
  4 GiB attempt exhausted its heap. One test run overlapping other checks had an
  unrelated sequenceAutoLock worker failure; the final complete run passed.
- Deployed managed runtime and frontend to the existing Pi plugins. Checked
  real MCP status for idle imaging/jobs before the PINS restart. Preserved the
  config hash and existing model. Backups are on the Pi under
  `/home/pi/.local/share/pins-ai-tests/chat-fix-20261012`.
- Live MCP discovery reports 32 tools; the audited chat gateway exposes 31.
  Real read-only `get_status` succeeded. Missing/untrusted browser origins
  returned 401. German camera+mount chat produced a camera preview and queued
  mount action without inference. The preview was rejected. These live checks
  passed again after the final frontend deployment.
- Playwright against the Pi UI verified Start → experimental modal → chat,
  the two-device preview, and older approval controls disappearing when a new
  request replaces them.

No live equipment action was approved or executed. Hardware completion and
next-approval behavior were tested with simulated MCP responses. Native mobile
builds and general free-form multilingual model accuracy were not evaluated.
The model remains checkpoint 700 and loads on demand; checkpoint 400 remains
unpromoted. Subsequent commits include the source changes and this report.
