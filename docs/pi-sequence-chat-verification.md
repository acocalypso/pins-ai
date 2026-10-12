# Thinking indicator, scrolling and sequence planning verification

Date: 2026-10-12. AI code commit `750504b`; frontend commit `bccbce0d`.
This supplements the earlier one-thread measurements in
`pi-chat-verification.md`; those historical measurements remain unchanged.

## Local verification

174 gateway checks passed with simulated observatory operations. Five curriculum
tests passed, including train-only placement and proposal-only sequence plans.
Frontend lint, type checking, 863 tests, formatting, i18n checks and production
build passed. Functionality, maintainability and UI review findings were fixed.

Playwright tested the production frontend against the Pi's real session gateway,
with delayed chat replies and a long HTTP 500 error simulated only in the browser.
No inference or equipment command was sent by those layout fixtures. The spinner
and accessible “Model is thinking” status appeared; drafting stayed enabled and
Send stayed disabled during requests. Debug stayed visible. After six replies and
the error, document height still equalled viewport height:

| Viewport | Document height | Composer bottom | Conversation viewport after replies |
|---|---:|---:|---:|
| 390 × 844 | 844 px | 750 px | 224 px |
| 844 × 390 | 390 px | 304 px | 60 px |
| 640 × 360 | 360 px | 274 px | 30 px |

At 640 × 360 the thinking status temporarily consumes the remaining conversation
height, while the input and cancel button stay visible. The surrounding app
remounts route views on orientation changes, so tests reconnected after rotation.
Physical Android/iOS keyboards were not tested.

## Sequence capability

Upstream unstable inspected at `6fef6c4bd73a8e1f24a36235dba6cbb9eff3d3d2`.
`add_target` creates a DSO container and Smart Exposure blocks in the loaded
sequence. The requested count is 300 and duration is 300 seconds; omit `filter`
to avoid a filter change. MCP has no name-to-coordinate resolver or new blank
sequence-root tool. Verified J2000 RA/Dec and append/replace intent are needed.
See `sequence-planning.md` for the new train-only examples and retraining limits.

## Live Pi verification

[Linux ARM64 build 38278189186](https://github.com/acocalypso/pins/actions/runs/38278189186)
succeeded. Four SHA-256-verified Debian packages from release
`unstable-38278189186-1` were installed; PINS restarted and remained active. The
packaged frontend returned HTTP 200 at port 5000. Trusted device/workstation chat
access still needs no manually entered token. Missing/untrusted origins returned
401. The gateway exposed 31 tools, completed a real read-only MCP query and
rejected a connection preview without executing it.

The native process was observed with `--threads 2 --threads-batch 2`.
Short automatic chat probes omitted `toolNames`:

| Prompt | Outcome | Elapsed |
|---|---|---:|
| Show the current equipment status | Real `get_status` result | 8.730 s |
| Is my equipment connected? | Text: “No, PINS reports that all equipment is not connected.” | 3.890 s |
| Zeige den aktuellen Gerätestatus. | Real `get_status` result; shared prompt cache | 1.240 s |

Cold status used 291 input tokens: prefill 67.55 tokens/s and decoding 18.63
tokens/s. These are short probes with a changed system prompt, not a controlled
thread-count benchmark. Peak measured model RSS was 974,992 KiB (952.1 MiB),
and peak temperature across these three probes was 65.0°C. The model was unloaded
afterward. The 75°C cutoff and existing memory limits were retained.

The exact M31 prompt completed in 18.172 s but returned HTTP 400:
**“Choose exactly one target plan.”** The adapter proposed an invalid plan rather
than asking for missing information; full gateway validation refused it before
execution. Temperature afterward was 66.65°C. No target was added and imaging
was not started. The model was unloaded again. This remains a diagnostic adapter
failure; adding training examples has not updated its weights. Retrain and
evaluate the new curriculum before claiming this request works reliably.

Real equipment mutations, active-imaging contention, sustained thermal behavior
and physical mobile keyboards remain unverified.
