# Plugin interface v0.1

The plugin owns `http://127.0.0.1:4783`. No public listener, CORS access, or second
MCP server is implemented. A future frontend needs an authenticated, authorized
host proxy; that proxy and UI are outside this phase.

Every request requires `Authorization: Bearer <api.key contents>`. The key is
generated with 256 bits of randomness and stored owner-readable only in
`~/.local/share/pins-ai/api.key`. Do not put it in URLs or repository files.

| Method / path | Behavior |
| --- | --- |
| GET `/status` | State, enablement, model identifier, owned PID, error, RAM/RSS/swap, capabilities |
| GET `/diagnostics` | Last 40 bounded native log lines; authenticated local debugging only |
| POST `/load` | Verify hash and load configured model, respecting memory limits |
| POST `/unload` | Cancel current inference and release weights/child |
| POST `/cancel` | Same immediate cancellation and release behavior |
| POST `/reload` | Validate the local config file, cancel/unload old runtime, apply new config; optional preload |
| POST `/chat` | Stateless compact history in, ordered SSE out; one active inference |

No endpoint accepts a tool definition, hardware command or arbitrary model path.
Switch models by editing the local config file then calling `/reload`; a remote
caller cannot modify that file through this interface. Port changes need a PINS
restart. Configuration errors are rejected before an existing runtime is stopped.

`/chat` accepts `Content-Type: application/json`:

```json
{"messages":[{"role":"user","content":"Erkläre Autoguiding kurz."}],"locale":"de"}
```

Only `user` and `assistant` roles are accepted. At most eight messages, 2000
characters per message, 6000 total characters and 16 KiB JSON. The final message
must be a user message. Unknown fields, `system`/`tool` roles, and embedded tool
definitions are rejected. Excess token context is rejected by llama.cpp with an
explicit error; history is not silently truncated. Each request supplies its own
history. No session database or cross-user conversation memory is retained.

SSE events:

- `capabilities`: always `mcpAvailable:false`, `toolExecutionEnabled:false`,
  `informationalOnly:true`.
- `token`: `{ "text":"...", "done":false, "timings":null }`.
- `metrics`: optional native usage/timing object in `timings`.
- `done`: stream completed successfully (terminal event).
- `error`: `{ "status":409, "error":"..." }`, terminal failure event.

Errors before streaming use HTTP 400 (bad input), 401 (authentication), 409
(concurrent load), 408 (timeout), 429 (eight request slots occupied), or 503
(disabled/unavailable/failed runtime). A concurrent chat fails with an SSE `error`
409 after capabilities; clients must inspect terminal events, not just HTTP 200.
Responses stream in order with awaited writes and a bounded socket-write deadline.
Disconnects cancel and kill inference when the next write detects the broken socket;
the overall turn deadline also bounds a stalled peer. Cancelling releases the model;
the next request performs a cold lazy load.

Chat is informational and model-generated. It is instructed to explain that
equipment actions and live state are unavailable; small models can still produce
incorrect prose. No prose or generated JSON can ever execute an action in this
phase. Tool accuracy, confirmations and live MCP integration are deferred, not
represented by fabricated tool fixtures.
