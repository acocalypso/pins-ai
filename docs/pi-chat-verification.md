# Pi chat verification — 2026-10-12

Installed all four checksum-verified Debian packages from Linux ARM64 build
[38271540485](https://github.com/acocalypso/pins/actions/runs/38271540485), release
`unstable-38271540485-1`. AI source: `82c5d0f`; frontend source: `6dc8376d`.
The package includes the unchanged diagnostic checkpoint-700 adapter and model.
PINS and its existing MCP server were running during the checks.

## Timeout repair

The previous packaged runtime targeted generic ARMv8. A prior probe starting
near 65°C returned HTTP 408 after 14.526 seconds, near the configured 75°C
cutoff. The same pinned llama.cpp version built natively for the Pi answered
an equipment-status question in 11.309 seconds. The replacement package targets
Pi 5 ARMv8.2 with FP16/dot-product instructions. This is a Pi 5 CPU target,
not a generic ARM64 binary for older processors.

Model batch threads and OpenMP threads are bounded to one. Native full-prompt
prefix caching is enabled. Guard failures retain their diagnostic cause and
kill the model child without permanently cancelling future inference. The
75°C cutoff and physical RAM/process-swap safeguards remain enabled.

## Automatic chat probes

Requests omitted `toolNames`. The first two shared a server-held conversation;
the German request used a new session. No equipment actions were authorized.

| Message | HTTP | Response | Wall time |
| --- | --- | --- | --- |
| Show the current equipment status | 200 | Native `get_status` call and real MCP result | 12.206 s |
| Is my equipment connected? | 200 | Model text using the preceding tool result | 6.634 s |
| Zeige den aktuellen Gerätestatus. | 200 | Native `get_status` call and real MCP result | 1.900 s |

Observed peak model RSS: **974,896 KiB**; peak CPU temperature: **52.35°C**.
The first turn processed 265 prompt tokens at 35.30 tokens/s and generated
16 tokens at 13.15 tokens/s. The second turn reused 261 cached prefix tokens;
the fresh German session reused only the matching prompt prefix, not prior
conversation messages. The model was unloaded after the probes.

The Pi had been restarted and was substantially cooler than in the earlier
failed probe. These are conservative smoke tests, not a controlled sustained
thermal benchmark. The reboot also left its clock at October 6; NTP was
enabled and synchronization confirmed before browser/approval verification.
Elapsed probe times use a monotonic clock and are unaffected by this correction.

## Browser and gateway

The packaged frontend at `http://192.168.178.109:5000` passed Start → experimental
warning → Continue → connected chat without a manually entered token. Typing
“Is my equipment connected?” selected `get_status` automatically and displayed
its real MCP result with the model ready. Normal chat has no tool-scope selector.
The known equipment-status shortcut issues its read-only query directly.

Missing/untrusted origins returned 401. The trusted test client could discover
31 exposed tools, query status, and reject a connection preview. No connection,
mount motion, exposure or other equipment mutation was executed. `set_switch`
was not exposed because no verified writable switch capability was available.

168 gateway checks passed with simulated observatory operations. Frontend
validation passed: 863 tests, typecheck (8 GiB Node heap), lint, formatting,
i18n validation and production build. Functionality, UI and maintainability
reviews passed after clarification-chain and multilingual routing corrections.

## Remaining limits

Checkpoint 700 is diagnostic. The English follow-up said **“Yes, the equipment
is not connected.”** The negative fact agrees with the tool result, but the
opening “Yes” is misleading. Model prose is explicitly unverified; structured
MCP data remains the source of truth. Better fine-tuning/evaluation is still
needed for reliable natural-language explanations.

Tool retrieval uses bounded multilingual aliases, not an unrestricted semantic
router. Unrecognized phrasing may need clarification. Real hardware mutations,
all multilingual operations, active imaging contention, sustained hot operation,
and Android/iOS devices were not tested. A forced thermal-cutoff recovery test
was not verified; no temperature limit was raised for these checks.
