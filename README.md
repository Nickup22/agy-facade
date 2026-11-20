# agy-facade

An unofficial, local HTTP adapter that lets OpenAI-style clients talk to an installed Google Antigravity CLI (`agy`). It runs the CLI as a subprocess and translates its `stream-json` events into chat completion responses. It does not implement Google's network protocol or manage Google sign-in or OAuth tokens. It is not affiliated with Google or OpenAI.

**Status: experimental, single-user and local-only.** This is the classic `stream-json` implementation, not the separate ACP prototype. The public snapshot was tested offline with a mock backend; live compatibility with current `agy`, individual models, and media is not confirmed. Check the [Antigravity terms](https://antigravity.google/terms) before use: invoking the installed CLI is not a promise of provider approval, continued compatibility, or quota availability.

## Why use it?

- It is one Python file with no third-party Python dependencies. If your client supports a custom OpenAI base URL, it can use the same local interface for multiple models exposed by `agy`.
- Chat supports ordinary JSON replies and incremental SSE streaming. A warmed process pool reduces startup overhead; a fresh process is used for each turn and discarded afterward.
- The request includes the conversation history, so the caller remains in charge of what context to send. The adapter assembles roles and media into a single CLI turn.
- Inline images and audio can be staged for the CLI. In practice, what the model reads depends on the current CLI, its permissions, and the model. The server does not verify that the model looked at an attachment.

## Start locally

You need Python 3 and an installed, working `agy` CLI. The included checks were run on Python 3.13 on Linux; older versions and other OSes are not verified. Set up and authenticate `agy` separately using the provider's instructions, then verify `agy models` works. The facade resolves `agy` from `PATH`, the `AGY_BIN` environment variable, or the `agy_bin` configuration field (the configuration takes precedence).

From a fresh local copy, on a POSIX shell:

```sh
umask 077
python3 agy_facade.py keys add local-client
python3 agy_facade.py
```

The first command creates a **facade** key, not a Google credential. Copy it to your client and keep it private. The server creates a local `agy_config.json` with defaults if one does not exist; set `default_model` to a model your CLI actually exposes if the supplied default is unavailable. Default base URL: `http://127.0.0.1:8321/v1`. `--port` and `--config` can be used for local testing. The server only accepts `127.0.0.1` and does not start without an active key. Key management: `keys list`, `keys add <name>`, `keys disable <name>`, `keys enable <name>`, `keys remove <name>`.

In a second terminal, set `AGY_FACADE_KEY` privately and either omit `model` to use the configured default or set it to an ID from `/v1/models`:

```sh
curl -sS http://127.0.0.1:8321/v1/models \
  -H "Authorization: Bearer $AGY_FACADE_KEY"

curl -sS http://127.0.0.1:8321/v1/chat/completions \
  -H "Authorization: Bearer $AGY_FACADE_KEY" \
  -H 'Content-Type: application/json' \
  --data '{"messages":[{"role":"user","content":"Reply with one sentence about the Moon."}]}'
```

For streaming, add `"stream":true` to the request body and use `curl -N`. For the first few seconds the model list can be incomplete while the CLI catalog loads. Entries from the fallback catalog are not proof of access.

## API and trade-offs

| Route | What is implemented |
| --- | --- |
| `GET /` | Local status and process counters |
| `GET /v1/models` | Cached CLI catalog, with a fallback and optional `extra_models` |
| `POST /v1/chat/completions` | Text, JSON or SSE; inline image/audio input on a best-effort basis |
| `POST /v1/images/generations` | Experimental CLI image generation, disabled by default; `url` or `b64_json` result |
| `GET /v1/images/<filename>` | Download a generated image with the same facade authentication |

- **Partial OpenAI compatibility.** There is no Responses API, embeddings, transcription, or OpenAI function-calling contract. Don't assume sampling settings, structured outputs, or token limits are honored. Native role separation is lost when the history is assembled into one turn; usage fields depend on what the CLI reports.
- **Not a throughput server.** Chat and image generation share one FIFO turn queue. Only one request can acquire a CLI worker at a time, including when other processes are prewarmed. Upstream errors on a stream arrive after the HTTP headers.
- **Media is the least portable part.** Put inline data URIs or base64 in supported content blocks. The public snapshot does not fetch client-supplied media URLs or open client-supplied paths. Audio is not supported by every model. The adapter deduplicates attachments and drops older history media when selecting which files to send. Live media behavior is not covered by the offline tests.
- **Image generation is opt-in.** Set `enable_image_generation` to `true` only in a dedicated CLI profile/OS account for this project. It relies on an image appearing among new CLI session artifacts and may select the wrong artifact if another session is generating images at the same time. Standard CLI permission prompts and model availability can prevent completion. Prefer `b64_json` for clients that cannot attach auth headers to image downloads.

## Queue and overload

Authenticated POSTs enter FIFO order **after the complete body has been received**, so a slow upload does not block ready requests from taking a turn. Media is decoded only when the request reaches the head. This keeps one CLI turn active without pooling requests or sharing their SSE frames. The experimental image-generation routes use the same queue. The old `max_concurrent` setting does not enable multiple turns.

| Configuration | Default | Applies to |
| --- | ---: | --- |
| `max_queued_requests` | 16 | Waiting requests, excluding the active one; `0` disables waiting |
| `queue_timeout_sec` | 300 | Maximum wait to start a turn, separate from CLI startup and inference |
| `max_request_bytes` | 24 MiB | One request body (`413` if exceeded) |
| `max_inflight_body_bytes` | 96 MiB | Combined declared body size for uploads, waiting requests, and the active request; not a process memory cap |
| `body_timeout_sec` | 60 | Total time allowed to upload the body (`408` on timeout) |
| `http_io_timeout_sec` | 15 | Socket inactivity during upload or writes, not the maximum inference time |

Queue or byte-budget exhaustion returns `429` with `Retry-After`; expiry in the queue returns `503` with a distinct `queue_timeout` error and no CLI turn. An unavailable CLI worker can still return a different `503`. Connections stay open without response headers while waiting, including SSE clients: set the client's timeout to cover queue wait, CLI startup, pacing, and generation. Closing the connection drops a queued request; TCP write-half-close after upload is not supported. The server closes excess sockets above a separate 48-connection guard, which can surface as a connection error rather than HTTP `429`.

Authenticated `GET /` reports `queue.active`, `queue.waiting`, and body budget usage. When polling status from a separate connection, do not treat queue position as a completion estimate.

## Security and data

Use only with a trusted local client. Do not publish the listener through a tunnel, reverse proxy, or port forward. A bearer key is not tenant isolation, and the facade is not a sandbox. It runs the CLI under the local OS account with that account's access. The public snapshot does not override CLI permission checks. Text instructions in a prompt are not an access-control boundary. Keep CLI permissions restrictive and use a dedicated low-privilege account or isolated environment for experiments.

The CLI can send prompts and attachments to its provider. Uploaded media can remain in CLI session storage and the facade's `agy_workdir/`; generated output remains in `agy_images/`. Session cleanup is off by default and is not a data-retention guarantee. Keep those directories and `agy_keys.json`, config files and logs private. Routine chat logging omits prompt/response previews, but upstream errors and CLI state may still contain sensitive material. Do not share logs or directories without reviewing them.

## Checks and possible next steps

```sh
python3 -B -m unittest discover -s tests -v
```

The tests use temporary state and a mock CLI backend. They exercise HTTP auth, request handling, chat JSON and SSE, inline-media handling, FIFO order, overload, and cancellation without contacting Google. They are not an end-to-end certification against any specific CLI release.

Possible next steps, not promises: record a live compatibility matrix for specific CLI releases, tighten media retention and error reporting, and evaluate ACP as a separate transport once capability negotiation and media handling are reliable. The goal is a dependable local adapter, not a public proxy service.

No license has been selected for this public snapshot. Public visibility does not grant permission to redistribute or reuse the source.
