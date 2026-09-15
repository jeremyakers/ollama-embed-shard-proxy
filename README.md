# Ollama embedding shard proxy

A small HTTP proxy that splits one Ollama `/api/embed` input array across two compatible Ollama servers, runs both shards concurrently, and restores the original embedding order.

All other Ollama routes are passed to the first backend. The service is intended for trusted loopback use; it provides no authentication or TLS.

## Requirements

- Go 1.26 or newer to build.
- Exactly two Ollama backends serving the same model, dimensions, tokenizer, pooling, context, and inference settings.
- Backend vectors must be verified compatible before use. Do not mix embedding spaces.

## Build

```bash
go test -race ./...
go build -trimpath -o ~/.local/bin/ollama-embed-shard-proxy .
```

## Run

```bash
~/.local/bin/ollama-embed-shard-proxy \
  -listen 127.0.0.1:11435 \
  -backends http://ollama-a.internal:11434,http://ollama-b.internal:11434
```

Health check:

```bash
curl -fsS http://127.0.0.1:11435/healthz
```

The bundled systemd unit reads backend URLs from an environment file. `/healthz` is a local process-liveness check; backend availability remains visible through proxied Ollama routes.

`/api/embed` accepts either a string or string-array `input`. It preserves `truncate`, `keep_alive`, `dimensions`, and `options`, and forwards `Authorization` and `X-API-Key`. Backend URLs containing userinfo credentials are rejected. A one-item request uses the first backend. A shard failure cancels its sibling and returns HTTP 502 with the root backend error.

## systemd user service

```bash
mkdir -p ~/.config/systemd/user
mkdir -p ~/.config/ollama-embed-shard-proxy
cp ollama-embed-shard-proxy.service ~/.config/systemd/user/
cp ollama-embed-shard-proxy.env.example ~/.config/ollama-embed-shard-proxy/environment
$EDITOR ~/.config/ollama-embed-shard-proxy/environment
systemctl --user daemon-reload
systemctl --user enable --now ollama-embed-shard-proxy.service
```

To keep the user service running without an active login and start it at boot:

```bash
loginctl enable-linger "$USER"
```

Logs and status:

```bash
systemctl --user status ollama-embed-shard-proxy.service
journalctl --user -u ollama-embed-shard-proxy.service
```

## grepai

```yaml
embedder:
  provider: ollama
  endpoint: http://127.0.0.1:11435
```

Grepai must use Ollama's array-based `/api/embed` endpoint to benefit from sharding. Older grepai releases that call `/api/embeddings` continue to work through first-backend pass-through but receive no sharding speedup.

## Measured result

With two vector-identical GB10 Ollama nodes and 64 representative code chunks:

| Path | Median time |
| --- | ---: |
| One GB10 | 2.40 seconds |
| Sharded proxy | 1.26 seconds |

The full Qwen workspace rebuild completed through this proxy. The proxy exposed a masked context-error bug during that run; a regression test now locks the corrected root-error behavior.

## Deliberate limitations

- Exactly two static backends.
- At most 128 inputs, a 2 MiB request body, and a 16 MiB response per shard.
- No authentication, TLS, discovery, metrics, retries, or circuit breaker.
- Sharded backend redirects are not followed.
- Non-`/api/embed` routes always use the first backend.
- Backend health and vector compatibility are deployment responsibilities.

## License

MIT
