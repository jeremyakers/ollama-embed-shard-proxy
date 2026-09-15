package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	maxRequestBodySize  = 2 << 20
	maxResponseBodySize = 16 << 20
	maxInputCount       = 128
	shutdownTimeout     = 10 * time.Minute
)

type embedRequest struct {
	Model      string          `json:"model"`
	Input      embedInput      `json:"input"`
	Truncate   *bool           `json:"truncate,omitempty"`
	KeepAlive  json.RawMessage `json:"keep_alive,omitempty"`
	Dimensions *int            `json:"dimensions,omitempty"`
	Options    json.RawMessage `json:"options,omitempty"`
}

type embedInput []string

func (i *embedInput) UnmarshalJSON(data []byte) error {
	var inputs []string
	if err := json.Unmarshal(data, &inputs); err == nil {
		*i = inputs
		return nil
	}
	var input string
	if err := json.Unmarshal(data, &input); err != nil {
		return fmt.Errorf("input must be a string or string array: %w", err)
	}
	*i = []string{input}
	return nil
}

type embedResponse struct {
	Model           string      `json:"model,omitempty"`
	Embeddings      [][]float32 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration,omitempty"`
	LoadDuration    int64       `json:"load_duration,omitempty"`
	PromptEvalCount int         `json:"prompt_eval_count,omitempty"`
}

type shardResult struct {
	response embedResponse
	err      error
}

type shardRequest struct {
	embedRequest embedRequest
	inputs       []string
	headers      http.Header
}

type proxy struct {
	backends [2]*url.URL
	client   *http.Client
	fallback http.Handler
}

func newProxy(backends [2]*url.URL, client *http.Client) http.Handler {
	fallback := httputil.NewSingleHostReverseProxy(backends[0])
	direct := fallback.Director
	fallback.Director = func(request *http.Request) {
		direct(request)
		request.Host = backends[0].Host
	}
	if client.Transport != nil {
		fallback.Transport = client.Transport
	}
	fallback.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		http.Error(w, fmt.Sprintf("backend 0: %v", err), http.StatusBadGateway)
	}
	return &proxy{backends: backends, client: client, fallback: fallback}
}

func parseBackends(value string) ([2]*url.URL, error) {
	var backends [2]*url.URL
	parts := strings.Split(value, ",")
	if len(parts) != len(backends) {
		return backends, fmt.Errorf("expected exactly two comma-separated backends")
	}
	for i, part := range parts {
		backend, err := url.Parse(strings.TrimSpace(part))
		if err != nil {
			return backends, fmt.Errorf("parse backend %d: %w", i, err)
		}
		if (backend.Scheme != "http" && backend.Scheme != "https") || backend.Host == "" {
			return backends, fmt.Errorf("backend %d must be an HTTP URL with a host", i)
		}
		backends[i] = backend
	}
	return backends, nil
}

func validateTimeout(value time.Duration) error {
	if value <= 0 {
		return fmt.Errorf("timeout must be positive")
	}
	return nil
}

func (p *proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/healthz" {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.URL.Path != "/api/embed" {
		if p.client.Timeout <= 0 {
			p.fallback.ServeHTTP(w, r)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), p.client.Timeout)
		defer cancel()
		p.fallback.ServeHTTP(w, r.WithContext(ctx))
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request embedRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRequestBodySize))
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("invalid embed request: %v", err), http.StatusBadRequest)
		return
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		http.Error(w, "invalid embed request: trailing data", http.StatusBadRequest)
		return
	}
	if request.Model == "" || len(request.Input) == 0 {
		http.Error(w, "model and input are required", http.StatusBadRequest)
		return
	}
	if len(request.Input) > maxInputCount {
		http.Error(w, fmt.Sprintf("input count exceeds maximum of %d", maxInputCount), http.StatusRequestEntityTooLarge)
		return
	}

	middle := (len(request.Input) + 1) / 2
	started := time.Now()
	shards := [][]string{[]string(request.Input[:middle]), []string(request.Input[middle:])}
	results := make([]shardResult, len(shards))
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	var waitGroup sync.WaitGroup
	for i, inputs := range shards {
		if len(inputs) == 0 {
			continue
		}
		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()
			shard := shardRequest{embedRequest: request, inputs: inputs, headers: r.Header}
			results[i].response, results[i].err = p.embed(ctx, p.backends[i], shard)
			if results[i].err != nil {
				cancel()
			}
		}()
	}
	waitGroup.Wait()

	failureIndex := -1
	for i, result := range results {
		if result.err != nil && !errors.Is(result.err, context.Canceled) {
			failureIndex = i
			break
		}
		if result.err != nil && failureIndex == -1 {
			failureIndex = i
		}
	}
	if failureIndex >= 0 {
		http.Error(w, fmt.Sprintf("backend %d: %v", failureIndex, results[failureIndex].err), http.StatusBadGateway)
		return
	}

	combined := embedResponse{Model: request.Model, TotalDuration: time.Since(started).Nanoseconds()}
	for _, result := range results {
		combined.Embeddings = append(combined.Embeddings, result.response.Embeddings...)
		if result.response.LoadDuration > combined.LoadDuration {
			combined.LoadDuration = result.response.LoadDuration
		}
		combined.PromptEvalCount += result.response.PromptEvalCount
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(combined); err != nil {
		return
	}
}

func (p *proxy) embed(ctx context.Context, backend *url.URL, shard shardRequest) (embedResponse, error) {
	request := shard.embedRequest
	request.Input = embedInput(shard.inputs)
	body, err := json.Marshal(request)
	if err != nil {
		return embedResponse{}, fmt.Errorf("marshal request: %w", err)
	}

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, backend.JoinPath("api", "embed").String(), bytes.NewReader(body))
	if err != nil {
		return embedResponse{}, fmt.Errorf("create request: %w", err)
	}
	httpRequest.Header = shard.headers.Clone()
	removeHopByHopHeaders(httpRequest.Header)
	httpRequest.Header.Set("Content-Type", "application/json")
	response, err := p.client.Do(httpRequest)
	if err != nil {
		return embedResponse{}, fmt.Errorf("send request: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, err := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		if err != nil {
			return embedResponse{}, fmt.Errorf("read error response: %w", err)
		}
		return embedResponse{}, fmt.Errorf("status %d: %s", response.StatusCode, body)
	}

	var result embedResponse
	limitedBody := &io.LimitedReader{R: response.Body, N: maxResponseBodySize + 1}
	decoder := json.NewDecoder(limitedBody)
	if err := decoder.Decode(&result); err != nil {
		return embedResponse{}, fmt.Errorf("decode response: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return embedResponse{}, fmt.Errorf("decode response: trailing data")
	}
	if limitedBody.N == 0 {
		return embedResponse{}, fmt.Errorf("decode response: body exceeds %d bytes", maxResponseBodySize)
	}
	if len(result.Embeddings) != len(shard.inputs) {
		return embedResponse{}, fmt.Errorf("expected %d embeddings, got %d", len(shard.inputs), len(result.Embeddings))
	}
	return result, nil
}

func removeHopByHopHeaders(headers http.Header) {
	for _, value := range headers.Values("Connection") {
		for _, name := range strings.Split(value, ",") {
			headers.Del(strings.TrimSpace(name))
		}
	}
	for _, name := range []string{
		"Accept-Encoding",
		"Connection",
		"Content-Encoding",
		"Content-Length",
		"Keep-Alive",
		"Proxy-Authenticate",
		"Proxy-Authorization",
		"Te",
		"Trailer",
		"Transfer-Encoding",
		"Upgrade",
	} {
		headers.Del(name)
	}
}

func serve(ctx context.Context, server *http.Server, listener net.Listener) error {
	serveResult := make(chan error, 1)
	go func() {
		serveResult <- server.Serve(listener)
	}()

	select {
	case err := <-serveResult:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer cancel()
		shutdownErr := server.Shutdown(shutdownCtx)
		serveErr := <-serveResult
		if errors.Is(serveErr, http.ErrServerClosed) {
			serveErr = nil
		}
		return errors.Join(shutdownErr, serveErr)
	}
}

func main() {
	listenAddress := flag.String("listen", "127.0.0.1:11435", "proxy listen address")
	backendList := flag.String("backends", "", "two comma-separated Ollama base URLs")
	requestTimeout := flag.Duration("timeout", 10*time.Minute, "backend request timeout")
	flag.Parse()

	if err := validateTimeout(*requestTimeout); err != nil {
		log.Fatal(err)
	}
	backends, err := parseBackends(*backendList)
	if err != nil {
		log.Fatal(err)
	}
	server := &http.Server{
		Addr:              *listenAddress,
		Handler:           newProxy(backends, &http.Client{Timeout: *requestTimeout}),
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       time.Minute,
	}
	listener, err := net.Listen("tcp", *listenAddress)
	if err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	log.Printf("listening on %s; sharding between %s and %s", listener.Addr(), backends[0], backends[1])
	if err := serve(ctx, server, listener); err != nil {
		log.Fatal(err)
	}
}
