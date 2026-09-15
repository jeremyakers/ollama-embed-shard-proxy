package main

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestProxy_shards_concurrently_and_preserves_order(t *testing.T) {
	// Given
	arrived := make(chan struct{})
	var arrivals atomic.Int64
	var firstInputs []string
	var secondInputs []string
	backend := func(inputs *[]string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var request struct {
				Input []string `json:"input"`
			}
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			*inputs = append([]string(nil), request.Input...)
			if arrivals.Add(1) == 2 {
				close(arrived)
			}
			select {
			case <-arrived:
			case <-r.Context().Done():
				return
			}
			embeddings := make([][]float32, len(request.Input))
			for i, input := range request.Input {
				embeddings[i] = []float32{float32(len(input))}
			}
			w.Header().Set("Content-Type", "application/json")
			response := struct {
				Embeddings [][]float32 `json:"embeddings"`
			}{Embeddings: embeddings}
			if err := json.NewEncoder(w).Encode(response); err != nil {
				t.Errorf("encode response: %v", err)
			}
		}))
	}
	first := backend(&firstInputs)
	second := backend(&secondInputs)
	t.Cleanup(first.Close)
	t.Cleanup(second.Close)
	firstURL := parseTestURL(t, first.URL)
	secondURL := parseTestURL(t, second.URL)
	handler := newProxy([2]*url.URL{firstURL, secondURL}, &http.Client{Timeout: time.Second})
	requestBody := `{"model":"test","input":["a","bb","ccc","dddd","eeeee"],"truncate":false}`
	request := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(requestBody))
	response := httptest.NewRecorder()

	// When
	handler.ServeHTTP(response, request)

	// Then
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", response.Code, response.Body.String())
	}
	if !reflect.DeepEqual(firstInputs, []string{"a", "bb", "ccc"}) {
		t.Fatalf("first inputs = %v, want [a bb ccc]", firstInputs)
	}
	if !reflect.DeepEqual(secondInputs, []string{"dddd", "eeeee"}) {
		t.Fatalf("second inputs = %v, want [dddd eeeee]", secondInputs)
	}
	var result struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	want := [][]float32{{1}, {2}, {3}, {4}, {5}}
	if !reflect.DeepEqual(result.Embeddings, want) {
		t.Fatalf("embeddings = %v, want %v", result.Embeddings, want)
	}
}

func TestProxy_passes_non_batch_routes_to_the_first_backend(t *testing.T) {
	// Given
	var secondRequests atomic.Int64
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			http.Error(w, "unexpected path", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{"models":[{"name":"test"}]}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	second := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		secondRequests.Add(1)
	}))
	t.Cleanup(first.Close)
	t.Cleanup(second.Close)
	firstURL := parseTestURL(t, first.URL)
	secondURL := parseTestURL(t, second.URL)
	handler := newProxy([2]*url.URL{firstURL, secondURL}, &http.Client{Timeout: time.Second})
	request := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	response := httptest.NewRecorder()

	// When
	handler.ServeHTTP(response, request)

	// Then
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", response.Code, response.Body.String())
	}
	if response.Body.String() != `{"models":[{"name":"test"}]}` {
		t.Fatalf("body = %q, want backend response", response.Body.String())
	}
	if secondRequests.Load() != 0 {
		t.Fatalf("second backend requests = %d, want 0", secondRequests.Load())
	}
}

func TestProxy_applies_timeout_to_pass_through_routes(t *testing.T) {
	// Given
	firstURL := parseTestURL(t, "http://first.example")
	secondURL := parseTestURL(t, "http://second.example")
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if _, ok := request.Context().Deadline(); !ok {
			return nil, errors.New("request has no deadline")
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{"models":[]}`)),
			Header:     make(http.Header),
			Request:    request,
		}, nil
	})
	handler := newProxy(
		[2]*url.URL{firstURL, secondURL},
		&http.Client{Transport: transport, Timeout: time.Second},
	)
	request := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	response := httptest.NewRecorder()

	// When
	handler.ServeHTTP(response, request)

	// Then
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", response.Code, response.Body.String())
	}
}

func TestProxy_health_is_local(t *testing.T) {
	// Given
	var backendRequests atomic.Int64
	backend := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		backendRequests.Add(1)
	}))
	t.Cleanup(backend.Close)
	backendURL := parseTestURL(t, backend.URL)
	handler := newProxy([2]*url.URL{backendURL, backendURL}, &http.Client{Timeout: time.Second})
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	response := httptest.NewRecorder()

	// When
	handler.ServeHTTP(response, request)

	// Then
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", response.Code)
	}
	if backendRequests.Load() != 0 {
		t.Fatalf("backend requests = %d, want 0", backendRequests.Load())
	}
}

func TestProxy_uses_only_the_first_backend_for_one_input(t *testing.T) {
	// Given
	var firstRequests atomic.Int64
	var secondRequests atomic.Int64
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		firstRequests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{"embeddings":[[1]]}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	second := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		secondRequests.Add(1)
	}))
	t.Cleanup(first.Close)
	t.Cleanup(second.Close)
	firstURL := parseTestURL(t, first.URL)
	secondURL := parseTestURL(t, second.URL)
	handler := newProxy([2]*url.URL{firstURL, secondURL}, &http.Client{Timeout: time.Second})
	request := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(`{"model":"test","input":["only"]}`))
	response := httptest.NewRecorder()

	// When
	handler.ServeHTTP(response, request)

	// Then
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", response.Code, response.Body.String())
	}
	if firstRequests.Load() != 1 || secondRequests.Load() != 0 {
		t.Fatalf("request counts = first:%d second:%d, want first:1 second:0", firstRequests.Load(), secondRequests.Load())
	}
}

func TestProxy_accepts_string_input_and_preserves_embed_options(t *testing.T) {
	// Given
	var received map[string]json.RawMessage
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{"embeddings":[[1]]}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	second := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("second backend must not receive a one-input request")
	}))
	t.Cleanup(first.Close)
	t.Cleanup(second.Close)
	handler := newProxy(
		[2]*url.URL{parseTestURL(t, first.URL), parseTestURL(t, second.URL)},
		&http.Client{Timeout: time.Second},
	)
	requestBody := `{"model":"test","input":"only","truncate":false,"keep_alive":-1,"dimensions":512,"options":{"num_ctx":4096}}`
	request := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(requestBody))
	response := httptest.NewRecorder()

	// When
	handler.ServeHTTP(response, request)

	// Then
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", response.Code, response.Body.String())
	}
	want := map[string]string{
		"input":      `["only"]`,
		"truncate":   `false`,
		"keep_alive": `-1`,
		"dimensions": `512`,
		"options":    `{"num_ctx":4096}`,
	}
	for field, value := range want {
		if string(received[field]) != value {
			t.Fatalf("field %s = %s, want %s", field, received[field], value)
		}
	}
}

func TestProxy_returns_bad_gateway_when_a_shard_fails(t *testing.T) {
	// Given
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "backend unavailable", http.StatusServiceUnavailable)
	}))
	second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{"embeddings":[[2]]}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	t.Cleanup(first.Close)
	t.Cleanup(second.Close)
	firstURL := parseTestURL(t, first.URL)
	secondURL := parseTestURL(t, second.URL)
	handler := newProxy([2]*url.URL{firstURL, secondURL}, &http.Client{Timeout: time.Second})
	request := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(`{"model":"test","input":["one","two"]}`))
	response := httptest.NewRecorder()

	// When
	handler.ServeHTTP(response, request)

	// Then
	if response.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body = %s", response.Code, response.Body.String())
	}
}

func TestProxy_rejects_trailing_request_data(t *testing.T) {
	// Given
	var backendRequests atomic.Int64
	backend := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		backendRequests.Add(1)
	}))
	t.Cleanup(backend.Close)
	backendURL := parseTestURL(t, backend.URL)
	handler := newProxy([2]*url.URL{backendURL, backendURL}, &http.Client{Timeout: time.Second})
	request := httptest.NewRequest(
		http.MethodPost,
		"/api/embed",
		strings.NewReader(`{"model":"test","input":["one"]} trailing`),
	)
	response := httptest.NewRecorder()

	// When
	handler.ServeHTTP(response, request)

	// Then
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", response.Code)
	}
	if backendRequests.Load() != 0 {
		t.Fatalf("backend requests = %d, want 0", backendRequests.Load())
	}
}

func TestProxy_rejects_trailing_backend_response_data(t *testing.T) {
	// Given
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{"embeddings":[[1]]} trailing`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	t.Cleanup(backend.Close)
	backendURL := parseTestURL(t, backend.URL)
	handler := newProxy([2]*url.URL{backendURL, backendURL}, &http.Client{Timeout: time.Second})
	request := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(`{"model":"test","input":["one"]}`))
	response := httptest.NewRecorder()

	// When
	handler.ServeHTTP(response, request)

	// Then
	if response.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body = %s", response.Code, response.Body.String())
	}
}

func TestProxy_reports_the_root_error_instead_of_a_canceled_sibling(t *testing.T) {
	// Given
	firstURL := parseTestURL(t, "http://first.example")
	secondURL := parseTestURL(t, "http://second.example")
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Host == firstURL.Host {
			<-request.Context().Done()
			return nil, request.Context().Err()
		}
		return &http.Response{
			StatusCode: http.StatusBadRequest,
			Body:       io.NopCloser(strings.NewReader("input exceeds the context length")),
			Header:     make(http.Header),
			Request:    request,
		}, nil
	})
	handler := newProxy([2]*url.URL{firstURL, secondURL}, &http.Client{Transport: transport, Timeout: time.Second})
	request := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(`{"model":"test","input":["one","two"]}`))
	response := httptest.NewRecorder()

	// When
	handler.ServeHTTP(response, request)

	// Then
	if response.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", response.Code)
	}
	if !strings.Contains(response.Body.String(), "backend 1: status 400: input exceeds the context length") {
		t.Fatalf("body = %q, want backend 1 context error", response.Body.String())
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func parseTestURL(t *testing.T, value string) *url.URL {
	t.Helper()
	parsed, err := url.Parse(value)
	if err != nil {
		t.Fatalf("parse test URL %q: %v", value, err)
	}
	return parsed
}
