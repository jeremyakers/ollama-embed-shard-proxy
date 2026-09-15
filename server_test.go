package main

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

func TestServe_shuts_down_when_context_is_canceled(t *testing.T) {
	// Given
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	server := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
		}),
		ReadHeaderTimeout: time.Second,
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- serve(ctx, server, listener)
	}()
	response, err := http.Get("http://" + listener.Addr().String())
	if err != nil {
		cancel()
		t.Fatalf("health request: %v", err)
	}
	if err := response.Body.Close(); err != nil {
		cancel()
		t.Fatalf("close health response: %v", err)
	}

	// When
	cancel()

	// Then
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("serve() error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("serve() did not stop after cancellation")
	}
}

func TestProxy_sets_read_deadline_for_embed_requests(t *testing.T) {
	// Given
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(embedResponse{Embeddings: [][]float32{{1}}}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(backend.Close)
	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatalf("parse backend URL: %v", err)
	}
	handler := newProxy([2]*url.URL{backendURL, backendURL}, &http.Client{Timeout: time.Second})
	request := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(`{"model":"test","input":["one"]}`))
	response := &readDeadlineRecorder{ResponseRecorder: httptest.NewRecorder()}

	// When
	handler.ServeHTTP(response, request)

	// Then
	if !response.deadlineSet {
		t.Fatal("embed request read deadline was not set")
	}
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", response.Code)
	}
}

type readDeadlineRecorder struct {
	*httptest.ResponseRecorder
	deadlineSet bool
}

func (r *readDeadlineRecorder) SetReadDeadline(deadline time.Time) error {
	if !deadline.IsZero() {
		r.deadlineSet = true
	}
	return nil
}
