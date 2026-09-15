package main

import "testing"

func TestParseBackends_requires_two_HTTP_URLs(t *testing.T) {
	tests := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{name: "valid", value: "http://ollama-a.example:11434,http://ollama-b.example:11434"},
		{name: "one backend", value: "http://ollama-a.example:11434", wantErr: true},
		{name: "unsupported scheme", value: "ftp://ollama-a.example,http://ollama-b.example:11434", wantErr: true},
		{name: "missing host", value: "http:///one,http://ollama-b.example:11434", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// When
			backends, err := parseBackends(test.value)

			// Then
			if test.wantErr && err == nil {
				t.Fatalf("parseBackends(%q) error = nil, want error", test.value)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("parseBackends(%q) error = %v", test.value, err)
			}
			if !test.wantErr && len(backends) != 2 {
				t.Fatalf("backend count = %d, want 2", len(backends))
			}
		})
	}
}
