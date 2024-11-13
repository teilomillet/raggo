package raggo

import (
	"context"
	"net/http"
	"time"

	"github.com/teilomillet/raggo/rag"
)

// Loader represents the main interface for loading files
type Loader interface {
	LoadURL(ctx context.Context, url string) (string, error)
	LoadFile(ctx context.Context, path string) (string, error)
	LoadDir(ctx context.Context, dir string) ([]string, error)
}

// loaderWrapper wraps the internal loader
type loaderWrapper struct {
	internal *rag.Loader
}

// LoaderOption is a functional option for configuring a Loader
type LoaderOption = rag.LoaderOption

// WithHTTPClient sets a custom HTTP client for the Loader
func WithHTTPClient(client *http.Client) LoaderOption {
	return rag.WithHTTPClient(client)
}

// SetTimeout sets a custom timeout for the Loader
func SetTimeout(timeout time.Duration) LoaderOption {
	return rag.WithTimeout(timeout)
}

// SetTempDir sets the temporary directory for downloaded files
func SetTempDir(dir string) LoaderOption {
	return rag.WithTempDir(dir)
}

// NewLoader creates a new Loader with the given options
func NewLoader(opts ...LoaderOption) Loader {
	return &loaderWrapper{internal: rag.NewLoader(opts...)}
}

func (lw *loaderWrapper) LoadURL(ctx context.Context, url string) (string, error) {
	return lw.internal.LoadURL(ctx, url)
}

func (lw *loaderWrapper) LoadFile(ctx context.Context, path string) (string, error) {
	return lw.internal.LoadFile(ctx, path)
}

func (lw *loaderWrapper) LoadDir(ctx context.Context, dir string) ([]string, error) {
	return lw.internal.LoadDir(ctx, dir)
}
