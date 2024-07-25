package raggo

import (
	"github.com/teilomillet/raggo/internal/rag"
)

// LogLevel represents the severity of a log message
type LogLevel = rag.LogLevel

// Log levels
const (
	LogLevelOff   = rag.LogLevelOff
	LogLevelError = rag.LogLevelError
	LogLevelWarn  = rag.LogLevelWarn
	LogLevelInfo  = rag.LogLevelInfo
	LogLevelDebug = rag.LogLevelDebug
)

// Logger interface for logging messages
type Logger = rag.Logger

// SetLogLevel sets the global log level for the raggo package
func SetLogLevel(level LogLevel) {
	rag.SetGlobalLogLevel(level)
}

// Debug logs a debug message
func Debug(msg string, keysAndValues ...interface{}) {
	rag.GlobalLogger.Debug(msg, keysAndValues...)
}

// Info logs an info message
func Info(msg string, keysAndValues ...interface{}) {
	rag.GlobalLogger.Info(msg, keysAndValues...)
}

// Warn logs a warning message
func Warn(msg string, keysAndValues ...interface{}) {
	rag.GlobalLogger.Warn(msg, keysAndValues...)
}

// Error logs an error message
func Error(msg string, keysAndValues ...interface{}) {
	rag.GlobalLogger.Error(msg, keysAndValues...)
}
