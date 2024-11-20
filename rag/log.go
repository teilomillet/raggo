// Package rag provides a flexible logging system for the Raggo framework.
// It supports multiple log levels, structured logging with key-value pairs,
// and can be easily extended with custom logger implementations.
package rag

import (
	"fmt"
	"log"
	"os"
	"strings"
)

// LogLevel represents the severity level of a log message.
// Higher values indicate more verbose logging.
type LogLevel int

const (
	// LogLevelOff disables all logging
	LogLevelOff LogLevel = iota
	// LogLevelError enables only error messages
	LogLevelError
	// LogLevelWarn enables error and warning messages
	LogLevelWarn
	// LogLevelInfo enables error, warning, and info messages
	LogLevelInfo
	// LogLevelDebug enables all messages including debug
	LogLevelDebug
)

// Logger defines the interface for logging operations.
// Implementations must support multiple severity levels and
// structured logging with key-value pairs.
type Logger interface {
	// Debug logs a message at debug level with optional key-value pairs
	Debug(msg string, keysAndValues ...interface{})
	// Info logs a message at info level with optional key-value pairs
	Info(msg string, keysAndValues ...interface{})
	// Warn logs a message at warning level with optional key-value pairs
	Warn(msg string, keysAndValues ...interface{})
	// Error logs a message at error level with optional key-value pairs
	Error(msg string, keysAndValues ...interface{})
	// SetLevel changes the current logging level
	SetLevel(level LogLevel)
}

// DefaultLogger provides a basic implementation of the Logger interface
// using the standard library's log package. It supports:
// - Multiple log levels
// - Structured logging with key-value pairs
// - Output to os.Stderr by default
// - Standard timestamp format
type DefaultLogger struct {
	logger *log.Logger
	level  LogLevel
}

// NewLogger creates a new DefaultLogger instance with the specified log level.
// The logger writes to os.Stderr using the standard log package format:
// timestamp + message + key-value pairs.
func NewLogger(level LogLevel) Logger {
	return &DefaultLogger{
		logger: log.New(os.Stderr, "", log.LstdFlags),
		level:  level,
	}
}

// SetLevel updates the logging level of the DefaultLogger.
// Messages below this level will not be logged.
func (l *DefaultLogger) SetLevel(level LogLevel) {
	l.level = level
}

// log is an internal helper that handles the actual logging operation.
// It checks the log level and formats the message with key-value pairs.
func (l *DefaultLogger) log(level LogLevel, msg string, keysAndValues ...interface{}) {
	if level <= l.level {
		l.logger.Printf("%s: %s %v", level, msg, keysAndValues)
	}
}

// Debug logs a message at debug level. This level should be used for
// detailed information needed for debugging purposes.
func (l *DefaultLogger) Debug(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelDebug, msg, keysAndValues...)
}

// Info logs a message at info level. This level should be used for
// general operational information.
func (l *DefaultLogger) Info(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelInfo, msg, keysAndValues...)
}

// Warn logs a message at warning level. This level should be used for
// potentially harmful situations that don't prevent normal operation.
func (l *DefaultLogger) Warn(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelWarn, msg, keysAndValues...)
}

// Error logs a message at error level. This level should be used for
// error conditions that affect normal operation.
func (l *DefaultLogger) Error(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelError, msg, keysAndValues...)
}

// String returns the string representation of a LogLevel.
// This is used for formatting log messages and configuration.
func (l LogLevel) String() string {
	return [...]string{"OFF", "ERROR", "WARN", "INFO", "DEBUG"}[l]
}

// UnmarshalText implements the encoding.TextUnmarshaler interface.
// It allows LogLevel to be configured from string values in configuration
// files or environment variables.
func (l *LogLevel) UnmarshalText(text []byte) error {
	switch strings.ToUpper(string(text)) {
	case "OFF":
		*l = LogLevelOff
	case "ERROR":
		*l = LogLevelError
	case "WARN":
		*l = LogLevelWarn
	case "INFO":
		*l = LogLevelInfo
	case "DEBUG":
		*l = LogLevelDebug
	default:
		return fmt.Errorf("invalid log level: %s", string(text))
	}
	return nil
}

// GlobalLogger is the package-level logger instance used by default.
// It can be accessed and modified by other packages using the rag framework.
var GlobalLogger Logger

// init initializes the global logger with a default configuration.
// By default, it logs at INFO level to os.Stderr.
func init() {
	GlobalLogger = NewLogger(LogLevelInfo)
}

// SetGlobalLogLevel sets the log level for the global logger instance.
// This function provides a convenient way to control logging verbosity
// across the entire application.
func SetGlobalLogLevel(level LogLevel) {
	GlobalLogger.SetLevel(level)
}
