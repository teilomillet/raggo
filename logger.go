// Package raggo provides a high-level logging interface for the Raggo framework,
// built on top of the core rag package logging system. It offers:
//   - Multiple severity levels (Debug, Info, Warn, Error)
//   - Structured logging with key-value pairs
//   - Global log level control
//   - Consistent logging across the framework
package raggo

import (
	"github.com/teilomillet/raggo/rag"
)

// LogLevel represents the severity of a log message.
// It is used to control which messages are output and
// to indicate the importance of logged information.
//
// Available levels (from least to most severe):
//   - LogLevelDebug: Detailed information for debugging
//   - LogLevelInfo:  General operational messages
//   - LogLevelWarn:  Warning conditions
//   - LogLevelError: Error conditions
//   - LogLevelOff:   Disable all logging
type LogLevel = rag.LogLevel

// Log levels define the available logging severities.
// Higher levels include messages from all lower levels.
const (
	// LogLevelOff disables all logging output
	LogLevelOff = rag.LogLevelOff

	// LogLevelError enables only error messages
	// Use for conditions that prevent normal operation
	LogLevelError = rag.LogLevelError

	// LogLevelWarn enables warning and error messages
	// Use for potentially harmful situations
	LogLevelWarn = rag.LogLevelWarn

	// LogLevelInfo enables info, warning, and error messages
	// Use for general operational information
	LogLevelInfo = rag.LogLevelInfo

	// LogLevelDebug enables all message types
	// Use for detailed debugging information
	LogLevelDebug = rag.LogLevelDebug
)

// Logger interface defines the logging operations available.
// It supports structured logging with key-value pairs for
// better log aggregation and analysis.
//
// Example usage:
//
//	logger.Debug("Processing document",
//	    "filename", "example.pdf",
//	    "size", 1024,
//	    "chunks", 5)
type Logger = rag.Logger

// SetLogLevel sets the global log level for the raggo package.
// Messages below this level will not be logged.
//
// Example usage:
//
//	// Enable all logging including debug
//	raggo.SetLogLevel(raggo.LogLevelDebug)
//	
//	// Only log errors
//	raggo.SetLogLevel(raggo.LogLevelError)
//	
//	// Disable all logging
//	raggo.SetLogLevel(raggo.LogLevelOff)
func SetLogLevel(level LogLevel) {
	rag.SetGlobalLogLevel(level)
}

// Debug logs a message at debug level with optional key-value pairs.
// Debug messages provide detailed information for troubleshooting.
//
// Example usage:
//
//	raggo.Debug("Processing chunk",
//	    "id", chunkID,
//	    "size", chunkSize,
//	    "overlap", overlap)
func Debug(msg string, keysAndValues ...interface{}) {
	rag.GlobalLogger.Debug(msg, keysAndValues...)
}

// Info logs a message at info level with optional key-value pairs.
// Info messages provide general operational information.
//
// Example usage:
//
//	raggo.Info("Document added successfully",
//	    "collection", collectionName,
//	    "documentID", docID)
func Info(msg string, keysAndValues ...interface{}) {
	rag.GlobalLogger.Info(msg, keysAndValues...)
}

// Warn logs a message at warning level with optional key-value pairs.
// Warning messages indicate potential issues that don't prevent operation.
//
// Example usage:
//
//	raggo.Warn("High memory usage detected",
//	    "usedMB", memoryUsed,
//	    "threshold", threshold)
func Warn(msg string, keysAndValues ...interface{}) {
	rag.GlobalLogger.Warn(msg, keysAndValues...)
}

// Error logs a message at error level with optional key-value pairs.
// Error messages indicate serious problems that affect normal operation.
//
// Example usage:
//
//	raggo.Error("Failed to connect to vector store",
//	    "error", err,
//	    "retries", retryCount)
func Error(msg string, keysAndValues ...interface{}) {
	rag.GlobalLogger.Error(msg, keysAndValues...)
}
