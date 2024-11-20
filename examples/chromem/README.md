# Chromem Example

This example demonstrates how to use the Chromem vector database with Raggo's SimpleRAG interface.

## Prerequisites

1. Go 1.16 or later
2. OpenAI API key (set as environment variable `OPENAI_API_KEY`)

## Running the Example

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

2. Run the example:
```bash
go run main.go
```

## What it Does

1. Creates a new SimpleRAG instance with Chromem as the vector database
2. Creates sample documents about natural phenomena
3. Adds the documents to the database
4. Performs a semantic search using the query "Why is the sky blue?"
5. Prints the response based on the relevant documents found

## Expected Output

```
Question: Why is the sky blue?

Answer: The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight travels through Earth's atmosphere, it collides with gas molecules. These molecules scatter blue wavelengths of light more strongly than red wavelengths, which is why we see the sky as blue.
```

## Configuration

The example uses the following configuration:
- Vector Database: Chromem (persistent mode)
- Collection Name: knowledge-base
- Embedding Model: text-embedding-3-small
- Chunk Size: 200 characters
- Chunk Overlap: 50 characters
- Top K Results: 1
- Minimum Score: 0.1

## Notes

- The database is stored in `./data/chromem.db`
- Sample documents are created in the `./data` directory
- The example uses persistent storage mode for Chromem
