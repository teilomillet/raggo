#!/bin/bash

SOURCE_DIR="testdata"
TARGET_DIR="benchmark_data"
DESIRED_COUNT=5000

# Clean the target directory
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo "Cleaned $TARGET_DIR"

# Get list of PDF files in source directory
pdf_files=($(ls "$SOURCE_DIR"/*.pdf))
num_pdfs=${#pdf_files[@]}

if [ $num_pdfs -eq 0 ]; then
    echo "No PDF files found in $SOURCE_DIR"
    exit 1
fi

# Duplicate PDFs
for ((i=0; i<DESIRED_COUNT; i++)); do
    source_file="${pdf_files[i % num_pdfs]}"
    target_file="$TARGET_DIR/$(basename "${source_file%.*}")_copy$i.pdf"
    cp "$source_file" "$target_file"
    echo "Copied $source_file to $target_file"
done

echo "Duplicated $DESIRED_COUNT PDF files in $TARGET_DIR"
