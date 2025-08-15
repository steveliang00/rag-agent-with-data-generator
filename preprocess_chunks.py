#!/usr/bin/env python3
"""
Temporary script to preprocess PDF chunks and save them to a file.
This allows generate_dataset.py to load pre-processed chunks instead of 
processing the PDF every time.
"""

import pickle
import os
import sys

# Add src directory to Python path

from src.create_database import DocumentManager

def main():
    print("Starting PDF preprocessing...")
    
    # Initialize document manager
    document_manager = DocumentManager()
    
    # Load and split documents
    print("Loading documents...")
    docs = document_manager.load_docs()
    print(f"Loaded {len(docs)} documents")
    
    print("Splitting documents into chunks...")
    chunks = document_manager.split_text(docs)
    print(f"Created {len(chunks)} chunks")
    
    # Save chunks to file
    output_file = "processed_chunks.pkl"
    print(f"Saving chunks to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"✓ Successfully saved {len(chunks)} chunks to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    print("\nYou can now run generate_dataset.py without re-processing the PDF!")

if __name__ == "__main__":
    main()
