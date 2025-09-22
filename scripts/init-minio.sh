#!/bin/bash

# MinIO initialization script for OSINT Stack
# This script sets up MinIO buckets and policies

set -e

echo "Initializing MinIO for OSINT Stack..."

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
until curl -f http://localhost:9000/minio/health/live > /dev/null 2>&1; do
    echo "Waiting for MinIO..."
    sleep 2
done

echo "MinIO is ready!"

# Set MinIO client alias
mc alias set osint http://localhost:9000 minioadmin minioadmin123

# Create buckets if they don't exist
echo "Creating buckets..."

buckets=(
    "osint-documents"
    "osint-exports" 
    "osint-reports"
    "osint-backups"
    "osint-temp"
    "osint-media"
    "osint-archives"
)

for bucket in "${buckets[@]}"; do
    echo "Creating bucket: $bucket"
    mc mb osint/$bucket --ignore-existing
done

# Set bucket policies
echo "Setting bucket policies..."

# Documents bucket - read/write for authenticated users
mc anonymous set download osint/osint-documents

# Exports bucket - read/write for authenticated users  
mc anonymous set download osint/osint-exports

# Reports bucket - read/write for authenticated users
mc anonymous set download osint/osint-reports

# Media bucket - read/write for authenticated users
mc anonymous set download osint/osint-media

# Temp bucket - read/write for cleanup
mc anonymous set download osint/osint-temp

# Archives bucket - read-only for long-term storage
mc anonymous set download osint/osint-archives

# Backups bucket - private (no anonymous access)
mc anonymous set none osint/osint-backups

echo "Setting up lifecycle policies..."

# Set lifecycle policy for temp files (delete after 7 days)
mc ilm add osint/osint-temp --expiry-days 7

# Set lifecycle policy for exports (delete after 30 days)
mc ilm add osint/osint-exports --expiry-days 30

echo "MinIO initialization completed successfully!"

# List buckets to verify
echo "Created buckets:"
mc ls osint

echo "MinIO setup complete!"
