"""
MinIO Storage Service for OSINT Stack

This module provides a comprehensive storage service using MinIO for:
- File uploads and downloads
- Export data storage
- Document storage
- Backup and archival
- Temporary file management
"""

import os
import io
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, BinaryIO, Union
from pathlib import Path
import mimetypes
import logging

from minio import Minio
from minio.error import S3Error, InvalidResponseError
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel

from .config import settings

logger = logging.getLogger(__name__)


class StorageObject(BaseModel):
    """Model for storage object metadata"""
    bucket_name: str
    object_name: str
    size: int
    last_modified: datetime
    etag: str
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class StorageService:
    """MinIO Storage Service for OSINT Stack"""
    
    def __init__(self):
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        self._ensure_buckets()
    
    def _ensure_buckets(self):
        """Ensure required buckets exist"""
        buckets = [
            "osint-documents",      # For uploaded documents
            "osint-exports",        # For exported data
            "osint-reports",        # For generated reports
            "osint-backups",        # For system backups
            "osint-temp",          # For temporary files
            "osint-media",         # For images, videos, etc.
            "osint-archives"       # For long-term storage
        ]
        
        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
            except S3Error as e:
                logger.error(f"Failed to create bucket {bucket}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to create storage bucket: {e}")
    
    def upload_file(
        self,
        file: Union[UploadFile, BinaryIO, bytes],
        bucket_name: str,
        object_name: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> StorageObject:
        """
        Upload a file to MinIO storage
        
        Args:
            file: File to upload (UploadFile, BinaryIO, or bytes)
            bucket_name: Target bucket name
            object_name: Object name (auto-generated if None)
            content_type: MIME type (auto-detected if None)
            metadata: Additional metadata
            
        Returns:
            StorageObject with file metadata
        """
        try:
            # Generate object name if not provided
            if object_name is None:
                if hasattr(file, 'filename') and file.filename:
                    object_name = f"{uuid.uuid4()}_{file.filename}"
                else:
                    object_name = f"{uuid.uuid4()}_{datetime.now().isoformat()}"
            
            # Determine content type
            if content_type is None:
                if hasattr(file, 'content_type') and file.content_type:
                    content_type = file.content_type
                else:
                    content_type, _ = mimetypes.guess_type(object_name)
                    content_type = content_type or 'application/octet-stream'
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'uploaded_at': datetime.now().isoformat(),
                'original_filename': getattr(file, 'filename', 'unknown')
            })
            
            # Upload file
            if hasattr(file, 'file'):
                # UploadFile object
                file_data = file.file
                file_size = os.fstat(file.file.fileno()).st_size
            elif isinstance(file, bytes):
                # Bytes object
                file_data = io.BytesIO(file)
                file_size = len(file)
            else:
                # BinaryIO object
                file_data = file
                file_data.seek(0, 2)  # Seek to end
                file_size = file_data.tell()
                file_data.seek(0)  # Reset to beginning
            
            # Upload to MinIO
            result = self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=file_data,
                length=file_size,
                content_type=content_type,
                metadata=metadata
            )
            
            logger.info(f"Uploaded file {object_name} to bucket {bucket_name}")
            
            return StorageObject(
                bucket_name=bucket_name,
                object_name=object_name,
                size=file_size,
                last_modified=datetime.now(),
                etag=result.etag,
                content_type=content_type,
                metadata=metadata
            )
            
        except S3Error as e:
            logger.error(f"Failed to upload file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    
    def download_file(self, bucket_name: str, object_name: str) -> bytes:
        """
        Download a file from MinIO storage
        
        Args:
            bucket_name: Source bucket name
            object_name: Object name to download
            
        Returns:
            File content as bytes
        """
        try:
            response = self.client.get_object(bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error(f"Failed to download file {object_name}: {e}")
            raise HTTPException(status_code=404, detail=f"File not found: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            raise HTTPException(status_code=500, detail=f"Download failed: {e}")
    
    def get_file_info(self, bucket_name: str, object_name: str) -> StorageObject:
        """
        Get file metadata from MinIO storage
        
        Args:
            bucket_name: Bucket name
            object_name: Object name
            
        Returns:
            StorageObject with file metadata
        """
        try:
            stat = self.client.stat_object(bucket_name, object_name)
            return StorageObject(
                bucket_name=bucket_name,
                object_name=object_name,
                size=stat.size,
                last_modified=stat.last_modified,
                etag=stat.etag,
                content_type=stat.content_type,
                metadata=stat.metadata
            )
        except S3Error as e:
            logger.error(f"Failed to get file info for {object_name}: {e}")
            raise HTTPException(status_code=404, detail=f"File not found: {e}")
    
    def list_files(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        recursive: bool = True
    ) -> List[StorageObject]:
        """
        List files in a bucket
        
        Args:
            bucket_name: Bucket name
            prefix: Object name prefix filter
            recursive: Whether to list recursively
            
        Returns:
            List of StorageObject metadata
        """
        try:
            objects = self.client.list_objects(
                bucket_name,
                prefix=prefix,
                recursive=recursive
            )
            
            files = []
            for obj in objects:
                files.append(StorageObject(
                    bucket_name=bucket_name,
                    object_name=obj.object_name,
                    size=obj.size,
                    last_modified=obj.last_modified,
                    etag=obj.etag,
                    content_type=getattr(obj, 'content_type', None),
                    metadata=getattr(obj, 'metadata', None)
                ))
            
            return files
        except S3Error as e:
            logger.error(f"Failed to list files in bucket {bucket_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list files: {e}")
    
    def delete_file(self, bucket_name: str, object_name: str) -> bool:
        """
        Delete a file from MinIO storage
        
        Args:
            bucket_name: Bucket name
            object_name: Object name to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.remove_object(bucket_name, object_name)
            logger.info(f"Deleted file {object_name} from bucket {bucket_name}")
            return True
        except S3Error as e:
            logger.error(f"Failed to delete file {object_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")
    
    def generate_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: timedelta = timedelta(hours=1),
        method: str = "GET"
    ) -> str:
        """
        Generate a presigned URL for file access
        
        Args:
            bucket_name: Bucket name
            object_name: Object name
            expires: URL expiration time
            method: HTTP method (GET, PUT, POST, DELETE)
            
        Returns:
            Presigned URL
        """
        try:
            url = self.client.presigned_url(
                method=method,
                bucket_name=bucket_name,
                object_name=object_name,
                expires=expires
            )
            return url
        except S3Error as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate URL: {e}")
    
    def copy_file(
        self,
        source_bucket: str,
        source_object: str,
        dest_bucket: str,
        dest_object: str
    ) -> StorageObject:
        """
        Copy a file within MinIO storage
        
        Args:
            source_bucket: Source bucket name
            source_object: Source object name
            dest_bucket: Destination bucket name
            dest_object: Destination object name
            
        Returns:
            StorageObject of the copied file
        """
        try:
            # Copy object
            copy_result = self.client.copy_object(
                bucket_name=dest_bucket,
                object_name=dest_object,
                source=Minio.build_object_name(source_bucket, source_object)
            )
            
            # Get metadata of copied object
            return self.get_file_info(dest_bucket, dest_object)
            
        except S3Error as e:
            logger.error(f"Failed to copy file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to copy file: {e}")
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age
        
        Args:
            max_age_hours: Maximum age in hours for temp files
            
        Returns:
            Number of files deleted
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            temp_files = self.list_files("osint-temp")
            
            deleted_count = 0
            for file in temp_files:
                if file.last_modified < cutoff_time:
                    self.delete_file("osint-temp", file.object_name)
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} temporary files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
            return 0


# Global storage service instance
storage_service = StorageService()


# Convenience functions for common operations
def upload_document(file: UploadFile) -> StorageObject:
    """Upload a document to the documents bucket"""
    return storage_service.upload_file(file, "osint-documents")


def upload_export(data: bytes, filename: str) -> StorageObject:
    """Upload exported data to the exports bucket"""
    return storage_service.upload_file(
        data, 
        "osint-exports", 
        object_name=filename,
        content_type="application/octet-stream"
    )


def upload_report(file: Union[UploadFile, bytes], filename: str) -> StorageObject:
    """Upload a report to the reports bucket"""
    return storage_service.upload_file(
        file, 
        "osint-reports", 
        object_name=filename
    )


def get_file_url(bucket_name: str, object_name: str, expires_hours: int = 1) -> str:
    """Get a presigned URL for file access"""
    return storage_service.generate_presigned_url(
        bucket_name, 
        object_name, 
        expires=timedelta(hours=expires_hours)
    )
