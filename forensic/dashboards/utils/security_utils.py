"""
Security utilities for input validation and sanitization.
"""

import re
import html
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import mimetypes
import magic
import logging

from .error_handling import ValidationError


# Patterns for input validation
SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
    r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
    r"(--|\#|\/\*|\*\/)",
    r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT)\b)",
    r"([\"\'][^\"\']*[\"\'])"
]

XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe[^>]*>.*?</iframe>",
    r"<object[^>]*>.*?</object>",
    r"<embed[^>]*>.*?</embed>",
    r"<link[^>]*>.*?</link>",
    r"<meta[^>]*>.*?</meta>"
]

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {
    'json', 'csv', 'xlsx', 'pdf', 'txt', 'log', 'html'
}

# Maximum file sizes (in bytes)
MAX_FILE_SIZES = {
    'json': 50 * 1024 * 1024,  # 50MB
    'csv': 100 * 1024 * 1024,  # 100MB
    'xlsx': 50 * 1024 * 1024,  # 50MB
    'pdf': 20 * 1024 * 1024,   # 20MB
    'txt': 10 * 1024 * 1024,   # 10MB
    'log': 10 * 1024 * 1024,   # 10MB
    'html': 5 * 1024 * 1024    # 5MB
}


def sanitize_input(user_input: str, allow_html: bool = False) -> str:
    """
    Sanitize user input to prevent XSS and injection attacks.
    
    Args:
        user_input: Raw user input
        allow_html: Whether to allow HTML tags
    
    Returns:
        Sanitized input string
    
    Raises:
        ValidationError: If input contains dangerous content
    """
    if not isinstance(user_input, str):
        user_input = str(user_input)
    
    # Check for SQL injection patterns
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            raise ValidationError("Input contains potentially dangerous SQL patterns")
    
    # Check for XSS patterns
    if not allow_html:
        for pattern in XSS_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                raise ValidationError("Input contains potentially dangerous script content")
    
    # HTML escape if not allowing HTML
    if not allow_html:
        user_input = html.escape(user_input)
    
    # URL decode to prevent encoding bypasses
    user_input = urllib.parse.unquote(user_input)
    
    # Limit length to prevent DoS
    if len(user_input) > 10000:  # 10KB limit
        raise ValidationError("Input too long")
    
    return user_input.strip()


def validate_filename(filename: str) -> str:
    """
    Validate and sanitize filename.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    
    Raises:
        ValidationError: If filename is invalid
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")
    
    # Remove path components
    filename = Path(filename).name
    
    # Check for dangerous characters
    if not SAFE_FILENAME_PATTERN.match(filename):
        raise ValidationError("Filename contains invalid characters")
    
    # Check length
    if len(filename) > 255:
        raise ValidationError("Filename too long")
    
    # Check for reserved names (Windows)
    reserved_names = [
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
        'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
        'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    ]
    
    name_without_ext = Path(filename).stem.upper()
    if name_without_ext in reserved_names:
        raise ValidationError("Filename uses reserved name")
    
    return filename


def validate_file_extension(filename: str) -> str:
    """
    Validate file extension.
    
    Args:
        filename: Filename to check
    
    Returns:
        File extension (lowercase)
    
    Raises:
        ValidationError: If extension is not allowed
    """
    extension = Path(filename).suffix.lower().lstrip('.')
    
    if not extension:
        raise ValidationError("File must have an extension")
    
    if extension not in ALLOWED_EXTENSIONS:
        raise ValidationError(f"File type '{extension}' not allowed. "
                            f"Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
    
    return extension


def validate_file_content(file_content: bytes, filename: str) -> bool:
    """
    Validate file content using magic numbers and MIME type detection.
    
    Args:
        file_content: File content as bytes
        filename: Original filename
    
    Returns:
        True if file content is valid
    
    Raises:
        ValidationError: If file content is invalid
    """
    # Check file size
    extension = validate_file_extension(filename)
    max_size = MAX_FILE_SIZES.get(extension, 1024 * 1024)  # 1MB default
    
    if len(file_content) > max_size:
        raise ValidationError(f"File too large. Maximum size for {extension} files: "
                            f"{max_size // (1024*1024)}MB")
    
    # Check MIME type using python-magic
    try:
        detected_mime = magic.from_buffer(file_content, mime=True)
        expected_mimes = get_expected_mime_types(extension)
        
        if detected_mime not in expected_mimes:
            logging.getLogger(__name__).warning(
                f"MIME type mismatch for {filename}: detected {detected_mime}, "
                f"expected one of {expected_mimes}"
            )
            # Don't raise error for now, just log warning
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not detect MIME type for {filename}: {e}")
    
    # Additional content validation based on file type
    if extension == 'json':
        validate_json_content(file_content)
    elif extension == 'csv':
        validate_csv_content(file_content)
    
    return True


def get_expected_mime_types(extension: str) -> List[str]:
    """
    Get expected MIME types for file extension.
    
    Args:
        extension: File extension
    
    Returns:
        List of expected MIME types
    """
    mime_mapping = {
        'json': ['application/json', 'text/json', 'text/plain'],
        'csv': ['text/csv', 'application/csv', 'text/plain'],
        'xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
        'pdf': ['application/pdf'],
        'txt': ['text/plain'],
        'log': ['text/plain', 'text/x-log'],
        'html': ['text/html', 'application/xhtml+xml']
    }
    
    return mime_mapping.get(extension, ['application/octet-stream'])


def validate_json_content(content: bytes) -> bool:
    """
    Validate JSON file content.
    
    Args:
        content: File content as bytes
    
    Returns:
        True if valid JSON
    
    Raises:
        ValidationError: If JSON is invalid
    """
    import json
    
    try:
        # Try to decode as UTF-8
        text_content = content.decode('utf-8')
        
        # Check for potentially malicious content
        if len(text_content) > 50 * 1024 * 1024:  # 50MB limit for JSON
            raise ValidationError("JSON file too large")
        
        # Try to parse JSON
        json.loads(text_content)
        
        return True
        
    except UnicodeDecodeError:
        raise ValidationError("File is not valid UTF-8")
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON format: {str(e)}")


def validate_csv_content(content: bytes) -> bool:
    """
    Validate CSV file content.
    
    Args:
        content: File content as bytes
    
    Returns:
        True if valid CSV
    
    Raises:
        ValidationError: If CSV is invalid
    """
    import csv
    import io
    
    try:
        # Try to decode as UTF-8
        text_content = content.decode('utf-8')
        
        # Check for extremely large files
        if len(text_content) > 100 * 1024 * 1024:  # 100MB limit
            raise ValidationError("CSV file too large")
        
        # Try to parse CSV
        csv_reader = csv.reader(io.StringIO(text_content))
        
        # Read first few rows to validate structure
        row_count = 0
        for row in csv_reader:
            row_count += 1
            if row_count > 1000:  # Don't validate entire file
                break
        
        return True
        
    except UnicodeDecodeError:
        raise ValidationError("File is not valid UTF-8")
    except csv.Error as e:
        raise ValidationError(f"Invalid CSV format: {str(e)}")


def validate_file_upload(file_content: bytes, filename: str, 
                        max_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Comprehensive file upload validation.
    
    Args:
        file_content: File content as bytes
        filename: Original filename
        max_size: Optional custom maximum size
    
    Returns:
        Dictionary with validation results
    
    Raises:
        ValidationError: If file validation fails
    """
    validation_results = {
        'filename': None,
        'extension': None,
        'size': len(file_content),
        'mime_type': None,
        'is_valid': False
    }
    
    try:
        # Validate filename
        clean_filename = validate_filename(filename)
        validation_results['filename'] = clean_filename
        
        # Validate extension
        extension = validate_file_extension(filename)
        validation_results['extension'] = extension
        
        # Override max size if provided
        if max_size:
            if len(file_content) > max_size:
                raise ValidationError(f"File too large. Maximum size: {max_size} bytes")
        
        # Validate content
        validate_file_content(file_content, filename)
        
        # Detect MIME type
        try:
            detected_mime = magic.from_buffer(file_content, mime=True)
            validation_results['mime_type'] = detected_mime
        except Exception:
            validation_results['mime_type'] = 'unknown'
        
        validation_results['is_valid'] = True
        
        logging.getLogger(__name__).info(
            f"File validation successful: {filename} ({extension}, {len(file_content)} bytes)"
        )
        
        return validation_results
        
    except ValidationError as e:
        logging.getLogger(__name__).warning(f"File validation failed for {filename}: {str(e)}")
        raise


def sanitize_path(path: Union[str, Path]) -> Path:
    """
    Sanitize file path to prevent directory traversal attacks.
    
    Args:
        path: File path to sanitize
    
    Returns:
        Sanitized Path object
    
    Raises:
        ValidationError: If path is invalid
    """
    path = Path(path)
    
    # Resolve to absolute path and check for traversal
    try:
        resolved_path = path.resolve()
    except Exception:
        raise ValidationError("Invalid path")
    
    # Check for path traversal attempts
    if '..' in path.parts:
        raise ValidationError("Path traversal not allowed")
    
    # Check for absolute paths from user input
    if path.is_absolute():
        raise ValidationError("Absolute paths not allowed")
    
    return resolved_path


def validate_ip_address(ip_address: str) -> bool:
    """
    Validate IP address format.
    
    Args:
        ip_address: IP address string
    
    Returns:
        True if valid IP address
    
    Raises:
        ValidationError: If IP address is invalid
    """
    import ipaddress
    
    try:
        ipaddress.ip_address(ip_address)
        return True
    except ValueError:
        raise ValidationError("Invalid IP address format")


def validate_user_input_length(input_value: str, min_length: int = 0, 
                              max_length: int = 1000) -> bool:
    """
    Validate user input length.
    
    Args:
        input_value: Input string to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
    
    Returns:
        True if length is valid
    
    Raises:
        ValidationError: If length is invalid
    """
    if len(input_value) < min_length:
        raise ValidationError(f"Input too short. Minimum length: {min_length}")
    
    if len(input_value) > max_length:
        raise ValidationError(f"Input too long. Maximum length: {max_length}")
    
    return True


def create_secure_headers() -> Dict[str, str]:
    """
    Create security headers for HTTP responses.
    
    Returns:
        Dictionary of security headers
    """
    return {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.plot.ly; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self';"
        ),
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }