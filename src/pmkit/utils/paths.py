"""
Path utilities for PM-Kit.

Production-ready path operations with security, thread-safety, and atomic writes.
All operations use pathlib.Path for cross-platform compatibility.
"""

import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union, Set

from pmkit.exceptions import PMKitError, ValidationError


# Thread-safe lock for directory operations
_dir_lock = threading.RLock()
_created_dirs: Set[Path] = set()


class PathSecurityError(PMKitError):
    """Raised when a path operation would violate security constraints."""
    
    def __init__(self, message: str, path: Optional[Path] = None):
        suggestion = "Ensure paths stay within project boundaries and don't contain '..' traversals"
        context = {"path": str(path)} if path else {}
        super().__init__(message, suggestion, context)


class PathOperationError(PMKitError):
    """Raised when a path operation fails."""
    
    def __init__(self, message: str, path: Optional[Path] = None):
        suggestion = "Check file permissions and disk space"
        context = {"path": str(path)} if path else {}
        super().__init__(message, suggestion, context)


def find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find the project root by walking up the directory tree.
    
    Looks for .pmkit or .git directory to identify project root.
    
    Args:
        start_path: Starting directory (defaults to current directory)
        
    Returns:
        Path to project root, or None if not found
        
    Example:
        >>> root = find_project_root()
        >>> if root:
        ...     print(f"Project root: {root}")
        ... else:
        ...     print("Not in a project directory")
    """
    current = Path(start_path or Path.cwd()).resolve()
    
    # Walk up the directory tree
    for directory in [current] + list(current.parents):
        # Check for .pmkit directory first (PM-Kit project)
        if (directory / ".pmkit").is_dir():
            return directory
            
        # Fall back to .git directory (any git project)
        if (directory / ".git").is_dir():
            return directory
    
    return None


def get_pmkit_dir(create: bool = True) -> Path:
    """
    Get the .pmkit directory in the project root.
    
    Args:
        create: Create the directory if it doesn't exist
        
    Returns:
        Path to .pmkit directory
        
    Raises:
        PathOperationError: If project root not found or creation fails
        
    Example:
        >>> pmkit_dir = get_pmkit_dir()
        >>> context_dir = pmkit_dir / "context"
        >>> ensure_directory(context_dir)
    """
    root = find_project_root()
    if not root:
        raise PathOperationError(
            "Not in a PM-Kit project directory"
        )
    
    pmkit_dir = root / ".pmkit"
    
    if create and not pmkit_dir.exists():
        ensure_directory(pmkit_dir)
    
    return pmkit_dir


def ensure_directory(
    directory: Path,
    parents: bool = True,
    mode: int = 0o755
) -> bool:
    """
    Thread-safe directory creation with proper permissions.
    
    Args:
        directory: Path to directory to create
        parents: Create parent directories if needed
        mode: Unix permissions for the directory
        
    Returns:
        True if directory was created, False if it already existed
        
    Raises:
        PathOperationError: If creation fails
        PathSecurityError: If path validation fails
        
    Example:
        >>> output_dir = Path("data/output")
        >>> if ensure_directory(output_dir):
        ...     print(f"Created directory: {output_dir}")
        ... else:
        ...     print(f"Directory already exists: {output_dir}")
    """
    directory = directory.resolve()
    
    # Security check
    validate_path(directory)
    
    with _dir_lock:
        # Check cache first
        if directory in _created_dirs:
            return False
        
        try:
            # Attempt to create directory
            directory.mkdir(parents=parents, exist_ok=True, mode=mode)
            _created_dirs.add(directory)
            
            # Verify it's actually a directory
            if not directory.is_dir():
                raise PathOperationError(
                    f"Path exists but is not a directory: {directory}",
                    directory
                )
            
            return True
            
        except PermissionError as e:
            raise PathOperationError(
                f"Permission denied creating directory: {directory}",
                directory
            ) from e
        except OSError as e:
            # Handle race condition where directory might have been created
            # by another thread/process between our check and creation
            if directory.is_dir():
                _created_dirs.add(directory)
                return False
            raise PathOperationError(
                f"Failed to create directory: {directory} - {e}",
                directory
            ) from e


def safe_write(
    file_path: Path,
    content: Union[str, bytes],
    backup: bool = True,
    encoding: str = "utf-8",
    mode: Optional[int] = None
) -> Optional[Path]:
    """
    Atomic file write with automatic backup if file exists.
    
    Uses temporary file + atomic rename to ensure file is either
    completely written or not modified at all.
    
    Args:
        file_path: Path to file to write
        content: Content to write (str or bytes)
        backup: Create backup if file exists
        encoding: Text encoding (ignored for bytes)
        mode: Unix permissions for the file
        
    Returns:
        Path to backup file if created, None otherwise
        
    Raises:
        PathOperationError: If write fails
        PathSecurityError: If path validation fails
        ValidationError: If content type doesn't match mode
        
    Example:
        >>> config_path = Path("config.yaml")
        >>> backup = safe_write(config_path, "key: value\\n")
        >>> if backup:
        ...     print(f"Backup created: {backup}")
    """
    file_path = file_path.resolve()
    
    # Security check
    validate_path(file_path)
    
    # Ensure parent directory exists
    ensure_directory(file_path.parent)
    
    # Create backup if requested and file exists
    backup_path = None
    if backup and file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + ".backup")
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
        except OSError as e:
            raise PathOperationError(
                f"Failed to create backup: {backup_path} - {e}",
                backup_path
            ) from e
    
    # Determine write mode
    is_binary = isinstance(content, bytes)
    write_mode = "wb" if is_binary else "w"
    
    # Write atomically using temporary file
    temp_fd = None
    temp_path = None
    
    try:
        # Create temporary file in same directory (for atomic rename)
        temp_fd, temp_name = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp"
        )
        temp_path = Path(temp_name)
        
        # Write content
        with os.fdopen(temp_fd, write_mode) as f:
            if is_binary:
                if not isinstance(content, bytes):
                    raise ValidationError(
                        "Binary mode requires bytes content",
                        field="content"
                    )
                f.write(content)
            else:
                if not isinstance(content, str):
                    raise ValidationError(
                        "Text mode requires string content",
                        field="content"
                    )
                f.write(content)
            
            # Force write to disk
            f.flush()
            os.fsync(f.fileno())
        
        # Set permissions if specified
        if mode is not None:
            os.chmod(temp_path, mode)
        
        # Atomic rename (replace existing file)
        temp_path.replace(file_path)
        
        return backup_path
        
    except Exception as e:
        # Clean up temporary file on failure
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass  # Best effort cleanup
        
        # Re-raise with context
        if isinstance(e, (ValidationError, PathOperationError, PathSecurityError)):
            raise
        raise PathOperationError(
            f"Failed to write file: {file_path} - {e}",
            file_path
        ) from e
    finally:
        # Ensure file descriptor is closed
        if temp_fd is not None and temp_path is None:
            try:
                os.close(temp_fd)
            except OSError:
                pass  # Already closed


def validate_path(path: Path, base_dir: Optional[Path] = None) -> None:
    """
    Security check to prevent path traversal attacks.
    
    Ensures the path doesn't escape the base directory (if provided)
    and doesn't contain dangerous patterns.
    
    Args:
        path: Path to validate
        base_dir: Optional base directory to enforce boundaries
        
    Raises:
        PathSecurityError: If path validation fails
        
    Example:
        >>> base = Path("/safe/directory")
        >>> user_path = base / "user_input.txt"
        >>> validate_path(user_path, base)  # OK
        >>> 
        >>> evil_path = base / "../../../etc/passwd"
        >>> validate_path(evil_path, base)  # Raises PathSecurityError
    """
    # Resolve to absolute path (follows symlinks)
    resolved_path = path.resolve()
    
    # Check for null bytes (common attack vector)
    if "\x00" in str(path):
        raise PathSecurityError(
            "Path contains null bytes",
            path
        )
    
    # Check against base directory if provided
    if base_dir is not None:
        base_resolved = base_dir.resolve()
        
        # Ensure path is within base directory
        try:
            resolved_path.relative_to(base_resolved)
        except ValueError:
            raise PathSecurityError(
                f"Path escapes base directory: {path}",
                path
            )
    
    # Check for suspicious patterns
    path_str = str(path)
    dangerous_patterns = [
        "..",  # Parent directory traversal
        "~",   # Home directory expansion
    ]
    
    for pattern in dangerous_patterns:
        if pattern in path_str:
            # Allow .. only if the resolved path is still within bounds
            if pattern == ".." and base_dir:
                # Already checked with relative_to above
                continue
            elif pattern == "~" and not path_str.startswith("~"):
                # Allow ~ in middle of filename
                continue
            else:
                raise PathSecurityError(
                    f"Path contains suspicious pattern '{pattern}': {path}",
                    path
                )


@contextmanager
def atomic_write_context(
    file_path: Path,
    mode: str = "w",
    encoding: str = "utf-8",
    backup: bool = True
):
    """
    Context manager for atomic file writing.
    
    Provides a file handle that writes to a temporary file,
    then atomically moves it to the target path on success.
    
    Args:
        file_path: Target file path
        mode: Write mode ('w' for text, 'wb' for binary)
        encoding: Text encoding (ignored for binary mode)
        backup: Create backup if file exists
        
    Yields:
        File handle for writing
        
    Raises:
        PathOperationError: If operation fails
        PathSecurityError: If path validation fails
        
    Example:
        >>> import json
        >>> config_path = Path("config.json")
        >>> with atomic_write_context(config_path) as f:
        ...     json.dump({"key": "value"}, f, indent=2)
    """
    file_path = file_path.resolve()
    
    # Security check
    validate_path(file_path)
    
    # Ensure parent directory exists
    ensure_directory(file_path.parent)
    
    # Create backup if requested
    backup_path = None
    if backup and file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + ".backup")
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
        except OSError as e:
            raise PathOperationError(
                f"Failed to create backup: {backup_path}",
                backup_path
            ) from e
    
    # Create temporary file
    temp_file = None
    temp_path = None
    
    try:
        temp_file = tempfile.NamedTemporaryFile(
            mode=mode,
            encoding=encoding if "b" not in mode else None,
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=False
        )
        temp_path = Path(temp_file.name)
        
        # Yield the file handle
        yield temp_file
        
        # Ensure content is written
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()
        temp_file = None
        
        # Atomic rename
        temp_path.replace(file_path)
        
    except Exception as e:
        # Clean up temporary file on failure
        if temp_file:
            temp_file.close()
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        
        # Restore backup if it was created
        if backup_path and backup_path.exists() and not file_path.exists():
            try:
                import shutil
                shutil.move(str(backup_path), str(file_path))
            except OSError:
                pass  # Best effort
        
        # Re-raise with context
        if isinstance(e, (PathOperationError, PathSecurityError)):
            raise
        raise PathOperationError(
            f"Failed to write file: {file_path} - {e}",
            file_path
        ) from e
    finally:
        # Ensure file is closed
        if temp_file:
            temp_file.close()


def safe_read(
    file_path: Path,
    encoding: str = "utf-8",
    default: Optional[Union[str, bytes]] = None
) -> Union[str, bytes, None]:
    """
    Safely read a file with validation and error handling.
    
    Args:
        file_path: Path to file to read
        encoding: Text encoding (use 'binary' for bytes)
        default: Default value if file doesn't exist
        
    Returns:
        File contents or default value
        
    Raises:
        PathSecurityError: If path validation fails
        PathOperationError: If read fails (except FileNotFoundError)
        
    Example:
        >>> config = safe_read(Path("config.yaml"), default="key: value\\n")
        >>> print(config)
    """
    file_path = file_path.resolve()
    
    # Security check
    validate_path(file_path)
    
    try:
        if encoding == "binary":
            return file_path.read_bytes()
        else:
            return file_path.read_text(encoding=encoding)
    except FileNotFoundError:
        return default
    except PermissionError as e:
        raise PathOperationError(
            f"Permission denied reading file: {file_path}",
            file_path
        ) from e
    except OSError as e:
        raise PathOperationError(
            f"Failed to read file: {file_path} - {e}",
            file_path
        ) from e


def get_safe_filename(name: str, max_length: int = 255) -> str:
    """
    Convert a string to a safe filename.
    
    Removes/replaces characters that are problematic in filenames.
    
    Args:
        name: Original name
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
        
    Example:
        >>> safe = get_safe_filename("My PRD: Version 2.0")
        >>> print(safe)  # "My_PRD_Version_2.0"
    """
    # Replace problematic characters
    replacements = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
        "\n": "_",
        "\r": "_",
        "\t": "_",
    }
    
    safe_name = name
    for char, replacement in replacements.items():
        safe_name = safe_name.replace(char, replacement)
    
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip(". ")
    
    # Ensure not empty
    if not safe_name:
        safe_name = "unnamed"
    
    # Truncate if too long
    if len(safe_name) > max_length:
        # Keep extension if present
        if "." in safe_name:
            base, ext = safe_name.rsplit(".", 1)
            max_base = max_length - len(ext) - 1
            safe_name = base[:max_base] + "." + ext
        else:
            safe_name = safe_name[:max_length]
    
    return safe_name


# Convenience function for clearing the directory cache
def clear_directory_cache() -> None:
    """Clear the thread-safe directory creation cache."""
    with _dir_lock:
        _created_dirs.clear()