"""
Unified Logger for the personal assistant system.
Provides a unified interface for logging structured data to the graph store.
"""
from __future__ import annotations
import csv
import json
import uuid
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from jsonschema import FormatChecker

def ensure_timezone_aware(dt: Optional[datetime] = None) -> datetime:
    """Ensure datetime is timezone-aware. Use UTC if no timezone specified.
    
    Args:
        dt: A datetime object (naive or aware) or None
        
    Returns:
        A timezone-aware datetime object (in UTC)
    """
    if dt is None:
        return datetime.now(timezone.utc)
    if not hasattr(dt, 'tzinfo') or dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# Import jsonschema for validation
try:
    import jsonschema
except ImportError:
    jsonschema = None

# Try to import GraphStore - might be in different locations
try:
    from ..graph.store import GraphStore
except ImportError:
    try:
        from assistant.graph.store import GraphStore
    except ImportError:
        GraphStore = None


class ValidationError(ValueError):
    """Raised when log entry validation fails."""
    pass


class ImportError(Exception):
    """Raised when CSV import encounters errors."""
    pass


# Define the log entry schema for validation
LOG_ENTRY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["id", "category", "timestamp"],
    "properties": {
        "id": {"type": "string"},
        "category": {
            "type": "string",
            "enum": ["nutrition", "medication", "exercise", "mood", "sleep", "task", "energy", "focus", "social", "custom"]
        },
        "timestamp": {"type": "string", "format": "date-time"},
        "value": {
            "oneOf": [
                {"type": "number"},
                {"type": "string"},
                {"type": "boolean"},
                {"type": "object"}
            ]
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "source": {"type": "string", "enum": ["manual", "imported", "inferred", "device"]},
        "metadata": {"type": "object"}
    }
}


class UnifiedLogger:
    """
    Unified logger that creates graph nodes for log entries.
    Each log entry becomes a node with type="log" in the graph store.
    """

    def __init__(self, data_dir: Optional[str] = None, graph_store: Optional[Any] = None):
        """
        Initialize the UnifiedLogger.

        Args:
            data_dir: Path to directory for GraphStore (if graph_store not provided)
            graph_store: Existing GraphStore instance to use
        """
        if graph_store is not None:
            self.graph_store = graph_store
        elif data_dir is not None:
            if GraphStore is None:
                raise ImportError("GraphStore not available")
            self.graph_store = GraphStore(data_dir)
        else:
            raise ValueError("Either data_dir or graph_store must be provided")

        # Add debug flag for logging
        self.debug_mode = False

    # ADD THESE METHODS TO THE EXISTING CLASS:

    def info(self, message: str, *args, **kwargs) -> None:
        """Log an info message to stdout."""
        print(f"INFO: {message}", *args)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log a debug message to stdout if debug mode is enabled."""
        if self.debug_mode:
            print(f"DEBUG: {message}", *args)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log a warning message to stderr."""
        print(f"WARNING: {message}", *args, file=sys.stderr)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log an error message to stderr."""
        print(f"ERROR: {message}", *args, file=sys.stderr)
    
    def log(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log an entry by creating a graph node.
        
        Args:
            entry: Log entry dict with id, category, timestamp, value, etc.
            
        Returns:
            Created node dict
            
        Raises:
            ValidationError or jsonschema.exceptions.ValidationError: If entry is invalid
        """
        # Generate ID if not provided
        if "id" not in entry or not entry["id"]:
            entry = dict(entry)  # Make a copy
            entry["id"] = str(uuid.uuid4())
        
        # Apply defaults before validation
        entry_with_defaults = dict(entry)
        if "confidence" not in entry_with_defaults or entry_with_defaults.get("confidence") == "":
            entry_with_defaults["confidence"] = 1.0
        elif entry_with_defaults["confidence"] is not None:
            try:
                entry_with_defaults["confidence"] = float(entry_with_defaults["confidence"])
            except (TypeError, ValueError):
                pass  # Let validation catch it
                
        if "source" not in entry_with_defaults or entry_with_defaults.get("source") == "":
            entry_with_defaults["source"] = "manual"
        
        # Validate with jsonschema if available
        if jsonschema:
            try:
                jsonschema.validate(instance=entry_with_defaults, schema=LOG_ENTRY_SCHEMA,  format_checker=FormatChecker())
            except jsonschema.exceptions.ValidationError as e:
                # Re-raise as-is for tests that expect this specific exception
                raise
        else:
            # Fallback validation if jsonschema not available
            if not isinstance(entry, dict):
                raise ValidationError("Entry must be a dictionary")
            
            if "category" not in entry:
                raise ValidationError("Field 'category' is required")
            
            if "timestamp" not in entry:
                raise ValidationError("Field 'timestamp' is required")
                
            # Validate category
            valid_categories = [
                "nutrition", "medication", "exercise", "mood", "sleep",
                "task", "energy", "focus", "social", "custom"
            ]
            if entry["category"] not in valid_categories:
                raise ValidationError(f"Invalid category: {entry['category']}")
            
            # Validate timestamp
            timestamp_str = entry["timestamp"]
            try:
                # Parse to validate format
                if timestamp_str.endswith("Z"):
                    timestamp_str = timestamp_str[:-1] + "+00:00"
                ensure_timezone_aware(datetime.fromisoformat(timestamp_str))
            except Exception:
                raise ValidationError(f"Invalid timestamp format: {entry['timestamp']}")
            
            # Validate value (reject arrays)
            if "value" in entry and isinstance(entry["value"], list):
                raise ValidationError("Value cannot be an array")
        
        # Build node with all log data in attrs
        node_data = {
            "id": entry_with_defaults["id"],
            "type": "log",
            "text": f"Log entry: {entry_with_defaults['category']}",
            "tags": self._generate_tags(entry_with_defaults["category"]),
            "attrs": {
                "id": entry_with_defaults["id"],
                "category": entry_with_defaults["category"],
                "timestamp": entry_with_defaults["timestamp"],
                "value": entry_with_defaults.get("value"),
                "confidence": entry_with_defaults.get("confidence", 1.0),
                "source": entry_with_defaults.get("source", "manual"),
                "metadata": entry_with_defaults.get("metadata", {})
            },
            "created_at": entry_with_defaults["timestamp"]
        }
        
        # Create node in graph store
        self.graph_store.create_node(node_data)
        
        # Persist immediately for better data safety
        if hasattr(self.graph_store, 'persist'):
            self.graph_store.persist()
        elif hasattr(self.graph_store, 'save'):
            self.graph_store.save()
        
        # Return the created node
        return self.get_node(entry_with_defaults["id"])
    
    def _generate_tags(self, category: str) -> List[str]:
        """Generate auto-tags for a log entry based on category."""
        # Using underscore instead of colon to comply with tag pattern
        tags = []
        if category:
            # Add sanitized tag
            sanitized_tag = f"log_{category}".lower()
            # Ensure it matches the pattern ^[a-z0-9_-]+$
            sanitized_tag = sanitized_tag.replace(":", "_")
            tags.append(sanitized_tag)
        return tags
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID from the graph store."""
        return self.graph_store.get_node(node_id)
    
    def import_csv(self, csv_path: str) -> Dict[str, Any]:
        """
        Import log entries from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Import report with 'imported' and 'errors' lists
        """
        path = Path(csv_path)
        if not path.exists():
            raise ImportError(f"File not found: {csv_path}")
        
        imported = []
        errors = []
        
        with open(path, 'r', encoding='utf-8', newline="" ) as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    # Parse metadata if it's JSON
                    if "metadata" in row and row["metadata"]:
                        try:
                            row["metadata"] = json.loads(row["metadata"])
                        except json.JSONDecodeError:
                            row["metadata"] = {}
                    
                    # Parse value if it's JSON
                    if "value" in row and row["value"]:
                        try:
                            row["value"] = json.loads(row["value"])
                        except json.JSONDecodeError:
                            # Keep as string if not valid JSON
                            pass
                    
                    # Handle empty confidence
                    if "confidence" in row and row["confidence"] == "":
                        row.pop("confidence", None)
                    
                    # Set source to 'imported' for CSV imports
                    row["source"] = "imported"
                    
                    # Log the entry
                    node = self.log(row)
                    imported.append(node)
                    
                except Exception as e:
                    errors.append({
                        "row": row_num,
                        "error": str(e),
                        "data": row
                    })
        
        # Persist after import
        if hasattr(self.graph_store, 'persist'):
            self.graph_store.persist()
        elif hasattr(self.graph_store, 'save'):
            self.graph_store.save()
        
        return {
            "imported": imported,
            "errors": errors
        }
    
    def get_daily_summary(self, date_str: str, tz: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregated summary of log entries for a specific day.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            tz: Timezone (defaults to UTC)
            
        Returns:
            Dict with categories as keys, containing count, sum, avg for numeric values
        """
        # Parse the date
        try:
            target_date = ensure_timezone_aware(datetime.fromisoformat(date_str)).date()
        except Exception:
            raise ValueError(f"Invalid date format: {date_str}")
        
        # Search for log nodes
        all_nodes = self.graph_store.search_nodes(limit=10000)  # Get many nodes
        
        summary = {}
        
        for node in all_nodes:
            if node.get("type") != "log":
                continue
                
            attrs = node.get("attrs", {})
            timestamp_str = attrs.get("timestamp")
            if not timestamp_str:
                continue
            
            # Parse timestamp and check if it's on the target date
            try:
                if timestamp_str.endswith("Z"):
                    timestamp_str = timestamp_str[:-1] + "+00:00"
                dt = ensure_timezone_aware(datetime.fromisoformat(timestamp_str))
                
                # Convert to UTC if needed
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                
                # Check if same date (in UTC)
                if dt.date() != target_date:
                    continue
                    
            except Exception:
                continue
            
            # Aggregate by category
            category = attrs.get("category")
            if not category:
                continue
                
            if category not in summary:
                summary[category] = {
                    "count": 0,
                    "values": []
                }
            
            summary[category]["count"] += 1
            
            # Collect values for numeric aggregation
            value = attrs.get("value")
            if value is not None and isinstance(value, (int, float)):
                summary[category]["values"].append(value)
        
        # Calculate aggregations
        for category, data in summary.items():
            if data["values"]:
                data["sum"] = sum(data["values"])
                data["avg"] = data["sum"] / len(data["values"])
            # Remove the temporary values list
            data.pop("values", None)
        
        return summary


def main(inputs: Dict[str, Any], outputs: Dict[str, Any], openai_client: Any = None) -> Any:
    """
    Main entry point for the unified logger module.
    """
    # Create a logger instance if data_dir is provided
    data_dir = inputs.get("data_dir")
    graph_store = inputs.get("graph_store")
    
    if graph_store:
        logger = UnifiedLogger(graph_store=graph_store)
    elif data_dir:
        logger = UnifiedLogger(data_dir=data_dir)
    else:
        raise ValueError("Either data_dir or graph_store must be provided in inputs")
    
    outputs["logger"] = logger
    return logger
