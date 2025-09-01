from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
import uuid
import re

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


# Public API functions/classes (must remain available):
# - create_node_id() -> str
# - create_edge_id() -> str  
# - class Node
# - class Edge
# - class LogEntry (ADDED - tests expect this)
# - node_to_dict(node: Node) -> dict
# - edge_to_dict(edge: Edge) -> dict
# - dict_to_node(data: dict) -> Node
# - dict_to_edge(data: dict) -> Edge
# - main(inputs: Dict[str, Any], outputs: Dict[str, Any], openai_client: Any | None = None) -> Any

NODE_TYPES: List[str] = [
    "thought",
    "note",
    "log",
    "task",
    "goal",
    "person",
    "concept",
    "reference",
]

EDGE_KINDS: List[str] = [
    "relates_to",
    "follows",
    "contradicts",
    "supports",
    "implements",
    "blocks",
    "contains",
    "references",
]

# LogEntry categories from the schema
LOG_CATEGORIES: List[str] = [
    "nutrition",
    "medication", 
    "exercise",
    "mood",
    "sleep",
    "task",
    "energy",
    "focus",
    "social",
    "custom",
]

LOG_SOURCES: List[str] = [
    "manual",
    "imported",
    "inferred",
    "device",
]

_UUID_PATTERN = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
)
_TAG_PATTERN = re.compile(r"^[a-z0-9_-]+$")

T = TypeVar("T")


def _now_iso() -> str:
    """
    Return current UTC time as an ISO-8601 string without explicit timezone info.
    Using naive UTC (datetime.utcnow()) to match test expectations.
    """
    return datetime.utcnow().isoformat()


def create_node_id() -> str:
    """Generate a UUID string suitable for node.id."""
    return str(uuid.uuid4())


def create_edge_id() -> str:
    """Generate a UUID string suitable for edge.id."""
    return str(uuid.uuid4())

def create_log_id() -> str:
    """Generate a UUID string suitable for log_entry.id."""
    return str(uuid.uuid4())


class ValidationError(ValueError):
    """Raised when schema validation fails for nodes/edges/log entries."""
    pass


def _is_valid_uuid_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return _UUID_PATTERN.fullmatch(value) is not None


def _is_valid_iso_datetime(value: Any) -> bool:
    """
    Accept both naive and timezone-aware ISO strings parsable by datetime.fromisoformat.
    """
    if not isinstance(value, str):
        return False
    try:
        ensure_timezone_aware(datetime.fromisoformat(value))
        return True
    except Exception:
        return False


def _validate_attrs(attrs: Any) -> None:
    if attrs is None:
        return
    if not isinstance(attrs, dict):
        raise ValidationError("Field 'attrs' must be an object/dict if provided.")
    # arbitrary content allowed


def _validate_tags(tags: Any) -> None:
    if tags is None:
        return
    if not isinstance(tags, list):
        raise ValidationError("Field 'tags' must be a list of strings or null.")
    for t in tags:
        if not isinstance(t, str):
            raise ValidationError("Each tag must be a string.")
        if not _TAG_PATTERN.fullmatch(t):
            raise ValidationError(f"Tag '{t}' does not match required pattern '^[a-z0-9_-]+$'.")


def _ensure_iso_string(value: Optional[str], field_name: str) -> str:
    if value is None:
        raise ValidationError(f"Field '{field_name}' is required and must be an ISO datetime string.")
    if not _is_valid_iso_datetime(value):
        raise ValidationError(f"Field '{field_name}' must be a valid ISO datetime string.")
    return value


def _iso_to_dt(s: str) -> datetime:
    return ensure_timezone_aware(datetime.fromisoformat(s))


def _dt_to_iso(dt: datetime) -> str:
    return dt.isoformat()


@dataclass
class Node:
    """
    Data model for a Node in the personal assistant graph.
    Fields:
      - id: UUID string
      - type: one of NODE_TYPES
      - text: optional textual content
      - tags: optional list of tag strings (pattern [a-z0-9_-]+)
      - attrs: optional dict of arbitrary metadata
      - created_at: ISO datetime string
      - updated_at: ISO datetime string or None
    """
    id: str
    type: str
    text: Optional[str] = None
    tags: Optional[List[str]] = None
    attrs: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: Optional[str] = None

    # internal flag to prevent changes to immutable fields after init
    _sealed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Validate id
        if not isinstance(self.id, str) or not _is_valid_uuid_string(self.id):
            raise ValidationError(f"Field 'id' must be a valid UUID string; got: {self.id!r}")
        # Validate type
        if not isinstance(self.type, str):
            raise ValidationError("Field 'type' must be a string.")
        if self.type not in NODE_TYPES:
            raise ValidationError(f"Field 'type' must be one of {NODE_TYPES}; got: {self.type!r}")
        # Validate tags and attrs
        _validate_tags(self.tags)
        _validate_attrs(self.attrs)
        # Validate created_at
        if not isinstance(self.created_at, str) or not _is_valid_iso_datetime(self.created_at):
            raise ValidationError("Field 'created_at' must be a valid ISO datetime string.")
        # Validate updated_at if present
        if self.updated_at is not None and not _is_valid_iso_datetime(self.updated_at):
            raise ValidationError("Field 'updated_at' must be a valid ISO datetime string or null.")
        # Seal to prevent changes to id and created_at
        self._sealed = True

    def __setattr__(self, name: str, value: Any) -> None:
        # Allow setting anything before sealed or setting the _sealed flag itself
        if name in ("_sealed",):
            object.__setattr__(self, name, value)
            return
        # If sealed, disallow changing id or created_at
        if getattr(self, "_sealed", False) and name in ("id", "created_at"):
            current = getattr(self, name, None)
            # If attribute exists and value would change, block
            if current is not None and value != current:
                raise AttributeError(f"Attribute '{name}' is immutable after creation.")
        object.__setattr__(self, name, value)

    def touch(self) -> None:
        """
        Update the updated_at timestamp to a new ISO datetime string.
        Ensures strictly monotonic increase relative to previous updated_at if present.
        """
        now = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = _dt_to_iso(now)
            return
        prev_dt = _iso_to_dt(self.updated_at)
        # Ensure strictly greater than previous; if not, add microsecond increments
        if now <= prev_dt:
            now = prev_dt + timedelta(microseconds=1)
        self.updated_at = _dt_to_iso(now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Instance method for serialization (tests expect this)."""
        return node_to_dict(self)

    @classmethod
    def from_dict(cls: Type["Node"], data: Dict[str, Any]) -> "Node":
        """
        Construct and validate a Node from a dict. Auto-generates id and created_at if missing.
        """
        if not isinstance(data, dict):
            raise ValidationError("Node data must be an object/dict.")
        d = dict(data)  # shallow copy
        # id
        id_val = d.get("id")
        if id_val is None:
            id_val = create_node_id()
        if not isinstance(id_val, str) or not _is_valid_uuid_string(id_val):
            raise ValidationError(f"Field 'id' must be a valid UUID string; got: {id_val!r}")
        # type
        type_val = d.get("type")
        if type_val is None:
            raise ValidationError("Field 'type' is required for Node.")
        # text
        text_val = d.get("text")
        # tags
        tags_val = d.get("tags")
        # attrs
        attrs_val = d.get("attrs")
        # created_at
        created_at_val = d.get("created_at")
        if created_at_val is None:
            created_at_val = _now_iso()
        elif not _is_valid_iso_datetime(created_at_val):
            raise ValidationError("Field 'created_at' must be a valid ISO datetime string.")
        # updated_at
        updated_at_val = d.get("updated_at")
        if updated_at_val is not None and not _is_valid_iso_datetime(updated_at_val):
            raise ValidationError("Field 'updated_at' must be a valid ISO datetime string or null.")
        return cls(
            id=id_val,
            type=type_val,
            text=text_val,
            tags=tags_val,
            attrs=attrs_val,
            created_at=created_at_val,
            updated_at=updated_at_val,
        )


@dataclass
class Edge:
    """
    Data model for an Edge in the graph.
    Fields:
      - id: UUID string
      - src: source node id (string - relaxed from UUID requirement)
      - dst: destination node id (string - relaxed from UUID requirement)
      - kind: one of EDGE_KINDS
      - weight: float between 0.0 and 1.0 inclusive (default 1.0)
      - attrs: optional dict of arbitrary metadata
      - created_at: ISO datetime string
    
    DESIGN NOTE: We're relaxing the UUID requirement for src/dst because:
    1. Tests use simple strings like "s" and "d" for testing
    2. In practice, we may want edges to reference external IDs
    3. The schema in the test file just says "string" not "uuid pattern"
    """
    id: str
    src: str
    dst: str
    kind: str
    weight: float = 1.0
    attrs: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=_now_iso)

    _sealed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Validate id (still needs to be UUID)
        if not isinstance(self.id, str) or not _is_valid_uuid_string(self.id):
            raise ValidationError(f"Field 'id' must be a valid UUID string; got: {self.id!r}")
        # CHANGED: src/dst now just need to be non-empty strings
        if not isinstance(self.src, str) or not self.src:
            raise ValidationError(f"Field 'src' must be a non-empty string; got: {self.src!r}")
        if not isinstance(self.dst, str) or not self.dst:
            raise ValidationError(f"Field 'dst' must be a non-empty string; got: {self.dst!r}")
        # Validate kind
        if not isinstance(self.kind, str) or self.kind not in EDGE_KINDS:
            raise ValidationError(f"Field 'kind' must be one of {EDGE_KINDS}; got: {self.kind!r}")
        # Validate weight
        try:
            w = float(self.weight)
        except Exception:
            raise ValidationError("Field 'weight' must be a number between 0.0 and 1.0 inclusive.")
        if not (0.0 <= w <= 1.0):
            raise ValidationError("Field 'weight' must be between 0.0 and 1.0 inclusive.")
        self.weight = w
        # attrs
        _validate_attrs(self.attrs)
        # created_at
        if not isinstance(self.created_at, str) or not _is_valid_iso_datetime(self.created_at):
            raise ValidationError("Field 'created_at' must be a valid ISO datetime string.")
        self._sealed = True

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_sealed",):
            object.__setattr__(self, name, value)
            return
        if getattr(self, "_sealed", False) and name in ("id", "created_at"):
            current = getattr(self, name, None)
            if current is not None and value != current:
                raise AttributeError(f"Attribute '{name}' is immutable after creation.")
        object.__setattr__(self, name, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Instance method for serialization (tests expect this)."""
        return edge_to_dict(self)

    @classmethod
    def from_dict(cls: Type["Edge"], data: Dict[str, Any]) -> "Edge":
        """
        Construct and validate an Edge from a dict. Auto-generates id and created_at if missing.
        """
        if not isinstance(data, dict):
            raise ValidationError("Edge data must be an object/dict.")
        d = dict(data)
        id_val = d.get("id")
        if id_val is None:
            id_val = create_edge_id()
        if not isinstance(id_val, str) or not _is_valid_uuid_string(id_val):
            raise ValidationError(f"Field 'id' must be a valid UUID string; got: {id_val!r}")
        src_val = d.get("src")
        dst_val = d.get("dst")
        if src_val is None or dst_val is None:
            raise ValidationError("Fields 'src' and 'dst' are required for Edge.")
        # CHANGED: Just check they're non-empty strings
        if not isinstance(src_val, str) or not src_val:
            raise ValidationError(f"Field 'src' must be a non-empty string; got: {src_val!r}")
        if not isinstance(dst_val, str) or not dst_val:
            raise ValidationError(f"Field 'dst' must be a non-empty string; got: {dst_val!r}")
        kind_val = d.get("kind")
        if kind_val is None:
            raise ValidationError("Field 'kind' is required for Edge.")
        weight_val = d.get("weight", 1.0)
        attrs_val = d.get("attrs")
        created_at_val = d.get("created_at")
        if created_at_val is None:
            created_at_val = _now_iso()
        elif not _is_valid_iso_datetime(created_at_val):
            raise ValidationError("Field 'created_at' must be a valid ISO datetime string.")
        return cls(
            id=id_val,
            src=src_val,
            dst=dst_val,
            kind=kind_val,
            weight=weight_val,
            attrs=attrs_val,
            created_at=created_at_val,
        )


@dataclass
class LogEntry:
    """
    Data model for a LogEntry - tracking timestamped events.
    Fields:
      - id: UUID string 
      - category: one of LOG_CATEGORIES
      - timestamp: ISO datetime string (when the event occurred)
      - value: optional value (can be number, string, bool, or object - but NOT array)
      - confidence: float between 0.0 and 1.0 (default 1.0)
      - source: one of LOG_SOURCES (default "manual")
      - metadata: optional dict of arbitrary metadata
    
    Note: LogEntry uses 'metadata' while Node/Edge use 'attrs' for extra data.
    """
    id: str
    category: str
    timestamp: str
    value: Optional[Union[float, str, bool, Dict[str, Any]]] = None
    confidence: float = 1.0
    source: str = "manual"
    metadata: Optional[Dict[str, Any]] = None
    
    _sealed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Validate id
        if not isinstance(self.id, str) or not _is_valid_uuid_string(self.id):
            raise ValidationError(f"Field 'id' must be a valid UUID string; got: {self.id!r}")
        # Validate category
        if not isinstance(self.category, str) or self.category not in LOG_CATEGORIES:
            raise ValidationError(f"Field 'category' must be one of {LOG_CATEGORIES}; got: {self.category!r}")
        # Validate timestamp
        if not isinstance(self.timestamp, str) or not _is_valid_iso_datetime(self.timestamp):
            raise ValidationError("Field 'timestamp' must be a valid ISO datetime string.")
        # Validate value (reject arrays)
        if self.value is not None:
            if isinstance(self.value, list):
                raise ValidationError("Field 'value' cannot be an array/list.")
            # Accept number, string, bool, dict
            if not isinstance(self.value, (int, float, str, bool, dict)):
                raise ValidationError("Field 'value' must be number, string, boolean, or object.")
        # Validate confidence
        try:
            c = float(self.confidence)
        except Exception:
            raise ValidationError("Field 'confidence' must be a number between 0.0 and 1.0.")
        if not (0.0 <= c <= 1.0):
            raise ValidationError("Field 'confidence' must be between 0.0 and 1.0.")
        self.confidence = c
        # Validate source
        if not isinstance(self.source, str) or self.source not in LOG_SOURCES:
            raise ValidationError(f"Field 'source' must be one of {LOG_SOURCES}; got: {self.source!r}")
        # Validate metadata
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValidationError("Field 'metadata' must be an object/dict if provided.")
        self._sealed = True

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_sealed",):
            object.__setattr__(self, name, value)
            return
        if getattr(self, "_sealed", False) and name in ("id", "timestamp"):
            current = getattr(self, name, None)
            if current is not None and value != current:
                raise AttributeError(f"Attribute '{name}' is immutable after creation.")
        object.__setattr__(self, name, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Instance method for serialization (tests expect this)."""
        return log_to_dict(self)

    @classmethod
    def from_dict(cls: Type["LogEntry"], data: Dict[str, Any]) -> "LogEntry":
        """
        Construct and validate a LogEntry from a dict. Auto-generates id if missing.
        """
        if not isinstance(data, dict):
            raise ValidationError("LogEntry data must be an object/dict.")
        d = dict(data)
        # id
        id_val = d.get("id")
        if id_val is None:
            id_val = create_log_id()
        if not isinstance(id_val, str) or not _is_valid_uuid_string(id_val):
            raise ValidationError(f"Field 'id' must be a valid UUID string; got: {id_val!r}")
        # category (required)
        category_val = d.get("category")
        if category_val is None:
            raise ValidationError("Field 'category' is required for LogEntry.")
        # timestamp (required)
        timestamp_val = d.get("timestamp")
        if timestamp_val is None:
            raise ValidationError("Field 'timestamp' is required for LogEntry.")
        # optional fields
        value_val = d.get("value")
        confidence_val = d.get("confidence", 1.0)
        source_val = d.get("source", "manual")
        metadata_val = d.get("metadata")
        
        return cls(
            id=id_val,
            category=category_val,
            timestamp=timestamp_val,
            value=value_val,
            confidence=confidence_val,
            source=source_val,
            metadata=metadata_val,
        )


def _clean_dict_for_json(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values and internal fields from dict for JSON serialization."""
    cleaned = {}
    for k, v in d.items():
        # Skip internal fields and None values
        if k.startswith("_") or v is None:
            continue
        cleaned[k] = v
    return cleaned


def node_to_dict(node: Node) -> Dict[str, Any]:
    """
    Serialize a Node to a plain dict suitable for JSON serialization.
    Excludes None values to comply with JSON schema.
    """
    if not isinstance(node, Node):
        raise ValidationError("node_to_dict expects a Node instance.")
    d = asdict(node)
    return _clean_dict_for_json(d)


def edge_to_dict(edge: Edge) -> Dict[str, Any]:
    """
    Serialize an Edge to a plain dict suitable for JSON serialization.
    Excludes None values to comply with JSON schema.
    """
    if not isinstance(edge, Edge):
        raise ValidationError("edge_to_dict expects an Edge instance.")
    d = asdict(edge)
    return _clean_dict_for_json(d)


def log_to_dict(log_entry: LogEntry) -> Dict[str, Any]:
    """
    Serialize a LogEntry to a plain dict suitable for JSON serialization.
    Excludes None values to comply with JSON schema.
    """
    if not isinstance(log_entry, LogEntry):
        raise ValidationError("log_to_dict expects a LogEntry instance.")
    d = asdict(log_entry)
    return _clean_dict_for_json(d)


def dict_to_node(data: Dict[str, Any]) -> Node:
    """
    Convert a dict to a Node, validating and auto-populating missing defaults.
    """
    return Node.from_dict(data)


def dict_to_edge(data: Dict[str, Any]) -> Edge:
    """
    Convert a dict to an Edge, validating and auto-populating missing defaults.
    """
    return Edge.from_dict(data)


def dict_to_log(data: Dict[str, Any]) -> LogEntry:
    """
    Convert a dict to a LogEntry, validating and auto-populating missing defaults.
    """
    return LogEntry.from_dict(data)


def main(inputs: Dict[str, Any], outputs: Dict[str, Any], openai_client: Any | None = None) -> Any:
    """
    Entrypoint to satisfy the public API contract. This module primarily provides models.
    For compatibility, main returns a dict describing available exports and echoes inputs/outputs.
    """
    # No global state or side effects required by tests; simply return a descriptor.
    return {
        "status": "ok",
        "provided_inputs": inputs,
        "expected_outputs": list(outputs.keys()) if isinstance(outputs, dict) else outputs,
    }
