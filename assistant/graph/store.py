from __future__ import annotations
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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


# Minimal logger stub. This is intentionally tiny; replace with real logger in production.
class UnifiedLogger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        print("INFO:", *args)

    def debug(self, *args: Any, **kwargs: Any) -> None:
        print("DEBUG:", *args)

    def warning(self, *args: Any, **kwargs: Any) -> None:
        print("WARN:", *args)

    def error(self, *args: Any, **kwargs: Any) -> None:
        print("ERROR:", *args)


# Allowed enums from schemas (kept representative)
_ALLOWED_NODE_TYPES = {
    "thought",
    "note",
    "log",
    "task",
    "goal",
    "person",
    "concept",
    "reference",
}

_ALLOWED_EDGE_KINDS = {
    "relates_to",
    "follows",
    "contradicts",
    "supports",
    "implements",
    "blocks",
    "contains",
    "references",
}

_TAG_PATTERN = re.compile(r"^[a-z0-9_-]+$")


def _generate_uuid() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    # Return an ISO 8601 UTC timestamp with Z suffix
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class GraphStore:
    """
    JSONL-backed persistent graph store with in-memory indices.

    - Persists nodes.jsonl and edges.jsonl under data_dir
    - Loads entire graph into memory on init
    - Thread-safe via RLock
    - All data stored locally in JSONL
    """

    NODES_FILENAME = "nodes.jsonl"
    EDGES_FILENAME = "edges.jsonl"
    TEMP_SUFFIX = ".tmp"
    BAK_SUFFIX = ".bak"

    def __init__(self, data_dir: Path | str, logger: Optional[Any] = None) -> None:
        """
        Initialize the GraphStore. Loads existing data files if present.

        Args:
            data_dir: directory path to store nodes.jsonl and edges.jsonl
            logger: optional logger implementing info/debug/warning/error
        """
        self.data_dir = Path(data_dir)
        # Use RLock for reentrant locking
        self.lock = threading.RLock()
        # Primary stores
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[str, Dict[str, Any]] = {}
        # Indices
        self._tag_index: Dict[str, set] = {}
        self._out_adj: Dict[str, set] = {}  # node_id -> set(edge_id)
        self._in_adj: Dict[str, set] = {}  # node_id -> set(edge_id)
        # Lowercased text cache for substring search
        self._text_cache: Dict[str, str] = {}

        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # logger
        self.logger = logger if logger is not None else UnifiedLogger()

        # Load existing data
        self.load()

        # Provide aliases for older APIs
        self.add_node = self.create_node
        self.put_node = self.create_node
        self.add_edge = self.create_edge
        self.put_edge = self.create_edge
        self.remove_edge = self.delete_edge  # Add alias for delete_edge

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the graph."""
        with self.lock:
            return [dict(node) for node in self._nodes.values()]

    @property
    def tag_index(self) -> Dict[str, set]:
        """Expose tag index for tests."""
        return self._tag_index
    
    @property
    def adj_out(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Expose adjacency list for outgoing edges.
        Returns edge objects, not just IDs (tests expect this).
        """
        with self.lock:
            result = {}
            for node_id, edge_ids in self._out_adj.items():
                result[node_id] = [self._edges[eid] for eid in edge_ids if eid in self._edges]
            return result
    
    @property
    def adj_in(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Expose adjacency list for incoming edges.
        Returns edge objects, not just IDs (tests expect this).
        """
        with self.lock:
            result = {}
            for node_id, edge_ids in self._in_adj.items():
                result[node_id] = [self._edges[eid] for eid in edge_ids if eid in self._edges]
            return result

    # ----------------------
    # Internal helper methods
    # ----------------------
    def _nodes_path(self) -> Path:
        return self.data_dir / self.NODES_FILENAME

    def _edges_path(self) -> Path:
        return self.data_dir / self.EDGES_FILENAME

    def _nodes_temp_path(self) -> Path:
        return self._nodes_path().with_suffix(self._nodes_path().suffix + self.TEMP_SUFFIX)

    def _edges_temp_path(self) -> Path:
        return self._edges_path().with_suffix(self._edges_path().suffix + self.TEMP_SUFFIX)

    def _backup_path(self, target: Path) -> Path:
        return target.with_suffix(target.suffix + self.BAK_SUFFIX)

    def _validate_tags(self, tags: List[str]) -> List[str]:
        normalized: List[str] = []
        for t in tags:
            if not isinstance(t, str):
                raise ValueError(f"Tag must be string, got {type(t)}")
            t_norm = t.lower()
            if not _TAG_PATTERN.match(t_norm):
                raise ValueError(f"Tag '{t}' does not match pattern {_TAG_PATTERN.pattern}")
            normalized.append(t_norm)
        return normalized

    def _index_node(self, node: Dict[str, Any]) -> None:
        node_id = node["id"]
        tags = node.get("tags") or []
        for t in tags:
            s = self._tag_index.setdefault(t, set())
            s.add(node_id)
        # text cache
        text = node.get("text") or ""
        self._text_cache[node_id] = text.lower()

    def _unindex_node(self, node: Dict[str, Any]) -> None:
        node_id = node["id"]
        tags = node.get("tags") or []
        for t in tags:
            s = self._tag_index.get(t)
            if s:
                s.discard(node_id)
                if not s:
                    self._tag_index.pop(t, None)
        self._text_cache.pop(node_id, None)

    def _index_edge(self, edge: Dict[str, Any]) -> None:
        edge_id = edge["id"]
        src = edge["src"]
        dst = edge["dst"]
        self._out_adj.setdefault(src, set()).add(edge_id)
        self._in_adj.setdefault(dst, set()).add(edge_id)

    def _unindex_edge(self, edge: Dict[str, Any]) -> None:
        edge_id = edge["id"]
        src = edge["src"]
        dst = edge["dst"]
        s = self._out_adj.get(src)
        if s:
            s.discard(edge_id)
            if not s:
                self._out_adj.pop(src, None)
        s = self._in_adj.get(dst)
        if s:
            s.discard(edge_id)
            if not s:
                self._in_adj.pop(dst, None)

    def _atomic_replace(self, temp_path: Path, target_path: Path) -> None:
        """
        Atomically replace target_path with temp_path while attempting to preserve a backup
        in case of failures so we can restore original if needed.
        """
        backup = self._backup_path(target_path)
        try:
            # If target exists, move it to backup first
            if target_path.exists():
                os.replace(str(target_path), str(backup))
            # Move temp into place
            os.replace(str(temp_path), str(target_path))
        except Exception as e:
            # Attempt to restore backup if present
            try:
                if backup.exists():
                    os.replace(str(backup), str(target_path))
            except Exception:
                # If restoration failed, log and re-raise original exception
                self.logger.error("Failed to restore backup after atomic replace failure", e)
            raise
        else:
            # Success: remove backup if present
            try:
                if backup.exists():
                    os.remove(str(backup))
            except Exception:
                self.logger.debug("Failed to remove backup file", backup)

    # ----------------------
    # Persistence
    # ----------------------
    def load(self) -> None:
        """
        Load nodes and edges from JSONL files into memory.
        Partial/corrupted trailing lines are tolerated and ignored.
        """
        with self.lock:
            self._nodes.clear()
            self._edges.clear()
            self._tag_index.clear()
            self._in_adj.clear()
            self._out_adj.clear()
            self._text_cache.clear()

            # Load nodes
            nodes_path = self._nodes_path()
            if nodes_path.exists():
                try:
                    with nodes_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                # Corrupted trailing line: stop processing further
                                self.logger.warning("Skipping corrupted line in nodes file")
                                break
                            nid = obj.get("id")
                            if not nid:
                                continue
                            self._nodes[nid] = obj
                            self._index_node(obj)
                except Exception as e:
                    self.logger.error("Failed to load nodes", e)

            # Load edges
            edges_path = self._edges_path()
            if edges_path.exists():
                try:
                    with edges_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                self.logger.warning("Skipping corrupted line in edges file")
                                break
                            eid = obj.get("id")
                            if not eid:
                                continue
                            self._edges[eid] = obj
                            self._index_edge(obj)
                except Exception as e:
                    self.logger.error("Failed to load edges", e)

    def save(self) -> None:
        """
        Persist current in-memory nodes and edges to disk atomically with temp files and backups.
        """
        with self.lock:
            nodes_temp = self._nodes_temp_path()
            edges_temp = self._edges_temp_path()
            nodes_path = self._nodes_path()
            edges_path = self._edges_path()

            # Write nodes temp
            try:
                with nodes_temp.open("w", encoding="utf-8") as f:
                    for node in self._nodes.values():
                        f.write(json.dumps(node, separators=(",", ":")) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                # If writing temp fails, ensure no temp left
                try:
                    if nodes_temp.exists():
                        nodes_temp.unlink()
                except Exception:
                    pass
                self.logger.error("Failed to write nodes temp file", e)
                raise

            # Write edges temp
            try:
                with edges_temp.open("w", encoding="utf-8") as f:
                    for edge in self._edges.values():
                        f.write(json.dumps(edge, separators=(",", ":")) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                try:
                    if edges_temp.exists():
                        edges_temp.unlink()
                except Exception:
                    pass
                self.logger.error("Failed to write edges temp file", e)
                # cleanup nodes temp as well to avoid half-written temps
                try:
                    if nodes_temp.exists():
                        nodes_temp.unlink()
                except Exception:
                    pass
                raise

            # Both temps written, perform atomic replacements with backup/restore fallback
            try:
                self._atomic_replace(nodes_temp, nodes_path)
                self._atomic_replace(edges_temp, edges_path)
            except Exception as e:
                self.logger.error("Failed during atomic replace of data files", e)
                # Attempt cleanup of temps
                try:
                    if nodes_temp.exists():
                        nodes_temp.unlink()
                except Exception:
                    pass
                try:
                    if edges_temp.exists():
                        edges_temp.unlink()
                except Exception:
                    pass
                raise
    
    # Alias for backward compatibility
    def persist(self) -> None:
        """Alias for save() to match test expectations."""
        self.save()

    # ----------------------
    # Node operations
    # ----------------------
    def create_node(self, type: str, text: str = "", tags: List[str] = None, attrs: dict = None) -> str:
        """
        Create a node and persist in memory. Returns node id.

        This function is tolerant of being called with a single dict bound to the first parameter:
        e.g., create_node(node_dict) where node_dict contains id/type/text/tags/attrs/created_at.
        """
        with self.lock:
            # Support being passed a pre-built node dict as first param
            if isinstance(type, dict):
                node = dict(type)  # copy
                nid = node.get("id") or _generate_uuid()
                node["id"] = nid
                if "created_at" not in node:
                    node["created_at"] = _now_iso()
                # normalize tags
                node_tags = node.get("tags") or []
                node["tags"] = self._validate_tags(node_tags) if node_tags else []
                # ensure type present and valid-ish
                node_type = node.get("type") or "note"
                node["type"] = node_type
            else:
                node_type = type if isinstance(type, str) else "note"
                if node_type not in _ALLOWED_NODE_TYPES:
                    # allow unknown types but log
                    self.logger.debug("Unknown node type", node_type)
                nid = _generate_uuid()
                node = {
                    "id": nid,
                    "type": node_type,
                    "text": text or "",
                    "tags": self._validate_tags(tags) if tags else [],
                    "attrs": dict(attrs) if attrs else {},
                    "created_at": _now_iso(),
                }

            if nid in self._nodes:
                # overwrite existing node id: unindex old and index new
                old = self._nodes[nid]
                self._unindex_node(old)
            self._nodes[nid] = node
            self._index_node(node)
            return nid

    def get_node(self, node_id: str) -> Optional[dict]:
        with self.lock:
            n = self._nodes.get(node_id)
            return dict(n) if n is not None else None

    def update_node(self, node_id: str, text: str = None, tags: List[str] = None, attrs: dict = None) -> bool:
        with self.lock:
            node = self._nodes.get(node_id)
            if not node:
                return False
            # unindex old tags/text
            self._unindex_node(node)
            if text is not None:
                node["text"] = text
            if tags is not None:
                node["tags"] = self._validate_tags(tags)
            if attrs is not None:
                node["attrs"] = dict(attrs)
            # reindex
            self._index_node(node)
            return True

    def delete_node(self, node_id: str, cascade: bool = True) -> bool:
        """
        Delete node by id. If cascade is True, also delete connected edges.
        """
        with self.lock:
            node = self._nodes.pop(node_id, None)
            if not node:
                return False
            # remove from tag index and text cache
            self._unindex_node(node)
            # remove edges
            out_edges = list(self._out_adj.get(node_id, set()))
            in_edges = list(self._in_adj.get(node_id, set()))
            to_remove = set(out_edges) | set(in_edges)
            if cascade:
                for eid in to_remove:
                    edge = self._edges.pop(eid, None)
                    if edge:
                        self._unindex_edge(edge)
            else:
                # If not cascade, just detach edges (keep edges but remove references to node)
                for eid in to_remove:
                    edge = self._edges.get(eid)
                    if not edge:
                        continue
                    # remove adjacency references to this node
                    if edge["src"] == node_id:
                        self._out_adj.get(node_id, set()).discard(eid)
                        edge["src"] = None
                    if edge["dst"] == node_id:
                        self._in_adj.get(node_id, set()).discard(eid)
                        edge["dst"] = None
            # cleanup adj dicts for node_id
            self._out_adj.pop(node_id, None)
            self._in_adj.pop(node_id, None)
            return True

    # ----------------------
    # Edge operations
    # ----------------------
    def create_edge(
        self,
        src_id: str,
        dst_id: str = None,
        kind: str = None,
        weight: float = 1.0,
        attrs: dict = None,
    ) -> str:
        """
        Create an edge and persist in memory. Returns edge id.

        This function is tolerant of being called with a single dict bound to the first parameter:
        e.g., create_edge(edge_dict) where edge_dict contains id/src/dst/kind/weight/attrs/created_at.
        """
        with self.lock:
            # Allow a dict passed as first arg (tests may call with edge dict)
            if isinstance(src_id, dict):
                edge = dict(src_id)
                eid = edge.get("id") or _generate_uuid()
                edge["id"] = eid
                if "created_at" not in edge:
                    edge["created_at"] = _now_iso()
                # normalize keys
                edge.setdefault("weight", 1.0)
                edge.setdefault("attrs", {})
                # ensure src/dst present
                if "src" not in edge or "dst" not in edge:
                    raise ValueError("Edge dict must contain 'src' and 'dst'")
                # kind default
                if "kind" not in edge:
                    edge["kind"] = kind or "relates_to"
            else:
                if dst_id is None or kind is None:
                    raise ValueError("dst_id and kind are required when not passing an edge dict")
                if kind not in _ALLOWED_EDGE_KINDS:
                    self.logger.debug("Unknown edge kind", kind)
                eid = _generate_uuid()
                edge = {
                    "id": eid,
                    "src": src_id,
                    "dst": dst_id,
                    "kind": kind,
                    "weight": float(weight) if weight is not None else 1.0,
                    "attrs": dict(attrs) if attrs else {},
                    "created_at": _now_iso(),
                }

            # ensure nodes exist (it's allowed to reference non-existent nodes, but index accordingly)
            # If same id exists, unindex previous
            if eid in self._edges:
                old = self._edges[eid]
                self._unindex_edge(old)
            self._edges[eid] = edge
            self._index_edge(edge)
            return eid

    def get_edge(self, edge_id: str) -> Optional[dict]:
        with self.lock:
            e = self._edges.get(edge_id)
            return dict(e) if e is not None else None

    def delete_edge(self, edge_id: str) -> bool:
        """
        Delete edge by id. Does not affect nodes.
        """
        with self.lock:
            edge = self._edges.pop(edge_id, None)
            if not edge:
                return False
            # Remove from indices
            self._unindex_edge(edge)
            return True

    def get_edges(self, src_id: str = None, dst_id: str = None, kind: str = None) -> List[dict]:
        with self.lock:
            results: List[dict] = []
            # Narrow search using adjacency if possible
            candidates: Optional[set] = None
            if src_id:
                candidates = set(self._out_adj.get(src_id, set()))
            if dst_id:
                dst_set = set(self._in_adj.get(dst_id, set()))
                candidates = dst_set if candidates is None else (candidates & dst_set)
            if candidates is None:
                # fallback to all edges
                edges_iter = list(self._edges.values())
            else:
                edges_iter = [self._edges[eid] for eid in candidates if eid in self._edges]

            for edge in edges_iter:
                if kind and edge.get("kind") != kind:
                    continue
                results.append(dict(edge))
            # sort by created_at descending
            results.sort(key=lambda e: e.get("created_at", ""), reverse=True)
            return results

    # ----------------------
    # Search and neighbors
    # ----------------------
    def search_nodes(
            self,
            tags: List[str] = None,
            text: str = None,
            node_type: str = None,
            created_after: Union[str, datetime] = None,
            created_before: Union[str, datetime] = None,
            limit: int = 10,
    ) -> List[dict]:
        """
        Search nodes by AND tags, case-insensitive substring text, node_type, and date range.

        Args:
            tags: List of tags (AND condition)
            text: Substring to search in node text
            node_type: Filter by specific node type
            created_after: Return nodes created after this datetime (string ISO or datetime object)
            created_before: Return nodes created before this datetime (string ISO or datetime object)
            limit: Maximum number of results to return

        Returns:
            List of matching nodes sorted by created_at descending
        """
        with self.lock:
            candidate_ids: Optional[set] = None

            # Filter by tags
            if tags:
                norm = self._validate_tags(tags)
                for t in norm:
                    ids = self._tag_index.get(t, set())
                    if candidate_ids is None:
                        candidate_ids = set(ids)
                    else:
                        candidate_ids &= ids
                # If no nodes match tags, early return
                if not candidate_ids:
                    return []

            # Filter by text
            if text:
                tq = text.lower()
                matched = {nid for nid, txt in self._text_cache.items() if tq in txt}
                candidate_ids = matched if candidate_ids is None else (candidate_ids & matched)
                if not candidate_ids:
                    return []

            # Filter by node type
            if node_type:
                matched_type = {nid for nid, node in self._nodes.items() if node.get("type") == node_type}
                candidate_ids = matched_type if candidate_ids is None else (candidate_ids & matched_type)
                if not candidate_ids:
                    return []

            # If still None, take all ids
            if candidate_ids is None:
                candidate_ids = set(self._nodes.keys())

            # Get nodes for candidates
            results = [self._nodes[nid] for nid in candidate_ids]

            # Filter by date range if specified
            if created_after or created_before:
                filtered_results = []

                # Parse datetime parameters if they're strings
                after_dt = None
                before_dt = None

                if created_after:
                    if isinstance(created_after, str):
                        # Handle ISO format with Z or +00:00
                        after_dt = ensure_timezone_aware(datetime.fromisoformat(created_after.replace('Z', '+00:00')))
                    elif isinstance(created_after, datetime):
                        after_dt = created_after
                    else:
                        raise ValueError(f"created_after must be datetime or ISO string, got {type(created_after)}")

                if created_before:
                    if isinstance(created_before, str):
                        before_dt = ensure_timezone_aware(datetime.fromisoformat(created_before.replace('Z', '+00:00')))
                    elif isinstance(created_before, datetime):
                        before_dt = created_before
                    else:
                        raise ValueError(f"created_before must be datetime or ISO string, got {type(created_before)}")

                # Filter nodes by date
                for node in results:
                    created_at_str = node.get("created_at")
                    if not created_at_str:
                        continue  # Skip nodes without created_at

                    try:
                        # Parse node's created_at timestamp
                        node_dt = ensure_timezone_aware(datetime.fromisoformat(created_at_str.replace('Z', '+00:00')))

                        # Check date constraints
                        if after_dt and node_dt <= after_dt:
                            continue  # Node too old
                        if before_dt and node_dt >= before_dt:
                            continue  # Node too new

                        # Node passes date filters
                        filtered_results.append(node)

                    except (ValueError, AttributeError) as e:
                        # Skip nodes with invalid timestamps
                        self.logger.debug(f"Skipping node with invalid timestamp: {created_at_str}")
                        continue

                results = filtered_results

            # Sort by created_at desc
            results.sort(key=lambda n: n.get("created_at", ""), reverse=True)

            # Return copies
            return [dict(n) for n in results[:limit]]

    # Helper method to add to GraphStore for getting nodes in a date range
    def get_nodes_in_range(
            self,
            start_time: Union[str, datetime],
            end_time: Union[str, datetime],
            node_type: str = None,
            limit: int = 100
    ) -> List[dict]:
        """
        Convenience method to get nodes within a specific time range.

        Args:
            start_time: Start of time range (inclusive)
            end_time: End of time range (exclusive)
            node_type: Optional filter by node type
            limit: Maximum results

        Returns:
            List of nodes in the time range
        """
        return self.search_nodes(
            node_type=node_type,
            created_after=start_time,
            created_before=end_time,
            limit=limit
        )

    # Also add a method to get recent nodes easily
    def get_recent_nodes(
            self,
            hours: int = 24,
            node_type: str = None,
            limit: int = 50
    ) -> List[dict]:
        """
        Get nodes created in the last N hours.

        Args:
            hours: Number of hours to look back
            node_type: Optional filter by node type
            limit: Maximum results

        Returns:
            List of recent nodes
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return self.search_nodes(
            node_type=node_type,
            created_after=cutoff.isoformat(),
            limit=limit
        )

    def get_neighbors(self, node_id: str, edge_kind: str = None, direction: str = "both") -> List[dict]:
        """
        Return neighbor nodes (as dicts) connected to node_id.
        direction: 'in', 'out', or 'both'
        If edge_kind provided, only consider edges with that kind.
        """
        with self.lock:
            eids = set()
            if direction in ("out", "both"):
                eids |= set(self._out_adj.get(node_id, set()))
            if direction in ("in", "both"):
                eids |= set(self._in_adj.get(node_id, set()))

            neighbors = set()
            for eid in eids:
                edge = self._edges.get(eid)
                if not edge:
                    continue
                if edge_kind and edge.get("kind") != edge_kind:
                    continue
                src = edge.get("src")
                dst = edge.get("dst")
                if src and src != node_id:
                    neighbors.add(src)
                if dst and dst != node_id:
                    neighbors.add(dst)
            return [dict(self._nodes[nid]) for nid in neighbors if nid in self._nodes]



# ----------------------
# Entrypoint
# ----------------------
def main(inputs: Dict[str, Any], outputs: Dict[str, Any], openai_client: Any | None = None) -> Any:
    """
    Entrypoint required by the public API contract.

    Expects inputs:
      - data_dir: path to directory where nodes.jsonl & edges.jsonl are stored

    Outputs:
      - graph_store: GraphStore instance
      - logger: UnifiedLogger instance
    """
    data_dir = inputs.get("data_dir")
    logger = UnifiedLogger()
    store = GraphStore(data_dir, logger=logger)
    outputs["graph_store"] = store
    outputs["logger"] = logger
    return store
