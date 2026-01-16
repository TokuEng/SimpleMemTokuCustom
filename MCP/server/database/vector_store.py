"""
Single-tenant vector store for SimpleMem MCP Server
Uses LanceDB with S3-compatible storage (DigitalOcean Spaces)
"""

import re
from typing import List, Optional, Dict, Any
from datetime import datetime

import lancedb
import pandas as pd
import pyarrow as pa

from ..auth.models import MemoryEntry

# UUID validation pattern (standard UUID format)
UUID_PATTERN = re.compile(
    r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
    re.IGNORECASE
)


# LanceDB schema for memory entries
def get_memory_schema(embedding_dimension: int = 2560) -> pa.Schema:
    """Get PyArrow schema for memory entries"""
    return pa.schema([
        pa.field("entry_id", pa.string()),
        pa.field("lossless_restatement", pa.string()),
        pa.field("keywords", pa.list_(pa.string())),
        pa.field("timestamp", pa.string()),
        pa.field("location", pa.string()),
        pa.field("persons", pa.list_(pa.string())),
        pa.field("entities", pa.list_(pa.string())),
        pa.field("topic", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), embedding_dimension)),
        pa.field("created_at", pa.string()),
    ])


class SingleTenantVectorStore:
    """
    Single-tenant vector storage with S3-compatible backend.
    Uses a single LanceDB table for all memories.
    """

    def __init__(
        self,
        s3_path: str,
        storage_options: Dict[str, str],
        table_name: str = "memories",
        embedding_dimension: int = 2560,
    ):
        """
        Initialize single-tenant vector store with S3 storage.

        Args:
            s3_path: S3 path for LanceDB (e.g., "s3://bucket-name/lancedb")
            storage_options: S3 credentials and configuration
            table_name: Name of the memories table (default: "memories")
            embedding_dimension: Dimension of embedding vectors (default: 2560)
        """
        self.s3_path = s3_path
        self.storage_options = storage_options
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self._table = None

        # Connect to LanceDB with S3 storage
        self.db = lancedb.connect(s3_path, storage_options=storage_options)

    def _get_table(self) -> Any:
        """Get or create the memories table"""
        if self._table is None:
            if self.table_name in self.db.table_names():
                self._table = self.db.open_table(self.table_name)
            else:
                # Create new table with schema
                schema = get_memory_schema(self.embedding_dimension)
                self._table = self.db.create_table(self.table_name, schema=schema)
        return self._table

    async def add_entries(
        self,
        entries: List[MemoryEntry],
        embeddings: List[List[float]],
    ) -> int:
        """
        Add memory entries to the table

        Args:
            entries: List of MemoryEntry objects
            embeddings: List of embedding vectors

        Returns:
            Number of entries added
        """
        if len(entries) != len(embeddings):
            raise ValueError("Number of entries must match number of embeddings")

        if not entries:
            return 0

        table = self._get_table()
        created_at = datetime.utcnow().isoformat()

        # Build records
        records = []
        for entry, embedding in zip(entries, embeddings):
            records.append({
                "entry_id": entry.entry_id,
                "lossless_restatement": entry.lossless_restatement,
                "keywords": entry.keywords or [],
                "timestamp": entry.timestamp or "",
                "location": entry.location or "",
                "persons": entry.persons or [],
                "entities": entry.entities or [],
                "topic": entry.topic or "",
                "vector": embedding,
                "created_at": created_at,
            })

        # Add to table
        table.add(records)
        return len(records)

    async def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 25,
    ) -> List[MemoryEntry]:
        """
        Perform semantic search using vector similarity

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of matching MemoryEntry objects
        """
        table = self._get_table()

        try:
            # Check if table has data
            if table.count_rows() == 0:
                return []

            results = (
                table.search(query_embedding)
                .limit(top_k)
                .to_pandas()
            )

            entries = []
            for _, row in results.iterrows():
                entries.append(MemoryEntry(
                    entry_id=row["entry_id"],
                    lossless_restatement=row["lossless_restatement"],
                    keywords=list(row["keywords"]) if row["keywords"] is not None else [],
                    timestamp=row["timestamp"] if row["timestamp"] else None,
                    location=row["location"] if row["location"] else None,
                    persons=list(row["persons"]) if row["persons"] is not None else [],
                    entities=list(row["entities"]) if row["entities"] is not None else [],
                    topic=row["topic"] if row["topic"] else None,
                ))

            return entries

        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    async def keyword_search(
        self,
        keywords: List[str],
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        """
        Perform keyword-based search (BM25-style matching)

        Args:
            keywords: List of keywords to match
            top_k: Number of results to return

        Returns:
            List of matching MemoryEntry objects
        """
        table = self._get_table()

        try:
            if table.count_rows() == 0:
                return []

            # Load all entries for keyword matching
            df = table.to_pandas()

            # Score each entry
            scores = []
            for idx, row in df.iterrows():
                score = 0
                entry_keywords = set(k.lower() for k in (row["keywords"] or []))
                entry_text = row["lossless_restatement"].lower()

                for kw in keywords:
                    kw_lower = kw.lower()
                    # Keyword list match: 2 points
                    if kw_lower in entry_keywords:
                        score += 2
                    # Text match: 1 point
                    if kw_lower in entry_text:
                        score += 1

                scores.append((idx, score))

            # Sort by score and get top-k
            scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, score in scores[:top_k] if score > 0]

            entries = []
            for idx in top_indices:
                row = df.iloc[idx]
                entries.append(MemoryEntry(
                    entry_id=row["entry_id"],
                    lossless_restatement=row["lossless_restatement"],
                    keywords=list(row["keywords"]) if row["keywords"] is not None else [],
                    timestamp=row["timestamp"] if row["timestamp"] else None,
                    location=row["location"] if row["location"] else None,
                    persons=list(row["persons"]) if row["persons"] is not None else [],
                    entities=list(row["entities"]) if row["entities"] is not None else [],
                    topic=row["topic"] if row["topic"] else None,
                ))

            return entries

        except Exception as e:
            print(f"Keyword search error: {e}")
            return []

    async def structured_search(
        self,
        persons: Optional[List[str]] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        timestamp_start: Optional[str] = None,
        timestamp_end: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        """
        Perform structured/metadata-based search

        Args:
            persons: Filter by person names
            location: Filter by location
            entities: Filter by entities
            timestamp_start: Start of timestamp range
            timestamp_end: End of timestamp range
            top_k: Number of results to return

        Returns:
            List of matching MemoryEntry objects
        """
        table = self._get_table()

        try:
            if table.count_rows() == 0:
                return []

            df = table.to_pandas()

            # Apply filters
            mask = [True] * len(df)

            if persons:
                persons_lower = set(p.lower() for p in persons)
                for i, row in df.iterrows():
                    row_persons = set(p.lower() for p in (row["persons"] or []))
                    if not persons_lower.intersection(row_persons):
                        mask[i] = False

            if location:
                location_lower = location.lower()
                for i, row in df.iterrows():
                    if mask[i] and row["location"]:
                        if location_lower not in row["location"].lower():
                            mask[i] = False
                    elif mask[i]:
                        mask[i] = False

            if entities:
                entities_lower = set(e.lower() for e in entities)
                for i, row in df.iterrows():
                    if mask[i]:
                        row_entities = set(e.lower() for e in (row["entities"] or []))
                        if not entities_lower.intersection(row_entities):
                            mask[i] = False

            if timestamp_start:
                for i, row in df.iterrows():
                    if mask[i] and row["timestamp"]:
                        if row["timestamp"] < timestamp_start:
                            mask[i] = False

            if timestamp_end:
                for i, row in df.iterrows():
                    if mask[i] and row["timestamp"]:
                        if row["timestamp"] > timestamp_end:
                            mask[i] = False

            # Get filtered results
            filtered_df = df[[m for m in mask]][:top_k]

            entries = []
            for _, row in filtered_df.iterrows():
                entries.append(MemoryEntry(
                    entry_id=row["entry_id"],
                    lossless_restatement=row["lossless_restatement"],
                    keywords=list(row["keywords"]) if row["keywords"] is not None else [],
                    timestamp=row["timestamp"] if row["timestamp"] else None,
                    location=row["location"] if row["location"] else None,
                    persons=list(row["persons"]) if row["persons"] is not None else [],
                    entities=list(row["entities"]) if row["entities"] is not None else [],
                    topic=row["topic"] if row["topic"] else None,
                ))

            return entries

        except Exception as e:
            print(f"Structured search error: {e}")
            return []

    async def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries from the table"""
        table = self._get_table()

        try:
            if table.count_rows() == 0:
                return []

            df = table.to_pandas()
            entries = []

            # Check if created_at column exists (may not exist in older tables)
            has_created_at = "created_at" in df.columns

            for _, row in df.iterrows():
                # Safely get created_at value (use pd.notna to handle pandas NA/NaN)
                created_at_val = None
                if has_created_at:
                    try:
                        val = row["created_at"]
                        created_at_val = val if pd.notna(val) else None
                    except Exception:
                        created_at_val = None

                entries.append(MemoryEntry(
                    entry_id=row["entry_id"],
                    lossless_restatement=row["lossless_restatement"],
                    keywords=list(row["keywords"]) if row["keywords"] is not None else [],
                    timestamp=row["timestamp"] if row["timestamp"] else None,
                    location=row["location"] if row["location"] else None,
                    persons=list(row["persons"]) if row["persons"] is not None else [],
                    entities=list(row["entities"]) if row["entities"] is not None else [],
                    topic=row["topic"] if row["topic"] else None,
                    created_at=created_at_val,
                ))

            return entries

        except Exception as e:
            print(f"Get all entries error: {e}")
            return []

    async def count_entries(self) -> int:
        """Count entries in the table"""
        table = self._get_table()
        try:
            return table.count_rows()
        except Exception:
            return 0

    async def clear_table(self) -> bool:
        """Clear all entries from the table"""
        try:
            self._table = None

            if self.table_name in self.db.table_names():
                self.db.drop_table(self.table_name)

            # Recreate empty table
            self._get_table()
            return True

        except Exception as e:
            print(f"Clear table error: {e}")
            return False

    async def delete_entries(self, entry_ids: List[str]) -> int:
        """Delete specific entries by their IDs

        Args:
            entry_ids: List of entry IDs to delete

        Returns:
            Number of entries deleted

        Raises:
            ValueError: If any entry_id is not a valid UUID format
        """
        if not entry_ids:
            return 0

        # Validate all entry IDs are valid UUIDs to prevent injection
        for eid in entry_ids:
            if not UUID_PATTERN.match(eid):
                raise ValueError(f"Invalid entry_id format (must be UUID): {eid}")

        table = self._get_table()
        try:
            # Build WHERE clause for deletion
            # LanceDB uses SQL-like syntax for deletion
            # Entry IDs are validated as UUIDs above, safe to interpolate
            placeholders = ", ".join([f"'{eid}'" for eid in entry_ids])
            where_clause = f"entry_id IN ({placeholders})"

            before_count = table.count_rows()
            table.delete(where_clause)
            after_count = table.count_rows()

            return before_count - after_count

        except Exception as e:
            print(f"Delete entries error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for the table"""
        try:
            table = self._get_table()
            count = table.count_rows()

            return {
                "mode": "single-tenant",
                "table_name": self.table_name,
                "entry_count": count,
                "embedding_dimension": self.embedding_dimension,
                "storage": "S3",
                "s3_path": self.s3_path,
            }
        except Exception as e:
            return {
                "mode": "single-tenant",
                "table_name": self.table_name,
                "entry_count": 0,
                "embedding_dimension": self.embedding_dimension,
                "storage": "S3",
                "s3_path": self.s3_path,
                "error": str(e),
            }

    def test_connection(self) -> tuple[bool, str]:
        """
        Test S3 connection by listing tables.

        Returns:
            Tuple of (success, message)
        """
        try:
            _ = self.db.table_names()
            return True, "S3 connection successful"
        except Exception as e:
            return False, f"S3 connection failed: {str(e)}"


# =============================================================================
# MULTI-TENANT VECTOR STORE (PRESERVED FOR REFERENCE)
# See multi-tenant-backup branch for original implementation
# =============================================================================
#
# class MultiTenantVectorStore:
#     """
#     Multi-tenant vector storage with per-user table isolation.
#     Each user gets their own LanceDB table for complete data isolation.
#     """
#
#     def __init__(
#         self,
#         db_path: str = "./data/lancedb",
#         embedding_dimension: int = 2560,
#     ):
#         self.db_path = db_path
#         self.embedding_dimension = embedding_dimension
#         os.makedirs(db_path, exist_ok=True)
#
#         # Connect to LanceDB (local path)
#         self.db = lancedb.connect(db_path)
#
#         # Cache for opened tables
#         self._tables: Dict[str, Any] = {}
#
#     def _get_table(self, table_name: str) -> Any:
#         """Get or create a user's table"""
#         if table_name not in self._tables:
#             if table_name in self.db.table_names():
#                 self._tables[table_name] = self.db.open_table(table_name)
#             else:
#                 schema = get_memory_schema(self.embedding_dimension)
#                 self._tables[table_name] = self.db.create_table(
#                     table_name,
#                     schema=schema,
#                 )
#         return self._tables[table_name]
#
#     # All methods had table_name as first parameter:
#     # async def add_entries(self, table_name: str, entries, embeddings)
#     # async def semantic_search(self, table_name: str, query_embedding, top_k)
#     # async def keyword_search(self, table_name: str, keywords, top_k)
#     # async def structured_search(self, table_name: str, persons, location, ...)
#     # async def get_all_entries(self, table_name: str)
#     # async def count_entries(self, table_name: str)
#     # async def clear_table(self, table_name: str)
#     # async def delete_table(self, table_name: str)
#     # def get_stats(self, table_name: str)
# =============================================================================
