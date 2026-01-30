"""
Memory Builder - Stage 1: Semantic Structured Compression

Converts raw dialogues into atomic memory entries through:
- Coreference resolution (remove pronouns)
- Temporal anchoring (absolute timestamps)
- Information extraction (keywords, persons, entities, etc.)

Single-tenant mode: Direct processing without table_name parameter.
"""

from typing import List, Optional
from datetime import datetime
import uuid

from ..auth.models import MemoryEntry, Dialogue
from ..integrations.openrouter import OpenRouterClient
from ..database.vector_store import SingleTenantVectorStore


class MemoryBuilder:
    """
    Builds atomic memory entries from dialogues.
    Single-tenant mode - uses shared vector store directly.
    """

    def __init__(
        self,
        openrouter_client: OpenRouterClient,
        vector_store: SingleTenantVectorStore,
        window_size: int = 40,  # Max dialogues per LLM call
        overlap_size: int = 2,
        temperature: float = 0.1,
        project_tags: Optional[dict] = None,
    ):
        self.client = openrouter_client
        self.vector_store = vector_store
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.temperature = temperature
        # Optional project tags for namespacing memories
        # Format: {"TAG": "Description", ...}
        self.project_tags = project_tags or {}

        # Context from previous processing for deduplication
        self._previous_context: str = ""
        self._total_processed = 0

    async def add_dialogue(
        self,
        speaker: str,
        content: str,
        timestamp: Optional[str] = None,
        auto_process: bool = True,
    ) -> dict:
        """
        Add a single dialogue and immediately process it into memory.

        Args:
            speaker: Speaker name
            content: Dialogue content
            timestamp: Optional ISO 8601 timestamp
            auto_process: Ignored (always processes immediately)

        Returns:
            Status dict with processing results
        """
        dialogue = Dialogue(
            dialogue_id=self._total_processed + 1,
            speaker=speaker,
            content=content,
            timestamp=timestamp or datetime.utcnow().isoformat(),
        )

        # Process immediately
        entries = await self._generate_memory_entries([dialogue])

        if not entries:
            return {
                "added": True,
                "processed": True,
                "entries_created": 0,
                "message": "No extractable information found",
            }

        # Generate embeddings
        texts = [entry.lossless_restatement for entry in entries]
        embeddings = await self.client.create_embedding(texts)

        # Store entries (no table_name parameter in single-tenant mode)
        count = await self.vector_store.add_entries(
            entries,
            embeddings,
        )

        # Update context for next call
        self._previous_context = self._build_context_summary(entries)
        self._total_processed += 1

        return {
            "added": True,
            "processed": True,
            "entries_created": count,
        }

    async def add_dialogues(
        self,
        dialogues: List[dict],
        auto_process: bool = True,
    ) -> dict:
        """
        Add multiple dialogues and process them immediately.

        Args:
            dialogues: List of dialogue dicts with speaker, content, timestamp
            auto_process: Ignored (always processes immediately)

        Returns:
            Status dict with processing info
        """
        if not dialogues:
            return {
                "added": 0,
                "entries_created": 0,
                "message": "No dialogues provided",
            }

        # Convert to Dialogue objects
        dialogue_objects = []
        for i, dlg in enumerate(dialogues):
            dialogue_objects.append(Dialogue(
                dialogue_id=self._total_processed + i + 1,
                speaker=dlg.get("speaker", ""),
                content=dlg.get("content", ""),
                timestamp=dlg.get("timestamp") or datetime.utcnow().isoformat(),
            ))

        total_entries = 0

        # Process in windows if too many dialogues
        for i in range(0, len(dialogue_objects), self.window_size):
            window = dialogue_objects[i:i + self.window_size]

            entries = await self._generate_memory_entries(window)

            if entries:
                # Generate embeddings
                texts = [entry.lossless_restatement for entry in entries]
                embeddings = await self.client.create_embedding(texts)

                # Store entries (no table_name parameter in single-tenant mode)
                count = await self.vector_store.add_entries(
                    entries,
                    embeddings,
                )
                total_entries += count

                # Update context for next window
                self._previous_context = self._build_context_summary(entries)

        self._total_processed += len(dialogues)

        return {
            "added": len(dialogues),
            "entries_created": total_entries,
        }

    async def _generate_memory_entries(
        self,
        dialogues: List[Dialogue],
    ) -> List[MemoryEntry]:
        """
        Generate atomic memory entries from dialogues using LLM

        Args:
            dialogues: List of Dialogue objects

        Returns:
            List of MemoryEntry objects
        """
        # Build dialogue text
        dialogue_text = self._format_dialogues(dialogues)

        # Build prompt
        prompt = self._build_extraction_prompt(dialogue_text)

        # Build system prompt (conditionally include project tags instruction)
        system_content = (
            "You are a professional information extraction assistant. "
            "Extract atomic, self-contained facts from dialogues. "
            "Each fact must be independently understandable without context. "
            "Always resolve pronouns to actual names and convert relative times to absolute timestamps."
        )
        if self.project_tags:
            tag_list = ", ".join(f"[{tag}]" for tag in self.project_tags.keys())
            system_content += f" IMPORTANT: Every fact MUST start with a project tag: {tag_list}."

        # Call LLM
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        # Retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )

                # Parse response
                data = self.client.extract_json(response)
                if data is None:
                    continue

                # Handle both list and dict with "entries" key
                entries_data = data if isinstance(data, list) else data.get("entries", [])

                entries = []
                for item in entries_data:
                    entry = MemoryEntry(
                        entry_id=str(uuid.uuid4()),
                        lossless_restatement=item.get("lossless_restatement", ""),
                        keywords=item.get("keywords", []),
                        timestamp=item.get("timestamp"),
                        location=item.get("location"),
                        persons=item.get("persons", []),
                        entities=item.get("entities", []),
                        topic=item.get("topic"),
                    )
                    if entry.lossless_restatement:
                        entries.append(entry)

                return entries

            except Exception as e:
                print(f"Memory extraction attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return []

        return []

    def _format_dialogues(self, dialogues: List[Dialogue]) -> str:
        """Format dialogues into readable text"""
        lines = []
        for dlg in dialogues:
            ts = f" [{dlg.timestamp}]" if dlg.timestamp else ""
            lines.append(f"{dlg.speaker}{ts}: {dlg.content}")
        return "\n".join(lines)

    def _build_extraction_prompt(self, dialogue_text: str) -> str:
        """Build the extraction prompt"""
        context_section = ""
        if self._previous_context:
            context_section = f"""
## Context from Previous Processing (for reference, avoid duplication):
{self._previous_context}

---
"""

        # Build project tags section if configured
        project_section = ""
        example_prefix = ""
        if self.project_tags:
            tag_lines = "\n".join(
                f"   - `[{tag}]` — {desc}"
                for tag, desc in self.project_tags.items()
            )
            project_section = f"""
2. **Project Namespacing (IMPORTANT!)**: Every memory MUST start with a project tag:
{tag_lines}
   - Choose the MOST relevant tag based on the content topic.
"""
            first_tag = list(self.project_tags.keys())[0]
            example_prefix = f"[{first_tag}] "

        # Adjust numbering based on whether project tags are included
        if self.project_tags:
            fact_num = "3"
            coref_num = "4"
            temporal_num = "5"
            extract_num = "6"
        else:
            fact_num = "2"
            coref_num = "3"
            temporal_num = "4"
            extract_num = "5"

        return f"""{context_section}## Dialogues to Process:
{dialogue_text}

---

## Extraction Requirements:

1. **Complete Coverage**: Capture ALL valuable information from the dialogues.
{project_section}
{fact_num}. **Self-Contained Facts**: Each entry must be independently understandable.
   - BAD: "He will meet Bob tomorrow" (Who is "he"? When is "tomorrow"?)
   - GOOD: "{example_prefix}Alice will meet Bob at Starbucks on 2025-01-15 at 14:00"

{coref_num}. **Coreference Resolution**: Replace ALL pronouns with actual names.
   - Replace: he, she, it, they, him, her, them, his, hers, their
   - With: The actual person's name or entity

{temporal_num}. **Temporal Anchoring**: Convert ALL relative times to absolute ISO 8601 format.
   - "tomorrow" -> Calculate actual date
   - "next week" -> Calculate actual date range
   - "in 2 hours" -> Calculate actual time

{extract_num}. **Information Extraction**: For each entry, extract:
   - `lossless_restatement`: Complete, unambiguous fact{" (starting with project tag)" if self.project_tags else ""}
   - `keywords`: Core terms for search (3-7 keywords)
   - `timestamp`: ISO 8601 format if mentioned
   - `location`: Specific location name
   - `persons`: All person names involved
   - `entities`: Companies, products, organizations, etc.
   - `topic`: Brief topic phrase (2-5 words)

## Output Format (JSON only, no other text):
{{
  "entries": [
    {{
      "lossless_restatement": "{example_prefix}Complete self-contained fact...",
      "keywords": ["keyword1", "keyword2", ...],
      "timestamp": "2025-01-15T14:00:00" or null,
      "location": "Starbucks, Downtown" or null,
      "persons": ["Alice", "Bob"],
      "entities": ["Company XYZ"] or [],
      "topic": "Meeting arrangement"
    }}
  ]
}}

Return ONLY valid JSON. No explanations or other text."""

    def _build_context_summary(self, entries: List[MemoryEntry]) -> str:
        """Build context summary from entries for next processing"""
        if not entries:
            return ""

        summaries = []
        for entry in entries[-5:]:  # Keep last 5 entries as context
            summaries.append(f"- {entry.lossless_restatement}")

        return "\n".join(summaries)

    def get_stats(self) -> dict:
        """Get processing statistics"""
        return {
            "total_dialogues_processed": self._total_processed,
        }


# =============================================================================
# MULTI-TENANT MEMORY BUILDER (PRESERVED FOR REFERENCE)
# See multi-tenant-backup branch for original implementation
# =============================================================================
#
# Key differences in multi-tenant mode:
# - Constructor takes table_name parameter
# - All vector_store calls include table_name:
#   await self.vector_store.add_entries(self.table_name, entries, embeddings)
# =============================================================================
