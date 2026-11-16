import re

from app.models.domain import Chunk, Citation

NOT_FOUND_MESSAGE = "I could not find information about this in the knowledge base."


class GroundingEngine:
    """
    Ensures answers are strictly grounded in retrieved context and verified with citations.
    Detects unanswerable queries based on similarity threshold and context absence.
    """

    def __init__(self, similarity_threshold: float = 0.35):
        self.similarity_threshold = similarity_threshold

    def format_context(self, chunks: list[tuple[Chunk, float]]) -> str:
        """
        Format retrieved chunks into structured context blocks with source tags.
        """
        blocks = []
        for chunk, score in chunks:
            filename = chunk.metadata.get("filename", "unknown_file")
            page = chunk.page_number if chunk.page_number is not None else 1
            section = chunk.section_title or "General"
            meta_str = f'file="{filename}", page={page}, section="{section}"'
            blocks.append(f"[Document: {meta_str}]\n{chunk.text.strip()}")
        return "\n\n".join(blocks)

    def extract_citations(
        self,
        answer: str,
        retrieved_chunks: list[tuple[Chunk, float]],
    ) -> list[Citation]:
        """
        Extract citations from the generated answer and map them to retrieved chunks.
        """
        citations: list[Citation] = []
        seen_keys = set()

        # Regex for [Doc: <filename>, Page: <page>, Section: <section>]
        cite_patterns = re.findall(
            r"\[Doc:\s*([^,\]]+)(?:,\s*Page:\s*([^,\]]+))?(?:,\s*Section:\s*([^,\]]+))?\]",
            answer,
        )

        for filename, page_str, sec_str in cite_patterns:
            filename = filename.strip()
            page = int(page_str.strip()) if page_str and page_str.strip().isdigit() else None
            section = sec_str.strip() if sec_str else None

            # Find matching chunk
            matched_chunk = None
            matched_score = 0.0
            for chunk, score in retrieved_chunks:
                c_fn = chunk.metadata.get("filename", "")
                if c_fn == filename or filename.lower() in c_fn.lower():
                    matched_chunk = chunk
                    matched_score = score
                    break

            if matched_chunk:
                key = (matched_chunk.document_id, page, section)
                if key not in seen_keys:
                    seen_keys.add(key)
                    snippet = matched_chunk.text[:200] + (
                        "..." if len(matched_chunk.text) > 200 else ""
                    )
                    citations.append(
                        Citation(
                            document_id=matched_chunk.document_id,
                            document_name=filename,
                            page_number=page or matched_chunk.page_number,
                            section_title=section or matched_chunk.section_title,
                            snippet=snippet,
                            score=matched_score,
                        )
                    )

        # If LLM didn't emit formal brackets but retrieved high-confidence chunks,
        # create fallback citations from top chunks
        if not citations and retrieved_chunks and NOT_FOUND_MESSAGE not in answer:
            for chunk, score in retrieved_chunks[:2]:
                if score >= self.similarity_threshold:
                    fn = chunk.metadata.get("filename", "document")
                    citations.append(
                        Citation(
                            document_id=chunk.document_id,
                            document_name=fn,
                            page_number=chunk.page_number,
                            section_title=chunk.section_title,
                            snippet=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                            score=score,
                        )
                    )

        return citations

    def check_is_found(self, answer: str, retrieved_chunks: list[tuple[Chunk, float]]) -> bool:
        if not retrieved_chunks:
            return False

        max_score = max(s for _, s in retrieved_chunks)
        if max_score < self.similarity_threshold:
            return False

        lower_ans = answer.lower()
        negative_phrases = [
            "could not find",
            "not found in the knowledge base",
            "no information",
            "cannot find",
            "not mentioned in the provided",
            "does not contain",
        ]
        if any(p in lower_ans for p in negative_phrases):
            return False

        return True
