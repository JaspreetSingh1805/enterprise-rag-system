import re

from app.adapters.llm.base import BaseLLMAdapter, LLMResponse
from app.core.logging import get_logger

logger = get_logger("mock_llm_adapter")


class MockLLMAdapter(BaseLLMAdapter):
    """
    Deterministic Mock LLM adapter for offline CI/CD and automated testing.
    Strictly follows grounding instructions and generates citations when context matches.
    """

    def __init__(self, model_name: str = "mock-gpt-4o"):
        self.model = model_name

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        logger.debug(f"MockLLM generating response for model {self.model}")

        # Check if this is a query rewrite request
        if (
            "query reformulation" in (system_prompt or "").lower()
            or "rewrite" in prompt.lower()
            and "search query" in prompt.lower()
        ):
            # Extract question to rewrite
            query_match = re.search(r"Question:\s*(.+)", prompt, re.IGNORECASE)
            rewritten = query_match.group(1).strip() if query_match else prompt.strip()
            # Clean up abbreviations / filler
            rewritten = (
                rewritten.replace("Can you tell me about", "").replace("what is", "").strip()
            )
            return LLMResponse(
                text=rewritten or "company policy overview",
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(rewritten.split()) + 2,
                total_tokens=len(prompt.split()) + len(rewritten.split()) + 2,
                model_name=self.model,
            )

        # RAG Generation Request
        # Check if context was provided in prompt
        context_blocks = re.findall(
            r"\[Document:\s*([^\]]+)\]\s*(.*?)(?=\[Document:|\Z)", prompt, re.DOTALL
        )

        # Determine question
        question_match = re.search(r"Question:\s*(.+?)(?:\n\n|\Z)", prompt, re.DOTALL)
        question = question_match.group(1).strip() if question_match else ""

        # Check for unanswerable trigger words or absent context
        unanswerable_keywords = [
            "mars rover",
            "competitor",
            "alien",
            "lottery",
            "weather in tokyo",
            "stock price of apple",
            "unrelated",
            "secret recipe",
            "nonexistent",
        ]
        is_unanswerable = not context_blocks or any(
            k in question.lower() for k in unanswerable_keywords
        )

        if is_unanswerable:
            response_text = "I could not find information about this in the knowledge base."
        else:
            # Construct a grounded answer citing the matched documents
            cited_sources = []
            answer_parts = []

            for doc_meta, content in context_blocks[:3]:
                # Extract clean snippet and citations
                meta_parts = dict(re.findall(r'(\w+)=["\']?([^"\',]+)["\']?', doc_meta))
                filename = meta_parts.get("file", meta_parts.get("filename", "document"))
                page = meta_parts.get("page", "1")
                section = meta_parts.get("section", "General")

                cite_tag = f"[Doc: {filename}, Page: {page}, Section: {section}]"
                cited_sources.append(cite_tag)

                # Take key sentences from content
                sentences = [s.strip() for s in content.split(".") if len(s.strip()) > 15]
                if sentences:
                    answer_parts.append(f"{sentences[0]}. {cite_tag}")

            if answer_parts:
                response_text = "Based on the provided documentation: " + " ".join(answer_parts)
            else:
                response_text = "I could not find information about this in the knowledge base."

        prompt_tokens = max(10, len(prompt.split()))
        completion_tokens = max(5, len(response_text.split()))

        return LLMResponse(
            text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model_name=self.model,
        )
