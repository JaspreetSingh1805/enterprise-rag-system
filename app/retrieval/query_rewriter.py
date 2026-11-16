from app.adapters.llm.base import BaseLLMAdapter
from app.core.logging import get_logger

logger = get_logger("query_rewriter")


class QueryRewriter:
    """
    Reformulates and expands queries using LLM to maximize retrieval recall.
    """

    def __init__(self, llm_adapter: BaseLLMAdapter):
        self.llm = llm_adapter

    async def rewrite(self, query: str, rewrite_prompt: str) -> str:
        try:
            prompt = f"Question: {query}\n\nRewritten search query:"
            res = await self.llm.generate(
                prompt=prompt,
                system_prompt=rewrite_prompt,
                temperature=0.0,
                max_tokens=60,
            )
            rewritten = res.text.strip().strip('"').strip("'")
            if rewritten and len(rewritten) > 2:
                logger.info(f"Rewrote query '{query}' -> '{rewritten}'")
                return rewritten
            return query
        except Exception as e:
            logger.warning(f"Query rewrite failed: {str(e)}; falling back to original query")
            return query
