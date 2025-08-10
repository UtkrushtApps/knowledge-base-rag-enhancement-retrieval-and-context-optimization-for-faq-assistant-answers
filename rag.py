import time
import tiktoken
from typing import List, Tuple, Dict, Any

class RAGPipeline:
    def __init__(self, chroma_db, embed_query_fn, llm_complete_fn, model_name="gpt-3.5-turbo", max_context_tokens=3000, top_k=5):
        """
        :param chroma_db: Vector DB object with .query() method
        :param embed_query_fn: function that returns embedding for a query string
        :param llm_complete_fn: function that invokes LLM with prompt and returns response
        :param model_name: LLM model name string (for tiktoken)
        :param max_context_tokens: Reserved number of tokens for context in prompt
        :param top_k: Number of top FAQ hits to retrieve
        """
        self.chroma_db = chroma_db
        self.embed_query_fn = embed_query_fn
        self.llm_complete_fn = llm_complete_fn
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.top_k = top_k
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def encode_query(self, query: str) -> List[float]:
        """Encode query string to embedding vector (dense retrieval)"""
        return self.embed_query_fn(query)

    def retrieve_faqs(self, query_embedding: List[float], top_k: int) -> Tuple[List[Dict], float]:
        """Retrieve top-k FAQ entries based on embedding similarity. Returns entries and latency (sec)."""
        t0 = time.time()
        results = self.chroma_db.query(query_embeddings=[query_embedding], n_results=top_k)
        retrieval_time = time.time() - t0
        # Each FAQ: {"question": str, "answer": str, ...}
        # Assume returned as list of dicts
        faqs = []
        if 'documents' in results and results['documents']:
            docs = results['documents'][0]
            metadatas = results.get('metadatas', [None])[0] if 'metadatas' in results else [None] * len(docs)
            # Each doc is text, each metadata has question/answer
            for i, doc in enumerate(docs):
                meta = metadatas[i] if metadatas and i < len(metadatas) and metadatas[i] else {}
                faq_item = meta.copy() if meta else {}
                if 'answer' not in faq_item:
                    faq_item['answer'] = doc
                faqs.append(faq_item)
        return faqs, retrieval_time

    def context_token_budgeting(self, faqs: List[Dict], user_query: str, prompt_prefix: str) -> Tuple[List[Tuple[int, Dict]], int]:
        """
        Given faqs and other prompt parts, select as many as can fit in token budget for context window.
        Returns context [(citation_number, faq_dict)], and token count used.
        """
        total_tokens = 0
        # Reserve tokens for prompt, instructions, and user query
        total_tokens += self._count_tokens(prompt_prefix + "\n" + user_query) + 50  # slack for LLM response
        context = []
        citation_num = 1
        for faq in faqs:
            block = f"Q: {faq.get('question', 'N/A')}\nA: {faq.get('answer', 'N/A')}".strip()
            block_tokens = self._count_tokens(block)
            if total_tokens + block_tokens > self.max_context_tokens:
                break
            context.append((citation_num, faq))
            total_tokens += block_tokens
            citation_num += 1
        return context, total_tokens

    def build_context_prompt(self, context_blocks: List[Tuple[int, Dict]], user_query: str) -> str:
        """
        Craft the prompt with context blocks, user question, and citation instructions.
        Format citations as e.g. [1] for each source.
        """
        prompt = "You are an FAQ assistant. Use the context to answer, citing sources as [1], [2], etc. Only use provided context, and do not invent information.\n\n"
        if context_blocks:
            prompt += "Context:\n"
            for citation_num, faq in context_blocks:
                prompt += f"[{citation_num}] Q: {faq.get('question', 'N/A')}\n[{citation_num}] A: {faq.get('answer', 'N/A')}\n"
            prompt += "\n"
        else:
            prompt += "No relevant FAQ context found.\n\n"
        prompt += f"User Question: {user_query.strip()}\n"
        prompt += ("\nAnswer (cite sources with [n]): ")
        return prompt

    def _count_tokens(self, text: str) -> int:
        """Estimate token count for a string using tiktoken."""
        return len(self.tokenizer.encode(text))

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Full RAG retrieval -> context -> prompt -> LLM run. Returns dict with citations, answer, and logging info.
        """
        log_info = {}

        # Step 1: Encode query
        query_embedding = self.encode_query(user_query)

        # Step 2: Retrieve top-k FAQ entries
        faqs, retrieval_time = self.retrieve_faqs(query_embedding, self.top_k)
        log_info['retrieval_latency_s'] = retrieval_time
        log_info['faq_hits'] = len(faqs)

        # Step 3: Assemble context blocks up to token budget
        prompt_prefix = "You are an FAQ assistant. Use the context to answer, citing sources as [1], [2], etc. Only use provided context, and do not invent information.\n\n"
        context_blocks, context_tokens = self.context_token_budgeting(faqs, user_query, prompt_prefix)
        log_info['context_faq_count'] = len(context_blocks)
        log_info['context_tokens_used'] = context_tokens

        # Step 4: Assemble full prompt
        full_prompt = self.build_context_prompt(context_blocks, user_query)
        log_info['prompt_token_count'] = self._count_tokens(full_prompt)

        # Step 5: Call LLM
        answer = self.llm_complete_fn(full_prompt)

        # Step 6: Return answer, context sources, and logs
        return {
            'answer': answer,
            'context_sources': [
                {
                    'citation': f"[{citation_num}]",
                    'question': faq.get('question', 'N/A'),
                    'answer': faq.get('answer', 'N/A')
                } for citation_num, faq in context_blocks
            ],
            'log': log_info
        }
