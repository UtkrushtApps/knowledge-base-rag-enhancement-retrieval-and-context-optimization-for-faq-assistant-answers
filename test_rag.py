import pytest
from unittest.mock import MagicMock
from rag import RAGPipeline

def mock_embed_query_fn(query):
    # Just returns a fake embedding (vector of fixed numbers)
    return [1.0] * 1536

def mock_llm_complete_fn(prompt):
    # Return input prompt (for test visibility)
    return f"MOCK_ANSWER (PROMPT:\n{prompt})"

def mock_chroma_query(**kwargs):
    # For retrieval, return 3 FAQ docs and metadatas
    docs = [
        "A1 Doc text",
        "A2 Doc text",
        "A3 Doc text"
    ]
    metadatas = [
        {'question': 'How to reset password?', 'answer': 'Go to Account > Reset.'},
        {'question': 'How to contact support?', 'answer': 'Email support@example.com.'},
        {'question': 'What is the refund policy?', 'answer': 'Refund within 30 days.'}
    ]
    return {'documents': [docs], 'metadatas': [metadatas]}

class MockChromaDB:
    def query(self, *args, **kwargs):
        return mock_chroma_query(**kwargs)

def test_rag_pipeline_happy_path():
    pipeline = RAGPipeline(
        chroma_db=MockChromaDB(),
        embed_query_fn=mock_embed_query_fn,
        llm_complete_fn=mock_llm_complete_fn,
        model_name="gpt-3.5-turbo",
        max_context_tokens=120,
        top_k=2,
    )
    user_query = "How do I reset my password?"
    result = pipeline.run(user_query)
    # Check answer string was built
    assert result['answer'].startswith("MOCK_ANSWER")
    # Citations match number of context_faq_count
    assert len(result['context_sources']) == result['log']['context_faq_count']
    # Logging fields present
    assert 'retrieval_latency_s' in result['log']
    assert 'faq_hits' in result['log']
    assert 'context_tokens_used' in result['log']
    print("Test result:", result)

def test_rag_pipeline_token_budget():
    pipeline = RAGPipeline(
        chroma_db=MockChromaDB(),
        embed_query_fn=mock_embed_query_fn,
        llm_complete_fn=mock_llm_complete_fn,
        model_name="gpt-3.5-turbo",
        max_context_tokens=20,  # Tiny budget
        top_k=3,
    )
    user_query = "Query with no room"
    result = pipeline.run(user_query)
    # Depending on prompt length, may have 0 or 1 context sources
    assert result['log']['context_faq_count'] <= 1
    print("Token budget result:", result)
