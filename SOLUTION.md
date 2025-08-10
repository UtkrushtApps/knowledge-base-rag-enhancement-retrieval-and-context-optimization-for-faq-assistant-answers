# Solution Steps

1. Import required libraries and tiktoken for token counting, ensuring OpenAI-style token estimation for GPT models.

2. Implement RAGPipeline class, initializing with chroma vector DB object, embedding function, LLM completion function, model configuration, context and retrieval parameters, and a tiktoken tokenizer instance.

3. Create the query encoder (encode_query) that generates dense vector embeddings using the provided encode function for the user query.

4. Implement the retrieval method (retrieve_faqs) to query Chroma DB for top-K semantically similar FAQ entries and record retrieval latency.

5. Write context_token_budgeting to assemble as many FAQ entries as can fit within the context token limit, counting tokens with tiktoken, and returning context entries with citation markers.

6. Craft the context prompt (build_context_prompt) that includes context blocks with [n] citation markers, ensures the prompt instructs the model to cite answers, and appends user question and answer instructions.

7. Add a private token counting helper method (_count_tokens) based on the tiktoken tokenizer.

8. Implement the run() method that orchestrates the pipeline: encodes query, retrieves FAQ, assembles context, builds prompt, invokes the LLM, and logs retrieval/generation stats (latency, token usage, number of hits).

9. For testing, mock the Chroma DB query, embedding, and LLM calls, then verify that retrieval, context assembly, and prompt building work as expected under different token budgets and retrieval settings.

10. Validate that all logging fields are populated and markers/citations appear as required in the response.

