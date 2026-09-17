"""
Prompt templates for the agents.
"""

KEYWORD_EXTRACTION_PROMPT = """Extract key search terms from the following natural language query.
Focus on:
- Entity names (people, places, organizations)
- Class concepts (types of things)
- Property concepts (relationships, attributes)
- Specific values or constraints

Query: {query}

Return a JSON list of keywords."""


SPARQL_GENERATION_PROMPT = """You are a SPARQL expert. Generate a valid SPARQL query based on the user's question and the provided schema context.

## Schema Context
{schema_context}

## User Question
{user_query}

## Instructions
1. Use ONLY classes and properties from the schema context
2. Declare all necessary prefixes
3. Use meaningful variable names
4. Apply appropriate filters for constraints
5. Consider the query type (SELECT, ASK, CONSTRUCT)

## Output Format
Return a JSON object with:
- "query": The complete SPARQL query
- "reasoning": Brief explanation of your approach
- "confidence": Your confidence score (0.0-1.0)

Generate the query now."""


MODEL_SELECTION_PROMPT = """Analyze the retrieved schema models and select the most relevant ones for answering the user's query.

## User Query
{user_query}

## Retrieved Models
{retrieved_models}

## Instructions
1. Identify which models contain relevant classes and properties
2. Consider relationships between models
3. Rank models by relevance
4. Explain your selection

Return a JSON object with:
- "selected_models": List of selected model identifiers
- "reasoning": Explanation of your selection
- "relevance_scores": Dict mapping model IDs to relevance scores (0-1)"""


VALIDATION_PROMPT = """Evaluate the SPARQL query results and determine the best query.

## Original Question
{user_query}

## Query Variants and Results
{query_results}

## Instructions
1. Check syntax validity for each query
2. Compare result sizes and contents
3. Assess agreement between variants
4. Consider which best answers the original question

Return a JSON object with:
- "best_query_index": Index of the best query (0-indexed)
- "confidence": Overall confidence score (0-1)
- "reasoning": Explanation of your selection
- "issues": List of any issues found"""


BACK_TRANSLATION_PROMPT = """Translate this SPARQL query back into a natural language question.
Be concise and capture the core intent of the query.

SPARQL Query:
{sparql_query}

Natural language question:"""
