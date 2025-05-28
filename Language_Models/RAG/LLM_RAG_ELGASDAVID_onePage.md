# Project Title: Intelligent Knowledge Base with Retrieval-Augmented Generation

## Problem and Solution

Large language models (LLMs) like GPT-3.5 are powerful, but they are limited by the static training data they were exposed to. As a result, they often underperform in specialized, high-knowledge domains such as aerospace, biotech, or industrial diagnostics—domains where context evolves, and the latest knowledge isn't part of a frozen training set. To resolve these issues, supplemental information can be added to aid in knowledge transfer. This process is known as Retrieval-Augmented Generation (RAG).

## Dataset and Technology

The primary dataset for this effort was built by leveraging Python with Beautiful Soup to gather information from a publicly available online forum. The data was processed through SentenceTransformer embeddings, FAISS vector search, and OpenAI GPT-3.5, ultimately served in a Streamlit web interface allowing users to explore the results.

## Uses and Benefits

This project demonstrates how enterprises can enhance commercial LLMs with domain-specific knowledge. By implementing RAG, organizations can:

- Extend capabilities into specialized domains without retraining
- Improve response accuracy for technical queries
- Keep knowledge current by updating the corpus independently
- Preserve institutional knowledge
- Reduce hallucinations by grounding responses in verified sources

## Drawbacks

RAG systems face several limitations:

- Response quality depends on corpus accuracy, potentially propagating misinformation
- Semantic matching limitations when terminology differs between queries and sources
- Higher computational costs for both vector database management and commercial LLM APIs
- Increased architectural complexity compared to standard LLM implementations
- Inheritance of biases present in the original community discussions

## Challenges

The initial challenge with this effort was Beautiful Soup programming specific to the forum platform (XenForo 2.3). Additionally, attempts to train a custom RAG model produced unusable results. Migrating to a semantic model combining vector database with a commercial generator yielded better outcomes. Finally, creating an inexpensive, ephemeral model proved difficult as commercial ML hosting solutions struggled with the implementation.

## Results and Impact

The results show that the supplemental RAG corpus contained a richness of data not found in the LLM. This satisfies the original use case for this effort.

### Response Comparison

#### RAG Response
To replace the steering wheel, you need to remove the horn button cover first. You can do this by carefully getting your fingernail underneath and popping it off from the front. Once the horn button cover is removed, you can proceed with replacing the steering wheel. If you need a new steering wheel, you can purchase one from Walloth Nesch, which still lists new ones for sale on their website. The Alpina steering wheels are 380mm in size, slightly smaller than the stock 400mm wheel. Make sure to check if you need the hub and horn button as well when purchasing a new steering wheel.

#### Pure OpenAI Response
To replace a steering wheel, you will need to first disconnect the battery to prevent any electrical issues. Then, remove the airbag module by unscrewing it from the steering wheel. Next, use a steering wheel puller tool to remove the steering wheel from the steering column. Once the old steering wheel is removed, you can then install the new steering wheel by reversing the steps. Be sure to torque tighten all screws and bolts to ensure the steering wheel is properly installed.

*Note that the OpenAI response references an airbag which is not a technology available in these older automobiles.

## Demo Resources

- **YouTube**: https://www.youtube.com/watch?v=gojyE5mE0RA
