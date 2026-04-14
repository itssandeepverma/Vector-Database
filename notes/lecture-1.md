# Lecture 1 Notes: Semantic Search and Vector Databases

## Lesson Goal

The goal of Lesson 1 is to build a simple semantic search system.

Instead of matching exact words, semantic search tries to understand the meaning of a sentence. In this notebook, questions from the Quora dataset are converted into embeddings, stored in Pinecone, and then searched using vector similarity.

## What Is Semantic Search?

Traditional keyword search looks for exact terms. If you search for:

`which city has the highest population in the world?`

a keyword system may miss related sentences that use different wording such as:

`which city is the most populated in the world?`

Semantic search solves this by converting text into vectors. Sentences with similar meaning get vectors that are close to each other in vector space.

That means the system can return relevant results even when the words are different.

## Core Idea of the Lesson

The lesson shows this pipeline:

1. Load text data.
2. Convert each sentence into an embedding.
3. Store embeddings in a vector database.
4. Convert a user query into an embedding.
5. Search for the nearest vectors.
6. Return the original matching text.

This is the basic retrieval pattern used in many modern AI systems, including semantic search, recommendation systems, and RAG pipelines.

## What the Notebook Uses

This project is centered on [`Lesson_1_Semantic_Search.ipynb`](/Users/sandeep/Desktop/Projects/Vector-Databases/Lesson_1_Semantic_Search.ipynb) and uses:

- `datasets` to load the Quora dataset.
- `sentence-transformers` to convert text into embeddings.
- `torch` to check whether CUDA is available.
- `pinecone` to store and search vectors.
- [`DLAIUtils.py`](/Users/sandeep/Desktop/Projects/Vector-Databases/DLAIUtils.py) to load environment variables and build an index name.

## Step-by-Step Flow in This Repo

### 1. Import Packages

The notebook imports the required libraries for data loading, embedding generation, and vector storage.

Important imports:

- `load_dataset` from Hugging Face datasets
- `SentenceTransformer`
- `Pinecone` and `ServerlessSpec`
- `torch`

### 2. Load the Dataset

The notebook loads a slice of the Quora training dataset:

```python
dataset = load_dataset('quora', split='train[240000:290000]')
```

This dataset contains question pairs. The notebook extracts question text from the records and flattens them into a single list.

Then it removes duplicates:

```python
question = list(set(questions))
```

This gives a cleaner set of candidate questions to store in the vector database.

### 3. Load the Embedding Model

The lesson uses:

`all-MiniLM-L6-v2`

This is a sentence-transformers model that maps each sentence into a `384`-dimensional dense vector.

Why this matters:

- Every sentence becomes a fixed-length numeric representation.
- Similar meanings should produce similar vectors.
- The vector dimension must match the database index dimension.

### 4. Convert a Query into a Vector

The notebook tests the model with:

```python
query = 'which city is the most populated in the world?'
xq = model.encode(query)
```

At this point:

- `query` is text
- `xq` is a dense numeric vector
- the vector represents the meaning of the sentence

This vector is what will later be compared against all stored vectors.

### 5. Connect to Pinecone

The notebook loads the Pinecone API key using the helper class in [`DLAIUtils.py`](/Users/sandeep/Desktop/Projects/Vector-Databases/DLAIUtils.py).

```python
utils = Utils()
PINECONE_API_KEY = utils.get_pinecone_api_key()
pinecone = Pinecone(api_key=PINECONE_API_KEY)
```

It also creates an index name:

```python
INDEX_NAME = utils.create_dlai_index_name('dl-ai')
```

That helper uses environment variables from `.env`.

### 6. Create the Vector Index

The notebook creates a Pinecone index with:

```python
pinecone.create_index(
    name=INDEX_NAME,
    dimension=model.get_sentence_embedding_dimension(),
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)
```

Important settings:

- `dimension`: must match the embedding model output, which is `384`
- `metric='cosine'`: compares vectors by cosine similarity
- `ServerlessSpec`: tells Pinecone where to host the index

If the dimension does not match the model output, inserts and queries will fail.

### 7. Prepare Records for Storage

The notebook stores only the first `10000` questions:

```python
vector_limit = 10000
questions = question[:vector_limit]
```

Then it processes the data in batches:

```python
batch_size = 200
```

For each batch it creates:

- `ids`: unique string IDs
- `metadatas`: dictionaries containing the original text
- `xc`: embeddings for the batch

Example structure:

```python
ids = ['0', '1', '2']
metadatas = [{'text': 'question 1'}, {'text': 'question 2'}]
xc = model.encode(...)
```

### 8. Upsert into Pinecone

The notebook writes vectors using:

```python
records = zip(ids, xc, metadatas)
index.upsert(vectors=records)
```

`upsert` means:

- insert the vector if it is new
- update it if the same ID already exists

Each stored item contains:

- an ID
- the vector values
- metadata, here the original question text

This is important because the vector database returns matching items, and metadata lets us recover readable content.

### 9. Query the Vector Database

The helper function in the notebook does this:

```python
embedding = model.encode(query).tolist()
results = index.query(
    top_k=10,
    vector=embedding,
    include_metadata=True,
    include_values=False
)
```

Meaning of each part:

- `top_k=10`: return the 10 nearest matches
- `vector=embedding`: search using the query embedding
- `include_metadata=True`: return the original text
- `include_values=False`: do not return raw vector values

Finally, the notebook prints the score and matched text.

## What Is a Vector Database?

A vector database is a system designed to store embeddings and quickly find the nearest ones.

Instead of searching rows by exact text match, it searches numeric vectors by similarity.

### Why Not Use a Normal Database?

A normal relational database is good for:

- exact match lookups
- filters
- joins
- structured records

But it is not built for efficient nearest-neighbor search across thousands or millions of high-dimensional vectors.

A vector database is optimized for:

- storing dense vectors
- similarity search
- metadata filtering
- large-scale retrieval

## How a Vector Database Works

### 1. Text Becomes Numbers

The embedding model converts text into a high-dimensional vector such as:

`[0.12, -0.04, 0.88, ...]`

This vector captures semantic information.

### 2. Similar Texts Cluster Together

Questions with similar meaning should be near each other in vector space.

For example:

- `how do i bake a cake?`
- `how do i make chocolate cake?`

These may use different words, but their embeddings are still close.

### 3. Query Uses the Same Embedding Model

The query must be encoded with the same model used for the stored data.

This is critical. If you embed the database with one model and the query with another, the vector spaces may not align correctly.

### 4. Similarity Metric Finds Nearest Neighbors

The system compares the query vector with stored vectors.

Common similarity metrics:

- cosine similarity
- dot product
- Euclidean distance

This lesson uses cosine similarity.

## Why Cosine Similarity Is Used Here

Cosine similarity measures the angle between vectors, not just their raw magnitude.

That makes it useful for text embeddings because we care more about directional similarity in meaning than absolute size.

In simple terms:

- higher cosine score means more similar meaning
- lower cosine score means less similar meaning

## What Pinecone Is Doing in This Lesson

Pinecone is the hosted vector database used in the notebook.

Its role is:

1. Create an index that knows the vector size and similarity metric.
2. Store embeddings with IDs and metadata.
3. Run fast nearest-neighbor search for a query embedding.
4. Return the top matching records.

In this lesson, Pinecone is not generating embeddings. The embedding model does that. Pinecone stores and retrieves them.

## Data Model Used in This Project

Each record effectively looks like this:

```python
{
  "id": "123",
  "values": [embedding numbers...],
  "metadata": {
    "text": "original question"
  }
}
```

This is a common pattern:

- ID for uniqueness
- vector for semantic search
- metadata for readable output and filtering

## Why Metadata Matters

Without metadata, the database would return only IDs and maybe scores.

Metadata allows the application to:

- show the original text
- attach document titles, URLs, authors, or tags
- apply filters later

In this notebook, metadata is simple:

```python
{'text': text}
```

But in real applications metadata is often much richer.

## What Happens When `run_query()` Is Called

When the notebook runs:

```python
run_query('which city has the highest population in the world?')
```

the system does the following:

1. Encode the query into a 384-dimensional vector.
2. Send that vector to Pinecone.
3. Pinecone compares it with stored vectors.
4. Pinecone returns the closest matches and similarity scores.
5. The notebook prints the original matched questions from metadata.

This is the end-to-end semantic search loop.

## Why This Matters for AI Applications

This lesson introduces the retrieval layer used in many real systems:

- semantic search engines
- question-answering systems
- document retrieval
- recommendation systems
- Retrieval-Augmented Generation (RAG)

In RAG, the same pattern is used:

1. embed documents
2. store them in a vector database
3. embed the user query
4. retrieve the most relevant chunks
5. send those chunks to an LLM

## Key Takeaways

- Semantic search matches meaning, not just keywords.
- Embeddings convert text into vectors.
- The notebook uses `all-MiniLM-L6-v2`, which outputs 384-dimensional vectors.
- Pinecone stores vectors and performs similarity search.
- The same embedding model must be used for both stored data and queries.
- Metadata is what makes retrieved vectors useful to humans.
- Vector databases are a core building block for modern retrieval systems.

## Simple Summary

Lesson 1 teaches how to build a semantic search system by:

- loading Quora questions
- converting them into embeddings
- storing those embeddings in Pinecone
- querying Pinecone with a new embedded sentence
- retrieving semantically similar questions

This is the foundation for understanding vector databases and how they support intelligent search.
