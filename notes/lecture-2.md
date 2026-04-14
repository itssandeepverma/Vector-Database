# Lecture 2 Notes: Retrieval Augmented Generation (RAG)

## Lesson Goal

The goal of Lesson 2 is to build a simple Retrieval Augmented Generation pipeline.

Instead of asking a language model to answer only from its built-in knowledge, the system first retrieves relevant documents from a vector database and then includes that retrieved context in the prompt.

This improves factual grounding and makes the answer more specific to the retrieved material.

## What Is RAG?

RAG stands for Retrieval Augmented Generation.

It combines two steps:

1. Retrieve relevant documents or passages from an external knowledge source.
2. Generate an answer using those retrieved documents as context.

This is useful because large language models do not store every fact perfectly, and they can hallucinate when asked about specific topics. Retrieval gives the model external evidence to work from.

## Core Idea of the Lesson

The lesson shows this pipeline:

1. Load a dataset of precomputed article embeddings.
2. Store those vectors and article text in Pinecone.
3. Convert a user query into an embedding with OpenAI.
4. Search Pinecone for the most relevant articles.
5. Build a prompt that includes the retrieved context.
6. Ask the language model to answer using that context.

This is the standard structure behind many document chat, question-answering, and knowledge assistant systems.

## What the Notebook Uses

This lesson is centered on [`Lesson_2_Retrieval_Augmented_Generation.ipynb`](/Users/sandeep/Desktop/Projects/Vector-Databases/Lesson_2_Retrieval_Augmented_Generation.ipynb) and uses:

- `pandas` to load the article dataset from CSV
- `ast` to parse stringified metadata and vector values
- `OpenAI` to create query embeddings and generate the final answer
- `Pinecone` to store and query vectors
- `tqdm` to show progress while uploading records
- [`DLAIUtils.py`](/Users/sandeep/Desktop/Projects/Vector-Databases/DLAIUtils.py) to load API keys and build the index name

## Step-by-Step Flow in This Repo

### 1. Import Packages

The notebook imports:

- `load_dataset` from `datasets`
- `OpenAI`
- `Pinecone` and `ServerlessSpec`
- `tqdm`
- `pandas`
- `ast`

Not every imported library is used heavily later, but the important ones for the RAG pipeline are OpenAI, Pinecone, pandas, and the utility class.

### 2. Load the Pinecone API Key

The notebook creates a utility object and loads the Pinecone key:

```python
utils = Utils()
PINECONE_API_KEY = utils.get_pinecone_api_key()
```

That helper reads the key from `.env`.

### 3. Create a Fresh Pinecone Index

The notebook connects to Pinecone and creates an index:

```python
pinecone = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = utils.create_dlai_index_name('dl-ai')
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
  pinecone.delete_index(INDEX_NAME)

pinecone.create_index(
    name=INDEX_NAME,
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)

index = pinecone.Index(INDEX_NAME)
```

What this does:

- connects to your Pinecone account
- builds a course-specific index name
- deletes any older version of that index
- creates a new empty vector index
- returns a handle to the index for later inserts and queries

Important detail:

- `dimension=1536` must match the embedding size stored in the dataset and the embedding model used for queries

### 4. Load the Dataset

The notebook reads:

```python
max_articles_num = 500
df = pd.read_csv('./data/wiki.csv', nrows=max_articles_num)
```

This means:

- only the first `500` articles are loaded by default
- each row contains a text passage, metadata, and a vector
- the limit is kept small for faster experimentation

The lesson notes that increasing `max_articles_num` can improve retrieval quality because the search space becomes richer.

### 5. Prepare Records for Pinecone

The dataset stores vectors and metadata as strings, so the notebook converts them back to Python objects:

```python
meta = ast.literal_eval(row['metadata'])
prepped.append({
    'id': row['id'],
    'values': ast.literal_eval(row['values']),
    'metadata': meta
})
```

Each Pinecone record contains:

- `id`: unique identifier
- `values`: embedding vector
- `metadata`: extra information, especially the article text

### 6. Upsert the Data into Pinecone

The notebook uploads records in batches:

```python
if len(prepped) >= 250:
    index.upsert(prepped)
    prepped = []
```

`upsert` means:

- insert if the ID is new
- update if the ID already exists

Batching is used because sending records one by one is inefficient.

### 7. Inspect Index Stats

The notebook runs:

```python
index.describe_index_stats()
```

This is a quick sanity check that confirms vectors were loaded into the index.

### 8. Connect to OpenAI

The notebook loads the OpenAI API key and builds a client:

```python
OPENAI_API_KEY = utils.get_openai_api_key()
openai_client = OpenAI(api_key=OPENAI_API_KEY)
```

Then it defines an embedding helper:

```python
def get_embeddings(articles, model="text-embedding-ada-002"):
   return openai_client.embeddings.create(input=articles, model=model)
```

This function converts text into embeddings so queries can be searched against the vectors already stored in Pinecone.

### 9. Retrieve Relevant Articles

For a query such as:

```python
query = "what is the berlin wall?"
embed = get_embeddings([query])
res = index.query(
    vector=embed.data[0].embedding,
    top_k=3,
    include_metadata=True
)
```

The system:

- embeds the query
- searches Pinecone for the top 3 nearest vectors
- returns metadata from the matched records

The retrieved text is then extracted with:

```python
text = [r['metadata']['text'] for r in res['matches']]
```

This is the retrieval part of RAG.

### 10. Build the Prompt

The notebook turns the retrieved passages into a grounded prompt:

```python
contexts = [x['metadata']['text'] for x in res['matches']]

prompt_start = (
    "Answer the question based on the context below.\n\n"+
    "Context:\n"
)

prompt_end = (
    f"\n\nQuestion: {query}\nAnswer:"
)

prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end
```

This is the augmentation part of RAG.

The model is not answering from the bare question alone. It is answering from:

- the user question
- the retrieved context passages
- the instruction to stay grounded in that context

### 11. Generate the Final Answer

The notebook sends the prompt to OpenAI:

```python
res = openai_client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    temperature=0,
    max_tokens=636
)
```

This is the generation part of RAG.

Because the prompt contains retrieved context, the answer is more likely to be specific and relevant than a plain standalone completion.

## Why RAG Helps

RAG improves language model applications in several ways:

- it grounds answers in retrieved evidence
- it reduces hallucinations
- it allows the system to use external knowledge not baked into model weights
- it makes it possible to answer questions over your own documents

This is why RAG is a common pattern for:

- document Q&A
- internal knowledge assistants
- support bots
- research tools
- retrieval-based chat applications

## Key Components of a RAG System

This lesson demonstrates the main parts of a RAG pipeline:

### Retriever

The retriever finds relevant documents by vector similarity.

In this notebook:

- OpenAI embeddings are used for the query
- Pinecone is used to find nearest stored vectors

### Knowledge Store

The knowledge store is the external dataset being searched.

In this notebook:

- the knowledge source is a wiki article CSV
- article text is stored as metadata alongside vector embeddings

### Generator

The generator turns the retrieved context into a final answer.

In this notebook:

- the OpenAI completion model reads the retrieved passages and produces the response

## Important Design Detail

The stored vectors and query vectors must belong to the same embedding space.

That means:

- the dimension must match
- the model family used for query embeddings should match the vectors already stored

If stored vectors were created by one embedding model and queries are created by a different incompatible model, retrieval quality can collapse.

## Practical Takeaway

Semantic search alone returns relevant passages.

RAG adds one more step:

- after retrieval, the passages are inserted into a prompt so a language model can generate a grounded answer

So the evolution from Lecture 1 to Lecture 2 is:

1. Lecture 1: retrieve similar text
2. Lecture 2: retrieve similar text and then generate an answer from it

That is the core progression from vector search to a working RAG application.
