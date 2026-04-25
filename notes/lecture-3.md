# Lecture 3 Notes: Recommender Systems

## Lesson Goal

The goal of Lesson 3 is to build a simple recommendation system using embeddings and a vector database.

Instead of recommending items from hand-written rules or exact keyword matches, the system represents news titles and article content as vectors. A user query can then be matched against the most similar vectors in Pinecone.

This makes it possible to recommend articles that are semantically related to a topic even when the wording is not identical.

## What Is a Recommender System?

A recommender system suggests items that are likely to be relevant to a user or a search term.

In this lesson, the recommendation logic is based on vector similarity:

1. Convert items into embeddings.
2. Convert the search term into an embedding.
3. Find the nearest stored vectors.
4. Return the matching titles.

This is a content-based recommender because the recommendations come from the content of the news data, not from user click history or ratings.

## Core Idea of the Lesson

The lesson shows two related recommendation pipelines:

1. Embed only the news titles and recommend articles based on title similarity.
2. Embed chunks of the full article text and recommend articles based on deeper content similarity.

The second approach is usually stronger because it uses much more of the article meaning than the title alone.

## What the Notebook Uses

This lesson is centered on [`Lesson_3_Recommender_Systems.ipynb`](/Users/sandeep/Desktop/Projects/Vector-Databases/Lesson_3_Recommender_Systems.ipynb) and uses:

- `pandas` to load the news dataset from CSV
- `OpenAI` to create embeddings
- `Pinecone` to store and search vectors
- `RecursiveCharacterTextSplitter` from LangChain to split long articles into chunks
- `tqdm` to show upload progress
- [`DLAIUtils.py`](/Users/sandeep/Desktop/Projects/Vector-Databases/DLAIUtils.py) to load API keys and build the index name

## Step-by-Step Flow in This Repo

### 1. Import Packages

The notebook imports the libraries needed for:

- data loading
- text chunking
- embedding generation
- vector storage

Important imports:

- `RecursiveCharacterTextSplitter`
- `OpenAI`
- `Pinecone` and `ServerlessSpec`
- `pandas`
- `tqdm`

### 2. Load API Keys and Initialize Clients

The notebook uses the utility helper to read API keys:

```python
utils = Utils()
PINECONE_API_KEY = utils.get_pinecone_api_key()
OPENAI_API_KEY = utils.get_openai_api_key()
```

Then it builds an OpenAI client:

```python
openai_client = OpenAI(api_key=OPENAI_API_KEY)
```

It also defines an embedding helper:

```python
def get_embeddings(articles, model="text-embedding-ada-002"):
   return openai_client.embeddings.create(input=articles, model=model)
```

This helper is reused for both titles and article chunks.

### 3. Load and Inspect the Dataset

The notebook reads from:

```python
./data/all-the-news-3.csv
```

It first prints the CSV header and then loads a small sample:

```python
df = pd.read_csv('./data/all-the-news-3.csv', nrows=99)
```

This is just an inspection step to confirm the dataset layout before indexing a larger amount of data.

### 4. Create a Pinecone Index

The notebook connects to Pinecone, deletes any existing index with the same generated name, and creates a new one:

```python
INDEX_NAME = utils.create_dlai_index_name('dl-ai')
pinecone = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
  pinecone.delete_index(INDEX_NAME)

pinecone.create_index(
    name=INDEX_NAME,
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-west-2')
)
```

Important settings:

- `dimension=1536`: must match the embedding size returned by the embedding model used here
- `metric='cosine'`: compares items by semantic similarity
- `region='us-west-2'`: chooses where the serverless index is hosted

### 5. Create Embeddings for News Titles

The first recommender version uses only titles.

The notebook processes the dataset in chunks:

```python
CHUNK_SIZE = 400
TOTAL_ROWS = 10000
chunks = pd.read_csv('./data/all-the-news-3.csv', chunksize=CHUNK_SIZE, nrows=TOTAL_ROWS)
```

For each chunk:

- extract the `title` column
- create embeddings for the titles
- build Pinecone records with an `id`, vector `values`, and title metadata
- upsert the records into Pinecone

Example record structure:

```python
{
    'id': '0',
    'values': embeddings.data[i].embedding,
    'metadata': {'title': titles[i]}
}
```

The metadata stores the original title so the recommendations are readable when results come back.

### 6. Query the Title-Based Recommender

The notebook defines:

```python
def get_recommendations(pinecone_index, search_term, top_k=10):
  embed = get_embeddings([search_term]).data[0].embedding
  res = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True)
  return res
```

This does the full recommendation step:

1. embed the user search term
2. query Pinecone for nearest vectors
3. return the matched titles from metadata

It tests the recommender with:

```python
reco = get_recommendations(index, 'obama')
```

This produces titles that are semantically related to the query term.

### 7. Rebuild the Index for Full Article Content

After the title-based version, the notebook creates a fresh index again:

```python
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
  pinecone.delete_index(name=INDEX_NAME)

pinecone.create_index(
    name=INDEX_NAME,
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-west-2')
)

articles_index = pinecone.Index(INDEX_NAME)
```

This means the notebook replaces the title-only index with a new index that will store article-content embeddings instead.

### 8. Split Articles into Chunks

Long article text is split before embedding:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20
)
```

Why this matters:

- long documents are easier to embed in smaller parts
- chunking preserves more specific meaning
- overlap helps avoid losing context at chunk boundaries

### 9. Embed and Store Article Chunks

The notebook starts with:

```python
news_data_rows_num = 100
```

This small limit keeps the demo faster, but the notebook notes that using `200`, `400`, `700`, or `1000` rows can improve result quality.

Then it loops through the articles:

```python
df = pd.read_csv('./data/all-the-news-3.csv', nrows=news_data_rows_num)
articles_list = df['article'].tolist()
titles_list = df['title'].tolist()
```

For each article:

1. split the text into chunks
2. embed those chunks
3. upsert each chunk into Pinecone
4. store the article title as metadata

The helper used for storage is:

```python
def embed(embeddings, title, prepped, embed_num):
  for embedding in embeddings.data:
    prepped.append({
        'id': str(embed_num),
        'values': embedding.embedding,
        'metadata': {'title': title}
    })
    embed_num += 1
    if len(prepped) >= 100:
        articles_index.upsert(prepped)
        prepped.clear()
  return embed_num
```

Important detail:

- many chunk vectors can point to the same article title
- that lets one article be represented by multiple semantic pieces

### 10. Query the Content-Based Recommender

The same `get_recommendations` function is reused, but now it queries the article-chunk index:

```python
reco = get_recommendations(articles_index, 'obama', top_k=100)
```

Because multiple chunks from the same article may match, the notebook removes duplicate titles:

```python
seen = {}
for r in reco.matches:
    title = r.metadata['title']
    if title not in seen:
        print(f'{r.score} : {title}')
        seen[title] = '.'
```

This produces a cleaner final recommendation list.

## Why Full-Content Recommendations Are Better

Title embeddings are fast and simple, but titles do not always capture all the important meaning in an article.

Embedding the article body improves recommendations because:

- more context is available to the embedding model
- the system can match on details that are missing from the headline
- different chunks of the same article can match different aspects of a query

This usually produces recommendations that are more relevant than title-only search.

## Important Design Details

There are a few practical details that matter in this notebook:

- the index dimension must match the embedding size
- query embeddings and stored vectors must come from the same embedding model family
- chunking long text improves retrieval quality for large documents
- duplicate article titles need to be filtered when multiple chunks from the same article match

If these details are ignored, recommendations can become noisy or fail entirely.

## Practical Takeaway

This lesson extends the same vector-search idea from the earlier notebooks into recommendation.

So the progression across the course becomes:

1. Lecture 1: retrieve semantically similar text
2. Lecture 2: retrieve supporting text and generate an answer
3. Lecture 3: retrieve semantically similar content as recommendations

That is the core shift in this notebook: use embeddings not just for search or grounded generation, but also for recommending content that is meaningfully related to a user query.
