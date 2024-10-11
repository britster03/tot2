# ir_system.py

import json
import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics import dcg_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import logging
import os

# ========================
# setup logging
# ========================

def setup_logging(log_path='logs/ir_system.log'):
    """
    set up logging configuration.
    """
    # ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # configure logging to write messages to the log file
    logging.basicConfig(
        filename=log_path,
        filemode='a',  # append mode
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO  # log messages with level INFO and above
    )

# ========================
# download nltk data
# ========================

def download_nltk_data():
    """
    download necessary nltk data.
    """
    try:
        # try to find the 'punkt' tokenizer data
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # if not found, download it
        logging.info("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
        logging.info("NLTK 'punkt' tokenizer downloaded successfully.")

# ========================
# data loading
# ========================

def load_corpus(corpus_path):
    """
    generator to load and yield documents one by one from corpus.jsonl.
    """
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # parse each line as json and yield the document
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # log any json decoding errors and continue
                logging.error(f"JSON decoding failed: {e}")
                continue

def load_queries(queries_path):
    """
    load queries from a json lines file.
    """
    queries = []
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # parse each line as json and extract query_id and query text
                obj = json.loads(line)
                queries.append({'query_id': obj['query_id'], 'query': obj['query']})
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding failed for query: {e}")
                continue
    # return the queries as a pandas dataframe
    return pd.DataFrame(queries)

def load_qrels(qrel_path):
    """
    load relevance judgments from a qrel file.
    """
    qrels = {}
    with open(qrel_path, 'r', encoding='utf-8') as f:
        for line in f:
            # split each line into parts
            parts = line.strip().split()
            if len(parts) == 4:
                query_id, _, doc_id, relevance = parts
                if relevance == '1':
                    # add the doc_id to the list of relevant documents for the query_id
                    if query_id in qrels:
                        qrels[query_id].append(doc_id)
                    else:
                        qrels[query_id] = [doc_id]
                # ignore documents with relevance not equal to '1'
            else:
                logging.warning(f"Unexpected QREL format: {line.strip()}")
    return qrels

# ========================
# bm25 indexing
# ========================

def build_bm25_index(corpus_path, bm25_index_path):
    """
    build and save bm25 index.
    """
    tokenized_corpus = []
    doc_ids = []
    logging.info("Starting BM25 indexing.")
    # iterate over documents in the corpus
    for doc in tqdm(load_corpus(corpus_path), desc="Building BM25 Index"):
        if 'text' in doc and 'doc_id' in doc:
            # tokenize the document text into words
            tokens = word_tokenize(doc['text'].lower())
            tokenized_corpus.append(tokens)
            doc_ids.append(doc['doc_id'])
        else:
            logging.warning(f"Document missing 'text' or 'doc_id': {doc}")
    
    if not tokenized_corpus:
        logging.error("No documents found for BM25 indexing.")
        return
    
    # create a bm25 index using the tokenized corpus
    bm25 = BM25Okapi(tokenized_corpus)
    # save the bm25 index and doc_ids to a file
    joblib.dump({'bm25': bm25, 'doc_ids': doc_ids}, bm25_index_path)
    logging.info(f"BM25 index built and saved to {bm25_index_path}")

# ========================
# embedding indexing with faiss
# ========================

def build_faiss_index(corpus_path, embedding_model_name, faiss_index_path, faiss_id_path, batch_size=1000):
    """
    build and save faiss index for embeddings with id mapping.
    """
    # load the pre-trained sentence transformer model
    embedding_model = SentenceTransformer(embedding_model_name)
    # get the dimension of the embeddings from the model
    dim = embedding_model.get_sentence_embedding_dimension()
    
    # initialize a faiss index for inner product similarity (cosine similarity)
    index = faiss.IndexFlatIP(dim)
    # enable id mapping in faiss index
    index = faiss.IndexIDMap(index)
    
    doc_ids = []  # list to store doc_ids
    batch_texts = []  # list to store texts for batch processing
    faiss_ids = []  # list to store unique integer ids for faiss
    current_id = 0  # counter for faiss ids
    
    logging.info("Starting Embedding-based indexing with FAISS.")
    
    # iterate over documents in the corpus
    for idx, doc in enumerate(tqdm(load_corpus(corpus_path), desc="Building FAISS Index")):
        if 'text' in doc and 'doc_id' in doc:
            batch_texts.append(doc['text'])
            doc_ids.append(doc['doc_id'])
            faiss_ids.append(current_id)
            current_id += 1
            
            if len(batch_texts) == batch_size:
                # encode the batch of texts to embeddings
                embeddings = embedding_model.encode(batch_texts, convert_to_tensor=False, show_progress_bar=False)
                embeddings = np.array(embeddings).astype('float32')
                # normalize the embeddings to unit length
                faiss.normalize_L2(embeddings)
                # add embeddings and their ids to the faiss index
                index.add_with_ids(embeddings, np.array(faiss_ids))
                # reset the batch lists
                batch_texts = []
                faiss_ids = []
                logging.info(f"Processed {idx + 1} documents for FAISS.")
        else:
            logging.warning(f"Document missing 'text' or 'doc_id': {doc}")
    
    # process any remaining documents in the batch
    if batch_texts:
        embeddings = embedding_model.encode(batch_texts, convert_to_tensor=False, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        index.add_with_ids(embeddings, np.array(faiss_ids))
        logging.info(f"Processed {idx + 1} documents for FAISS.")
    
    # save the faiss index to a file
    faiss.write_index(index, faiss_index_path)
    logging.info(f"FAISS embedding index built and saved to {faiss_index_path}")
    
    # save the mapping from faiss ids to document ids
    joblib.dump(doc_ids, faiss_id_path)
    logging.info(f"FAISS doc_id mapping saved to {faiss_id_path}")

# ========================
# retrieval and ranking
# ========================

def bm25_retrieve(query, bm25_obj, top_k=1000):
    """
    retrieve top_k documents using bm25.
    """
    # tokenize the query text
    tokens = word_tokenize(query.lower())
    # get bm25 scores for all documents
    scores = bm25_obj['bm25'].get_scores(tokens)
    # get indices of documents sorted by score in descending order
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_indices]
    # get the doc_ids corresponding to the top indices
    top_doc_ids = [bm25_obj['doc_ids'][i] for i in top_indices]
    return top_doc_ids, top_scores

def embedding_retrieve(query, embedding_model, faiss_index, faiss_id_list, top_k=1000):
    """
    retrieve top_k documents using embedding-based retrieval with faiss.
    """
    # encode the query into an embedding
    query_embedding = embedding_model.encode([query], convert_to_tensor=False, show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype('float32')
    # normalize the query embedding
    faiss.normalize_L2(query_embedding)
    # search the faiss index for the top_k most similar documents
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # map faiss indices back to document ids
    top_doc_ids = []
    for idx in indices[0]:
        if idx != -1 and idx < len(faiss_id_list):
            doc_id = faiss_id_list[idx]
            top_doc_ids.append(doc_id)
        else:
            logging.warning(f"Invalid FAISS index returned: {idx}")
            top_doc_ids.append("-1")  # placeholder for invalid doc_id
    
    return top_doc_ids, distances[0]

# ========================
# score combination
# ========================

def combine_scores(bm25_scores, embedding_scores, alpha=0.5):
    """
    normalize bm25 and embedding scores and return them separately.
    """
    bm25_min, bm25_max = np.min(bm25_scores), np.max(bm25_scores)
    embedding_min, embedding_max = np.min(embedding_scores), np.max(embedding_scores)
    
    # normalize bm25 scores to range [0,1]
    if bm25_max - bm25_min == 0:
        bm25_norm = np.zeros_like(bm25_scores)
    else:
        bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-6)
    
    # normalize embedding scores to range [0,1]
    if embedding_max - embedding_min == 0:
        embedding_norm = np.zeros_like(embedding_scores)
    else:
        embedding_norm = (embedding_scores - embedding_min) / (embedding_max - embedding_min + 1e-6)
    
    return bm25_norm, embedding_norm

def retrieve_and_rank(query, bm25_obj, embedding_model, faiss_index, faiss_id_list, top_k=1000, alpha=0.5):
    """
    retrieve and rank documents for a single query by combining bm25 and embedding scores.
    """
    # retrieve documents using bm25
    bm25_doc_ids, bm25_scores = bm25_retrieve(query, bm25_obj, top_k)
    
    # retrieve documents using embeddings
    embedding_doc_ids, embedding_scores = embedding_retrieve(query, embedding_model, faiss_index, faiss_id_list, top_k)
    
    # normalize the scores
    bm25_norm, embedding_norm = combine_scores(bm25_scores, embedding_scores, alpha)
    
    # combine the scores into a single dictionary
    doc_score_dict = {}
    
    # add bm25 scores to the dictionary with weighting
    for doc_id, score in zip(bm25_doc_ids, bm25_norm):
        doc_score_dict[doc_id] = doc_score_dict.get(doc_id, 0) + alpha * score
    
    # add embedding scores to the dictionary with weighting
    for doc_id, score in zip(embedding_doc_ids, embedding_norm):
        doc_score_dict[doc_id] = doc_score_dict.get(doc_id, 0) + (1 - alpha) * score
    
    # sort the documents by their combined scores in descending order
    ranked_docs = sorted(doc_score_dict.items(), key=lambda item: item[1], reverse=True)[:top_k]
    ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs]
    return ranked_doc_ids  # return the list of ranked doc_ids

def batch_retrieve_and_rank(queries_df, bm25_obj, embedding_model, faiss_index, faiss_id_list, top_k=1000, alpha=0.5, batch_size=100):
    """
    retrieve and rank documents for multiple queries in batches.
    
    returns:
    - dict: mapping from query_id to ranked list [doc_id, ...]
    """
    ranked_results = {}
    total_queries = len(queries_df)
    logging.info("Starting batch retrieval and ranking.")
    
    # process queries in batches to save memory
    for i in tqdm(range(0, total_queries, batch_size), desc="Batch Processing"):
        batch = queries_df.iloc[i:i+batch_size]
        for idx, row in batch.iterrows():
            query_id = row['query_id']
            query = row['query']
            # retrieve and rank documents for the query
            ranked_list = retrieve_and_rank(query, bm25_obj, embedding_model, faiss_index, faiss_id_list, top_k, alpha)
            ranked_results[query_id] = ranked_list
    
    return ranked_results

# ========================
# evaluation metrics
# ========================

def reciprocal_rank(ranked_list, true_doc_ids):
    """
    compute reciprocal rank for a single query.
    """
    for idx, doc_id in enumerate(ranked_list, start=1):
        if doc_id in true_doc_ids:
            # return the reciprocal of the rank position
            return 1.0 / idx
    return 0.0  # return zero if none of the relevant docs are found

def success_at_k(ranked_list, true_doc_ids, k=10):
    """
    compute success@k for a single query.
    """
    top_k = ranked_list[:k]
    # return 1 if any relevant doc is in the top_k, else 0
    return int(any(doc_id in true_doc_ids for doc_id in top_k))

def compute_dcg(ranked_list, true_doc_ids, k=1000):
    """
    compute dcg for a single query.
    """
    # create a relevance list where relevant docs are marked with 1
    relevance = [1 if doc_id in true_doc_ids else 0 for doc_id in ranked_list[:k]]
    # compute dcg score using sklearn's function
    return dcg_score([relevance], [relevance], k=k)

def evaluate_metrics(ranked_results, qrels, k=10):
    """
    evaluate retrieval performance across all queries.
    
    returns:
    - dict: average metrics
    """
    rr_total = 0.0  # sum of reciprocal ranks
    dcg_total = 0.0  # sum of dcg scores
    success_total = 0  # count of successes at k
    count = 0  # number of queries evaluated
    
    for query_id, ranked_list in ranked_results.items():
        true_doc_ids = qrels.get(query_id, [])
        if not true_doc_ids:
            logging.warning(f"No relevance judgments for query_id: {query_id}")
            continue  # skip queries without relevance judgments
        # compute metrics for this query
        rr = reciprocal_rank(ranked_list, true_doc_ids)
        dcg = compute_dcg(ranked_list, true_doc_ids, k=k)
        success = success_at_k(ranked_list, true_doc_ids, k=k)
        
        # accumulate the metrics
        rr_total += rr
        dcg_total += dcg
        success_total += success
        count += 1
    
    if count == 0:
        logging.error("No queries with relevance judgments found.")
        return {}
    
    # compute average metrics
    avg_rr = rr_total / count
    avg_dcg = dcg_total / count
    avg_success = success_total / count
    
    return {
        'Average Reciprocal Rank (RR)': avg_rr,
        'Average DCG': avg_dcg,
        f'Average Success@{k}': avg_success
    }

# ========================
# main function
# ========================

def main(args):
    # set up logging
    setup_logging()
    logging.info("IR System Started.")
    
    # download necessary nltk data
    download_nltk_data()
    
    # check if indices need to be built or loaded
    if args.build_bm25:
        # build the bm25 index
        build_bm25_index(args.corpus_path, args.bm25_index_path)
    else:
        # check if the bm25 index file exists
        if not os.path.exists(args.bm25_index_path):
            logging.error(f"BM25 index file not found at {args.bm25_index_path}.")
            return
    
    if args.build_faiss:
        # build the faiss index
        build_faiss_index(args.corpus_path, args.embedding_model, args.faiss_index_path, args.faiss_id_path, args.batch_size)
    else:
        # check if the faiss index files exist
        if not os.path.exists(args.faiss_index_path):
            logging.error(f"FAISS index file not found at {args.faiss_index_path}.")
            return
        if not os.path.exists(args.faiss_id_path):
            logging.error(f"FAISS doc_id mapping file not found at {args.faiss_id_path}.")
            return
    
    # load queries and qrels for evaluation
    queries_df = load_queries(args.queries_path)
    qrels = load_qrels(args.qrel_path)
    logging.info(f"Loaded {len(queries_df)} queries.")
    
    # load the bm25 index
    bm25_obj = joblib.load(args.bm25_index_path)
    logging.info("Loaded BM25 index.")
    
    # load the faiss index and doc_id mapping
    faiss_index = faiss.read_index(args.faiss_index_path)
    faiss_id_list = joblib.load(args.faiss_id_path)
    logging.info("Loaded FAISS index and doc_id mapping.")
    
    # initialize the embedding model
    embedding_model = SentenceTransformer(args.embedding_model)
    logging.info(f"Loaded embedding model: {args.embedding_model}")
    
    # perform retrieval and ranking for all queries
    ranked_results = batch_retrieve_and_rank(
        queries_df,
        bm25_obj,
        embedding_model,
        faiss_index,
        faiss_id_list,
        top_k=args.top_k,
        alpha=args.alpha,
        batch_size=args.batch_size
    )
    logging.info("Completed retrieval and ranking.")
    
    # evaluate the retrieval performance
    metrics = evaluate_metrics(ranked_results, qrels, k=args.k)
    if metrics:
        logging.info("Evaluation Metrics:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        # print the metrics to console
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        logging.error("No evaluation metrics to display.")
    
    # save the ranked results to a file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for query_id, ranked_list in ranked_results.items():
            # write the query_id and the list of ranked doc_ids
            f.write(f"{query_id}\t{','.join(ranked_list)}\n")
    logging.info(f"Ranked results saved to {args.output_path}")
    
    logging.info("IR System Finished Successfully.")

# ========================
# argument parser
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToT Known-Item Identification IR System")
    parser.add_argument('--corpus_path', type=str, default='data/corpus.jsonl', help='Path to corpus.jsonl')
    parser.add_argument('--queries_path', type=str, default='data/queries.jsonl', help='Path to queries.jsonl')
    parser.add_argument('--qrel_path', type=str, default='data/qrel.txt', help='Path to qrel.txt')
    parser.add_argument('--bm25_index_path', type=str, default='models/bm25_index.pkl', help='Path to BM25 index file')
    parser.add_argument('--faiss_index_path', type=str, default='models/faiss_index.faiss', help='Path to FAISS index file')
    parser.add_argument('--faiss_id_path', type=str, default='models/faiss_id_list.pkl', help='Path to FAISS doc_id mapping file')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    parser.add_argument('--top_k', type=int, default=1000, help='Number of top documents to retrieve for each query')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weighting factor to combine BM25 and Embedding scores (0 <= alpha <= 1)')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of queries to process in a batch')
    parser.add_argument('--k', type=int, default=10, help='Rank cutoff for Success@k')
    parser.add_argument('--output_path', type=str, default='data/ranked_results.txt', help='Path to save ranked results')
    parser.add_argument('--build_bm25', action='store_true', help='Flag to build BM25 index')
    parser.add_argument('--build_faiss', action='store_true', help='Flag to build FAISS embedding index')
    
    args = parser.parse_args()
    
    main(args)
