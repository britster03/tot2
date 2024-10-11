# test.py

import faiss
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

def test_single_query(query, embedding_model_name, faiss_index_path, faiss_id_path, top_k=10):
    # load the faiss index from the file
    faiss_index = faiss.read_index(faiss_index_path)
    # load the mapping from faiss ids to doc_ids
    faiss_id_list = joblib.load(faiss_id_path)
    
    # initialize the embedding model
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # encode the query to get its embedding
    query_embedding = embedding_model.encode([query], convert_to_tensor=False, show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype('float32')
    # normalize the query embedding
    faiss.normalize_L2(query_embedding)
    
    # perform the search on the faiss index
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # map the indices to doc_ids
    top_doc_ids = []
    for idx in indices[0]:
        if idx != -1 and 0 <= idx < len(faiss_id_list):
            doc_id = faiss_id_list[idx]
            top_doc_ids.append(doc_id)
        else:
            print(f"Invalid FAISS index returned: {idx}")
            top_doc_ids.append("-1")  # placeholder for invalid doc_id
    
    # print the retrieved doc_ids for the query
    print(f"Retrieved doc_ids for query '{query}': {top_doc_ids}")

if __name__ == "__main__":
    # define the test query (replace with an actual query if needed)
    test_query = "Alaric II (, , \"ruler of all\"; ; – August 507) was the King of the Visigoths from 484 until 507. He succeeded his father Euric as king of the Visigoths in Toulouse on 28 December 484; he was the great-grandson of the more famous Alaric I, who sacked Rome in 410."
    embedding_model_name = 'all-MiniLM-L6-v2'  # name of the embedding model used
    faiss_index_path = 'models/faiss_index.faiss'  # path to the faiss index file
    faiss_id_path = 'models/faiss_id_list.pkl'  # path to the faiss id mapping file
    top_k = 10  # number of top documents to retrieve
    
    # run the test function with the provided parameters
    test_single_query(test_query, embedding_model_name, faiss_index_path, faiss_id_path, top_k)
