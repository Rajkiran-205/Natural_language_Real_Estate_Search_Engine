import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import os
import time
import copy

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
CLEANED_DATA_PATH = 'cleaned_properties.json' 
INDEX_PATH = 'property_index_real.faiss'
INDEX_MAP_PATH = 'index_to_prop_id_real.json'

# --- Module 1: Semantic Parser  ---

def parse_price(query):
    price_info = {}
    price_pattern = re.compile(
        r'(under|below|over|above|around)?\s*([\d\.]+)\s*(l|lakh|cr|crore)', 
        re.IGNORECASE
    )
    match = price_pattern.search(query)
    if match:
        qualifier, amount_str, unit = match.groups()
        amount = float(amount_str)
        if unit.lower() in ['l', 'lakh']: amount *= 100000
        elif unit.lower() in ['cr', 'crore']: amount *= 10000000
        if qualifier and qualifier.lower() in ['under', 'below']: price_info['max'] = int(amount)
        elif qualifier and qualifier.lower() in ['over', 'above']: price_info['min'] = int(amount)
        else: price_info['max'] = int(amount)
    return price_info

def parse_query(query):
    original_query = query
    structured_query = {"filters": {}, "semantic_query": ""}
    cities = ['Gurgaon','Noida','Ghaziabad','Greater-Noida','Bangalore','Mumbai','Pune','Hyderabad','Kolkata','Chennai',
              'New-Delhi','Ahmedabad','Navi-Mumbai','Thane','Faridabad','Bhubaneswar','Bokaro-Steel-City','Vijayawada','Vrindavan',
              'Bhopal','Gorakhpur','Jamshedpur','Agra','Allahabad','Jodhpur','Aurangabad','Jaipur','Mangalore','Nagpur','Guntur',
              'Navsari','Palghar','Salem','Haridwar','Durgapur','Madurai','Manipal','Patna','Ranchi','Raipur','Sonipat','Kottayam',
              'Kozhikode','Thrissur','Tirupati','Trivandrum','Trichy','Udaipur','Vapi','Varanasi','Vadodara','Visakhapatnam',
              'Surat','Kanpur','Kochi','Mysore','Goa','Bhiwadi','Lucknow','Nashik','Guwahati','Chandigarh','Indore','Coimbatore','Dehradun']
    semantic_query_text = original_query
    for city in cities:
        city_pattern = r'\b' + city.replace('-', r'\s?-?') + r'\b'
        if re.search(city_pattern, semantic_query_text, re.IGNORECASE):
            structured_query["filters"]["location"] = {"city": city.replace('-', ' ')}
            semantic_query_text = re.sub(city_pattern, '', semantic_query_text, flags=re.IGNORECASE)
            break
    bhk_match = re.search(r'(\d+)\s*(bhk|bedroom|bed)', semantic_query_text, re.IGNORECASE)
    if bhk_match:
        structured_query["filters"]["bhk"] = int(bhk_match.group(1))
        semantic_query_text = semantic_query_text.replace(bhk_match.group(0), "", 1)
    property_types = ["apartment", "flat", "villa", "house", "plot"]
    for p_type in property_types:
        type_pattern = r'\b' + p_type + r's?\b'
        if re.search(type_pattern, semantic_query_text, re.IGNORECASE):
            structured_query["filters"]["property_type"] = p_type
            semantic_query_text = re.sub(type_pattern, '', semantic_query_text, flags=re.IGNORECASE)
            break
    if re.search(r'\b(sale|buy)\b', semantic_query_text, re.IGNORECASE):
        structured_query["filters"]["status"] = "sale"
        semantic_query_text = re.sub(r'\b(sale|buy)\b', '', semantic_query_text, flags=re.IGNORECASE)
    price_filter = parse_price(original_query)
    if price_filter:
        structured_query["filters"]["price"] = price_filter
        price_match = re.search(r'(under|below|over|above|around)?\s*([\d\.]+)\s*(l|lakh|cr|crore)', semantic_query_text, re.IGNORECASE)
        if price_match: semantic_query_text = semantic_query_text.replace(price_match.group(0), "", 1)
    semantic_part = ' '.join(semantic_query_text.split())
    stopwords = ["show me", "find", "a", "an", "the", "with", "in", "for", "is", "near", "me", "of"]
    for word in stopwords:
        semantic_part = re.sub(r'\b' + word + r'\b', '', semantic_part, flags=re.IGNORECASE)
    structured_query["semantic_query"] = ' '.join(semantic_part.split()).strip()
    return structured_query

# --- Module 2: Retrieval (with Smart Scoring) ---

def index_properties(data_path, model):
    with open(data_path, 'r', encoding='utf-8') as f: properties = json.load(f)
    index_to_prop_id, texts_to_embed = {}, []
    for i, prop in enumerate(properties):
        title = prop.get('title') or ''
        description = prop.get('description') or ''
        texts_to_embed.append(f"{title}. {description}")
        index_to_prop_id[i] = prop['property_id']
    embeddings = model.encode(texts_to_embed, convert_to_tensor=False, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(INDEX_MAP_PATH, 'w') as f: json.dump(index_to_prop_id, f)
    return index, properties, index_to_prop_id

def retrieve_and_score(parsed_query, model, index, all_properties, index_to_prop_id):
    """
    NEW: Retrieves properties by scoring them against filters and semantic query,
    instead of just doing a hard filter.
    """
    filters = parsed_query.get('filters', {})
    semantic_query = parsed_query.get('semantic_query')

    # 1. Get semantic scores for all documents
    query_embedding = model.encode([semantic_query])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k=len(all_properties))
    
    semantic_scores = {index_to_prop_id[str(idx)]: dist for idx, dist in zip(indices[0], distances[0])}

    # 2. Calculate a "filter score" for each property
    scored_properties = []
    for prop in all_properties:
        filter_score = 1.0
        
        # Location is a hard filter - it must match if specified
        if 'location' in filters:
            if prop.get('location', {}).get('city', '').lower() != filters['location']['city'].lower():
                continue # Skip properties not in the desired city
        
        # Score based on BHK
        if 'bhk' in filters:
            if prop.get('bhk') == filters['bhk']:
                filter_score *= 1.0 # Perfect match
            else:
                filter_score *= 0.5 # Mismatch, penalize
        
        # Score based on price
        if 'price' in filters and prop.get('price'):
            price_filter = filters['price']
            prop_price = prop['price']
            if 'max' in price_filter and prop_price > price_filter['max']:
                # Penalize based on how much it's over budget
                over_by_percent = (prop_price - price_filter['max']) / price_filter['max']
                filter_score *= max(0, 1 - over_by_percent) # Score drops to 0 if 100% over budget
        
        # 3. Combine scores
        prop_id = prop['property_id']
        # Weights can be tuned to prioritize semantic vs. filter matches
        final_score = (0.6 * semantic_scores.get(prop_id, 0)) + (0.4 * filter_score)
        
        scored_properties.append((prop_id, final_score))

    # 4. Sort by the final combined score
    scored_properties.sort(key=lambda x: x[1], reverse=True)
    
    return [prop_id for prop_id, score in scored_properties[:25]] # Return top 25 candidates


# --- Module 3: Reranker  ---

def rerank_properties(original_query, retrieved_ids, all_properties, model):
    if not retrieved_ids: return []
    properties_map = {p['property_id']: p for p in all_properties}
    pairs = []
    for prop_id in retrieved_ids:
        prop = properties_map.get(prop_id)
        if prop: 
            title = prop.get('title') or ''
            description = prop.get('description') or ''
            pairs.append([original_query, f"{title}. {description}"])
    if not pairs: return []
    scores = model.predict(pairs, show_progress_bar=True)
    id_score_pairs = sorted(list(zip(retrieved_ids, scores)), key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in id_score_pairs]

# --- Main Search Pipeline ---

class SearchEngine:
    def __init__(self):
        print("Initializing search engine...")
        self.retrieval_model = SentenceTransformer(MODEL_NAME)
        self.reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
        if not os.path.exists(INDEX_PATH) or not os.path.exists(CLEANED_DATA_PATH):
            print("ERROR: No index or data file found. Please run data transformation and indexing first.")
            self.properties = []
            return
        print("Loading existing index and data...")
        self.index = faiss.read_index(INDEX_PATH)
        with open(CLEANED_DATA_PATH, 'r', encoding='utf-8') as f: self.properties = json.load(f)
        with open(INDEX_MAP_PATH, 'r') as f: self.index_to_prop_id = json.load(f)
        print("Search engine ready.")

    def search(self, query):
        if not self.properties:
            print("Search engine not initialized.")
            return

        print(f"\n\n{'='*50}\n--- Executing Search for: '{query}' ---\n{'='*50}")
        
        # 1. Parse Query
        parsed_query = parse_query(query)
        print(f"\n1. Parsed Query:\n{json.dumps(parsed_query, indent=2)}")
        
        # 2. Retrieve Properties using the new scoring method
        retrieved_ids = retrieve_and_score(parsed_query, self.retrieval_model, self.index, self.properties, self.index_to_prop_id)
        print(f"\n2. Retrieved IDs (before reranking): {len(retrieved_ids)} candidates found via smart scoring.")

        # 3. Rerank Properties
        final_ids = rerank_properties(query, retrieved_ids, self.properties, self.reranker_model)
        print(f"\n3. Reranked IDs: {len(final_ids)} candidates after reranking.")
        
        # 4. Fetch and display final results
        print("\n--- Top 5 Search Results ---")
        if not final_ids:
            print("No matching properties found.")
            return

        final_results = [p for p in sorted([p for p in self.properties if p['property_id'] in final_ids], key=lambda x: final_ids.index(x['property_id']))]
        for i, prop in enumerate(final_results[:5]):
            print(f"\n{i+1}. {prop['title']}")
            print(f"   Price: \u20b9{prop.get('price', 0):,}")
            print(f"   Location: {prop.get('location', {}).get('locality', 'N/A')}, {prop.get('location', {}).get('city', 'N/A')}")
            print(f"   URL: {prop['source_url']}")
        
        return final_results

if __name__ == '__main__':
    if not os.path.exists(CLEANED_DATA_PATH):
        print("ERROR: 'cleaned_properties.json' not found. Please run the data transformation script first.")
    else:
        if not os.path.exists(INDEX_PATH):
            print("Creating index from scraped data for the first time... (This may take a while)")
            retrieval_model = SentenceTransformer(MODEL_NAME)
            index_properties(CLEANED_DATA_PATH, retrieval_model)
            print("Index created successfully.")
    
        engine = SearchEngine()
        if engine.properties:
            engine.search("3 bhk flat for sale in Gurgaon under 2 crore")
            engine.search("show me a luxury apartment in a good society")
            engine.search("a house near a metro station in Noida")
