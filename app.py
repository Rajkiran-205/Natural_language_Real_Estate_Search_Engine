import streamlit as st
import pandas as pd
# We will import the SearchEngine class from your latest pipeline script
# Note: Your pipeline must be saved as a .py file (e.g., search_pipeline_main.py) for this to work.
from search_pipeline_main import SearchEngine, index_properties 
from sentence_transformers import SentenceTransformer
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Property Search",
    page_icon="🧠",
    layout="wide"
)

# --- Model and Engine Loading ---
# Use Streamlit's caching to load the models and engine only once for performance.
@st.cache_resource
def load_search_engine():
    """
    Loads the SearchEngine and its components. Handles initial index creation if needed.
    Returns the engine instance.
    """
    CLEANED_DATA_PATH = 'cleaned_properties.json'
    INDEX_PATH = 'property_index_real.faiss'
    
    if not os.path.exists(CLEANED_DATA_PATH):
        st.error("`cleaned_properties.json` not found! Please run the data transformation script first.")
        return None

    if not os.path.exists(INDEX_PATH):
        st.warning("Search index not found. Creating a new one... (This might take a moment on the first startup)")
        try:
            retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
            index_properties(CLEANED_DATA_PATH, retrieval_model)
            st.success("Search index created successfully! The app will now reload.")
            # Rerun to load the engine now that the index exists
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to create search index: {e}")
            return None

    engine = SearchEngine()
    return engine

# --- Main App UI ---
st.title("🧠 Intelligent Property Search Engine")
st.markdown("Our advanced engine understands your needs. Try queries like _'3 bhk flat in Gurgaon under 2 crore'_ or _'show me a luxury apartment in a good society'_.")

# Load the engine
engine = load_search_engine()

if engine and engine.properties:
    # Create a search bar within a form for a cleaner experience
    with st.form(key='search_form'):
        query = st.text_input("What kind of property are you looking for?", placeholder="e.g., a house near a metro station in Noida")
        submit_button = st.form_submit_button(label='✨ Find Properties')

    if submit_button and query:
        with st.spinner("Analyzing your query and finding the best matches..."):
            # Perform the search using our advanced pipeline
            results = engine.search(query)
        
        st.success(f"Found {len(results)} relevant properties for you.")
        
        # Display results in a refined layout
        if results:
            # Create columns for a cleaner, grid-like layout
            cols = st.columns(2)
            for i, prop in enumerate(results[:10]): # Display top 10 results
                col = cols[i % 2]
                with col:
                    # Use a container with a border for each result card
                    with st.container():
                        st.subheader(f"{i+1}. {prop.get('title', 'N/A')}")
                        
                        price = prop.get('price', 0)
                        if price > 10000000: # If price is in crores
                            price_display = f"₹{price/10000000:.2f} Cr"
                        else: # Otherwise, display in lakhs
                            price_display = f"₹{price/100000:.2f} Lakh"
                        
                        st.markdown(f"**Price:** {price_display}")
                        
                        location = prop.get('location', {})
                        st.markdown(f"**Location:** {location.get('locality', 'N/A')}, {location.get('city', 'N/A')}")
                        
                        st.markdown(f"**Configuration:** {prop.get('bhk', 'N/A')} BHK")
                        
                        st.link_button("View Original Listing ↗️", prop.get('source_url', ''))
                        
                        with st.expander("See more details"):
                            st.write(prop.get('description', 'No description available.'))
                        
                        st.markdown("---") # Visual separator
        else:
            st.warning("No properties found matching your query. Our engine tried its best, but you might want to try a broader search.")
            
elif not engine:
    st.error("The search engine could not be initialized. Please check the console for errors.")

