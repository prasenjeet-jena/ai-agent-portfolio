import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

# LangChain components for chunking and embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# ChromaDB for our local vector database
import chromadb

# ==============================================================================
# Step 0: Setup and Authentication
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
env_path = os.path.join(project_root, ".env")

load_dotenv(dotenv_path=env_path)

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# ==============================================================================
# Step 1: Defined the URLs we want to scrape
# ==============================================================================
github_docs_urls = [
    "https://docs.github.com/en/repositories",
    "https://docs.github.com/en/pull-requests",
    "https://docs.github.com/en/actions",
    "https://docs.github.com/en/authentication",
    "https://docs.github.com/en/organizations"
]

# ==============================================================================
# Step 2: Initialize Chunking Configuration
# ==============================================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# ==============================================================================
# Step 3: Initialize Embeddings
# ==============================================================================
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# ==============================================================================
# Step 4: Initialize the Vector Database (ChromaDB)
# ==============================================================================
db_path = os.path.join(current_dir, "data", "chroma_db")
os.makedirs(db_path, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=db_path)

# ==============================================================================
# Helper Function for Deep Scraping
# ==============================================================================
def get_valid_links(soup, base_url):
    """
    Finds all valid documentation links on the current page to visit next.
    We only keep links that stay within the GitHub docs domain.
    """
    valid_links = []
    # Find every anchor tag with an href attribute
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Convert relative URLs (like /en/actions/overview) to full URLs
        full_url = urljoin(base_url, href)
        # Remove any anchors (e.g., #some-section) so we don't scrape the same page twice
        full_url = full_url.split('#')[0]
        
        parsed_url = urlparse(full_url)
        # Only keep links going to docs.github.com/en
        if parsed_url.netloc == "docs.github.com" and parsed_url.path.startswith("/en"):
            valid_links.append(full_url)
            
    return valid_links

# ==============================================================================
# Main Pipeline Logic
# ==============================================================================
def run_ingestion():
    all_chunks_data = []

    print("\n🚀 Starting GitHub Document Ingestion Pipeline...\n")
    
    # 0. Delete existing collection to start fresh
    try:
        chroma_client.delete_collection(name="github_docs")
        print("🗑️  Deleted existing ChromaDB collection 'github_docs' to start fresh.")
    except ValueError:
        # It's totally fine if the collection doesn't exist yet!
        pass
        
    # Create a fresh collection
    collection = chroma_client.create_collection(name="github_docs")

    # Tracking for our "deep" scrape
    visited_urls = set()
    urls_to_visit = list(github_docs_urls)  # Start with our seed URLs
    max_pages = 50
    page_count = 0

    # We loop until we either run out of links, or hit our 50 page limit
    while urls_to_visit and page_count < max_pages:
        url = urls_to_visit.pop(0)
        
        # Deduplication: Don't scrape the same page twice!
        if url in visited_urls:
            continue
            
        visited_urls.add(url)
        page_count += 1
        
        print(f"📄 Scraping page {page_count}/{max_pages}: {url}")
        
        # Fetch the webpage
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code != 200:
            print(f"⚠️ Failed to fetch {url}. Status code: {response.status_code}")
            continue
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Before we destroy the DOM to get the text, find sub-pages for future visits
        new_links = get_valid_links(soup, url)
        for link in new_links:
            # We only add links to the queue if we haven't visited them already, and they aren't already queued
            if link not in visited_urls and link not in urls_to_visit:
                urls_to_visit.append(link)
        
        # 2. Clean the webpage to extract only readable content
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
            
        main_content = soup.find('main')
        if not main_content:
            main_content = soup.find('body')
            
        if not main_content:
            print(f"⚠️ Could not find parseable text for {url}")
            continue
            
        raw_text = main_content.get_text(separator=' ', strip=True)
        
        # 3. Chunk the text
        chunks = text_splitter.split_text(raw_text)
        print(f"   ✂️ Created {len(chunks)} chunks.")
        
        for chunk in chunks:
            all_chunks_data.append({
                "text": chunk,
                "url": url
            })

    # Prepare data for insertion into ChromaDB
    documents = []
    metadatas = []
    ids = []
    
    for i, data in enumerate(all_chunks_data):
        documents.append(data["text"])
        metadatas.append({"source": data["url"]})
        ids.append(f"chunk_{i}")
        
    total_chunks_stored = len(documents)

    # 4. Embed and Store
    if total_chunks_stored > 0:
        print(f"\n🧠 Embedding and storing {total_chunks_stored} total chunks into ChromaDB... (This may take a moment)")
        
        embeddings_list = embeddings_model.embed_documents(documents)
        
        collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
        print("✅ Storage complete!\n")
    else:
        print("⚠️ No chunks were generated. Exiting.")
        return

    # ==============================================================================
    # Step 5: Verification and Testing
    # ==============================================================================
    print("==================================================")
    print("🔍 VERIFICATION REPORT")
    print("==================================================")
    
    count = collection.count()
    print(f"📊 Total chunks successfully stored in database: {count}\n")
    
    print("📝 Printing 3 sample chunks:")
    for i in range(min(3, len(documents))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Source URL: {metadatas[i]['source']}")
        print(f"Text Snippet: {documents[i][:150]}...")
        
    print("\n--------------------------------------------------")
    
    test_query = "how do I protect a branch"
    print(f"🎯 Running Test Search Query: '{test_query}'")
    
    query_embedding = embeddings_model.embed_query(test_query)
    
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print("\n🏆 Top 3 Most Similar Results:")
    for j in range(len(search_results['documents'][0])):
        result_text = search_results['documents'][0][j]
        result_url = search_results['metadatas'][0][j]['source']
        print(f"\nResult {j+1} (Source: {result_url}):")
        print(f"{result_text[:200]}...")

if __name__ == "__main__":
    run_ingestion()
