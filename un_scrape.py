from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
import math
import requests
import os
import fitz

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from read_pdf import read_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import conversational_retrieval
import fitz

# Set up the Selenium WebDriver
driver = webdriver.Chrome()  

# Open the web page
driver.get('https://documents.un.org/')

# Allow the page to load
time.sleep(2) 

# Use Selenium to find the input field and button
input_field = driver.find_element(By.ID, 'title') 
print(input_field)
button = driver.find_element(By.ID, 'btnSearch') 

input_field.clear() 
input_field.send_keys('tamil')  

# Scroll to the button
driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)

time.sleep(1)

# Click the button
button.click()

time.sleep(2)

first_span = driver.find_element(By.CSS_SELECTOR, '.search-criteria > span')

pagenum = int(first_span.find_elements(By.TAG_NAME, "b")[-1].text)

def read_pdf(pdf_file):
    with fitz.open(stream=pdf_file, filetype="pdf") as file:
        return [page.get_text().lower() for page in file]


# Allow time for the page to update or navigate 
def next():
    """
    Executes button click to next page of resources
    """

    span_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//span[@title='Navigate to next page']"))
    )

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    time.sleep(1)

    span_element.click()

    time.sleep(2)

# Use BeautifulSoup to parse the updated page content


links = []
only_links = []
def linkpull():
    """
    pulls all links from active page
    """
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    search_items = soup.find_all('div', class_='search-results-item')
    count = 0
    for search_item in search_items:
        
        # Find all <a> elements with the class 'icofont-ui-file' within the container
        symbol = search_item.find("div", class_="symbol")
        container = symbol.find("div", class_="text-align-container")
        link = container.find('a', class_='icofont-ui-file')
        
        link_names = search_item.find_all('h2')
        if not link in only_links:
        # Loop through each <a> element and get the href attribute
            links.append([link_names[-1].text, link.get('href')])
            only_links.append(link)
            count += 1
    print(count)
        

        

linkpull()
max = math.ceil(pagenum/20)-1
if max > 25:
    max = math.ceil(max/2)
    
for page in range(2):
    next()
    linkpull()

print(links)
print(len(links))

import ollama
import chromadb

from chromadb.config import Settings

client = chromadb.PersistentClient(path=".", settings=Settings(allow_reset=True))
collection = client.create_collection(name="un_docs")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
 


metadata = []
doc_id = 0
for link in links:
    print("here is download")
    url = link[-1]
    print(url)
    name = link[0]
    name = name.split("/")[0]
    print(name)
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    # Check if the request was successful
    if response.status_code == 200:
        # Open a file to write the PDF content
        doc_content = read_pdf(response.content)
        print("PDF downloaded successfully!")

        # Choose a batch size based on API capability and memory constraints
        
        batch_docs = []  # Stores text chunks for each batch
        batch_metadata = []  # Stores metadata for each chunk
        batch_ids = []  # Stores unique IDs for each chunk
        batch_embedding = []

        # Example loop through each document and its chunks
        for p, page_content in enumerate(doc_content):
            documents = text_splitter.split_text(page_content)
            
            for i, d in enumerate(documents):
                response = ollama.embeddings(model="nomic-embed-text", prompt=d)
                embedding = response["embedding"]
                batch_embedding.append(embedding)
                if not embedding:
                    print(f"Embedding failed for chunk {i}; skipping.")
                    continue
                # Add text chunk and metadata to the batch
                batch_docs.append(d)
                batch_metadata.append({
                    "title": name,
                    "link": url,
                    "chunk_index": i,
                    "page": p + 1
                })
                batch_ids.append(f"{doc_id}_{p+1}_{i}")

                # When the batch is full, send it for embedding
        
        try:
            
            # Add the entire batch to ChromaDB
            collection.add(
                ids=batch_ids,
                embeddings=batch_embedding,
                documents=batch_docs,
                metadatas=batch_metadata
            )

            # Clear the batch lists for the next set
            batch_docs.clear()
            batch_metadata.clear()
            batch_ids.clear()
            batch_embedding.clear()

        except Exception as e:
            print(f"Batch embedding error: {e}")
            continue
        
        # Ensure any remaining chunks are processed after loop completes
        

    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")
    doc_id += 1
# Close the WebDriver
print("hi")
print(collection.count())

driver.quit()