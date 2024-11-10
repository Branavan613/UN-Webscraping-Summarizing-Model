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
input_field.send_keys('abortion')  

# Scroll to the button
driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)

time.sleep(1) 


# Click the button
button.click()

time.sleep(2)

first_span = driver.find_element(By.CSS_SELECTOR, '.search-criteria > span')

pagenum = int(first_span.find_elements(By.TAG_NAME, "b")[-1].text)

def read_pdf(pdf_file):
    content = []
    pagenum = 1
    with fitz.open(stream = pdf_file, filetype = "pdf") as file:
        for page in file: 
            c = page.get_text()
            content.append([pagenum, c.lower()])
            pagenum += 1
    return content

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
soup = BeautifulSoup(driver.page_source, 'html.parser')

links = []

def linkpull():
    """
    pulls all links from active page
    """
    search_items = soup.find_all('div', class_='search-results-item')

    for search_item in search_items:
        # Find all <a> elements with the class 'icofont-ui-file' within the container
        symbol = search_item.find("div", class_="symbol")
        container = symbol.find("div", class_="text-align-container")
        link = container.find('a', class_='icofont-ui-file')
        
        link_names = search_item.find_all('h2')

        # Loop through each <a> element and get the href attribute
        links.append([link_names[-1].text, link.get('href')])

        

linkpull()
max = math.ceil(pagenum/20)-1
if max > 25:
    max = math.ceil(max/2)
    
for page in range(max):
    next()
    linkpull()
    
import ollama
import chromadb

from chromadb.config import Settings

client = chromadb.PersistentClient(path=".", settings=Settings(allow_reset=True))
collection = client.create_collection(name="un_docs")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
 

print(links)
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

        for p in doc_content:
            documents = text_splitter.split_text(p[1])
            
            for i, d in enumerate(documents):
                response = ollama.embeddings(model="nomic-embed-text", prompt=d)
                embedding = response["embedding"]
                
                # Create metadata dictionary
                metadata = {
                    "title": name,
                    "link": url,
                    "chunk_index": i,
                    "page": p[0]
                }
                
                # Add to collection with unique id for each chunk
                collection.add(
                    ids=[f"{doc_id}_{i}"],
                    embeddings=[embedding],
                    documents=[d],
                    metadatas=[metadata]
                )
        
        doc_id += 1
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")
# Close the WebDriver
print
driver.quit()