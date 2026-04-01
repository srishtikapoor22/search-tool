from pypdf import PdfReader
from transformers import pipeline


#text extraction from pdf
def extract(file_path):
    reader=PdfReader(file_path)
    text=""

    for i in range(1, len(reader.pages)):
        page_text=reader.pages[i].extract_text()
        if page_text:
            text += " ".join(page_text.split()) + "\n\n"
    return text    

#text chunking
def chunking(raw_text):
    paragraph=raw_text.split("\n")
    chunks=[]
    for para in paragraph:
        #remove whitespaces
        para=para.strip()

        if "abstract" in para.lower()[:100]: # Only check the start of the paragraph
            continue

        if len(para)>150:
            chunks.append(para)
    return chunks

#search
def pdf_search(chunks,query,threshhold=0.1):
    print("Loading model")
    classifier=pipeline("zero-shot-classification",model="facebook/bart-large-mnli")

    results=[]
    for chunk in chunks:
        result=classifier(chunk,candidate_labels=[query])
        score=result["scores"][0]

        if score>threshhold:
            results.append({
                "score": score,
                "text": chunk
            })

    results.sort(key=lambda x:x["score"],reverse=True)

    if not results:
        print("No relevant results found. Try lowering the threshold.")
            
    else:
        print(f"\nFound {len(results)} matches. Showing Top 5:\n")
    for i, match in enumerate(results[:5]):
        print(f"RANK {i+1} | CONFIDENCE: {round(match['score'], 4)}")
        print(f"CONTENT: {match['text'][:600]}...") # Limit text length for readability
        print("-" * 50)

if __name__=="__main__":
    file_path=input("Enter the path of the PDF file: ")
    query=input("Enter your search query: ")
    raw_text=extract(file_path)
    chunks=chunking(raw_text)
    search_results=pdf_search(chunks,query)