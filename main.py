import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings  # UPDATED import!
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import re
import requests
from urllib.parse import quote

from transformers import pipeline
import re

# Translation pipeline (NLLB-200)
translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="eng_Latn",
    tgt_lang="ory_Orya",
    max_length=400
)

def translate_to_odia(text, max_len=400):
    """
    Translates English text to Odia using NLLB-200 pipeline.
    Splits long text into sentences for better quality.
    """
    # Split text into sentences by period/question/exclamation
    # You can adjust this logic for your needs.
    sentences = re.split(r'(?<=[.?!])\s+', text)
    translated = []
    for sent in sentences:
        sent = sent.strip()
        if sent:
            try:
                # Truncate for NLLB context limit, if needed
                chunk = sent[:max_len]
                odia = translator(chunk)[0]['translation_text']
                translated.append(odia)
            except Exception as e:
                translated.append(f"[Trans Error: {str(e)}]")
    return ' '.join(translated)

# Initialize global variables
retriever = None
rag_application = None
document_text = None
pdf_name = None

# Function to load the PDF and initialize the RAG application
def load_pdf(file):
    global retriever, rag_application, document_text, pdf_name

    if not file:
        return "❌ No PDF uploaded", ""

    try:
        time.sleep(1)

        loader = PyPDFLoader(file.name)
        docs = loader.load()
        document_text = "\n".join([doc.page_content for doc in docs])
        pdf_name = file.name.split("/")[-1]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250, chunk_overlap=50
        )

        doc_splits = text_splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits, embedding=embedding
        )

        retriever = vectorstore.as_retriever(k=4)

        llm = ChatOllama(model="llama3", temperature=0)

        prompt = PromptTemplate(
            template="""Based on the content of the uploaded research paper as document, \
            provide a clear, concise, and accurate answer to the question below. \
            If necessary, include relevant details, key findings, and context from the paper to support the answer. \
            If the paper does not provide a specific answer, mention that the answer is not explicitly covered.\n\n            Question: {question}\n            Documents: {documents}\n            Answer:\n            """,
            input_variables=["question", "documents"],
        )

        rag_chain = prompt | llm | StrOutputParser()
        rag_application = RAGApplication(retriever, rag_chain)

        return "✅ PDF loaded successfully!", f"**Loaded File:** {pdf_name}"
    except Exception as e:
        return f"❌ Error loading PDF: {str(e)}", ""

# Function to summarize the paper
def summarize_paper():
    global document_text

    if not document_text:
        return "Please upload a PDF first!"

    try:
        prompt = (
            "Summarize the following research paper with focus on these points below:\n\n"
            "1. Short overview\n"
            "2. Methodology used\n"
            "3. Results\n"
            "4. Conclusion\n\n"
            f"Paper Content:\n{document_text[:7000]}"
        )

        llm = ChatOllama(model="llama3", temperature=0)
        response = llm.invoke(prompt)
        summary_english = response.content.strip()

        # Odia translation (sentence-wise)
        summary_odia = translate_to_odia(summary_english)

        return f"**English Summary:**\n{summary_english}\n\n**Odia Summary (ଓଡ଼ିଆ ସାରାଂଶ):**\n{summary_odia}"

    except Exception as e:
        return f"❌ Error generating summary: {str(e)}"

# Function to extract keywords using the LLM
def extract_keywords():
    global document_text

    if not document_text:
        return "Please upload a PDF first!"

    try:
        prompt = (
            "Extract the 5 most relevant keywords or key phrases from the following research paper content. "
            "Only provide the keywords separated by commas, without any additional text.\n\n"
            f"Paper Content:\n{document_text[:5000]}"
        )

        llm = ChatOllama(model="llama3", temperature=0)
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"❌ Error extracting keywords: {str(e)}"

# Function to generate IEEE-style citation using DOI
def generate_citation():
    global document_text

    if not document_text:
        return "Please upload a PDF first!"

    try:
        doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", document_text, re.IGNORECASE)
        if not doi_match:
            return "No DOI found in the document."

        doi = doi_match.group().strip()
        headers = {"Accept": "application/vnd.citationstyles.csl+json"}
        response = requests.get(f"https://doi.org/{doi}", headers=headers)

        if response.status_code != 200:
            return f"Failed to retrieve citation from DOI: {doi}"

        metadata = response.json()
        authors = metadata.get("author", [])
        title = metadata.get("title", ["No Title"])[0]
        container_title = metadata.get("container-title", ["Unknown Publication"])[0]
        year = metadata.get("issued", {}).get("date-parts", [[None]])[0][0]

        if authors:
            author_names = ", ".join([
                f"{author.get('given', 'Unknown')} {author.get('family', 'Author')}".strip()
                for author in authors
            ])
        else:
            author_names = "Unknown Author"

        citation = f"[1] {author_names}, \"{title}\", *{container_title}*, {year}. DOI: {doi}"
        return citation
    except Exception as e:
        return f"❌ Error generating citation: {str(e)}"

# Function to search for related papers using Semantic Scholar API
def find_related_papers():
    keywords = extract_keywords()
    if "❌" in keywords or "Please upload" in keywords:
        return keywords

    try:
        query = quote("+".join(keywords.split(", ")))
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,url&limit=5"
        response = requests.get(url)

        if response.status_code != 200:
            return f"❌ Error fetching related papers: {response.status_code}"

        papers = response.json().get("data", [])
        if not papers:
            return "No related papers found."

        result = "\n".join([f"- [{paper['title']}]({paper['url']})" for paper in papers])
        return result
    except Exception as e:
        return f"❌ Error finding related papers: {str(e)}"

# RAG Application Class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        documents = self.retriever.invoke(question)
        doc_texts = "\n".join([doc.page_content for doc in documents])
        return self.rag_chain.invoke({"question": question, "documents": doc_texts})

# New: Q/A handler with Odia translation!
def answer_question_in_both_langs(question):
    if not rag_application:
        return "Please upload a PDF first!"
    answer_en = rag_application.run(question)
    answer_or = translate_to_odia(answer_en)
    return f"**English:**\n{answer_en}\n\n**Odia (ଓଡ଼ିଆ):**\n{answer_or}"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="lime")) as app:
    gr.Markdown("""
    # 📄 **Research Paper Analysis Tool**
    Upload a research paper in PDF format and get summaries (now in Odia & English!), citations, keywords, related papers, and answers to your questions in both languages.
    """)

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF", type="filepath")
            load_btn = gr.Button("📤 Upload Document")
        with gr.Column():
            load_output = gr.Textbox(label="Status", interactive=False)

    load_btn.click(load_pdf, inputs=pdf_input, outputs=[load_output], show_progress="full")

    with gr.Tab("🔍 Ask Questions"):
        question_input = gr.Textbox(label="❓ Ask a Question", placeholder="Type your question here...")
        ask_btn = gr.Button("🔍 Get Answer")
        answer_output = gr.Textbox(label="Answer", lines=12, max_lines=18, interactive=False)
        ask_btn.click(answer_question_in_both_langs, inputs=question_input, outputs=answer_output)

    with gr.Tab("📝 Summarization"):
        summarize_btn = gr.Button("📝 Summarize Paper")
        summary_output = gr.Textbox(label="Summary", lines=14, max_lines=22, interactive=False)
        summarize_btn.click(summarize_paper, outputs=summary_output, show_progress="full")

    with gr.Tab("🔑 Keywords"):
        keyword_btn = gr.Button("🔑 Extract Keywords")
        keyword_output = gr.Textbox(label="Keywords", interactive=False)
        keyword_btn.click(extract_keywords, outputs=keyword_output, show_progress="full")

    with gr.Tab("📚 Citation"):
        citation_btn = gr.Button("📚 Generate Citation")
        citation_output = gr.Textbox(label="Citation", interactive=False)
        citation_btn.click(generate_citation, outputs=citation_output, show_progress="full")

    with gr.Tab("🔗 Related Papers"):
        related_btn = gr.Button("🔗 Find Related Papers")
        related_output = gr.Textbox(label="Related Papers", lines=15, max_lines=20, interactive=False)
        related_btn.click(find_related_papers, outputs=related_output, show_progress="full")

app.launch(share=True)
