# Odia Research Paper Summarization Tool

This tool provides **automatic research paper summarization and question answering** in both **English and Odia (ଓଡ଼ିଆ)**. Powered by LangChain, Gradio, HuggingFace NLLB-200, and open-source LLMs, it brings inclusive GenAI to Indian languages.

---

## Features

- Upload any research paper in PDF format.
- Summarize in **English** and **Odia** side by side.
- Ask questions about the paper and get answers in **both languages**.
- Extract keywords, generate citations, and discover related works.
- User-friendly **Gradio web interface**.

---

## Setup Instructions

### 1. **Clone the Repository** (or place code in a new folder)

If using GitHub:
```bash
git clone https://github.com/sivajeet/English_Odia_Research_Summary_Tool.git
cd English_Odia_Research_Summary_Tool

## Python Environment:
Requires Python 3.9–3.12 (3.10+ recommended)
Check with: python3 --version

## Create Virtual Environment (Recommended)
python3 -m venv .venv
source .venv/bin/activate

## Install Dependencies
With requirements.txt:
pip install -r requirements.txt

## If you use local LLMs via Ollama, also:
pip install ollama

## Download Models (First Run)
On the first run, the NLLB-200 model will be downloaded automatically.

This may take a few minutes and requires internet.

## Usage
Run your main script (replace main.py if named differently):
python main.py


# Example Workflow
- Upload a research PDF.
- Click Summarize Paper. Get English & Odia summaries, side by side!
- Use Ask a Question to get answers in both languages.
- Extract Keywords, auto-generate Citation, and find Related Papers.

# Example Output
**English Summary:**
The paper presents a fine-grained taxonomy of code review feedback in TypeScript projects...

**Odia Summary (ଓଡ଼ିଆ ସାରାଂଶ):**
ପତ୍ରଟି ଟାଇପସ୍କ୍ରିପ୍ଟ ପ୍ରୋଜେକ୍ଟରେ କୋଡ୍ ସମୀକ୍ଷା ମତାମତର ସୂକ୍ଷ୍ମ ବର୍ଗୀକରଣ ପ୍ରଦାନ କରେ...


# Contact
Sivajeet Chand (Demo Author)
Email: sivajeet.chand@tum.de
For questions, file an Issue or open a Pull Request.

