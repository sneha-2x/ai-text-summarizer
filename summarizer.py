# summarizer.py
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import PyPDF2

# ---------------- NLTK setup (offline-ready) ----------------
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

# ---------------- Abstractive summarization ----------------
DEFAULT_MODEL = "facebook/bart-large-cnn"
ALT_MODEL = "t5-small"

def load_abstractive(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=False)
    pipe = pipeline("summarization", model=mdl, tokenizer=tok)
    return tok, mdl, pipe

def chunk_text(tokenizer, text: str, max_input_tokens: int = 950):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_input_tokens):
        chunk_tokens = tokens[i:i+max_input_tokens]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
    return chunks if chunks else [""]

def abstractive_summary(text: str, model_name: str = DEFAULT_MODEL,
                        min_length: int = 60, max_length: int = 180) -> str:
    tok, mdl, pipe = load_abstractive(model_name)
    context_limit = 1024 if "bart" in model_name else 512
    chunks = chunk_text(tok, text, max_input_tokens=int(0.9 * context_limit))
    partials = []
    for ch in chunks:
        out = pipe(ch, min_length=min_length, max_length=max_length)[0]["summary_text"]
        partials.append(out.strip())
    combined = " ".join(partials)
    if len(partials) > 1:
        out2 = pipe(combined, min_length=min_length, max_length=max_length)[0]["summary_text"]
        return out2.strip()
    return combined.strip()

# ---------------- Extractive summarization ----------------
def extractive_summary(text: str, sentences: int = 4) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    sents = summarizer(parser.document, sentences)
    return " ".join(str(s) for s in sents).strip()

# ---------------- PDF reading ----------------
def read_pdf_streamlit(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# ---------------- Main summarizer ----------------
def summarize(text: str, mode: str = "abstractive", model: str = DEFAULT_MODEL,
              min_length: int = 60, max_length: int = 180, sentences: int = 4) -> str:
    if mode == "extractive":
        return extractive_summary(text, sentences=sentences)
    else:
        return abstractive_summary(text, model_name=model, min_length=min_length, max_length=max_length)
