# AI Text Summarizer

A Python application that summarizes large texts using modern NLP. It supports both **abstractive** (Transformer-based) and **extractive** (LexRank) summarization, batch processing, an optional Streamlit web UI, and ROUGE evaluation.

---

## Features

- Abstractive summarization using **t5-small**
- Extractive summarization using LexRank (Sumy)
- Batch summarization of multiple `.txt` files
- Optional Streamlit web interface
- ROUGE metrics (ROUGE-1/2/Lsum) evaluation
- Clear, well-structured Python code

---

## Setup

1.**Clone the repo**

Open your terminal or command prompt and run:

```bash
git clone https://github.com/sneha-2x/ai-text-summarizer.git
```
2.**Set the path**
```bash
cd ai_text_summarizer
```
3. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

4. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

> The first run of an abstractive model will download pre-trained weights.

---

## Usage (CLI)

### Single text or file
```bash
python summarizer.py --input "path/to/article.txt" --model bart --min-length 60 --max-length 200
```
Or pass raw text:
```bash
python summarizer.py --text "Paste your long text here..." --model t5 --max-length 130
```

### Extractive mode
```bash
python summarizer.py --input article.txt --extractive --sentences 4
```

### Batch folder of .txt files
```bash
python summarizer.py --batch-dir data/input_texts --out data/summaries --model bart --max-length 180
```
This writes one `.summary.txt` per input file.

### Compare models quickly
```bash
python summarizer.py --input article.txt --compare
```
Prints summaries from `bart-large-cnn`, `t5-small`, and LexRank side-by-side.

---

## Streamlit Web App 
```bash
python -m streamlit run streamlit_app.py
```
- Paste text OR upload multiple `.txt` files
- Choose abstractive (BART/T5) or extractive (LexRank)
- Adjust length/beam parameters
- (Optional) Provide a reference summary to compute ROUGE

---

## ROUGE Evaluation

Evaluate candidate summaries with references using a CSV file (columns: `reference`, `candidate`):
```bash
python eval_rouge.py --csv data/pairs.csv
```

Or evaluate two folders (paired by filename):
```bash
python eval_rouge.py --refs data/references --cands data/summaries
```

---

## Model & Approach

- **Abstractive**: Uses Hugging Face `pipeline("summarization")` with 
  - `t5-small` – lightweight model for faster inference
  Long texts are automatically chunked to fit model context, summarized per-chunk, and then lightly compressed.

- **Extractive**: LexRank (unsupervised) selects the most central sentences using sentence similarity graphs.

- **Evaluation**: ROUGE-1/2/Lsum via `rouge-score` to compare summaries against human-written references.

---

## Repository Structure

```text
ai_text_summarizer/
├── README.md
├── requirements.txt
├── summarizer.py
├── streamlit_app.py
└── eval_rouge.py
```

---

## Notes / Tips

- GPU is optional; CPU works but will be slower for large batches.
- For very long documents, consider increasing `--max-length` a bit and using `--num-beams 4` or `6` for quality (slower).
- Summaries are aimed at **3–5 sentences** by default. Tune `--min-length`, `--max-length`, and `--sentences` for your needs.
