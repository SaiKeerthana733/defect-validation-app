import re
import string
import spacy
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Build stopword list but keep domain-relevant words
EN_STOP = set(stopwords.words("english"))
DOMAIN_KEEP = {"error", "bug", "issue", "null", "exception", "timeout", "crash", "login", "payment"}
EN_STOP = EN_STOP.difference(DOMAIN_KEEP)

stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, remove punctuation, numbers, extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\[.*?\]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def preprocess_text(text: str) -> str:
    """Full preprocessing: clean, tokenize, remove stopwords, lemmatize, stem."""
    base = clean_text(text)
    doc = nlp(base)
    tokens = []
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        lemma = t.lemma_.strip().lower()
        if lemma and lemma not in EN_STOP and len(lemma) > 1:
            tokens.append(stemmer.stem(lemma))
    return " ".join(tokens)