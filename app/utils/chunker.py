import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')

def chunk_text(text, max_tokens=512, overlap=50):
    """
    Chunk teks berdasarkan kalimat dengan overlap.
    Hitung token berdasarkan kata (kasar).
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        token_count = len(sent.split())
        if current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_length = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sent)
        current_length += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
