import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
from collections import defaultdict
from heapq import nlargest
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Remove unwanted characters, numbers, and extra spaces
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^A-Za-z0-9.,!?\'\n]', ' ', text)  # Remove special characters except common punctuation
    
    # Convert to lowercase
    text = text.lower()
    return text

def process_text(text, summary_percentage=0.3):
    # Preprocessing
    text = preprocess_text(text)
    
    # Tokenization
    tokenized_word = word_tokenize(text)
    
    # Stop words
    stop_words = set(stopwords.words('english'))
    additional_stopwords = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
    }
    stop_words.update(additional_stopwords)
    
    # Punctuation
    punctuation = set(string.punctuation)
    punctuation.add('\n')
    
    # Word frequencies
    word_frequencies = defaultdict(int)
    for w in tokenized_word:
        if w not in stop_words and w not in punctuation:
            word_frequencies[w] += 1
            
    # Normalize frequencies
    max_frequency = max(word_frequencies.values()) if word_frequencies else 1
    for word in word_frequencies:
        word_frequencies[word] /= max_frequency
        
    # Sentence tokenization and scoring
    tokenized_sentence = sent_tokenize(text)
    
    sentence_scores = {}
    for sent in tokenized_sentence:
        words = word_tokenize(sent)
        score = sum(word_frequencies.get(word, 0) for word in words)
        sentence_scores[sent] = score
        
    # Select top sentences
    select_length = max(1, int(len(tokenized_sentence) * summary_percentage))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    
    return summary, tokenized_word, tokenized_sentence

# Streamlit UI
st.set_page_config(page_title="📄 Document Summarizer", page_icon="📄", layout="centered")

# Custom CSS for better styling
st.markdown("""
<style>
    .summary-box {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: #ffffff;
    }
    .method-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    .method-card:hover {
        transform: translateY(-5px);
    }
    .explanation-box {
        background-color: #0e1117;
        border-left: 3px solid #00ff00;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("📄 Advanced Document Summarizer")
st.markdown("""
Welcome to the **Advanced Document Summarizer**! Before you begin, learn about our summarization methods:
""")

# Interactive Method Explanation Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="method-card">
        <h3>📋 Extractive Summarization</h3>
        <p>Click to learn more about this method</p>
    </div>
    """, unsafe_allow_html=True)
    show_extractive = st.button("Learn About Extractive")
    
with col2:
    st.markdown("""
    <div class="method-card">
        <h3>🤖 Abstractive Summarization</h3>
        <p>Click to learn more about this method</p>
    </div>
    """, unsafe_allow_html=True)
    show_abstractive = st.button("Learn About Abstractive")

if show_extractive:
    st.markdown("""
    <div class="explanation-box">
        <h4>Extractive Summarization 📋</h4>
        <p>This method works by:</p>
        <ul>
            <li>Identifying key sentences from the original text</li>
            <li>Preserving the exact wording of the source material</li>
            <li>Using advanced algorithms to score sentence importance</li>
            <li>Maintaining context and accuracy of information</li>
        </ul>
        <p><strong>Best for:</strong> Technical documents, research papers, and factual content where precise wording is important.</p>
    </div>
    """, unsafe_allow_html=True)

if show_abstractive:
    st.markdown("""
    <div class="explanation-box">
        <h4>Abstractive Summarization 🤖</h4>
        <p>This method works by:</p>
        <ul>
            <li>Generating new sentences that capture key ideas</li>
            <li>Rephrasing content in a more concise way</li>
            <li>Using AI to understand and reformulate information</li>
            <li>Creating more natural, human-like summaries</li>
        </ul>
        <p><strong>Best for:</strong> News articles, stories, and content where natural flow is priority.</p>
        <p><em>Coming soon!</em></p>
    </div>
    """, unsafe_allow_html=True)

# Main Interface
st.header("Input Your Text")
text_input = st.text_area("Paste your text here:", height=200, placeholder="Enter or paste the text you want to analyze...")

if text_input:
    # Analysis Options
    st.header("Text Analysis Tools")
    
    col1, col2 = st.columns(2)
    with col1:
        show_tokens = st.button("👀 Show Word Tokens")
    with col2:
        show_sentences = st.button("📝 Show Sentences")
    
    # Process text for tokens and sentences if requested
    summary_percentage = st.slider("Summary Length (% of original)", min_value=10, max_value=50, value=30) / 100
    _, tokens, sentences = process_text(text_input, summary_percentage)
    
    if show_tokens:
        with st.expander("Word Tokens", expanded=True):
            st.write(tokens)
            
    if show_sentences:
        with st.expander("Sentence Tokens", expanded=True):
            for i, sent in enumerate(sentences, 1):
                st.write(f"{i}. {sent}")
    
    # Summarization Options
    st.header("Generate Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        extractive_summary = st.button("📋 Generate Extractive Summary")
    with col2:
        abstractive_summary = st.button("🤖 Generate Abstractive Summary")
    
    if extractive_summary:
        with st.spinner("Generating extractive summary..."):
            summary, _, _ = process_text(text_input, summary_percentage)
            summary_text = " ".join(summary)
            
            st.markdown("### Your Summary")
            st.markdown(f'<div class="summary-box">{summary_text}</div>', unsafe_allow_html=True)
            
            # Summary Statistics in expandable section
            with st.expander("📊 View Summary Statistics"):
                original_words = len(text_input.split())
                summary_words = len(summary_text.split())
                reduction = round((1 - summary_words/original_words) * 100, 1)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Words", original_words)
                col2.metric("Summary Words", summary_words)
                col3.metric("Reduction", f"{reduction}%")
            
            # Download and Copy options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download Summary", summary_text, file_name="summary.txt", mime="text/plain")
            with col2:
                st.markdown(f"""
                <button onclick="navigator.clipboard.writeText(`{summary_text}`)" style="
                    background-color: #1f77b4;
                    border: none;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    font-size: 16px;
                    cursor: pointer;
                    border-radius: 4px;
                    width: 100%;">
                    📋 Copy Summary
                </button>
                """, unsafe_allow_html=True)
                
    elif abstractive_summary:
        st.info("🚧 Abstractive summarization is currently under development. Stay tuned for updates! 🔜")

else:
    st.info("👆 Start by pasting your text in the box above!")