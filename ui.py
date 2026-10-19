import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("📄 Ask Questions About Your Document")

document_text = ""

# Upload file
uploaded_file = st.file_uploader("Upload a file", type=["txt"])

if uploaded_file:
    document_text = uploaded_file.read().decode("utf-8")
    st.success("File uploaded successfully!")

# Ask question
question = st.text_input("Ask a question about your document")

if st.button("Ask"):
    if document_text == "":
        st.error("Upload file first")
    else:
        q = question.lower()

        if "summary" in q or "summarize" in q:
            answer = document_text[:500]
        else:
            # TF-IDF aur chunking logic
            chunks = [s.strip() for s in document_text.split('.') if len(s.strip()) > 10]

            if chunks and question:
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(chunks + [question])
                sim_scores = cosine_similarity(X[-1], X[:-1])
                best_idx = sim_scores.argmax()
                
                answer = f"Most relevant chunk found:\n\n{chunks[best_idx]}"
            else:
                answer = "Sorry, I couldn't find a relevant answer."

        st.write("🤖 Answer:")
        st.write(answer)
