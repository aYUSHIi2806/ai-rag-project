import streamlit as st

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
            if q in document_text.lower():
                answer = "Found in document:\n\n" + document_text[:500]
            else:
                answer = "Sorry, I couldn't find a relevant answer."

        st.write("🤖 Answer:")
        st.write(answer)