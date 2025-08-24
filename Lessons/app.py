import streamlit as st
from retrieval_to_pdf import generate_pdf_lesson

st.title("📘 AI Lesson Generator")

topic = st.text_input("Enter a topic (e.g., Newton's Second Law)")
difficulty = st.selectbox("Select difficulty level", ["beginner", "intermediate", "advanced"])

if st.button("Generate Lesson PDF"):
    if topic:
        with st.spinner("Generating lesson..."):
            pdf_path = generate_pdf_lesson(topic, difficulty, output_dir="outputs")

        st.success("✅ Lesson generated!")
        with open(pdf_path, "rb") as f:
            st.download_button("Download Lesson PDF", f, file_name=pdf_path.split("/")[-1])
    else:
        st.warning("Please enter a topic first.")
