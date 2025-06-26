import streamlit as st
import requests

st.title(" Adaptive Learning")

pdf_file = st.file_uploader("Upload your learning content PDF", type=["pdf"])
num_questions = st.number_input("How many MCQs per skill?", min_value=1, max_value=5, value=2)

if st.button("Generate MCQs") and pdf_file:
    with st.spinner(" Generating..."):
        try:
            response = requests.post(
                f"http://localhost:8000/run-crew/?num_questions={num_questions}",
                files={"file": (pdf_file.name, pdf_file, "application/pdf")},
            )

            if response.status_code != 200:
                st.error(f" Error {response.status_code}: {response.json().get('detail')}")
            else:
                result = response.json()
                st.success(" MCQs generated successfully!")

                st.write(f"**Total Skills:** {result['total_skills']}")
                st.write(f"**Expected MCQs:** {result['total_mcqs_expected']}")
                st.write(f"**Actual MCQs:** {result['actual_mcqs']}")
                st.write("---")

                for q in result["mc_questions"]:
                    st.markdown(f"**Q:** {q['question']}")
                    for i, a in enumerate(q['answers'], 1):
                        st.markdown(f"{i}. {a}")
                    st.markdown(f"**Topic:** {q['topic']} | **Difficulty:** {q['difficulty']}")
                    st.write("---")

        except Exception as e:
            st.error(f" Failed to contact the backend: {str(e)}")
