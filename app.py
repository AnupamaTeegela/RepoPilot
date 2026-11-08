
import streamlit as st

from backend import (
    clone_repository,
    extract_code_documents,
    split_code_documents,
    create_vector_store,
    answer_question,
    generate_project_overview
)

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RepoPilot - AI Codebase Assistant",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------
# 2. Initialize Session State
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "repo_loaded" not in st.session_state:
    st.session_state.repo_loaded = False

if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""

if "project_overview" not in st.session_state:
    st.session_state.project_overview = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# 3. Application Heading
# --------------------------------------------------

st.title("🚀 RepoPilot")

st.subheader("AI-Powered Codebase Intelligence Assistant")

st.write(
    "Understand any public GitHub repository using "
    "AI-powered code search and analysis."
)

st.divider()

# --------------------------------------------------
# 4. Connect GitHub Repository
# --------------------------------------------------

st.header("📂 Connect GitHub Repository")

repo_url = st.text_input(
    "Enter Public GitHub Repository URL",
    placeholder="https://github.com/username/repository"
)

if st.button("🔍 Analyze Repository", type="primary"):

    if not repo_url.strip():

        st.warning("Please enter a GitHub repository URL.")

    else:

        try:
            with st.spinner("Cloning repository..."):
                repo_path = clone_repository(repo_url.strip())

            with st.spinner("Extracting source code..."):
                documents = extract_code_documents(repo_path)

            if not documents:
                raise ValueError(
                    "No supported source code files were found."
                )

            with st.spinner("Splitting code into chunks..."):
                chunks = split_code_documents(documents)

            st.write("### Repository Processing Details")

            st.write("**Total files:**", len(documents))
            st.write("**Total code chunks:**", len(chunks))

            st.write("### Files Detected")

            for document in documents:
                st.write("📄", document["file_path"])

            with st.spinner(
                "Generating embeddings and building FAISS database..."
            ):
                vector_store = create_vector_store(chunks)

            # Save repository details
            st.session_state.vector_store = vector_store
            st.session_state.repo_loaded = True
            st.session_state.repo_url = repo_url.strip()

            # Clear previous repository's chat history
            st.session_state.chat_history = []
            st.session_state.project_overview = ""

            st.success("Repository processed successfully!")

            # Generate Project Overview
            with st.spinner("Generating project overview..."):
                overview = generate_project_overview(vector_store)

            st.session_state.project_overview = overview

            st.success("Project overview generated successfully!")

            st.write(f"**Files processed:** {len(documents)}")
            st.write(f"**Code chunks created:** {len(chunks)}")

        except Exception as error:

            st.error(f"Repository analysis failed: {error}")

# --------------------------------------------------
# 5. Project Overview
# --------------------------------------------------

st.divider()

if st.session_state.repo_loaded:

    st.header("📊 Project Overview")

    st.markdown(
        st.session_state.project_overview
        or "Project overview is not available."
    )

st.divider()

# --------------------------------------------------
# 6. Ask Questions About the Codebase
# --------------------------------------------------

st.header("💬 Ask Questions About Your Codebase")

if st.session_state.repo_loaded:

    st.success(
        f"Connected Repository: {st.session_state.repo_url}"
    )

    question = st.text_input(
        "Ask RepoPilot anything about the repository",
        placeholder="Where is authentication implemented?"
    )

    if st.button("🤖 Ask RepoPilot"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            try:
                with st.spinner(
                    "Searching code and generating answer..."
                ):

                    result = answer_question(
                        st.session_state.vector_store,
                        question
                    )

                st.subheader("🧠 AI Generated Answer")

                st.write(result["answer"])

                st.subheader("📁 Source Files")

                for source in result["sources"]:
                    st.code(source)

                # Save successful question and answer
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"]
                })

            except Exception as error:

                st.error(
                    f"Unable to generate answer: {error}"
                )

else:

    st.info(
        "Please connect and analyze a GitHub repository "
        "before asking questions."
    )

# --------------------------------------------------
# 7. Chat History
# --------------------------------------------------

st.divider()

st.header("🕘 Chat History")

if st.session_state.chat_history:

    for index, chat in enumerate(
        st.session_state.chat_history,
        start=1
    ):

        with st.expander(
            f"Question {index}: {chat['question']}"
        ):

            st.write("**Answer:**")
            st.write(chat["answer"])

else:

    st.info("No questions asked yet. Start chatting with RepoPilot!")

# --------------------------------------------------
# 8. Footer
# --------------------------------------------------

st.divider()

# st.caption(
#     "RepoPilot | Built with Python, LangChain, Groq, FAISS and Streamlit"
# )
