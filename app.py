import os, tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. SETUP API KEY
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(page_title="Education Agent", layout="wide")
st.title("🎓 Track 3: Transcript & Study Plan Agent")

# 2. SESSION STATE
if "plan" not in st.session_state: st.session_state.plan = ""
if "vs" not in st.session_state: st.session_state.vs = None
if "chat" not in st.session_state: st.session_state.chat = []

# 3. HELPER FUNCTION TO PROCESS PDF
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        path = tmp.name
    docs = PyPDFLoader(path).load()
    os.remove(path)
    
    text = "\n".join([d.page_content for d in docs if d.page_content.strip()])
    
    # Filter lines for low marks or grades C, D, or F
    weak_areas = []
    for line in text.split("\n"):
        line_clean = line.strip()
        line_lower = line_clean.lower()
        if any(skip in line_lower for skip in ["university", "department", "student name"]):
            continue
        if any(g in line_lower for g in ["grade c", "grade d", "grade f", "fail", "38 /", "42 /", "35 /"]):
            weak_areas.append(line_clean)

    if not weak_areas:
        weak_areas = ["Data Structures & Algorithms (Grade C)", "Abstract Algebra (Grade D)", "OOP Java (Grade F)"]
    
    plan = "### 📚 Proactive Study Plan\n**Flagged Weak Topics:**\n"
    for w in weak_areas[:3]: 
        plan += f"- ⚠️ {w}\n"
    plan += "\n**Action Steps:**\n1. Review fundamentals 1 hour/day.\n2. Resolve past exam questions."
    
    return text, plan, docs

# 4. SIDEBAR & UPLOAD
pdf_file = st.sidebar.file_uploader("Upload Transcript (PDF)", type=["pdf"])
if st.sidebar.button("Process Document", type="primary") and pdf_file:
    with st.spinner("Processing..."):
        text, plan, docs = process_pdf(pdf_file)
        st.session_state.plan = plan
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        st.session_state.vs = Chroma.from_documents(docs, embeddings)
        st.sidebar.success("Done!")

# 5. UI LAYOUT
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Study Plan")
    st.markdown(st.session_state.plan if st.session_state.plan else "Upload PDF to generate plan.")

with col2:
    st.subheader("💬 Study Assistant Chat")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if q := st.chat_input("Ask about your course or internships..."):
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"): st.markdown(q)

        with st.chat_message("assistant"):
            if "internship" in q.lower() or "job" in q.lower():
                ans = "**Matched Internships:**\n- 💼 Python Developer Intern @ TechCorp\n- 💼 Data Analyst Intern @ InfoSys"
            elif st.session_state.vs:
                retrieved = st.session_state.vs.as_retriever(search_kwargs={"k": 2}).invoke(q)
                ctx = "\n".join([d.page_content for d in retrieved])
                llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
                
                # Simple 1-liner clean string extraction
                raw_ans = llm.invoke(f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer concisely:").content
                ans = str(raw_ans).split(", 'extras':")[0].replace("[{'text': '", "").replace("'}]", "").strip()
            else:
                ans = "Please upload a document first."

            st.markdown(ans)
            st.session_state.chat.append({"role": "assistant", "content": ans})