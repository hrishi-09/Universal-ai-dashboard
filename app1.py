# =========================================================
# UNIVERSAL AI DATA VISUALISATION DASHBOARD
# Fixed & Improved Version
# =========================================================

# =========================================================
# IMPORTS
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from PIL import Image
from docx import Document
from collections import Counter

import re
import cv2
import tempfile
import os

# pypdf replaces the deprecated PyPDF2
try:
    from pypdf import PdfReader
except ImportError:
    try:
        import PyPDF2
        PdfReader = PyPDF2.PdfReader
    except ImportError:
        PdfReader = None

# pdf2docx is optional — may not be installed
try:
    from pdf2docx import Converter as PdfToDocxConverter
    PDF2DOCX_AVAILABLE = True
except ImportError:
    PDF2DOCX_AVAILABLE = False

# Word-to-PDF: use reportlab (cross-platform, no LibreOffice needed)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Transformers object detection is optional (requires torch)
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Universal AI Dashboard",
    page_icon="🚀",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.main { background-color: #0f172a; color: white; }
h1, h2, h3 { color: #38bdf8; }
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    font-size: 16px;
}
[data-testid="metric-container"] {
    background-color: #1e293b;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #334155;
}
[data-testid="stSidebar"] { background-color: #111827; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("🚀 Universal AI Data Visualisation Dashboard")

st.markdown("""
### Upload Any File And Automatically Get:

✅ Full Excel / CSV Analysis  
✅ Smart Visualisations  
✅ Word File AI Analysis  
✅ PDF Reader & Converter  
✅ AI Image Detection  
✅ Face Detection  
✅ Live Camera Detection  
✅ AI Generated Summaries  
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📂 Upload Files")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV, Excel, Word, PDF or Image",
    type=["csv", "xlsx", "docx", "pdf", "png", "jpg", "jpeg"]
)

# =========================================================
# SAMPLE DATASET
# =========================================================

sample_data = pd.DataFrame({
    "Transaction_ID": ["TXN1001", "TXN1002"],
    "Customer_ID":    ["CUST101", "CUST102"],
    "Bank":           ["HDFC", "ICICI"],
    "Card_Type":      ["Visa", "MasterCard"],
    "Gender":         ["M", "F"],
    "Age":            [25, 40],
    "City":           ["Delhi", "Mumbai"],
    "Amount":         [2500, 7800],
    "Fraud":          [0, 1]
})

st.sidebar.download_button(
    "⬇ Download Sample Dataset",
    sample_data.to_csv(index=False).encode("utf-8"),
    "sample_dataset.csv",
    "text/csv"
)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

@st.cache_data
def load_data(file_bytes, file_name):
    """Load CSV or Excel from bytes (cache-safe)."""
    import io
    buf = io.BytesIO(file_bytes)
    if file_name.endswith(".csv"):
        return pd.read_csv(buf, low_memory=False)
    return pd.read_excel(buf)


def read_word(file) -> str:
    """Extract plain text from a .docx file."""
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


def read_pdf(file) -> str:
    """Extract plain text from a PDF file."""
    if PdfReader is None:
        return "⚠ PDF reading library not installed. Run: pip install pypdf"
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text


def summarize_text(text: str, num_sentences: int = 5) -> str:
    """Extractive summarisation using word-frequency scoring."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    if len(sentences) <= num_sentences:
        return text

    words = re.findall(r'\w+', text.lower())
    word_freq = Counter(words)

    scored = {
        s: sum(word_freq[w] for w in re.findall(r'\w+', s.lower()))
        for s in sentences
    }

    top = sorted(scored, key=scored.get, reverse=True)[:num_sentences]
    return " ".join(top)


def grammar_check(text: str) -> str:
    """Very basic common-typo fixer."""
    replacements = {
        " teh ":     " the ",
        " adn ":     " and ",
        " recieve ": " receive ",
        " occured ": " occurred ",
        " seperate ": " separate ",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text


# =========================================================
# OBJECT DETECTION (lazy-loaded to avoid startup crash)
# =========================================================

@st.cache_resource
def load_object_detector():
    if not TRANSFORMERS_AVAILABLE:
        return None
    return hf_pipeline("object-detection", model="facebook/detr-resnet-50")


def detect_objects(image):
    detector = load_object_detector()
    if detector is None:
        return ["⚠ Transformers / PyTorch not installed."]
    results = detector(image)
    return [f"{obj['label']} ({round(obj['score'] * 100, 2)}%)" for obj in results]


def image_info(image):
    w, h = image.size
    return {"Width": w, "Height": h, "Color Mode": image.mode}


# =========================================================
# WORD → PDF CONVERTER
# =========================================================

def word_to_pdf_bytes(docx_file) -> bytes | None:
    """Convert a .docx to PDF bytes using reportlab."""
    if not REPORTLAB_AVAILABLE:
        return None

    # Extract text from the Word doc
    text = read_word(docx_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_path = tmp.name

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    for line in text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), styles["Normal"]))

    doc.build(story)

    with open(pdf_path, "rb") as f:
        data = f.read()

    os.unlink(pdf_path)
    return data


# =========================================================
# MAIN FILE HANDLER
# =========================================================

if uploaded_file:

    file_name = uploaded_file.name.lower()

    # =====================================================
    # CSV / EXCEL
    # =====================================================

    if file_name.endswith(".csv") or file_name.endswith(".xlsx"):

        st.header("📊 Excel / CSV Analysis")

        try:
            # Read bytes once so the cache key is stable
            file_bytes = uploaded_file.read()
            df = load_data(file_bytes, file_name)

            st.success("✅ Dataset Loaded Successfully")

            # Clean column names
            df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

            # Auto date detection — only on object columns to avoid data corruption
            for col in df.select_dtypes(include="object").columns:
                try:
                    converted = pd.to_datetime(df[col], infer_datetime_format=True)
                    # Only accept if most values parsed successfully
                    if converted.notna().mean() > 0.8:
                        df[col] = converted
                except Exception:
                    pass

            numeric_cols     = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            date_cols        = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

            # Metrics
            st.subheader("📌 Dataset Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows",            df.shape[0])
            c2.metric("Columns",         df.shape[1])
            c3.metric("Missing Values",  int(df.isnull().sum().sum()))
            c4.metric("Duplicate Rows",  int(df.duplicated().sum()))

            st.subheader("📂 Dataset")
            st.dataframe(df, use_container_width=True, height=500)

            st.subheader("🧠 Column Information")
            info_df = pd.DataFrame({
                "Column":        df.columns,
                "Datatype":      df.dtypes.astype(str).values,
                "Missing Values":df.isnull().sum().values,
                "Unique Values": df.nunique().values
            })
            st.dataframe(info_df, use_container_width=True)

            # Sidebar filters
            st.sidebar.subheader("🔍 Filters")
            filtered_df = df.copy()
            for col in categorical_cols[:5]:
                unique_values = df[col].dropna().unique().tolist()
                if len(unique_values) < 100:
                    selected = st.sidebar.multiselect(
                        f"Filter {col}", unique_values, default=unique_values
                    )
                    filtered_df = filtered_df[filtered_df[col].isin(selected)]

            # Missing values chart
            st.subheader("⚠ Missing Values")
            missing_df = pd.DataFrame({
                "Column":  df.columns,
                "Missing": df.isnull().sum().values
            })
            st.plotly_chart(
                px.bar(missing_df, x="Column", y="Missing", color="Missing"),
                use_container_width=True
            )

            # Statistics
            if numeric_cols:
                st.subheader("📋 Statistical Summary")
                st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)

            # Bar chart
            if categorical_cols and numeric_cols:
                st.subheader("📊 Bar Chart")
                col1, col2 = st.columns(2)
                cat_col = col1.selectbox("Category Column", categorical_cols)
                num_col = col2.selectbox("Numeric Column",  numeric_cols)
                st.plotly_chart(
                    px.bar(filtered_df, x=cat_col, y=num_col, color=cat_col),
                    use_container_width=True
                )

            # Pie chart
            if categorical_cols:
                st.subheader("🥧 Pie Chart")
                pie_col = st.selectbox("Select Column", categorical_cols, key="pie")
                pie_df  = filtered_df[pie_col].value_counts().reset_index()
                pie_df.columns = ["Category", "Count"]
                st.plotly_chart(
                    px.pie(pie_df, names="Category", values="Count"),
                    use_container_width=True
                )

            # Histogram
            if numeric_cols:
                st.subheader("📉 Histogram")
                hist_col = st.selectbox("Select Numeric Column", numeric_cols, key="hist")
                st.plotly_chart(
                    px.histogram(filtered_df, x=hist_col, nbins=40),
                    use_container_width=True
                )

            # Scatter plot
            if len(numeric_cols) >= 2:
                st.subheader("🔵 Scatter Plot")
                x_col = st.selectbox("X Axis", numeric_cols, key="x")
                y_col = st.selectbox("Y Axis", numeric_cols, key="y")
                st.plotly_chart(
                    px.scatter(filtered_df, x=x_col, y=y_col, color=y_col, size=y_col),
                    use_container_width=True
                )

            # Time series
            if date_cols and numeric_cols:
                st.subheader("📈 Time Series")
                st.plotly_chart(
                    px.line(filtered_df, x=date_cols[0], y=numeric_cols[0]),
                    use_container_width=True
                )

            # Correlation heatmap
            if len(numeric_cols) >= 2:
                st.subheader("🔥 Correlation Heatmap")
                corr = filtered_df[numeric_cols].corr()
                st.plotly_chart(
                    px.imshow(corr, text_auto=True),
                    use_container_width=True
                )

            # Download filtered
            st.subheader("⬇ Download Filtered Dataset")
            st.download_button(
                "Download CSV",
                filtered_df.to_csv(index=False).encode("utf-8"),
                "filtered_dataset.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")

    # =====================================================
    # WORD (.docx) ANALYSIS
    # =====================================================

    elif file_name.endswith(".docx"):

        st.header("📄 Word File Analysis")

        try:
            text = read_word(uploaded_file)

            st.subheader("📜 Extracted Text")
            st.text_area("Content", text, height=300)

            st.subheader("🧠 Summary")
            st.success(summarize_text(text))

            st.subheader("✍ Corrected Text")
            st.text_area("Corrected Content", grammar_check(text), height=300)

            words = re.findall(r'\w+', text.lower())
            c1, c2, c3 = st.columns(3)
            c1.metric("Words",        len(words))
            c2.metric("Characters",   len(text))
            c3.metric("Unique Words", len(set(words)))

            st.subheader("📊 Word Frequency")
            freq_df = pd.DataFrame(
                Counter(words).most_common(20),
                columns=["Word", "Frequency"]
            )
            st.plotly_chart(
                px.bar(freq_df, x="Word", y="Frequency", color="Frequency"),
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error reading Word file: {e}")

    # =====================================================
    # PDF ANALYSIS
    # =====================================================

    elif file_name.endswith(".pdf"):

        st.header("📕 PDF Analysis")

        try:
            text = read_pdf(uploaded_file)

            st.subheader("📜 PDF Content")
            st.text_area("Extracted Text", text, height=300)

            st.subheader("🧠 Summary")
            st.success(summarize_text(text))

            # PDF → DOCX
            if PDF2DOCX_AVAILABLE:
                st.subheader("🔄 PDF to Word")

                # Reset file pointer after read_pdf consumed it
                uploaded_file.seek(0)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(uploaded_file.read())
                    pdf_path = tmp_pdf.name

                docx_path = pdf_path.replace(".pdf", ".docx")

                try:
                    cv = PdfToDocxConverter(pdf_path)
                    cv.convert(docx_path)
                    cv.close()

                    with open(docx_path, "rb") as f:
                        st.download_button(
                            "⬇ Download DOCX",
                            f.read(),
                            file_name="converted.docx"
                        )
                except Exception as conv_err:
                    st.warning(f"PDF→DOCX conversion failed: {conv_err}")
                finally:
                    for path in [pdf_path, docx_path]:
                        if os.path.exists(path):
                            os.unlink(path)
            else:
                st.info("ℹ Install `pdf2docx` to enable PDF → Word conversion.")

        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    # =====================================================
    # IMAGE ANALYSIS
    # =====================================================

    elif file_name.endswith((".png", ".jpg", ".jpeg")):

        st.header("🖼 AI Image Analysis")

        try:
            image = Image.open(uploaded_file)

            st.image(image, use_container_width=True)

            # Object detection
            st.subheader("🤖 AI Object Detection")
            if TRANSFORMERS_AVAILABLE:
                with st.spinner("Detecting Objects..."):
                    objects = detect_objects(image)
                if objects:
                    for obj in objects:
                        st.success(obj)
                else:
                    st.warning("No objects detected.")
            else:
                st.info("ℹ Install `transformers` and `torch` to enable AI object detection.")

            # Image info
            st.subheader("📋 Image Information")
            info = image_info(image)
            st.dataframe(
                pd.DataFrame({"Property": list(info.keys()), "Value": list(info.values())}),
                use_container_width=True
            )

            # Face detection
            st.subheader("😀 Face Detection")
            img_array = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            img_drawn = img_array.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(img_drawn, (x, y), (x + w, y + h), (0, 255, 0), 3)

            st.image(
                img_drawn,
                caption=f"Faces Detected: {len(faces)}",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error analysing image: {e}")

# =========================================================
# LIVE CAMERA DETECTION
# FIX: The original infinite while-loop blocks Streamlit's
# thread. We capture one frame per rerun instead.
# =========================================================

st.markdown("---")
st.header("📷 Live Camera Detection")

face_cascade_cam = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

run_camera = st.checkbox("Start Camera")

if run_camera:
    camera = cv2.VideoCapture(0)
    success, frame = camera.read()
    camera.release()

    if not success:
        st.error("❌ Cannot access camera. Make sure it is connected and not in use.")
    else:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade_cam.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, "Human Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        st.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            caption=f"Faces Detected: {len(faces)}",
            use_container_width=True
        )

        # Auto-rerun to simulate a live feed
        import time
        time.sleep(0.1)
        st.rerun()

# =========================================================
# WORD → PDF CONVERTER
# =========================================================

st.markdown("---")
st.header("🔄 Word to PDF Converter")

word_upload = st.file_uploader(
    "Upload DOCX File",
    type=["docx"],
    key="word_to_pdf"
)

if word_upload:
    if REPORTLAB_AVAILABLE:
        try:
            pdf_bytes = word_to_pdf_bytes(word_upload)
            if pdf_bytes:
                st.download_button(
                    "⬇ Download PDF",
                    pdf_bytes,
                    file_name="converted.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Conversion produced an empty file.")
        except Exception as e:
            st.error(f"Conversion failed: {e}")
    else:
        st.warning(
            "⚠ `reportlab` is not installed. "
            "Run `pip install reportlab` to enable Word → PDF conversion."
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown("""
<center>

## 🚀 Universal AI Dashboard

Built Using: Streamlit · Plotly · Pandas · OpenCV · Transformers · Python

</center>
""", unsafe_allow_html=True)