import os

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline


BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "faiss_index.bin"
METADATA_PATH = BASE_DIR / "faiss_metadata.json"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

DEMO_CLAIMS = [
    "Online money gaming services shall be permitted without restriction.",
    "A foreign contribution may be accepted freely by any organisation without prior permission.",
    "Coastal vessels may operate on Indian coasts without any licence or registration.",
    "Public examinations may be conducted by any private agency without oversight.",
]


st.set_page_config(
    page_title="Policy Inconsistency Detector",
    page_icon="",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading FAISS index and policy metadata...")
def load_index_and_metadata():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        st.error(
            "Missing faiss_index.bin or faiss_metadata.json. Run the notebook through "
            "the save-index step first."
        )
        st.stop()

    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return index, metadata


@st.cache_resource(show_spinner="Loading sentence embedding model...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading NLI contradiction model...")
def load_nli_model():
    return pipeline(
        "text-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v1",
        truncation=True,
    )


def normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def split_into_clauses(text, min_chars=70, max_chars=900):
    text = normalize_text(text)
    if not text:
        return []

    raw_parts = re.split(r"(?<=[.!?;:])\s+", text)
    clauses = []
    for part in raw_parts:
        part = normalize_text(part)
        if len(part) < min_chars:
            continue
        while len(part) > max_chars:
            split_at = part.rfind(" ", 0, max_chars)
            split_at = split_at if split_at > 200 else max_chars
            clauses.append(part[:split_at].strip())
            part = part[split_at:].strip()
        if len(part) >= min_chars:
            clauses.append(part)
    return clauses


def extract_uploaded_text(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.read()

    if suffix == ".pdf":
        text_parts = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        return "\n".join(text_parts)

    return raw.decode("utf-8", errors="ignore")


def retrieve(query, index, metadata, embedder, k=8, exclude_doc_id=None):
    search_k = min(max(k * 5, k), len(metadata))
    q_embed = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(q_embed, search_k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        record = dict(metadata[int(idx)])
        if exclude_doc_id and record.get("doc_id") == exclude_doc_id:
            continue

        record["retrieval_score"] = float(score)
        results.append(record)

        if len(results) == k:
            break

    return results


def has_positive_policy_language(text):
    text = text.lower()
    negative_patterns = [
        r"\bno person shall\b",
        r"\bshall not\b",
        r"\bmust not\b",
        r"\bmay not\b",
        r"\bnot be permitted\b",
        r"\bnot permitted\b",
        r"\bprohibited\b",
        r"\bbanned?\b",
        r"\bbarred\b",
    ]
    for pattern in negative_patterns:
        text = re.sub(pattern, " ", text)

    return bool(
        re.search(
            r"\b(shall|must|required to|may|allowed|permitted|authorised|eligible|entitled)\b",
            text,
        )
    )


def has_negative_policy_language(text):
    return bool(
        re.search(
            r"\b(shall not|must not|may not|not be permitted|not permitted|prohibited|"
            r"ban|banned|barred|no person shall|without permission|punishable)\b",
            text.lower(),
        )
    )


def has_unrestricted_scope_language(text):
    return bool(
        re.search(
            r"\b(without restriction|without registration|without prior approval|"
            r"without approval|without licence|without license|without compliance|"
            r"freely|all .* may|any .* may)\b",
            text.lower(),
        )
    )


def has_conditional_scope_language(text):
    return bool(
        re.search(
            r"\b(subject to|only after|only if|provided that|unless|except|"
            r"registration|registered|licen[cs]e|approval|prior approval|"
            r"recognised|recognized|compliance|due diligence|permission)\b",
            text.lower(),
        )
    )


def classify_conflict_type(a, b):
    text = f"{a} {b}".lower()
    a_pos = has_positive_policy_language(a)
    a_neg = has_negative_policy_language(a)
    b_pos = has_positive_policy_language(b)
    b_neg = has_negative_policy_language(b)

    if (has_unrestricted_scope_language(a) and has_conditional_scope_language(b)) or (
        has_unrestricted_scope_language(b) and has_conditional_scope_language(a)
    ):
        return "scope"
    if (a_pos and not a_neg and b_neg and not b_pos) or (
        a_neg and not a_pos and b_pos and not b_neg
    ):
        return "direct"
    if re.search(r"\b(except|unless|only|subject to|provided that|notwithstanding)\b", text):
        return "scope"
    if re.search(r"\b(before|after|within|from|until|date|days|months|years)\b", text):
        return "temporal"
    return "indirect"


def heuristic_conflict_score(a, b):
    a_pos = has_positive_policy_language(a)
    a_neg = has_negative_policy_language(a)
    b_pos = has_positive_policy_language(b)
    b_neg = has_negative_policy_language(b)

    score = 0.0
    opposing_polarity = (a_pos and not a_neg and b_neg and not b_pos) or (
        a_neg and not a_pos and b_pos and not b_neg
    )
    scope_mismatch = (
        has_unrestricted_scope_language(a) and has_conditional_scope_language(b)
    ) or (
        has_unrestricted_scope_language(b) and has_conditional_scope_language(a)
    )

    if opposing_polarity:
        score += 0.55
    if scope_mismatch:
        score += 0.60
    if (opposing_polarity or scope_mismatch) and re.search(r"\b(except|unless|only|subject to|provided that)\b", f"{a} {b}".lower()):
        score += 0.15
    if (opposing_polarity or scope_mismatch) and re.search(r"\b(licen[cs]e|permission|approval|registration|prohibit|penalty|fine)\b", f"{a} {b}".lower()):
        score += 0.10
    return min(score, 0.85)


def run_nli(nli_model, a, b):
    result = nli_model(f"{a} </s> {b}")
    item = result[0] if isinstance(result, list) else result
    return item["label"].lower(), float(item["score"])


def analyze_claim(query, index, metadata, embedder, use_nli=True, k=8, threshold=0.50):
    candidates = retrieve(query, index, metadata, embedder, k=k)
    nli_model = load_nli_model() if use_nli else None
    rows = []

    for candidate in candidates:
        heuristic_score = heuristic_conflict_score(query, candidate["text"])
        nli_label = "not_run"
        nli_score = 0.0

        if nli_model is not None:
            try:
                nli_label, nli_score = run_nli(nli_model, query, candidate["text"])
            except Exception as exc:
                nli_label = f"nli_error: {exc.__class__.__name__}"

        model_conflict = "contrad" in nli_label and nli_score >= threshold
        heuristic_conflict = heuristic_score >= threshold
        is_conflict = model_conflict or heuristic_conflict

        rows.append(
            {
                "claim": query,
                "source_file": candidate.get("source_file", ""),
                "doc_type": candidate.get("doc_type", ""),
                "chunk": candidate.get("local_chunk_id", candidate.get("chunk_id", "")),
                "retrieval_score": candidate["retrieval_score"],
                "nli_label": nli_label,
                "nli_score": nli_score,
                "heuristic_score": heuristic_score,
                "conflict": is_conflict,
                "conflict_type": classify_conflict_type(query, candidate["text"]) if is_conflict else "none",
                "candidate_text": candidate["text"],
                "source_path": candidate.get("source_path", ""),
            }
        )

    return rows


def build_report(claims, rows):
    conflicts = [row for row in rows if row["conflict"]]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "claims_checked": claims,
        "retrieved_candidates": len(rows),
        "conflicts_found": len(conflicts),
        "results": conflicts,
        "all_candidates": rows,
    }


def save_report(report):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"frontend_policy_report_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def similarity_band(score):
    if score >= 0.70:
        return "High", "#1f8f4d"
    if score >= 0.50:
        return "Medium", "#c69214"
    return "Low", "#c2413b"


def render_similarity_badge(score):
    label, color = similarity_band(score)
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:0.25rem 0.6rem;
            border-radius:6px;
            background:{color};
            color:white;
            font-weight:700;
            margin:0.25rem 0 0.75rem 0;">
            {label} similarity: {score:.3f}
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_source_document(path):
    source_path = Path(path)
    if not source_path.exists() or not source_path.is_file():
        return None
    return source_path.read_text(encoding="utf-8", errors="ignore")


def render_result_cards(rows, only_conflicts=False):
    visible_rows = [row for row in rows if row["conflict"]] if only_conflicts else rows

    if not visible_rows:
        st.info("No rows to show for this filter.")
        return

    for row in visible_rows:
        label = "Potential inconsistency" if row["conflict"] else "Relevant document"
        with st.expander(
            f"{label}: {row['source_file']} | similarity {row['retrieval_score']:.3f}",
            expanded=row["conflict"],
        ):
            render_similarity_badge(row["retrieval_score"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("NLI", row["nli_label"])
            c2.metric("NLI score", f"{row['nli_score']:.3f}")
            c3.metric("Heuristic", f"{row['heuristic_score']:.3f}")
            c4.metric("Type", row["conflict_type"])

            if row["conflict"]:
                st.warning(
                    "Flagged because the checked clause and the retrieved source use opposing "
                    "policy language, such as permission vs prohibition, or broad permission vs "
                    "conditional restriction."
                )

            st.markdown("**Checked clause**")
            st.write(row["claim"])
            st.markdown("**Retrieved source clause**")
            st.caption(f"Source chunk: {row['source_file']} #{row['chunk']}")
            st.write(row["candidate_text"])


index, metadata = load_index_and_metadata()
embedder = load_embedder()

source_counts = Counter(record.get("source_file", "unknown") for record in metadata)
doc_type_counts = Counter(record.get("doc_type", "unknown") for record in metadata)

st.title("Policy Inconsistency Detector")
st.write(
    "Search Indian policy documents, check draft clauses against the corpus, and export "
    "source-attributed inconsistency reports."
)

with st.sidebar:
    st.header("Corpus")
    st.metric("Chunks indexed", f"{len(metadata):,}")
    st.metric("Documents", f"{len(source_counts):,}")
    st.write("Document types")
    st.dataframe(
        pd.DataFrame(doc_type_counts.items(), columns=["Type", "Chunks"]),
        hide_index=True,
        use_container_width=True,
    )

    st.header("Detection")
    top_k = st.slider("Relevant chunks", 3, 20, 8)
    threshold = st.slider("Contradiction threshold", 0.10, 0.95, 0.50, 0.05)
    use_nli = st.toggle("Use NLI model", value=True)
    st.caption("Turn NLI off for a faster heuristic-only demo.")


tab_search, tab_claim, tab_upload, tab_demo = st.tabs(
    ["Find Documents", "Check Claim", "Check Uploaded Document", "Demo Inconsistencies"]
)

with tab_search:
    st.subheader("Find relevant documents")
    search_query = st.text_input(
        "Search by policy topic or clause",
        "online gaming prohibition",
    )

    if st.button("Search documents", type="primary"):
        results = retrieve(search_query, index, metadata, embedder, k=top_k)
        st.session_state["search_results"] = results

    results = st.session_state.get("search_results", [])
    if results:
        grouped = defaultdict(
            lambda: {
                "best_score": 0.0,
                "matches": 0,
                "sample": "",
                "source_path": "",
            }
        )
        for item in results:
            row = grouped[item["source_file"]]
            row["best_score"] = max(row["best_score"], item["retrieval_score"])
            row["matches"] += 1
            if not row["sample"]:
                row["sample"] = item["text"][:260]
            if not row["source_path"]:
                row["source_path"] = item.get("source_path", "")

        doc_rows = [
            {
                "Document": doc,
                "Similarity": similarity_band(data["best_score"])[0],
                "Best score": round(data["best_score"], 3),
                "Matches": data["matches"],
                "Sample": data["sample"],
            }
            for doc, data in grouped.items()
        ]
        st.dataframe(pd.DataFrame(doc_rows), hide_index=True, use_container_width=True)

        st.markdown("### Download relevant documents")
        for doc, data in grouped.items():
            source_text = read_source_document(data["source_path"])
            label, _ = similarity_band(data["best_score"])
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.write(f"**{doc}**")
            c2.write(f"{label}: {data['best_score']:.3f}")
            if source_text:
                c3.download_button(
                    "Download",
                    source_text,
                    file_name=doc,
                    mime="text/plain",
                    key=f"download-{doc}",
                )
            else:
                c3.caption("File unavailable")

        st.markdown("### Matching chunks")
        for result in results:
            with st.expander(f"{result['source_file']} #{result['local_chunk_id']} | {result['retrieval_score']:.3f}"):
                render_similarity_badge(result["retrieval_score"])
                st.write(result["text"])


with tab_claim:
    st.subheader("Check a policy claim")
    claim = st.text_area(
        "Paste a proposed policy clause",
        "Online money gaming services shall be permitted without restriction.",
        height=120,
    )

    if st.button("Check claim", type="primary"):
        rows = analyze_claim(
            claim,
            index,
            metadata,
            embedder,
            use_nli=use_nli,
            k=top_k,
            threshold=threshold,
        )
        st.session_state["claim_rows"] = rows
        st.session_state["claim_report"] = build_report([claim], rows)

    rows = st.session_state.get("claim_rows", [])
    if rows:
        conflicts = [row for row in rows if row["conflict"]]
        st.metric("Potential inconsistencies", len(conflicts))
        render_result_cards(rows, only_conflicts=False)

        report = st.session_state["claim_report"]
        report_path = save_report(report)
        st.download_button(
            "Download JSON report",
            json.dumps(report, ensure_ascii=False, indent=2),
            file_name=report_path.name,
            mime="application/json",
        )
        st.caption(f"Saved locally: {report_path}")


with tab_upload:
    st.subheader("Check an uploaded draft document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT policy draft", type=["pdf", "txt"])
    max_clauses = st.slider("Clauses to check from upload", 1, 20, 6)

    if uploaded_file and st.button("Extract and check document", type="primary"):
        uploaded_text = extract_uploaded_text(uploaded_file)
        clauses = split_into_clauses(uploaded_text)[:max_clauses]

        all_rows = []
        for clause in clauses:
            all_rows.extend(
                analyze_claim(
                    clause,
                    index,
                    metadata,
                    embedder,
                    use_nli=use_nli,
                    k=top_k,
                    threshold=threshold,
                )
            )

        st.session_state["upload_clauses"] = clauses
        st.session_state["upload_rows"] = all_rows
        st.session_state["upload_report"] = build_report(clauses, all_rows)

    clauses = st.session_state.get("upload_clauses", [])
    rows = st.session_state.get("upload_rows", [])
    if clauses:
        st.write(f"Checked {len(clauses)} clauses from the uploaded document.")
        with st.expander("Extracted clauses"):
            for clause in clauses:
                st.write(f"- {clause}")

        conflicts = [row for row in rows if row["conflict"]]
        st.metric("Potential inconsistencies", len(conflicts))
        render_result_cards(rows, only_conflicts=True)

        report = st.session_state["upload_report"]
        report_path = save_report(report)
        st.download_button(
            "Download document report",
            json.dumps(report, ensure_ascii=False, indent=2),
            file_name=report_path.name,
            mime="application/json",
        )
        st.caption(f"Saved locally: {report_path}")


with tab_demo:
    st.subheader("Demo inconsistencies")
    st.write(
        "These sample draft clauses are intentionally broad. The system retrieves laws "
        "that restrict or prohibit the same activity, creating visible inconsistency cases."
    )

    demo_claim = st.selectbox("Choose a demo claim", DEMO_CLAIMS)

    if st.button("Run demo check", type="primary"):
        rows = analyze_claim(
            demo_claim,
            index,
            metadata,
            embedder,
            use_nli=use_nli,
            k=top_k,
            threshold=threshold,
        )
        st.session_state["demo_rows"] = rows
        st.session_state["demo_report"] = build_report([demo_claim], rows)

    rows = st.session_state.get("demo_rows", [])
    if rows:
        conflicts = [row for row in rows if row["conflict"]]
        st.metric("Potential inconsistencies", len(conflicts))
        render_result_cards(rows, only_conflicts=False)

        report = st.session_state["demo_report"]
        st.download_button(
            "Download demo report",
            json.dumps(report, ensure_ascii=False, indent=2),
            file_name="demo_policy_inconsistency_report.json",
            mime="application/json",
        )
