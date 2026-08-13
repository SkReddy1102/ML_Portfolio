# ============================================================
# APP: Bridge NLP → SQL Explorer (RAG + Embedding Intent + Domain Relevance)
# Pipeline: Domain Relevance → Intent → RAG → SQL → Execution
# ============================================================

# importing the required libraries
import streamlit as st
import pandas as pd
import json, os, re, datetime
from sqlalchemy import create_engine, text
import numpy as np
import sqlite3
import openai
from openai import OpenAI
from rapidfuzz import fuzz, process  # pip install rapidfuzz

# ------------------------------------------------------------
# Load dataset and create SQLite DB
# ------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(APP_DIR, "data", "cleaned_dataset.csv")

df = pd.read_csv(CSV_PATH)
print("Rows, Cols:", df.shape)

# Initialize client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = OPENAI_API_KEY.strip() if OPENAI_API_KEY else None
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def require_openai_client():
    """Show a Streamlit error instead of crashing when the API key is missing."""
    if client is None:
        st.error("Set the OPENAI_API_KEY environment variable before running this app.")
        st.stop()
    return client


def stop_for_openai_error(error: Exception):
    """Show user-friendly OpenAI API errors instead of a traceback."""
    if isinstance(error, openai.RateLimitError):
        st.error(
            "OpenAI API credits are exhausted for this account. "
            "Add credits in OpenAI Platform billing, then rerun the app."
        )
        st.info("Billing page: https://platform.openai.com/settings/organization/billing/")
        st.stop()

    if isinstance(error, openai.AuthenticationError):
        st.error("OpenAI API authentication failed. Check that OPENAI_API_KEY is valid.")
        st.stop()

    if isinstance(error, openai.APIConnectionError):
        st.error("Could not connect to the OpenAI API. Check your network, VPN, proxy, or firewall.")
        st.stop()

    raise error

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Create SQLite database and save table
DB_PATH = os.path.join(APP_DIR, "bridges.db")     # Database file
DB_URI  = f"sqlite:///{DB_PATH}"
TABLE_NAME = "bridges"     # Table inside the DB
TABLE = TABLE_NAME         # alias used later in code

# write data into SQLite
conn = sqlite3.connect(DB_PATH)
df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
conn.close()

print(f"Table '{TABLE_NAME}' created inside database '{DB_PATH}'")
print(df.head(10))

# Create a global SQLAlchemy engine (used by query_db and get_table_columns)
engine = create_engine(DB_URI)

# ============================================================
# 1. BASIC VALIDATION
# ============================================================

def is_gibberish(text: str) -> bool:
    """Very simple noise filter to avoid random keystrokes."""
    if len(text.strip()) < 3:
        return True
    if re.search(r"[^a-zA-Z0-9\s\?\.,%-]", text):
        return True
    if len(re.findall(r"[aeiou]", text.lower())) / max(len(text), 1) < 0.2:
        return True
    return False

# ============================================================
# 2. KEYWORDS & DOMAIN RELEVANCE (RULES + EMBEDDINGS)
# ============================================================

# A. Domain keywords tied to typical bridge data
BRIDGE_KEYWORDS = [
    "bridge", "structure", "year built", "condition", "county", "state",
    "owner", "deck", "width", "height", "traffic", "adt", "percent",
    "truck", "navigation", "reconstructed", "history", "operational",
    "load", "design load", "skew", "clearance"
]

# B. Policy / planning keywords (intent words, future-proof)
POLICY_KEYWORDS = [
    "safety", "unsafe", "risk", "critical", "priority", "structurally deficient",
    "poor condition", "failing", "deficient",
    "repair", "maintenance", "rehabilitation", "needs repair", "needs maintenance",
    "funding", "budget", "allocate", "investment", "planning",
    "infrastructure", "improvement program",
    "worst", "best", "top", "highest", "lowest", "rural", "urban", "region",
]

# C. Domain sentences for semantic relevance
BRIDGE_DOMAIN_SENTENCES = [
    "Questions about road and highway bridges, their condition, age, traffic, location, and structure.",
    "Infrastructure policy questions about which bridges need repair, are unsafe, or should be prioritized.",
    "Analysis of bridges by county, state, traffic volume, and structural ratings."
]

# ============================================================
# 3. EMBEDDING HELPERS (NEW OpenAI SDK)
# ============================================================

def get_embeddings(texts):
    """Return embeddings as numpy array using new OpenAI SDK."""
    if isinstance(texts, str):
        texts = [texts]

    openai_client = require_openai_client()
    try:
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
    except Exception as error:
        stop_for_openai_error(error)

    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype="float32")

def cosine_sim(vec, mat):
    """Cosine similarity between one vector and a matrix."""
    if mat.size == 0:
        return np.array([])
    v = vec / (np.linalg.norm(vec) + 1e-8)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return m @ v

@st.cache_resource
def build_bridge_domain_index():
    """Embeddings for general 'bridge domain' meaning."""
    return get_embeddings(BRIDGE_DOMAIN_SENTENCES)

def is_relevant_to_bridges(q: str, threshold: float = 0.55) -> bool:
    """
    1) Fast keyword relevance
    2) If no keyword match → semantic relevance using embeddings
    """
    q_lower = q.lower()

    # Fast keyword gate
    if any(k in q_lower for k in BRIDGE_KEYWORDS + POLICY_KEYWORDS):
        return True

    # Semantic domain relevance
    domain_embs = build_bridge_domain_index()
    q_vec = get_embeddings(q)[0]
    sims = cosine_sim(q_vec, domain_embs)
    if sims.size == 0:
        return False

    max_sim = float(sims.max())
    print(f"[DOMAIN-RELEVANCE] sim={max_sim:.3f}")
    return max_sim >= threshold

# ============================================================
# 4. DB & SCHEMA UTILITIES
# ============================================================

def query_db(sql: str) -> pd.DataFrame:
    """Run SQL query and return a pandas DataFrame."""
    return pd.read_sql(sql, engine)

def get_table_columns() -> list[str]:
    """Fetch actual column names from SQLite table schema."""
    with engine.connect() as conn:
        res = conn.execute(text(f"PRAGMA table_info({TABLE});"))
        cols = [row[1] for row in res]
    return cols

# ---------- Load states & counties from dataset ----------
def load_unique_values(column_name):
    try:
        with engine.connect() as conn:
            res = conn.execute(text(f'SELECT DISTINCT "{column_name}" FROM {TABLE}'))
            values = [row[0] for row in res if row[0]]
        return set(v.strip().lower() for v in values)
    except:
        return set()

STATE_SET = load_unique_values("1 - State Name")
COUNTY_SET = load_unique_values("3 - County Name")

# ---------- Fuzzy Matcher ----------
def fuzzy_find_matches(question: str, candidates: set, threshold=80):
    """
    Return all fuzzy matches where similarity >= threshold.
    Ex:
    'montgomerry' → ['montgomery county']
    """
    q = question.lower()
    matches = []

    for cand in candidates:
        score = fuzz.partial_ratio(q, cand)
        if score >= threshold:
            matches.append(cand)

    return matches

# ---------- Hierarchical Detection (with explicit county trigger) ----------
def detect_geo_entities(question: str):
    """
    Detect states & counties mentioned in the question.

    Rules:
    - States can be detected via exact or fuzzy match.
    - Counties are ONLY detected if the user actually mentions 'county' / 'counties'.
    Returns:
        {
          "states": [...],
          "counties": [...],
          "level": "state" | "county" | "mixed" | None
        }
    """
    q = question.lower()

    # States (exact + fuzzy)
    exact_states = [s for s in STATE_SET if s in q]
    fuzzy_states = fuzzy_find_matches(q, STATE_SET)
    states = sorted(set(exact_states + fuzzy_states))

    # Counties (only if word "county" appears)
    detected_counties = []
    if "county" in q or "counties" in q:
        exact_counties = [c for c in COUNTY_SET if c in q]
        fuzzy_counties = fuzzy_find_matches(q, COUNTY_SET)
        detected_counties = sorted(set(exact_counties + fuzzy_counties))

    counties = detected_counties

    # Hierarchy level
    if states and not counties:
        level = "state"
    elif counties and not states:
        level = "county"
    elif states and counties:
        level = "mixed"
    else:
        level = None

    return {
        "states": [s.title() for s in states],
        "counties": [c.title() for c in counties],
        "level": level
    }

# ============================================================
# 5. RAG INDEX (SCHEMA-AWARE – NO FEW-SHOTS)
# ============================================================

@st.cache_resource
def build_rag_index():
    """
    Build and cache embeddings for schema columns only.
    We tell the model explicitly to use the column names EXACTLY as written
    (including numbers, spaces, and hyphens) when writing SQL.
    """
    columns = get_table_columns()

    # Each schema doc is a natural-language hint + exact column name
    schema_docs = [
        (
            f"Column name (copy EXACTLY in SQL): {c}. "
            f"Use this exact name, including numbers, spaces, and hyphens."
        )
        for c in columns
    ]

    schema_embs = get_embeddings(schema_docs) if columns else np.zeros((0, 1536), dtype="float32")

    return {
        "columns": columns,
        "schema_docs": schema_docs,
        "schema_embs": schema_embs,
    }

def rag_retrieve(question: str, k_columns: int = 8):
    """
    Retrieve the most relevant schema docs for a question.
    Returns: schema_ctx (list of schema description strings).
    """
    idx = build_rag_index()
    q_vec = get_embeddings(question)[0]

    sims = cosine_sim(q_vec, idx["schema_embs"])
    if sims.size == 0:
        return []

    top_k = min(k_columns, len(sims))
    top_idx = sims.argsort()[::-1][:top_k]
    schema_ctx = [idx["schema_docs"][i] for i in top_idx]

    return schema_ctx

# ============================================================
# 6. EMBEDDING-BASED INTENT + LLM FALLBACK
# ============================================================

INTENT_LABELS = {
    "compare":  "Questions that compare two or more groups, counties, or regions.",
    "trend":    "Questions asking how something changes over time or across years.",
    "aggregate":"Questions about averages, totals, sums, counts, or distributions.",
    "ranking":  "Questions asking for top, bottom, highest, lowest, or ranking of bridges or counties.",
    "policy":   "Questions about priority, safety risk, critical bridges, or which bridges should be fixed first.",
    # 'filter' is reserved as a generic fallback
}

@st.cache_resource
def build_intent_index():
    labels = list(INTENT_LABELS.keys())
    descs  = list(INTENT_LABELS.values())
    intent_embs = get_embeddings(descs)
    return labels, intent_embs

def llm_intent_fallback(question: str) -> str:
    """
    Ask LLM to classify intent when embedding confidence is low.
    """
    label_list = ["compare", "trend", "aggregate", "ranking", "policy", "filter"]
    label_str = ", ".join(label_list)

    prompt = f"""
You are an assistant that classifies user questions into one intent.

Possible intents:
- compare: comparing two or more groups or regions
- trend: asking how something changes over time or years
- aggregate: asking for averages, totals, sums, or counts
- ranking: asking for top/bottom/highest/lowest
- policy: asking about priority, safety, risk, or which bridges to fix first
- filter: simple filtering or listing without extra analysis

Return ONLY ONE word from this list: {label_str}

Question: {question}
Answer with just the intent label.
""".strip()

    try:
        openai_client = require_openai_client()
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You classify questions into one of the predefined intent labels."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        raw = resp.choices[0].message.content.strip().lower()

        for label in label_list:
            if label in raw:
                return label
    except Exception as e:
        print(f"[INTENT-LLM-FALLBACK-ERROR] {e}")

    return "filter"

def classify_intent(question: str, threshold: float = 0.60, use_llm_fallback: bool = True) -> str:
    """
    Hybrid intent detection:
      1) Embedding-based similarity against intent descriptions
      2) If similarity < threshold:
           - if use_llm_fallback: call LLM
           - else: return 'filter'
    """
    labels, intent_embs = build_intent_index()
    q_vec = get_embeddings(question)[0]
    sims = cosine_sim(q_vec, intent_embs)

    if sims.size == 0:
        return "filter"

    best_idx = int(np.argmax(sims))
    best_label = labels[best_idx]
    max_sim = float(sims[best_idx])
    print(f"[INTENT-EMB] best='{best_label}', sim={max_sim:.3f}")

    if max_sim < threshold:
        if use_llm_fallback:
            llm_label = llm_intent_fallback(question)
            print(f"[INTENT-LLM-FALLBACK] -> '{llm_label}'")
            return llm_label
        else:
            print("[INTENT] Low confidence, falling back to 'filter'")
            return "filter"

    return best_label

# ============================================================
# 7. SQL CLEANING, SAFETY & GENERATION (Schema-only RAG)
# ============================================================

def clean_sql(raw: str) -> str:
    """
    Clean raw SQL from the model:
      - remove markdown fences (```sql, ```)
      - remove leading junk (l, sql, numbers, spaces)
      - remove lines before SELECT
    """
    if raw is None:
        return ""

    sql = raw.strip()

    # Remove leading ```sql or ``` blocks
    if sql.lower().startswith("```sql"):
        sql = sql[5:].strip()
    elif sql.startswith("```"):
        sql = sql[3:].strip()

    # Remove trailing ```
    if sql.endswith("```"):
        sql = sql[:-3].strip()

    # Split lines and remove junk before SELECT
    lines = sql.split("\n")
    clean_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip junk lines like: '', 'l', 'sql', '```'
        if stripped == "" or stripped.lower() in ["l", "sql", "`", "```"]:
            continue

        clean_lines.append(stripped)

    # Rejoin
    sql = "\n".join(clean_lines).strip()

    # If SQL does not start with SELECT, force-select from first SELECT line
    if "select" in sql.lower():
        idx = sql.lower().index("select")
        sql = sql[idx:].strip()

    return sql

# ============================================================
# Force important columns always to appear in SELECT output
# ============================================================

def enforce_context_columns(sql: str, table: str) -> str:
    """
    Ensures key context columns are always included in the SELECT statement
    for clarity. We always include state, but do NOT force county — that is
    controlled by the prompt and the question.
    """

    MUST_HAVE_COLUMNS = []

    if not MUST_HAVE_COLUMNS:
        return sql

    cleaned = sql.strip()

    # If LLM wrapped in ```sql ... ```
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # drop first and last ``` lines
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Work with a single-line string for simplicity
    cleaned = " ".join(cleaned.split())
    upper   = cleaned.upper()

    # If it's not a SELECT, just return
    if not upper.startswith("SELECT"):
        return cleaned

    # Handle SELECT *
    if "SELECT *" in upper:
        # turn SELECT * INTO SELECT "1 - State Name", *
        return cleaned.replace(
            "*",
            ", ".join(MUST_HAVE_COLUMNS) + ", *",
            1
        )

    # Try to split on FROM in a case-insensitive way
    idx = upper.find(" FROM ")
    if idx == -1:
        return cleaned  # malformed SQL, do nothing

    select_part = cleaned[:idx]          # 'SELECT ...'
    rest_part   = cleaned[idx+1:]        # 'FROM ...'

    # Remove SELECT keyword and split cols
    cols_str   = select_part[len("SELECT"):].strip()
    select_cols = [c.strip() for c in cols_str.split(",") if c.strip()]

    # Inject required columns if missing
    for must in MUST_HAVE_COLUMNS:
        if must not in select_cols:
            select_cols.insert(0, must)

    new_select = "SELECT " + ", ".join(select_cols)
    return new_select + " " + rest_part

def validate_sql(sql: str) -> bool:
    sql = sql.strip().strip("`").strip()

    bad_words = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    if any(bad in sql.upper() for bad in bad_words):
        return False

    if not sql.upper().startswith("SELECT"):
        return False

    return True

# ============================================================
# 7b. PROMPT BUILDER
# ============================================================

def build_prompt(intent: str, question: str) -> str:
    """
    Build SQL prompt using:
      - schema RAG (top relevant columns)
      - hierarchical state & county detection
      - intent-specific SQL shaping (ranking / trend / compare / aggregate / filter)
      - safe SQL (no SELECT * and no DDL/DML)
    """

    # ---------------------------------------
    # 1) Retrieve schema context via RAG
    # ---------------------------------------
    schema_ctx = rag_retrieve(question, k_columns=7)
    schema_block = "\n".join(schema_ctx) if schema_ctx else "No schema found."

    # Extract column names only
    relevant_columns = [
        c.replace("Column:", "").strip().strip('"')
        for c in schema_ctx
    ]

    # ---------------------------------------
    # 2) Geographic entity detection
    # ---------------------------------------
    geo = detect_geo_entities(question)
    states = geo["states"]         # list of state names from data
    counties = geo["counties"]     # list of county names from data
    level = geo["level"]           # "state", "county", "mixed", or None

    # ---------------------------------------
    # 3) Geo rules given to LLM
    # ---------------------------------------
    if level == "state":
        # Pure state-level question (e.g., "Maryland vs Virginia")
        geo_hint = f"""
Detected states: {', '.join(states)}.
Filter using "1 - State Name" with LIKE for robustness.
Always include "1 - State Name" in SELECT.
Do NOT include "3 - County Name" unless the question explicitly asks about counties.
"""
    elif level == "county":
        # Pure county-level question (e.g., "compare Fairfax and Montgomery County")
        geo_hint = f"""
Detected counties: {', '.join(counties)}.
Filter using "3 - County Name" with LIKE (and "1 - State Name" if needed).
Always include BOTH "1 - State Name" and "3 - County Name" in SELECT.
"""
    elif level == "mixed":
        # Question explicitly touches both states and counties
        geo_hint = f"""
Detected states: {', '.join(states)}
Detected counties: {', '.join(counties)}

Rules:
- Apply BOTH filters:
    WHERE "1 - State Name" LIKE '%STATE%'
      AND "3 - County Name" LIKE '%COUNTY%'
- Always include both columns in SELECT.
- When aggregating, GROUP BY "1 - State Name", "3 - County Name".
"""
    else:
        geo_hint = "No specific state or county detected."

    # ---------------------------------------
    # 4) Aggregation guidance (depends on intent + geo level)
    # ---------------------------------------
    if intent == "compare":
        # Compare intent gets more precise rules based on what was detected
        if level == "state" and states:
            aggregation_hint = f"""
The user is asking to compare STATES.

COMPARE RULES (MANDATORY – STATE LEVEL):
- You are comparing these states: {', '.join(states)}.
- ALWAYS use an aggregation function such as AVG(), COUNT(), or SUM()
  on metrics relevant to "condition" or similar.
- NEVER return raw bridge rows — only summarized results per state.
- ALWAYS GROUP BY ONLY "1 - State Name".
- Do NOT include "3 - County Name" in GROUP BY or SELECT unless the
  question explicitly mentions counties.
- Select only the aggregated metrics needed to answer the comparison.
- The output must have one row per state.
"""
        elif level == "county" and counties:
            aggregation_hint = f"""
The user is asking to compare COUNTIES.

COMPARE RULES (MANDATORY – COUNTY LEVEL):
- You are comparing these counties: {', '.join(counties)}.
- ALWAYS use an aggregation function such as AVG(), COUNT(), or SUM()
  on metrics relevant to the comparison.
- NEVER return raw bridge rows — only summarized results per county.
- ALWAYS GROUP BY "3 - County Name" (and include "1 - State Name" if useful).
- Select only the aggregated metrics needed to answer the comparison.
- The output must have one row per county.
"""
        elif level == "mixed":
            aggregation_hint = """
The user is asking to compare at a state+county hierarchy.

COMPARE RULES (MANDATORY – MIXED LEVEL):
- Use aggregation functions such as AVG(), COUNT(), or SUM().
- NEVER return raw bridge rows — only summarized results per (state, county).
- ALWAYS GROUP BY BOTH "1 - State Name" AND "3 - County Name".
- The output should have one row per (state, county) pair.
"""
        else:
            # Fallback when geo detection fails but intent is compare
            aggregation_hint = """
The user is asking to compare regions.

COMPARE RULES (MANDATORY – GENERIC):
- Use aggregation functions such as AVG(), COUNT(), or SUM().
- NEVER return raw bridge rows — only summarized results per region.
- GROUP BY the appropriate region column(s) (state or county) inferred from the question.
- The output should have one row per region.
"""
    elif intent in ["ranking", "aggregate", "trend", "policy"]:
        aggregation_hint = """
The user is asking for a high-level summary (ranking / aggregate / trend / policy).

- Aggregation is allowed (COUNT, AVG, SUM, etc.).
- Use GROUP BY only when needed (e.g., per state or county).
- Use ORDER BY when ranking or ordering is implied.
- Use LIMIT only when "top N", "bottom N", or similar is stated.
"""
    else:
        # filter / detail-like intents
        aggregation_hint = """
The user is asking for a detailed listing.

- Do NOT use GROUP BY unless explicitly required.
- Do NOT use aggregation unless explicitly required.
- Return raw bridge rows (individual bridges) that match the filters.
"""

    # ---------------------------------------
    # 5) Intent-specific SQL shaping
    # ---------------------------------------
    def intent_specific_guidance(intent: str, question: str) -> str:
        q = question.lower()

        if intent == "ranking":
            return """
INTENT: RANKING
- Use ORDER BY with a meaningful numeric column (e.g., traffic, counts, or condition).
- If the question uses "top", "highest", "busiest", "most" → ORDER BY that column DESC.
- If the question uses "lowest", "least", "smallest" → ORDER BY that column ASC.
- Apply LIMIT N when the user says "top N", "first N", etc.
"""
        if intent == "trend":
            return """
INTENT: TREND
- ONLY return the year column and the aggregated metric.
- NEVER include state or county columns unless the question compares multiple states.
- Detect a year-like column (contains 'year' or 'yr').
- GROUP BY the year column.
- ORDER BY the year column ASC.
- Use aggregate metrics such as AVG(), COUNT(), or SUM() depending on the question.
"""

        if intent == "compare":
            return """
INTENT: COMPARE
- Return one row per region (state or county, depending on the question).
- MUST use GROUP BY region column(s) – no raw detail rows.
- Include at least one numeric metric for comparison (e.g., average condition, count of bridges).
"""
        if intent == "aggregate":
            return """
INTENT: AGGREGATE
- Use aggregate functions: COUNT, AVG, SUM, MIN, MAX as appropriate.
- GROUP BY dimension columns only when needed (e.g., by state or county).
"""
        # default: filter / detail
        return """
INTENT: FILTER / DETAIL
- Return raw bridge rows that match the filters.
- Do NOT aggregate unless the user explicitly asks for a summary.
"""

    intent_hint = intent_specific_guidance(intent, question)

    # ---------------------------------------
    # 6) Column relevance hint (safe SQL)
    # ---------------------------------------
    col_hint = (
        ", ".join([f'"{c}"' for c in relevant_columns])
        if relevant_columns else "NONE"
    )

    # ---------------------------------------
    # 7) Final prompt sent to LLM
    # ---------------------------------------
    prompt = f"""
You are an expert SQL generator that outputs strictly valid SQLite SQL.

TABLE TO USE:
- Only use this table: {TABLE}

RELEVANT COLUMNS (do NOT use SELECT *):
{schema_block}

GEOGRAPHIC GUIDANCE:
{geo_hint}

AGGREGATION LOGIC:
{aggregation_hint}

INTENT-SPECIFIC RULES:
{intent_hint}

COLUMN GUIDANCE:
- Use only the columns shown above (no hallucinations).
- Prefer selecting only the columns needed for the question.
- Relevant columns: {col_hint}

SQL SAFETY RULES:
- ALWAYS wrap column names in double quotes.
- NEVER use SELECT *.
- NEVER modify data.
- NEVER use UPDATE, DELETE, INSERT, DROP, ALTER, TRUNCATE.
- ONLY return the SQL query (no explanation).
- Start the query text with SELECT.

Detected intent: {intent}
User question: {question}

Write ONE safe SQL SELECT query:

SQL:
""".strip()

    return prompt

def generate_sql(intent: str, question: str) -> str:
    """Main LLM call for SQL generation."""
    prompt = build_prompt(intent, question)

    openai_client = require_openai_client()
    try:
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes *correct and safe SQL*."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
    except Exception as error:
        stop_for_openai_error(error)

    # Get raw content from the new OpenAI SDK
    sql_raw = resp.choices[0].message.content.strip()
    sql_clean = clean_sql(sql_raw)

    # Ensure context columns are handled after SQL cleanup
    sql_with_context = enforce_context_columns(sql_clean, TABLE_NAME)

    return sql_with_context

# ------------------------------------------------------------
# Reorder columns for display: state first, then county
# ------------------------------------------------------------
def reorder_columns_for_display(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    """
    Simple & robust:
    - Put all 'state' columns first
    - Then all 'county' columns
    - Then everything else in original order
    """
    if df is None or df.empty:
        return df

    cols = list(df.columns)

    state_cols  = [c for c in cols if "state" in c.lower()]
    county_cols = [c for c in cols if "county" in c.lower()]
    remaining   = [c for c in cols if c not in state_cols + county_cols]

    new_cols = state_cols + county_cols + remaining
    new_cols = [c for c in new_cols if c in df.columns]

    return df[new_cols]

# ------------------------------------------------------------
# Keyword overrides for intent
# ------------------------------------------------------------
def force_intent_from_keywords(question: str, default_intent: str) -> str:
    """
    Post-process the embedding-based intent using simple keyword rules.
    This gives us cheap, very reliable overrides for obvious patterns.
    """
    q = question.lower()

    # ---- COMPARE / DIFFERENCE ----
    compare_triggers = [
        "compare ", "comparison", "compare bridges",
        "difference between", "differ between", "differ across",
        "versus", " vs ", "vs.", " vs. ",
        "how does", "how do", "relative to"
    ]
    if any(t in q for t in compare_triggers):
        return "compare"

    # ---- TREND / TIME EVOLUTION ----
    trend_triggers = [
        "trend", "over time", "by year", "across years",
        "change from", "change over", "evolution", "historical"
    ]
    if any(t in q for t in trend_triggers):
        return "trend"

    # ---- RANKING / TOP-N ----
    ranking_triggers = [
        "top ", "bottom ", "busiest", "most traffic", "least traffic",
        "highest", "lowest", "biggest", "smallest", "rank", "sorted by"
    ]
    if any(t in q for t in ranking_triggers):
        return "ranking"

    # ---- Default: whatever the embedding classifier said ----
    return default_intent

# ============================================================
# 8. QUERY HISTORY (JSON LOG)
# ============================================================

def save_query_history(question: str, sql: str):
    path = os.path.join(APP_DIR, "data", "query_history.json")
    entry = {"time": str(datetime.datetime.now()), "question": question, "sql": sql}
    data = []
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data.append(entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data[-500:], f, indent=2)

def load_query_history():
    path = os.path.join(APP_DIR, "data", "query_history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

# ============================================================
# 9. STREAMLIT UI – BridgesAI Production Dashboard
# ============================================================

st.set_page_config(
    page_title="BridgesAI · Bridge Infrastructure Intelligence",
    page_icon="🌉",
    layout="wide",
)

# ---- HEADER / HERO SECTION ----
header_col1, header_col2 = st.columns([3, 1], vertical_alignment="center")

with header_col1:
    st.markdown(
        """
        <h1 style="margin-bottom:0.2rem;">BridgesAI</h1>
        <p style="color:#BBBBBB; font-size:0.95rem; margin-top:0.1rem;">
            NLP-driven analytics on bridge infrastructure data – 
            ask questions in plain language, get ranked, filterable, 
            and explainable SQL results.
        </p>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    st.metric("Database", TABLE_NAME)
    st.caption(f"Backend: SQLite · File: {DB_PATH}")

st.markdown("---")

# ---- HIGH-LEVEL DESCRIPTION ----
st.markdown(
    """
    **What BridgesAI does**

    - Understands infrastructure and policy-style questions about bridges  
    - Classifies your intent (ranking, trends, aggregates, policy, etc.)  
    - Uses schema-aware RAG to generate safe, read-only SQL on the bridges dataset  
    - Returns sortable tables and automatic charts for quick decision support  
    """
)

st.markdown("")

# -------------------------------------------------
# Helper: pick visualization based on intent + data
# -------------------------------------------------
def pick_viz_axes(intent: str, df: pd.DataFrame):
    """
    Decide whether to plot and which columns to use, based on intent.

    Returns:
        ( (chart_type, x_col, y_col), df_for_viz )
        - chart_type in {"line", "bar"} or None
        - df_for_viz is usually df (or a lightly sorted copy)

    If no suitable chart can be determined, returns:
        ((None, None, None), df)
    """
    if df is None or df.empty:
        return (None, None, None), df

    cols = list(df.columns)
    if len(cols) < 2:
        return (None, None, None), df

    # Separate numeric vs non-numeric columns
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols     = [c for c in cols if c not in numeric_cols]

    # Helper: detect year-like columns
    def is_year_like(name: str) -> bool:
        n = name.lower()
        return ("year" in n) or n.endswith("yr") or "yr " in n

    year_cols = [c for c in cols if is_year_like(c)]

    # Helper: pick a "good" numeric metric (not id/code/year)
    def pick_metric(candidates):
        for c in candidates:
            cl = c.lower()
            if any(k in cl for k in ["code", "id"]):
                continue
            if "year" in cl or "yr" in cl:
                continue
            return c
        return candidates[0] if candidates else None

    # ========== TREND: line chart over time ==========
    if intent == "trend":
        x = None

        # Prefer explicit year-like columns
        if year_cols:
            preferred = [c for c in year_cols if "average daily traffic" in c.lower()]
            x = preferred[0] if preferred else year_cols[0]

        # Fallback: first numeric column
        if x is None and numeric_cols:
            x = numeric_cols[0]

        metric_candidates = [c for c in numeric_cols if c != x]
        y = pick_metric(metric_candidates)

        if x is None or y is None:
            return (None, None, None), df

        return ("line", x, y), df.copy()

    # ========== RANKING / COMPARE / AGGREGATE / POLICY ==========
    if intent in ["ranking", "compare", "aggregate", "policy"]:
        if not numeric_cols:
            return (None, None, None), df

        preferred_cat_order = [
            "1 - State Name",
            "3 - County Name",
            "State",
            "County",
        ]
        x = None
        for cand in preferred_cat_order:
            for c in cat_cols:
                if c.lower() == cand.lower():
                    x = c
                    break
            if x:
                break

        if x is None and cat_cols:
            x = cat_cols[0]

        if x is None:
            return (None, None, None), df

        y = pick_metric(numeric_cols)
        if y is None:
            return (None, None, None), df

        return ("bar", x, y), df.copy()

    # ========== Other intents: no chart ==========
    return (None, None, None), df

# ---- MAIN LAYOUT: LEFT = QUERY, RIGHT = RESULTS ----
left_col, right_col = st.columns([1.1, 1.6])

# =======================
# LEFT: QUERY PANEL
# =======================
with left_col:
    st.subheader("Ask BridgesAI")

    suggestions = [
        "Top 10 bridges with highest average daily traffic",
        "Which counties have the most bridges in poor condition?",
        "Trend of bridges built from 1950 to 2020",
        "Compare average traffic between two counties",
        "List structurally deficient bridges with high traffic",
    ]

    sel = st.selectbox(
        "Suggested questions",
        ["(Select a template)"] + suggestions,
        index=0,
    )

    default_q = sel if sel != "(Select a template)" else ""
    question = st.text_input(
        "Or type your own query",
        value=default_q,
        placeholder="e.g., Show the top 5 counties by number of structurally deficient bridges",
    )

    st.caption(
        "Tip: You can ask about traffic, year built, condition ratings, counties, "
        "or policy-style priorities such as ‘at-risk’ or ‘needs repair’."
    )

    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

# =======================
# RIGHT: RESULTS PANEL
# =======================
with right_col:
    if run_btn and question:
        if is_gibberish(question):
            st.warning("⚠ Please enter a meaningful question (more than a few characters).")
        elif not is_relevant_to_bridges(question):
            st.warning("⚠ This question doesn’t seem related to bridges or this dataset.")
        else:
            with st.spinner("Analyzing intent, generating SQL, and querying the database…"):

                # ---------------- INTENT PROCESSING ----------------
                intent_raw = classify_intent(question, threshold=0.60, use_llm_fallback=True)
                intent = force_intent_from_keywords(question, intent_raw)

                # ---------------- SQL GENERATION ----------------
                sql = generate_sql(intent, question)

                if not validate_sql(sql):
                    st.error("⚠ Unsafe or invalid SQL detected — query blocked for safety.")
                    st.code(sql, language="sql")
                else:
                    # ---------------- SQL EXECUTION ----------------
                    try:
                        df_result = query_db(sql)
                    except Exception as e:
                        st.error(f"⚠ SQL execution failed: {e}")
                        st.code(sql, language="sql")
                        df_result = None

                    save_query_history(question, sql)

                    # ---------------- DISPLAY INTENT + SQL ----------------
                    st.markdown(f"### 🧭 Detected intent: `{intent}`")
                    st.markdown("#### Generated SQL")
                    st.code(sql, language="sql")

                    # ======================================================
                    # RESULTS + VISUALIZATION
                    # ======================================================
                    if df_result is not None and not df_result.empty:
                        st.markdown("#### Query results")

                        # TREND → keep column order exactly; others → reorder
                        if intent == "trend":
                            df_display = df_result.copy()
                        else:
                            df_display = reorder_columns_for_display(df_result, sql)

                        st.dataframe(df_display, use_container_width=True)

                        # ---- Decide if we should plot, and how ----
                        (viz_info, df_for_viz) = pick_viz_axes(intent, df_display)
                        chart_type, x_col, y_col = viz_info if viz_info else (None, None, None)

                        # ==================================================
                        # TREND VISUALIZATION (fixed & reliable)
                        # ==================================================
                        if intent == "trend" and chart_type and x_col and y_col:
                            df_trend = df_for_viz.copy()

                            # 1) Extract YEAR (4-digit) safely from the x_col
                            if x_col in df_trend.columns:
                                df_trend[x_col] = (
                                    df_trend[x_col]
                                    .astype(str)
                                    .str.extract(r"(\d{4})")[0]
                                )

                            # 2) Convert to numeric Year / Value
                            df_trend["Year"] = pd.to_numeric(df_trend[x_col], errors="coerce")
                            df_trend["Value"] = pd.to_numeric(df_trend[y_col], errors="coerce")

                            # 3) Drop missing and sort
                            df_trend = df_trend.dropna(subset=["Year", "Value"])

                            if df_trend.empty:
                                st.info("Trend visualization unavailable — required columns missing.")
                            else:
                                df_trend = df_trend.sort_values("Year")
                                st.markdown("##### Trend over time")
                                st.line_chart(df_trend.set_index("Year")["Value"])

                        # ==================================================
                        # NON-TREND VISUALIZATION
                        # ==================================================
                        elif chart_type and x_col and y_col:
                            if x_col in df_for_viz.columns and y_col in df_for_viz.columns:
                                if chart_type == "line":
                                    st.markdown(f"##### Trend: {y_col} over {x_col}")
                                    st.line_chart(df_for_viz.set_index(x_col)[y_col])
                                elif chart_type == "bar":
                                    st.markdown(f"##### Distribution: {y_col} by {x_col}")
                                    st.bar_chart(df_for_viz.set_index(x_col)[y_col])
                            else:
                                st.info("Chart could not be generated — missing required columns.")

                        # ==================================================
                        # DOWNLOAD BUTTON
                        # ==================================================
                        csv = df_display.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download results as CSV",
                            data=csv,
                            file_name="bridgesai_query_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    else:
                        st.info("No rows returned. Try relaxing filters or changing the question.")

    elif run_btn and not question:
        st.info("Enter a question or select a template on the left, then click **Run analysis**.")

# =======================
# SIDEBAR: ABOUT + HISTORY
# =======================
st.sidebar.title("BridgesAI")
st.sidebar.caption("Bridge Infrastructure Intelligence")

history = load_query_history()
if history:
    st.sidebar.markdown("#### Recent queries")
    for item in history[-10:][::-1]:
        st.sidebar.write(f"🕒 {item['time']}")
        st.sidebar.caption(f"**Q:** {item['question']}")
        st.sidebar.code(item["sql"], language="sql")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **System notes**

    - Backend: SQLite on local file `bridges.db`  
    - Access: read-only, safe SQL only (no UPDATE / DELETE / DDL)  
    - NLP: OpenAI embeddings + chat model with schema-aware RAG  
    """
)
