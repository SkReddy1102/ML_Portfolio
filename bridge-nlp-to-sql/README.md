# BridgesAI - NLP-to-SQL Bridge Infrastructure Explorer

BridgesAI is a Streamlit-based natural-language analytics application that allows users to ask questions about bridge infrastructure data and automatically converts those questions into safe SQLite queries.

The application combines OpenAI embeddings, intent classification, schema-aware retrieval, geographic entity detection, SQL generation, validation, and interactive visualization to make infrastructure data easier to explore for non-SQL users.

## Project Overview

Bridge datasets can contain many columns, inconsistent naming conventions, geographic fields, condition ratings, traffic information, and structural attributes.

Writing SQL manually for every analysis can be slow and difficult for non-technical users.

BridgesAI allows users to ask questions in natural language such as:

- Which counties have the most bridges in poor condition?
- Show the top 10 bridges with the highest average daily traffic.
- Compare average traffic between two counties.
- Show bridge construction trends over time.
- List structurally deficient bridges with high traffic.

The system interprets the request, identifies the relevant schema fields, generates a read-only SQL query, executes it against SQLite, and presents the results in an interactive Streamlit interface.

## System Pipeline

```text
Natural-Language Question
        |
        v
Domain Relevance Validation
        |
        v
Intent Classification
        |
        v
Geographic Entity Detection
        |
        v
Schema-Aware RAG
        |
        v
SQL Prompt Generation
        |
        v
LLM SQL Generation
        |
        v
SQL Cleaning & Safety Validation
        |
        v
SQLite Execution
        |
        v
Results Table
        |
        v
Automatic Visualization / CSV Export
```

## Key Features

### Natural-Language Querying

Users can query bridge infrastructure data without writing SQL manually.

### Domain Relevance Validation

The system checks whether a question is related to bridge infrastructure using:

- Domain-specific keywords
- OpenAI embeddings
- Cosine similarity

### Hybrid Intent Detection

Queries are classified into analytical intents including:

- Ranking
- Trend
- Compare
- Aggregate
- Policy
- Filter

The application combines embedding similarity, an LLM fallback, and keyword-based overrides to improve intent detection.

### Geographic Entity Recognition

The application identifies states and counties mentioned in user questions.

RapidFuzz is used for typo-tolerant matching so geographic names can still be detected when the user enters minor spelling errors.

### Schema-Aware RAG

The system retrieves the database columns most relevant to the user's question using embeddings.

Only relevant schema information is included in the SQL-generation prompt, helping reduce column hallucination and improve query accuracy.

### Safe SQL Generation

The application generates SQLite SELECT queries using an OpenAI chat model.

A validation layer blocks modifying operations including:

- DROP
- DELETE
- UPDATE
- INSERT
- ALTER
- TRUNCATE

Only read-only SELECT queries are allowed.

### SQLite Backend

The cleaned bridge dataset is automatically loaded into a local SQLite database when the application starts.

### Interactive Results

The Streamlit dashboard displays:

- Detected query intent
- Generated SQL
- Query results
- Automatic bar or line charts when appropriate
- Downloadable CSV results
- Recent query history

## Dataset

The project uses a cleaned bridge infrastructure dataset focused on bridges in Virginia and Maryland.

The dataset is derived from National Bridge Inventory data and contains approximately 18,500 bridge records with fields related to:

- State and county
- Bridge age
- Structural attributes
- Condition ratings
- Traffic volume
- Design characteristics
- Operational and clearance information

The cleaned dataset is stored in:

```text
data/cleaned_dataset.csv
```

## Technologies

- Python
- Streamlit
- OpenAI API
- OpenAI embeddings
- SQLite
- SQLAlchemy
- Pandas
- NumPy
- RapidFuzz
- RAG
- Natural Language Processing
- SQL

## Repository Structure

```text
bridge-nlp-to-sql/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- .gitignore
|
`-- data/
    `-- cleaned_dataset.csv
```

The application automatically creates:

```text
bridges.db
data/query_history.json
```

These generated files are excluded from Git using `.gitignore`.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## OpenAI API Key

The application reads the OpenAI API key from an environment variable.

In PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

Do not store API keys directly inside the source code or commit them to GitHub.

## Running the Application

From the project directory:

```bash
streamlit run app.py
```

The application will:

- Load `data/cleaned_dataset.csv`
- Create a local SQLite database
- Build the schema and intent embedding indexes
- Launch the BridgesAI Streamlit interface

## SQL Safety

BridgesAI is designed as a read-only analytics application.

Generated queries must begin with `SELECT`, and SQL commands that modify the database are blocked before execution.

The SQL-generation prompt also instructs the model to:

- Use only the bridge table
- Use retrieved schema columns
- Avoid `SELECT *`
- Wrap database column names correctly
- Generate only one SQL query
- Avoid data-modification operations

## Example Workflow

A user might ask:

```text
Which counties have the most bridges in poor condition?
```

BridgesAI then:

1. Validates that the question belongs to the bridge domain
2. Detects the analytical intent
3. Identifies relevant geographic entities
4. Retrieves relevant database columns
5. Generates a SQLite query
6. Checks the SQL for unsafe operations
7. Executes the query
8. Displays the results and visualization

## Academic Context

This project was developed as a graduate-level team project focused on applying natural-language processing and generative AI to infrastructure analytics.

The portfolio version presents the current working implementation of the application, including schema-aware RAG, intent detection, safe SQL generation, SQLite execution, and interactive visualization.

## Key Learning Outcomes

Through this project, I gained hands-on experience with:

- NLP-to-SQL system design
- Retrieval-Augmented Generation
- Embedding-based semantic similarity
- Prompt engineering
- SQL generation and validation
- SQLite and SQLAlchemy
- Natural-language intent classification
- Fuzzy geographic matching
- Streamlit application development
- LLM API integration
- Data visualization
- AI application safety controls

## Future Improvements

Potential improvements include:

- A verified few-shot SQL example retrieval layer
- Automated evaluation of generated SQL
- Execution-accuracy benchmarking
- Improved multi-turn conversational queries
- Expanded support for bridge data across additional states
- More advanced visualization recommendations
- Automated test coverage for SQL safety and intent classification
- Deployment using a hosted database and cloud application platform
