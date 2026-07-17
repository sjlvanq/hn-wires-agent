# Hacker News Wires Agent

> **🚧 Work in Progress** - This project is under active development. The ultimate goal is to build a **news curation assistant for content creation**.

A LangGraph-based conversational agent that enriches user conversations by associating inputs with relevant Hacker News articles using semantic search and vector embeddings.

This project was built to run on **small local models** - no need for expensive GPUs or cloud APIs. By breaking tasks into smaller pieces (like one agent just picking the right post, another just writing the response), each model only needs to handle simple jobs with limited context. The whole thing runs on your own machine, keeping your data private while still being smart enough to be useful.

The system follows a retrieval-selection-response pipeline:
1. **Retrieve**: Semantic search finds candidate posts
2. **Select**: LLM selects the most relevant post
3. **Fetch**: Retrieves full post details with wire analysis
4. **Respond**: Generates contextual response linking user input to selected post

## Screenshot

![Hacker News Wires Agent Screenshot](screenshot.png)

The project uses a SQLite database (`db.sqlite`) containing:
- Hacker News posts table
- Wire analysis summaries
- Vector embeddings for semantic search

The agent system used to construct this database is currently under review for publication.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Edit .env to configure Ollama models and database path
```

## Usage

```bash
python main.py
```

The agent will start an interactive CLI where you can ask questions and receive responses enriched with relevant Hacker News content.


## Configuration

Key environment variables in `.env`:
- `OLLAMA_BASE_URL`: Ollama API endpoint
- `OLLAMA_EMBEDDING_MODEL`: Model for embeddings (default: `all-minilm:33m`)
- `SELECTOR_LLM_MODEL`: Model for post selection
- `WRITER_LLM_MODEL`: Model for response generation
- `DATABASE_PATH`: Path to SQLite database

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Status

Experimental project for educational purposes.