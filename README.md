# LearnPath AI

LearnPath AI is an AI-powered learning assistant that helps students learn from their course materials through document-based question answering, quizzes, and progress tracking.

## Features

- Ingest course materials, including PDF, PPTX, and VTT files
- Ask questions grounded in retrieved learning material
- Generate extractive answers with source citations
- Practice through quizzes and assessments
- Track mastery and review learning topics
- Optional LLM integration with an offline extractive mode

## Tech Stack

- Python
- Flask
- Retrieval-based question answering
- HTML, CSS, and JavaScript frontend

## Prerequisites

- Python 3.11+ (use a version supported by the installed dependencies)
- pip
- Git

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/learnpath-ai.git
cd learnpath-ai
```

Create and activate a virtual environment on Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and configure an LLM provider. Keep API keys private. Offline extractive mode is available without an LLM key.

## Run the Application

```bash
python run.py
```

Open http://127.0.0.1:5000 in your browser.

## Run Tests

```bash
python -m pytest -q
```

## Run the Benchmark

```bash
python -m eval.run_benchmark --skip-simulation
```

## Security

Do not commit API keys, passwords, private course documents, uploaded user files, or local databases.

## Project Status

The existing automated test suite passed 147 tests in the development environment. Benchmark results depend on the evaluation dataset and configuration.
