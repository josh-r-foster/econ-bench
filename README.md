# EconBench

**EconBench** is a benchmark for testing economic preferences, rationality, and decision-making capabilities in Large Language Models (LLMs). It simulates classic behavioral economics experiments to evaluate how agents handle risk, time, and social interaction.

## Overview

EconBench evaluates models across three core dimensions of economic behavior:
1.  **Risk & Rationality**: Tests adherence to Expected Utility Theory and the Independence Axiom using the Marschak-Machina Triangle.
2.  **Social Preferences**: Measures altruism and fairness through Dictator and Ultimatum Games.
3.  **Time Preferences**: Elicits discount rates and tests for dynamic consistency (e.g., present bias) using intertemporal choices.

## Installation

### Prerequisites
- Python 3.8+
- API keys for the models you intend to test (OpenAI, Anthropic, Google Gemini, etc.)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/econ-bench.git
    cd econ-bench
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GOOGLE_API_KEY=...
    # Add other keys as needed
    ```

## Usage

### Running Experiments

The benchmark consists of three main task scripts located in `src/tasks/`.

#### 1. Rationality & Risk (Independence Axiom)
Tests for violations of the Independence Axiom using the Marschak-Machina Triangle.
```bash
python src/tasks/independence.py --model gpt-4o
```
*Arguments:*
- `--model`: Model ID (e.g., `gpt-4o`, `claude-3-5-sonnet-20240620`, `gemini-1.5-pro`).
- `--n-divisions`: Grid density for the triangle (default: 7).
- `--verbose`: Print full interactions.

#### 2. Social Preferences
Runs the Dictator Game and Ultimatum Game (Propsoer & Responder roles).
```bash
python src/tasks/social.py --model gpt-4o
```
*Arguments:*
- `--model`: Model ID.
- `--repetitions`: Number of trials per condition.
- `--responder-repetitions`: Number of trials for responder scans.

#### 3. Time Preferences
Elicits discount rates and consistency parameters (Beta-Delta model).
```bash
python src/tasks/time.py --model gpt-4o
```
*Arguments:*
- `--model`: Model ID.
- `--n-iterations`: Precision of bisection (default: 10).

### Viewing Results (Dashboard)

EconBench includes a web-based dashboard to visualize results.

1.  **Start the local server:**
    ```bash
    # From the project root
    python3 -m http.server 8000
    ```

2.  **Open the dashboard:**
    Navigate to `http://localhost:8000/web/` in your browser.
    
    *Note: The dashboard reads data from `web/data/`. Experiment scripts automatically save web-ready JSON files there.*

## Supported Models

EconBench uses a model registry (`src/models/registry.py`) to handle various providers:

- **OpenAI**: `gpt-4o`, `gpt-4-turbo`, `o1-preview`, etc.
- **Anthropic**: `claude-3-5-sonnet-20240620`, `claude-3-opus-20240229`, etc.
- **Google**: `gemini-1.5-pro`, `gemini-1.5-flash`.
- **Open Source (via HF/vLLM)**: `meta-llama/Llama-3.1-70B-Instruct`, `Qwen/Qwen3-8B` (requires local setup).

To add a new model, update `src/models/registry.py` or use the appropriate prefix for API-based models.

## Repository Structure

```
econ-bench/
├── src/
│   ├── models/            # LLM interfaces and wrappers
│   ├── tasks/             # Experiment scripts (independence.py, social.py, time.py)
│   └── tools/             # Analysis and utility tools
├── web/                   # Frontend dashboard (HTML/JS/CSS)
│   └── data/              # JSON data for the dashboard
├── data/                  # Raw experiment results (CSV/Logs)
├── scripts/               # Helper scripts (benchmarking, leaderboards)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## License

See `LICENSE` file for details.
