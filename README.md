# llmquizmaker

`llmquizmaker` converts the text of PDF documents into thought provoking quiz
questions using a local large language model.

## Setup

1. Install Python 3.10 or later.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Ensure you have access to a compatible model such as
`mistralai/Mistral-7B-Instruct-v0.2`.

## Usage

Start the interactive agent with one or more PDFs:

```bash
python question_agent.py docs/lesson1.pdf docs/lesson2.pdf
```

Once running you can use the built‑in commands:

* `add <file>` – load another PDF.
* `gen 5` – generate five questions per loaded chunk.
* `set style "Ask in the voice of Socrates"` – customise the question style.
* `show` – display current configuration.
* `quit` – exit the program.

Configuration defaults can be changed in `config.json`.
