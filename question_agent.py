#!/usr/bin/env python3
"""
Interactive PDF-to-question pipeline using a local Mistral-7B-Instruct model.
Run:

    python question_agent.py docs/lesson1.pdf docs/lesson2.pdf
"""
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import typer
import ruamel.yaml as yaml

from pdf_loader import pdf_to_text

# ---------- defaults ----------
CFG_FILE = "config.yaml"
DEFAULT_CFG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "load_4bit": True,
    "max_tokens": 512,
    "temperature": 0.7,
    "question_prompt": (
        "You are an expert tutor. Read the following passage and generate {n} "
        "clear, concise, thought-provoking questions that test deep understanding.\n\n"
        "{text}\n\nQUESTIONS:"
    ),
    "chunk_tokens": 2048,
}
# ------------------------------

app = typer.Typer(add_completion=False, invoke_without_command=True)  # Typer CLI


class QuestionAgent:
    """Holds the model once and serves questions on demand."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model, self.tokenizer = self._load_model()
        self.streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)
        self.device = next(self.model.parameters()).device
        self.corpus_chunks: List[str] = []

    # ---------- model ----------
    def _load_model(self):
        print("🔄  Loading model… (first time only)")
        kwargs = dict(
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        if self.cfg["load_4bit"]:
            kwargs.update(load_in_4bit=True)
        else:
            kwargs.update(load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name"])
        model = AutoModelForCausalLM.from_pretrained(self.cfg["model_name"], **kwargs)
        model.eval()
        return model, tokenizer

    # ---------- corpus ----------
    def add_pdf(self, path: Path):
        if not path.exists():
            print(f"❌  {path} not found.")
            return
        text = pdf_to_text(path)
        if not text.strip():
            print(f"⚠️  {path} appears empty.")
            return
        chunks = self._split_to_chunks(text)
        self.corpus_chunks.extend(chunks)
        print(
            f"✅  Added {path.name}: {len(chunks)} chunk(s), "
            f"{sum(len(c) for c in chunks):,} chars total."
        )

    def _split_to_chunks(self, text: str) -> List[str]:
        toks = self.tokenizer(text)["input_ids"]
        chunk_len = self.cfg["chunk_tokens"]
        return [
            self.tokenizer.decode(toks[i : i + chunk_len], skip_special_tokens=True)
            for i in range(0, len(toks), chunk_len)
        ]

    # ---------- generation ----------
    def _build_prompt(self, chunk: str, n: int) -> str:
        return self.cfg["question_prompt"].format(text=chunk, n=n)

    def ask(self, n_questions: int = 3):
        if not self.corpus_chunks:
            print("⚠️  No PDFs loaded yet.")
            return
        for chunk in self.corpus_chunks:
            prompt = self._build_prompt(chunk, n_questions)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            self.model.generate(
                **inputs,
                max_new_tokens=self.cfg["max_tokens"],
                temperature=self.cfg["temperature"],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=self.streamer,
            )
            # streamer prints as it generates – nothing else to do
        print("\n--- done ---\n")

    # ---------- config ----------
    def set_style(self, instruction: str):
        self.cfg["question_prompt"] = instruction + "\n\n{text}\n\nQUESTIONS:"
        print("📝  Question style updated.")

    def show_cfg(self):
        print(yaml.dump(self.cfg, Dumper=yaml.RoundTripDumper))

# ---------------- CLI commands ----------------
agent: Optional[QuestionAgent] = None  # will instantiate in main()


def _ensure_agent():
    global agent
    if agent is None:
        raise typer.Exit("💡  Agent not initialized (this should not happen).")


@app.command()
def add(path: str):
    """Add another PDF (can be run repeatedly)."""
    _ensure_agent()
    agent.add_pdf(Path(path))


@app.command()
def gen(n: int = typer.Argument(3, help="Number of questions per chunk.")):
    """Generate questions from currently loaded PDFs."""
    _ensure_agent()
    agent.ask(n)


@app.command()
def set(
    key: str = typer.Argument(..., help="`style` OR any cfg key"),
    value: str = typer.Argument(..., help="New value (enclose in quotes)")
):
    """Change config on the fly. Example:

        set style "Ask in the voice of Socrates, one question per line"
    """
    _ensure_agent()
    if key == "style":
        agent.set_style(value)
    else:
        agent.cfg[key] = yaml.safe_load(value)
        print(f"🔧  cfg[{key}] = {agent.cfg[key]}")


@app.command()
def show():
    """Print current configuration."""
    _ensure_agent()
    agent.show_cfg()


@app.command()
def quit():
    """Exit cleanly, freeing GPU memory."""
    raise typer.Exit()


# --------------- entry -----------------
def main(pdf_files: List[str]):
    global agent
    # read config if exists
    cfg = DEFAULT_CFG.copy()
    if Path(CFG_FILE).exists():
        cfg.update(yaml.safe_load(Path(CFG_FILE).read_text()))
    agent = QuestionAgent(cfg)
    for p in pdf_files:
        agent.add_pdf(Path(p))
    print(
        "\n💬  Type 'help' for commands. `add`, `gen`, `set`, `show`, `quit`.\n"
        "    Example: gen 5\n"
    )
    # hand control to Typer's REPL loop (blocks)
    app()

if __name__ == "__main__":
    main(sys.argv[1:])
