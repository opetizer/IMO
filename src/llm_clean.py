"""
llm_clean.py - Clean and format IMO participant lists using LLM
Updated: 2026-02-12 - Switched from Ollama/deepseek-r1 to Claude (Anthropic API)
"""
import re
import os
import json

# Try Anthropic first, fall back to Ollama
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from ollama import Client as OllamaClient
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"


def clean_text(text):
    """Basic regex cleaning."""
    text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', '', text)
    text = re.sub(r'\d{4}-\d{1,2}-\d{1,2}( \d{1,2}:\d{2}(:\d{2})?)?', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：\u201c\u201d\u2018\u2019（）《》【】,.!?;:"\'\(\)\[\]<> \n]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = '\n'.join(line.strip() for line in text.splitlines())
    return text


SYSTEM_PROMPT = """You are a specialized assistant for processing official IMO meeting participant lists.

Rules:
1. Remove all document metadata (file paths, headers, footers, page numbers, document codes, dates, session info).
2. Remove meeting summary info (committee name, chair/vice-chair introductions).
3. Format entries:
   - Country/organization names as first-level headings
   - Position group titles (Head of Delegation, Representatives, Advisers) as second-level headings  
   - Merge each person's name, title, and institution onto a single line, separated by ", "
4. Remove empty groups and compress excessive blank lines.
5. Preserve all sections (country delegations, international organizations, observers, secretariat).

Core principles:
- Maintain original order strictly
- Never modify, abbreviate, or change any names, titles, or institution names
- Output only the formatted list, no preambles or summaries"""


def ollama_clean(input_path, output_path):
    """Clean participant list using Claude (preferred) or Ollama fallback."""
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    cleaned_text = None

    # --- Try Claude first ---
    if HAS_ANTHROPIC and ANTHROPIC_API_KEY:
        print(f"Using Claude ({CLAUDE_MODEL}) for participant list cleaning...")
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Please clean and format the following participant list:\n\n{raw_text}"}],
                temperature=0.0,
            )
            cleaned_text = response.content[0].text.strip()
        except Exception as e:
            print(f"Claude API error: {e}. Falling back...")
            cleaned_text = None

    # --- Fallback to Ollama ---
    if cleaned_text is None and HAS_OLLAMA:
        print("Using Ollama (deepseek-r1:8b) fallback...")
        try:
            client = OllamaClient()
            prompt = SYSTEM_PROMPT + "\n\nPlease clean and format the following participant list:\n\n" + raw_text
            response = client.generate(model="deepseek-r1:8b", prompt=prompt)
            cleaned_text = response['response'].strip()
        except Exception as e:
            print(f"Ollama error: {e}")
            cleaned_text = None

    if cleaned_text is None:
        print("ERROR: No LLM backend available. Using regex cleaning only.")
        cleaned_text = clean_text(raw_text)

    # Remove <think> tags
    cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)

    if not cleaned_text:
        print("Warning: Cleaned text is empty, check input file.")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"Done! Saved to: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        ollama_clean(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python llm_clean.py <input_path> <output_path>")
        print(f"\nBackends: Anthropic={HAS_ANTHROPIC} (key={'set' if ANTHROPIC_API_KEY else 'unset'}), Ollama={HAS_OLLAMA}")
