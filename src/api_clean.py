"""
api_clean.py - LLM-based document parsing for IMO proposals
Updated: 2026-02-12 - Switched from Qwen to Claude (Anthropic API)
"""
import re
import os
import json

# Try Anthropic first, fall back to OpenAI-compatible
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from openai import OpenAI

# --- Configuration ---
# Priority: Anthropic Claude > OpenAI-compatible fallback
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Fallback: OpenAI-compatible endpoint (e.g., Qwen, local LLM)
FALLBACK_API_KEY = os.getenv("QWEN_API_KEY", os.getenv("OPENAI_API_KEY", ""))
FALLBACK_BASE_URL = os.getenv("QWEN_BASE_URL", "")

# Model selection
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # Best balance of cost/quality for structured extraction
FALLBACK_MODEL = os.getenv("LLM_MODEL", "qwen-plus")


def clean_text_regex(text):
    """Basic regex-based text cleaning."""
    text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', '', text)
    text = re.sub(r'\d{4}-\d{1,2}-\d{1,2}( \d{1,2}:\d{2}(:\d{2})?)?', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）《》【】,.!?;:"\'()\[\]<> \n]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = '\n'.join(line.strip() for line in text.splitlines())
    return text


SYSTEM_PROMPT = """
You are an expert document parsing AI, specializing in official documents from the International Maritime Organization (IMO), such as those for the Marine Environment Protection Committee (MEPC).

Your task is to receive the pre-cleaned text of **any** such document and convert it into a single, structured JSON object. You must **dynamically** find the information, not assume pre-set values.

**Instructions:**

1.  **Extract Metadata:** Dynamically parse the document header to find key metadata. If an item is not present, return `null`.
2.  **Extract Standard Sections:** Locate and extract the full text content for common logical sections (e.g., `Executive summary`, `Introduction`, `Action requested...`, `ANNEX`). If a section is not found, return `null`.
3.  **Reconstruct All Tables:** This is your most critical task.
    * Scan the entire document (especially any `ANNEX` sections) to locate **all** content that is formatted as a table.
    * These tables are often broken across lines and pages in the raw text.
    * You must reconstruct **each** table you find into a clean, well-formatted **Markdown string**.
    * Capture each table as an object within the `extracted_tables` array.

You must output **only** the completed JSON object based on the format below. Do not add any extra explanations or comments. The language should only be English. The output must be valid JSON.

**Output JSON Format:**
```json
{
"metadata": {
    "document_id": "[Extract from header, e.g., 'MEPC 78/3/1' or 'MEPC 80/INF.2']",
    "session": "[Extract from header, e.g., '78' or '80']",
    "agenda_item": "[Extract from header, e.g., '3' or null if not found]",
    "date": "[Extract from header, e.g., '8 July 2024', return "2024/7/8" format]",
    "title": "[Extract the main title, e.g., 'REDUCTION OF GHG EMISSION']",
    "subject": "[Extract the secondary title or subject, e.g., 'Draft amendments to MARPOL Annex II...']",
    "submitted_by": "[Extract the author/submitter, e.g., 'Submitted by China, Japan and South Korea', return a list of countries]"
},
"sections": {
    "summary": "[Extract text following 'Executive summary:', or 'SUMMARY', or null if not found]",
    "introduction": "[Extract text following the 'Introduction' heading, or null if not found]",
    "action_requested": "[Extract text following 'Action requested of the Committee' (or similar heading), or null if not found]",
    "annex_content": "[Extract all general text content found under the 'ANNEX' heading, *excluding* the tables themselves which are handled below. Null if no annex text.]"
},
"extracted_tables": [
    {
    "table_title": "[Extract the title for the first table found, e.g., 'Table 1 - Proposed Amendments']",
    "source_section": "[Identify where the table was found, e.g., 'ANNEX' or 'Body']",
    "markdown_content": "[The first table, fully reconstructed as a clean Markdown string]"
    }
]
}
```
"""


def llm_api_clean(input_path, output_path, api_key=None, base_url=None, model_name=None):
    """
    Parse and structure an IMO document using LLM.
    Priority: Claude (Anthropic) > OpenAI-compatible fallback.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found -> {input_path}")
        return

    user_prompt = "Please handle the following text to be processed according to your role and rules:\n\n" + raw_text

    cleaned_text = None

    # --- Try Claude (Anthropic) first ---
    if HAS_ANTHROPIC and ANTHROPIC_API_KEY:
        model = model_name or CLAUDE_MODEL
        print(f"Using Claude model '{model}' via Anthropic API...")
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            cleaned_text = response.content[0].text.strip()
            # Extract JSON from possible markdown code fence
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
        except Exception as e:
            print(f"Claude API error: {e}. Falling back...")
            cleaned_text = None

    # --- Fallback to OpenAI-compatible API ---
    if cleaned_text is None:
        _api_key = api_key or FALLBACK_API_KEY
        _base_url = base_url or FALLBACK_BASE_URL
        _model = model_name or FALLBACK_MODEL

        if not _api_key:
            print("Error: No API key available (neither ANTHROPIC_API_KEY nor fallback).")
            cleaned_text = json.dumps({})
        else:
            print(f"Using fallback model '{_model}' via OpenAI-compatible API...")
            try:
                client = OpenAI(api_key=_api_key, base_url=_base_url)
                response = client.chat.completions.create(
                    model=_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                cleaned_text = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Fallback API error: {e}")
                cleaned_text = json.dumps({})

    # Clean up think tags
    cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    print(f"Done! Cleaned file saved to: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        llm_api_clean(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python api_clean.py <input_path> <output_path>")
        print("\nEnvironment variables:")
        print("  ANTHROPIC_API_KEY  - Claude API key (preferred)")
        print("  QWEN_API_KEY       - Qwen/fallback API key")
        print("  QWEN_BASE_URL      - Fallback API base URL")
        print(f"\nCurrent config:")
        print(f"  Anthropic available: {HAS_ANTHROPIC}")
        print(f"  Anthropic key set: {bool(ANTHROPIC_API_KEY)}")
        print(f"  Fallback key set: {bool(FALLBACK_API_KEY)}")
