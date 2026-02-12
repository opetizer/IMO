#!/usr/bin/env python3
"""Extract Date, Subject and Originator_split from JSON outputs under `output/`.

Usage:
    python src/extract_metadata.py --input output --output output/all_proposals_metadata.csv

The script prefers `data_processed.json` then `data.json` then `data_parsed.json` in each directory
and will write a single CSV containing one row per proposal/document.
"""
import os
import json
import csv
import argparse
import logging

PREFERRED_FILES = ["data_processed.json"]


def setup_logger():
    logger = logging.getLogger("extract_metadata")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def find_candidate_files(root_dir, logger):
    """Walk the tree and pick one preferred JSON per directory (if any)."""
    candidates = []
    for current_dir, dirs, files in os.walk(root_dir):
        for fname in PREFERRED_FILES:
            if fname in files:
                candidates.append(os.path.join(current_dir, fname))
                break
    logger.info(f"Found {len(candidates)} candidate JSON files under {root_dir}")
    return candidates


def extract_from_json_file(json_path, logger):
    """Load json and return a list of document dicts or empty list on failure."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load JSON {json_path}: {e}")
        return []

    # Common patterns: data is a list of dicts, or a dict containing a list under a key
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try several likely keys
        for key in ('documents', 'items', 'data', 'results'):
            if key in data and isinstance(data[key], list):
                return data[key]
        # If dict's values are all dicts keyed by filename, flatten
        # Or if this dict looks like mapping filename->entry
        values = [v for v in data.values() if isinstance(v, dict)]
        if values:
            return values
    logger.warning(f"JSON at {json_path} does not contain a list of documents; skipping.")
    return []


def normalize_originator_split(value):
    if value is None:
        return ''
    if isinstance(value, list):
        try:
            # Flatten list items to strings and join
            return ' | '.join(str(x).strip() for x in value if x is not None)
        except Exception:
            return json.dumps(value, ensure_ascii=False)
    # Otherwise, return string
    return str(value)


def collect_metadata(json_files, logger):
    rows = []
    for jfile in json_files:
        docs = extract_from_json_file(jfile, logger)
        for entry in docs:
            if not isinstance(entry, dict):
                continue
            row = {
                'Council': jfile.split(os.sep)[1],
                'Session': jfile.split(os.sep)[2].split(' ')[1],
                'Date': entry.get('Date', '') or entry.get('date', ''),
                'Subject': entry.get('Subject', '') or entry.get('subject', ''),
                'Originator_split': normalize_originator_split(entry.get('Originator_split', entry.get('Originator split', entry.get('originator_split', None))))
            }
            rows.append(row)
    logger.info(f"Collected {len(rows)} rows of metadata.")
    return rows


def write_csv(rows, out_path, logger):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    headers = ['Council', 'Session', 'Date', 'Subject', 'Originator_split']
    try:
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        logger.info(f"Wrote CSV to {out_path}")
    except Exception as e:
        logger.error(f"Error writing CSV {out_path}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Date, Subject and Originator_split from output JSONs")
    parser.add_argument('--input', type=str, default='output', help='Root output directory to scan')
    parser.add_argument('--output', type=str, default=os.path.join('output', 'all_proposals_metadata.csv'), help='Output CSV path')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    logger.info(f"Scanning {args.input} for proposal JSON files...")
    json_files = find_candidate_files(args.input, logger)
    rows = collect_metadata(json_files, logger)

    if rows:
        write_csv(rows, args.output, logger)
    else:
        logger.warning("No rows collected; CSV will not be written.")


if __name__ == '__main__':
    main()
