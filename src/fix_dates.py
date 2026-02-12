#!/usr/bin/env python3
"""Fix Date fields in a JSON file to YY/MM/DD format.

Usage:
    python src/fix_dates.py input.json [-o output.json] [--inplace] [--key KEY]

By default it writes to input_fixed.json. Use --inplace to overwrite the original file.
"""
import argparse
import json
import logging
import os
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Common date formats to try (day/month/year, month/day/year, iso)
FORMATS = [
    "%d/%m/%Y",
    "%d/%m/%y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m/%d/%y",
]

DATE_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$")


def try_parse_date(s: str):
    s = s.strip()
    # quick sanity
    if not s or not any(ch.isdigit() for ch in s):
        return None

    # try known formats
    for fmt in FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass

    # try ISO-ish
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass

    # fallback: regex (day/month/year or month/day/year)
    m = DATE_RE.match(s)
    if m:
        a, b, c = m.groups()
        a_i, b_i, c_i = int(a), int(b), int(c)
        # if year is 2-digit, interpret as 2000+ for 00-69 else 1900+
        if c_i < 100:
            c_i = 2000 + c_i if c_i <= 69 else 1900 + c_i
        # assume a=day, b=month by default
        try:
            return datetime(year=c_i, month=b_i, day=a_i)
        except Exception:
            # try as month/day/year
            try:
                return datetime(year=c_i, month=a_i, day=b_i)
            except Exception:
                return None

    return None


def process(obj, key_name="Date", changed=0):
    """Recursively walk `obj` and update values for keys named `key_name`.

    Returns the number of changed fields.
    """
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == key_name and isinstance(v, str):
                dt = try_parse_date(v)
                if dt:
                    new = dt.strftime("%Y/%m/%d")
                    if new != v:
                        logging.info("%s: %s -> %s", k, v, new)
                        obj[k] = new
                        changed += 1
                else:
                    logging.debug("Could not parse date: %s", v)
            else:
                changed += process(v, key_name)
    elif isinstance(obj, list):
        for i in obj:
            changed += process(i, key_name)
    return changed


def main():
    p = argparse.ArgumentParser(description="Normalize Date fields in JSON to YY/MM/DD")
    p.add_argument("input", help="Input JSON file")
    p.add_argument("-o", "--output", help="Output file (default: <input>_fixed.json)")
    p.add_argument("--inplace", action="store_true", help="Overwrite the input file")
    p.add_argument("--key", default="Date", help="Key name to search for (default: Date)")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = process(data, key_name=args.key)
    logging.info("Total fields changed: %d", changed)

    if args.inplace:
        out_path = args.input
        # make a small backup
        bak = args.input + ".bak"
        if not os.path.exists(bak):
            os.rename(args.input, bak)
            logging.info("Backup created: %s", bak)
        else:
            logging.info("Backup already exists: %s", bak)
    else:
        out_path = args.output or (os.path.splitext(args.input)[0] + "_fixed.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logging.info("Wrote output: %s", out_path)


if __name__ == "__main__":
    main()
