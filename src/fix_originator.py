"""
Fix missing Originator/Title in data_processed.json by extracting from PDF filenames.
PDF filename pattern: "MSC 108-12-2 - Title text (Country1, Country2).pdf"

Usage:
    python src/fix_originator.py --committees MSC CCC SSE
    python src/fix_originator.py --committees MSC --dry-run
"""

import os
import re
import json
import argparse


def extract_from_filename(filename):
    """Extract Symbol, Title, and Originator from PDF filename.
    
    Pattern: "MSC 108-12-2 - Title text (Country1, Country2 and Country3).pdf"
    Also handles: "MSC 108-12-2 - Title text (Country1 and...).pdf" (truncated)
    """
    name = os.path.splitext(filename)[0]  # remove .pdf
    
    # Extract originator from parentheses at end
    originator = ''
    orig_match = re.search(r'\(([^)]+)\)\s*$', name)
    if orig_match:
        originator = orig_match.group(1).strip()
        # Clean up truncation markers
        originator = re.sub(r'\.\.\.$', '', originator).strip()
        originator = re.sub(r',\s*$', '', originator).strip()
        # Remove the originator part from name for title extraction
        name = name[:orig_match.start()].strip()
    
    # Extract symbol and title
    # Pattern: "MSC 108-12-2 - Title text" (note: " - " with spaces separates symbol from title)
    # Must use " - " (with spaces) to avoid splitting on hyphens in the symbol itself
    title_match = re.match(r'^(.+?)\s+-\s+(.+)$', name)
    if title_match:
        symbol_raw = title_match.group(1).strip()
        title = title_match.group(2).strip()
    else:
        symbol_raw = name.strip()
        title = ''
    
    # Normalize symbol: "MSC 108-12-2" -> "MSC 108/12/2"
    # But keep the raw form too for matching
    symbol_norm = re.sub(r'-', '/', symbol_raw)
    
    return {
        'symbol_raw': symbol_raw,
        'symbol_norm': symbol_norm,
        'title': title,
        'originator': originator
    }


def normalize_symbol(sym):
    """Normalize symbol for matching: MSC 108/12/2 -> MSC 108/12/2"""
    sym = sym.strip()
    # Remove Rev, Add suffixes for matching flexibility
    # But keep them for exact match first
    return sym


def match_and_update(data_processed, filename_info, logger_lines):
    """Match filename info to data_processed entries and update missing fields."""
    # Build lookup from filenames
    fn_lookup = {}
    for info in filename_info:
        sym = info['symbol_norm']
        fn_lookup[sym] = info
        # Also add lowercase version
        fn_lookup[sym.lower()] = info
    
    updated = 0
    matched = 0
    
    for doc in data_processed:
        sym = doc.get('Symbol', '').strip()
        if not sym:
            continue
        
        # Try exact match
        info = fn_lookup.get(sym) or fn_lookup.get(sym.lower())
        
        # Try with common variations
        if not info:
            # MSC 108/12/2/Rev.1 -> try MSC 108/12/2/REV.1
            for key, val in fn_lookup.items():
                if key.lower().replace(' ', '') == sym.lower().replace(' ', ''):
                    info = val
                    break
        
        if info:
            matched += 1
            changed = False
            
            # Update Originator if empty
            if not doc.get('Originator', '').strip() and info['originator']:
                doc['Originator'] = info['originator']
                changed = True
            
            # Update Title if empty
            if not doc.get('Title', '').strip() and info['title']:
                doc['Title'] = info['title']
                changed = True
            
            # Update Originator_split
            if changed and info['originator']:
                origs = split_originator(info['originator'])
                doc['Originator_split'] = origs
                updated += 1
    
    return matched, updated


def split_originator(orig_str):
    """Split originator string into list of individual countries/orgs."""
    if not orig_str:
        return []
    
    # Replace common separators
    # "Belgium, Cyprus, Denmark, France, Germany and Greece" -> list
    # "Liberia and ICS" -> list
    # "Australia, New Zealand, U..." -> list (handle truncation)
    
    parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', orig_str)
    result = []
    for p in parts:
        p = p.strip().strip('.')
        if p and len(p) > 1:
            result.append(p)
    return result


def process_committee(committee, data_dir, output_dir, dry_run=False):
    """Process one committee: extract from filenames, update data_processed.json."""
    print(f"\n{'='*60}")
    print(f"Processing {committee}...")
    print(f"{'='*60}")
    
    data_comm = os.path.join(data_dir, committee)
    output_comm = os.path.join(output_dir, committee)
    
    if not os.path.isdir(data_comm):
        print(f"  Data directory not found: {data_comm}")
        return
    
    # Get all session folders
    sessions = sorted([d for d in os.listdir(data_comm) 
                       if os.path.isdir(os.path.join(data_comm, d)) and d.startswith(committee.split('-')[0])])
    
    total_updated = 0
    total_matched = 0
    total_docs = 0
    
    for session in sessions:
        session_data = os.path.join(data_comm, session)
        session_output = os.path.join(output_comm, session)
        json_path = os.path.join(session_output, 'data_processed.json')
        
        if not os.path.exists(json_path):
            print(f"  {session}: No data_processed.json, skipping")
            continue
        
        # Get PDF filenames
        pdfs = [f for f in os.listdir(session_data) if f.lower().endswith('.pdf')]
        if not pdfs:
            print(f"  {session}: No PDFs found in data dir")
            continue
        
        # Extract info from filenames
        filename_info = [extract_from_filename(f) for f in pdfs]
        
        # Check how many have originators
        has_orig = sum(1 for fi in filename_info if fi['originator'])
        
        # Load data_processed.json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_docs += len(data)
        
        # Check if originator already exists
        existing_orig = sum(1 for d in data if d.get('Originator', '').strip())
        
        if existing_orig > 0 and existing_orig == len(data):
            print(f"  {session}: All {len(data)} docs already have Originator, skipping")
            total_matched += len(data)
            continue
        
        # Match and update
        logger_lines = []
        matched, updated = match_and_update(data, filename_info, logger_lines)
        total_matched += matched
        total_updated += updated
        
        print(f"  {session}: {len(data)} docs, {len(pdfs)} PDFs ({has_orig} with orig), matched={matched}, updated={updated}")
        
        if updated > 0 and not dry_run:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"    -> Saved updated data_processed.json")
    
    print(f"\n  Summary for {committee}:")
    print(f"    Total docs: {total_docs}")
    print(f"    Matched: {total_matched}")
    print(f"    Updated: {total_updated}")
    if dry_run:
        print(f"    (DRY RUN - no files modified)")


def regenerate_assignments(committee, output_dir):
    """Regenerate bertopic_assignments CSV with updated originator data."""
    bertopic_dir = os.path.join(output_dir, committee, 'bertopic')
    assignments_path = os.path.join(bertopic_dir, f'bertopic_assignments_{committee}.csv')
    
    if not os.path.exists(assignments_path):
        print(f"  No assignments file for {committee}")
        return
    
    import pandas as pd
    df = pd.read_csv(assignments_path, encoding='utf-8-sig')
    
    # Load all data_processed.json for this committee
    comm_dir = os.path.join(output_dir, committee)
    sessions = sorted([d for d in os.listdir(comm_dir) 
                       if os.path.isdir(os.path.join(comm_dir, d)) and d.startswith(committee.split('-')[0])])
    
    # Build symbol -> originator lookup
    orig_lookup = {}
    for session in sessions:
        json_path = os.path.join(comm_dir, session, 'data_processed.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for doc in data:
                sym = doc.get('Symbol', '')
                orig = doc.get('Originator', '')
                if sym and orig:
                    orig_lookup[sym] = orig
    
    # Update assignments
    updated = 0
    for idx, row in df.iterrows():
        sym = row.get('symbol', '')
        if sym in orig_lookup and (pd.isna(row.get('originator')) or not str(row.get('originator', '')).strip()):
            df.at[idx, 'originator'] = orig_lookup[sym]
            updated += 1
    
    if updated > 0:
        df.to_csv(assignments_path, index=False, encoding='utf-8-sig')
        print(f"  Updated {updated} rows in {assignments_path}")
    else:
        print(f"  No updates needed for assignments")
    
    return updated


def main():
    parser = argparse.ArgumentParser(description='Fix missing Originator from PDF filenames')
    parser.add_argument('--committees', nargs='+', default=['MSC', 'CCC', 'SSE', 'ISWG-GHG', 'MEPC'])
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    parser.add_argument('--regen-assignments', action='store_true', help='Also regenerate bertopic assignments CSV')
    args = parser.parse_args()
    
    for committee in args.committees:
        process_committee(committee, args.data_dir, args.output_dir, args.dry_run)
        
        if args.regen_assignments and not args.dry_run:
            print(f"\n  Regenerating assignments for {committee}...")
            regenerate_assignments(committee, args.output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
