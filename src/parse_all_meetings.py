#####
# Batch parse all meeting data.json files
# 批量解析所有会议的 data.json 并保存为 data_parsed.json
#
# Usage:
# python parse_all_meetings.py --meeting_folder "output/MEPC"
#####

import os
import sys
import json
import argparse
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from json_read import load_data


def natural_meeting_sort_key(name):
    """Sort meeting names naturally (e.g., MEPC 77, MEPC 78, ...)"""
    import re
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    return name


def find_meeting_dirs(base_folder):
    """Find all meeting subdirectories containing data.json"""
    items = []
    for entry in os.listdir(base_folder):
        path = os.path.join(base_folder, entry)
        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, 'data.json')):
                items.append(entry)
    items_sorted = sorted(items, key=natural_meeting_sort_key)
    return items_sorted


def parse_meeting(base_folder, meeting_dir, skip_existing=True):
    """Parse a single meeting's data.json and save as data_parsed.json"""
    input_path = os.path.join(base_folder, meeting_dir, 'data.json')
    output_path = os.path.join(base_folder, meeting_dir, 'data_parsed.json')

    # Skip if already exists
    if skip_existing and os.path.exists(output_path):
        return None, 'skipped'

    try:
        # Load and parse data
        df = load_data(input_path)

        if df.empty:
            return None, 'empty'

        # Convert to dict for JSON serialization
        data = df.to_dict('records')

        # Save as data_parsed.json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return len(data), 'success'

    except Exception as e:
        print(f"  Error processing {meeting_dir}: {e}")
        return None, 'error'


def main():
    parser = argparse.ArgumentParser(
        description='Batch parse all meeting data.json files to data_parsed.json'
    )
    parser.add_argument('--meeting_folder', type=str, required=True,
                        help='Path to folder containing meeting subfolders (e.g., output/MEPC)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip meetings that already have data_parsed.json (default: True)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing data_parsed.json files')
    args = parser.parse_args()

    base = args.meeting_folder

    # Find all meetings
    meetings = find_meeting_dirs(base)

    if not meetings:
        print(f'No meeting subfolders with data.json found in {base}')
        return

    print(f'Found {len(meetings)} meetings:')
    for m in meetings:
        print(f'  - {m}')

    # Skip existing if not forcing
    skip_existing = args.skip_existing and not args.force

    print(f'\nProcessing meetings...')
    print(f'Skip existing: {skip_existing}')
    print(f'Force overwrite: {args.force}')
    print()

    # Process each meeting
    results = {
        'success': [],
        'skipped': [],
        'error': [],
        'empty': []
    }

    for meeting in tqdm(meetings, desc="Parsing meetings"):
        count, status = parse_meeting(base, meeting, skip_existing=skip_existing)

        if status == 'success':
            results['success'].append((meeting, count))
            print(f"  [OK] {meeting}: {count} records")
        elif status == 'skipped':
            results['skipped'].append(meeting)
            print(f"  [SKIP] {meeting}: already exists")
        elif status == 'empty':
            results['empty'].append(meeting)
            print(f"  [EMPTY] {meeting}: no data")
        else:
            results['error'].append(meeting)
            print(f"  [ERROR] {meeting}: failed to parse")

    # Summary
    print(f'\n{"="*60}')
    print(f'Summary:')
    print(f'  Successfully processed: {len(results["success"])}')
    print(f'  Skipped: {len(results["skipped"])}')
    print(f'  Empty: {len(results["empty"])}')
    print(f'  Errors: {len(results["error"])}')
    print(f'{"="*60}')

    if results['success']:
        print(f'\nSuccessfully processed meetings:')
        for meeting, count in results['success']:
            print(f'  - {meeting}: {count} records')


if __name__ == '__main__':
    main()
