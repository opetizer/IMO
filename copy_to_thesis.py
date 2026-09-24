"""
Copy analysis outputs to thesis folder.
Only copies *.png, *.html, *.txt, *.csv files.
Mirrors the output structure.
Special handling for cooccurrence PNGs in meeting subdirectories.
"""
import os
import shutil
from pathlib import Path

SOURCE_ROOT = r"D:\university\Research\IMO\output"
TARGET_ROOT = r"D:\university\毕业设计\毕业论文\output"

# Extensions to copy
INCLUDE_EXTENSIONS = {'.png', '.html', '.txt', '.csv'}
# Extensions to exclude
EXCLUDE_EXTENSIONS = {'.json', '.jsonl', '.pkl', '.model', '.bin', '.npy'}

def copy_file(src, dst):
    """Copy file if it doesn't exist or is older than source."""
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  Copy: {src}")
    elif os.path.getmtime(src) > os.path.getmtime(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  Update: {src}")
    else:
        print(f"  Skip: {dst} (up to date)")

def main():
    print("=" * 60)
    print("Copying IMO analysis outputs to thesis folder...")
    print("=" * 60)

    # Directories to mirror (excluding meeting data subdirectories)
    main_dirs = [
        "MEPC/alliance",
        "MEPC/citation", 
        "MEPC/bertopic",
        "MEPC/visualization",
        "MSC/alliance",
        "MSC/citation",
        "MSC/bertopic",
        "MSC/visualization",
        "CCC/alliance",
        "CCC/citation",
        "CCC/bertopic",
        "CCC/visualization",
        "SSE/alliance",
        "SSE/citation",
        "SSE/bertopic",
        "SSE/visualization",
        "ISWG-GHG/alliance",
        "ISWG-GHG/citation",
        "ISWG-GHG/bertopic",
        "ISWG-GHG/visualization",
        "stance_analysis",
        "dynamic_analysis",
        "deep_analysis",
        "visualization",
    ]

    files_copied = 0
    files_skipped = 0

    for rel_dir in main_dirs:
        src_dir = os.path.join(SOURCE_ROOT, rel_dir)
        if not os.path.exists(src_dir):
            print(f"\nSkipping {rel_dir} (not found)")
            continue

        print(f"\nProcessing: {rel_dir}")
        target_dir = os.path.join(TARGET_ROOT, rel_dir)

        for root, dirs, files in os.walk(src_dir):
            for fname in files:
                fext = os.path.splitext(fname)[1].lower()

                # Skip excluded files
                if fext in EXCLUDE_EXTENSIONS:
                    continue

                # Only include specified extensions
                if fext not in INCLUDE_EXTENSIONS:
                    continue

                src_path = os.path.join(root, fname)
                rel_path = os.path.relpath(src_path, SOURCE_ROOT)
                dst_path = os.path.join(TARGET_ROOT, rel_path)

                try:
                    copy_file(src_path, dst_path)
                    files_copied += 1
                except Exception as e:
                    print(f"  Error copying {fname}: {e}")
                    files_skipped += 1

    # Special handling: cooccurrence PNGs from meeting subdirectories
    print("\n" + "=" * 60)
    print("Special handling: Cooccurrence graphs...")
    print("=" * 60)

    committees = ["MEPC", "MSC", "CCC", "SSE", "ISWG-GHG"]
    for comm in committees:
        meeting_dir = os.path.join(SOURCE_ROOT, comm)
        if not os.path.exists(meeting_dir):
            continue

        # Create cooccurrence output dir for this committee
        coocc_target = os.path.join(TARGET_ROOT, comm, "cooccurrence")
        os.makedirs(coocc_target, exist_ok=True)

        # Find all cooccurrence PNGs in meeting subdirectories
        for item in os.listdir(meeting_dir):
            item_path = os.path.join(meeting_dir, item)
            if not os.path.isdir(item_path):
                continue

            # Skip if this is a known output directory
            if item.lower() in ['alliance', 'citation', 'bertopic', 'visualization']:
                continue

            # Look for cooccurrence files in this meeting directory
            meeting_files = []
            if os.path.exists(os.path.join(item_path, 'cooccurrence_graph.png')):
                meeting_files.append('cooccurrence_graph.png')
            if os.path.exists(os.path.join(item_path, 'cooccurrence_overall.png')):
                meeting_files.append('cooccurrence_overall.png')

            # Also check per_agenda subdirectory
            per_agenda_dir = os.path.join(item_path, 'per_agenda')
            if os.path.exists(per_agenda_dir):
                for fname in os.listdir(per_agenda_dir):
                    if fname.endswith('.png'):
                        meeting_files.append(f'per_agenda/{fname}')

            for co_file in meeting_files:
                src = os.path.join(item_path, co_file)
                # Rename: MEPC_77 -> MEPC_77_cooccurrence.png
                safe_meeting_name = item.replace('/', '_').replace(' ', '_').replace('\\', '_')
                if co_file.startswith('per_agenda/'):
                    # per_agenda/cooccurrence_agenda_4.png -> MEPC_77_agenda_4_cooccurrence.png
                    base_name = co_file.replace('per_agenda/', '').replace('cooccurrence_agenda_', '').replace('.png', '')
                    dst_name = f"{safe_meeting_name}_agenda_{base_name}_cooccurrence.png"
                else:
                    # cooccurrence_graph.png or cooccurrence_overall.png
                    suffix = co_file.replace('.png', '').replace('cooccurrence_', '')
                    dst_name = f"{safe_meeting_name}_{suffix}_cooccurrence.png"
                    if dst_name.endswith('_cooccurrence_cooccurrence'):
                        dst_name = dst_name.replace('_cooccurrence_cooccurrence', '_cooccurrence')

                dst = os.path.join(coocc_target, dst_name)
                try:
                    copy_file(src, dst)
                    files_copied += 1
                except Exception as e:
                    print(f"  Error copying cooccurrence {item}/{co_file}: {e}")
                    files_skipped += 1

    # Summary
    print("\n" + "=" * 60)
    print("Copy complete!")
    print(f"  Files processed: {files_copied}")
    print(f"  Files skipped/errors: {files_skipped}")
    print("=" * 60)

if __name__ == "__main__":
    main()
