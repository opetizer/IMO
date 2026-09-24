from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_COMMITTEES = ["MEPC", "MSC", "CCC", "SSE", "ISWG-GHG"]
MODULE_LABELS = {
    "alliance": "Alliance Network",
    "citation": "Citation Network",
    "visualization": "Topic Visualization",
    "stance_analysis": "Country Stance",
    "dynamic_analysis": "Dynamic Topics",
    "deep_analysis": "Cross-Committee Deep Analysis",
    "misc": "Other Figures",
}


@dataclass(frozen=True)
class ImageRecord:
    source: Path
    target: Path
    committee: str
    module: str
    chart_key: str
    title: str


def _detect_committee(rel: Path) -> str:
    first = rel.parts[0] if rel.parts else "misc"
    if first in DEFAULT_COMMITTEES:
        return first
    return "cross-committee"


def _detect_module(rel: Path) -> str:
    parts = rel.parts
    if len(parts) >= 2 and parts[1] in {"alliance", "citation", "visualization"}:
        return parts[1]
    if parts and parts[0] in {"dynamic_analysis", "deep_analysis", "visualization", "stance_analysis"}:
        return parts[0]
    stem = rel.stem.lower()
    if "alliance" in stem:
        return "alliance"
    if "citation" in stem or "similarity_clusters" in stem:
        return "citation"
    if any(token in stem for token in ("topic_", "wordcloud", "country_trends", "committee_comparison", "cross_committee_volume")):
        return "visualization"
    return "misc"


def _normalize_chart_key(stem: str, committee: str) -> str:
    key = stem.lower()
    committee_key = committee.lower()
    key = key.replace(f"{committee_key}_", "")
    key = key.replace(f"_{committee_key}", "")
    key = key.replace("iswg-ghg_", "")
    key = key.replace("_iswg-ghg", "")
    key = key.replace("country_proposal_trends", "country_trends")
    key = key.replace("topic_visualization_interpretation", "visualization_interpretation")
    return key


def _title_from_chart_key(chart_key: str) -> str:
    return chart_key.replace("_", " ").replace("ghg", "GHG").title().replace("Iswg-Ghg", "ISWG-GHG")


def _record_priority(record: ImageRecord) -> tuple[int, int, int, str]:
    rel_parts = record.source.parts
    module_depth = 0 if record.module in rel_parts else 1
    prefix_bonus = 0 if record.source.stem.startswith(record.committee) else 1
    length_score = len(record.source.name)
    return (module_depth, prefix_bonus, length_score, str(record.source))


def run_step(command: list[str], cwd: Path) -> None:
    print(f"\n>>> Running: {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd), check=True)


def existing_committee_dirs(base_dir: Path, committees: list[str]) -> list[str]:
    return [committee for committee in committees if (base_dir / committee).exists()]


def rerun_pipeline(project_root: Path, base_dir: Path, committees: list[str], skip_bertopic: bool) -> None:
    python_exe = sys.executable
    src_dir = project_root / "src"
    available = existing_committee_dirs(base_dir, committees)
    if not available:
        raise RuntimeError(f"No committee folders found under {base_dir}")

    if not skip_bertopic:
        for committee in available:
            run_step(
                [
                    python_exe,
                    str(src_dir / "bertopic_model.py"),
                    "--meeting_folder",
                    str(base_dir / committee),
                    "--dynamic",
                ],
                project_root,
            )

    for committee in available:
        committee_dir = base_dir / committee
        run_step(
            [python_exe, str(src_dir / "alliance_network.py"), "--meeting_folder", str(committee_dir), "--per_topic"],
            project_root,
        )
        run_step(
            [python_exe, str(src_dir / "citation_network.py"), "--meeting_folder", str(committee_dir)],
            project_root,
        )

    run_step(
        [python_exe, str(src_dir / "country_stance.py"), "--base-dir", str(base_dir), "--committees", *available],
        project_root,
    )
    run_step(
        [python_exe, str(src_dir / "dynamic_topics.py"), "--base-dir", str(base_dir), "--committees", *available],
        project_root,
    )
    run_step(
        [python_exe, str(src_dir / "cross_committee_deep.py"), "--base-dir", str(base_dir), "--committees", *available],
        project_root,
    )
    run_step(
        [python_exe, str(src_dir / "topic_visualization.py"), "--base-dir", str(base_dir), "--committees", *available],
        project_root,
    )


def collect_images(project_root: Path, base_dir: Path, report_dir: Path) -> list[ImageRecord]:
    images_dir = report_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    selected: dict[tuple[str, str, str], ImageRecord] = {}
    seen_hashes: dict[str, ImageRecord] = {}
    for image_path in sorted(base_dir.rglob("*.png")):
        rel = image_path.relative_to(base_dir)
        committee = _detect_committee(rel)
        module = _detect_module(rel)
        chart_key = _normalize_chart_key(rel.stem, committee)
        target_name = f"{committee}__{module}__{chart_key}.png"
        record = ImageRecord(
            source=image_path,
            target=images_dir / target_name,
            committee=committee,
            module=module,
            chart_key=chart_key,
            title=_title_from_chart_key(chart_key),
        )

        digest = hashlib.sha1(image_path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            continue
        key = (committee, module, chart_key)
        existing = selected.get(key)
        if existing is None or _record_priority(record) < _record_priority(existing):
            selected[key] = record
        seen_hashes[digest] = record

    copied: list[ImageRecord] = []
    for record in sorted(selected.values(), key=lambda item: (item.committee, item.module, item.chart_key)):
        shutil.copy2(record.source, record.target)
        copied.append(record)

    print(f"\nCollected {len(copied)} PNG images into {images_dir}")
    return copied


def collect_html_outputs(base_dir: Path) -> list[Path]:
    return sorted(base_dir.rglob("*.html"))


def generate_report(project_root: Path, base_dir: Path, report_dir: Path, images: list[ImageRecord], html_files: list[Path]) -> Path:
    report_path = report_dir / "analysis_report.md"
    grouped_images: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for image in images:
        grouped_images[image.committee][image.module].append(image)

    lines: list[str] = []
    lines.append("# IMO Analysis Report")
    lines.append("")
    lines.append("This report is generated from the current analysis outputs in the repository. It reruns the available analysis scripts, collects all PNG figures into a unified images folder, and embeds those figures below.")
    lines.append("")
    lines.append("## Output Summary")
    lines.append("")
    lines.append(f"- Base output directory: `{base_dir}`")
    lines.append(f"- Collected PNG figures: {len(images)}")
    lines.append(f"- Available interactive HTML outputs: {len(html_files)}")
    lines.append(f"- Consolidated image folder: `{report_dir / 'images'}`")
    lines.append("")

    committee_order = [committee for committee in DEFAULT_COMMITTEES if committee in grouped_images]
    other_committees = sorted(committee for committee in grouped_images if committee not in DEFAULT_COMMITTEES)
    for committee in committee_order + other_committees:
        lines.append(f"## {committee}")
        lines.append("")
        modules = grouped_images[committee]
        module_order = [module for module in ("alliance", "citation", "visualization", "stance_analysis", "dynamic_analysis", "deep_analysis", "misc") if module in modules]
        for module in module_order:
            lines.append(f"### {MODULE_LABELS.get(module, module.title())}")
            lines.append("")
            for image in modules[module]:
                rel = image.target.relative_to(report_dir).as_posix()
                lines.append(f"#### {image.title}")
                lines.append("")
                lines.append(f"![{image.title}]({rel})")
                lines.append("")

    if html_files:
        lines.append("## Interactive Outputs")
        lines.append("")
        lines.append("The following HTML outputs were generated or preserved alongside the PNG figures:")
        lines.append("")
        for html in html_files:
            rel = html.relative_to(project_root).as_posix()
            lines.append(f"- `{rel}`")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated report: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun available IMO analyses, collect images, and build a markdown report.")
    parser.add_argument("--base-dir", default="output", help="Directory containing committee analysis outputs")
    parser.add_argument("--report-dir", default="paper/generated_report", help="Directory to save the consolidated images and report")
    parser.add_argument("--committees", nargs="+", default=DEFAULT_COMMITTEES, help="Committees to rerun")
    parser.add_argument("--skip-bertopic", action="store_true", help="Skip BERTopic reruns if topic modeling outputs already exist")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_dir = (project_root / args.base_dir).resolve()
    report_dir = (project_root / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    rerun_pipeline(project_root, base_dir, args.committees, args.skip_bertopic)
    images = collect_images(project_root, base_dir, report_dir)
    html_files = collect_html_outputs(base_dir)
    generate_report(project_root, base_dir, report_dir, images, html_files)


if __name__ == "__main__":
    main()