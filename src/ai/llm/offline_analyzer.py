from __future__ import annotations

from typing import List

from ai.llm.interface import AnalyzerAdapter, AnalyzerError


def _extract_lines(context: str) -> List[str]:
    """
    Split context into normalized lines for deterministic parsing.
    """
    if not isinstance(context, str):
        return []
    return [
        ln.rstrip()
        for ln in context.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]


def _find_line_prefix(lines: List[str], prefix: str) -> str | None:
    """
    Return the first line that starts with prefix; otherwise None.
    """
    p = prefix.strip()
    return next((ln[len(p) :].strip() for ln in lines if ln.startswith(p)), None)


def _parse_key_counts(raw: str) -> List[tuple[str, int]]:
    """
    Parse comma-separated items like 'IT(10), HR(8)' into [('IT', 10), ('HR', 8)].
    Robust to spacing; ignores malformed entries deterministically.
    """
    out: List[tuple[str, int]] = []
    if not raw:
        return out
    items = [x.strip() for x in raw.split(",") if x.strip()]
    for item in items:
        # Expect 'Label(count)'
        if "(" in item and item.endswith(")"):
            try:
                label, num = item.split("(", 1)
                label = label.strip()
                num = num[:-1].strip()
                count = int(float(num))
                out.append((label, count))
            except (ValueError, TypeError):
                # Ignore malformed segments
                continue
    # Deterministic ordering: by count desc, then label asc
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out


def _parse_distribution(raw: str) -> List[tuple[str, int]]:
    """
    Parse 'low(3), medium(5), high(2)' into [('medium',5), ('low',3), ('high',2)] (sorted by count desc).
    """
    return _parse_key_counts(raw)


class OfflineAnalyzer(AnalyzerAdapter):
    """
    Deterministic, rule-based analyzer producing structured markdown.

    - No network/files; pure transformation of provided context.
    - Parses header summaries like 'Top Departments:', 'Quality:', 'Complexity:' added by the sampling engine.
    - Produces stable sections suitable for offline CI validation.
    """

    def analyze(self, context: str, question: str, prompt_version: str, comparison_mode: bool = False) -> str:
        if not isinstance(context, str) or not context.strip():
            raise AnalyzerError("Context is required for offline analysis")
        if not isinstance(question, str) or not question.strip():
            raise AnalyzerError("Question is required for offline analysis")
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise AnalyzerError("prompt_version is required for offline analysis")

        lines = _extract_lines(context)

        # Extract high-level counts
        total_raw = _find_line_prefix(lines, "Total tickets:")
        selected_raw = _find_line_prefix(lines, "Selected for sampling:")
        try:
            total_tickets = int(float(total_raw or "0"))
        except (ValueError, TypeError):
            total_tickets = 0
        try:
            selected_tickets = int(float(selected_raw or "0"))
        except (ValueError, TypeError):
            selected_tickets = 0

        # Parse distributions from header
        tops_raw = _find_line_prefix(lines, "Top Departments:")
        quality_raw = _find_line_prefix(lines, "Quality:")
        complexity_raw = _find_line_prefix(lines, "Complexity:")

        top_departments = _parse_key_counts(tops_raw or "")
        quality_dist = _parse_distribution(quality_raw or "")
        complexity_dist = _parse_distribution(complexity_raw or "")

        # Build structured markdown deterministically
        md_sections: List[str] = []

        # Title and meta
        md_sections.append("# Service Desk Analysis")
        md_sections.append(f"Prompt Version: {prompt_version}")
        md_sections.append(f"Question: {question.strip()}")
        md_sections.append("")

        # Summary section
        md_sections.append("## Summary")
        md_sections.append(f"- Total Tickets: {total_tickets}")
        md_sections.append(f"- Sampled Tickets: {selected_tickets}")
        md_sections.append(f"- Comparison Mode: {'Enabled' if comparison_mode else 'Disabled'}")
        md_sections.append("")

        # Distributions
        md_sections.append("## Distributions")
        if top_departments:
            md_sections.append("### Top Departments")
            for dep, cnt in top_departments:
                md_sections.append(f"- {dep}: {cnt}")
        else:
            md_sections.append("### Top Departments")
            md_sections.append("- (none)")
        if quality_dist:
            md_sections.append("### Ticket Quality")
            for label, cnt in quality_dist:
                md_sections.append(f"- {label}: {cnt}")
        else:
            md_sections.append("### Ticket Quality")
            md_sections.append("- (none)")
        if complexity_dist:
            md_sections.append("### Resolution Complexity")
            for label, cnt in complexity_dist:
                md_sections.append(f"- {label}: {cnt}")
        else:
            md_sections.append("### Resolution Complexity")
            md_sections.append("- (none)")
        md_sections.append("")

        # Insights: simple deterministic heuristics
        md_sections.append("## Insights")
        if top_departments:
            dominant = top_departments[0][0]
            md_sections.append(f"- Highest volume observed in: {dominant}")
        else:
            md_sections.append("- No department volume data available.")
        if quality_dist:
            # Prefer worst quality to surface onboarding opportunity
            worst = sorted(quality_dist, key=lambda kv: (kv[1], kv[0]))[-1][0] if quality_dist else "n/a"
            md_sections.append(f"- Quality distribution includes: {', '.join([q for q, _ in quality_dist])}; focus on '{worst}'.")
        else:
            md_sections.append("- Quality distribution not available.")
        if complexity_dist:
            toughest = sorted(complexity_dist, key=lambda kv: (kv[1], kv[0]))[-1][0]
            md_sections.append(f"- Complexity hotspots detected in: {toughest}.")
        else:
            md_sections.append("- Complexity distribution not available.")
        md_sections.append("")

        # Recommendations: stable rule-based
        md_sections.append("## Recommendations")
        md_sections.append("- Establish targeted onboarding modules for top-volume departments.")
        md_sections.append("- Improve documentation for common low-quality tickets to uplift standards.")
        md_sections.append("- Create troubleshooting guides for high-complexity resolutions.")
        md_sections.append("- Track reassignment trends and address root causes.")
        md_sections.append("")

        # Onboarding KV draft (static fields, deterministic)
        md_sections.append("## Onboarding Key-Values")
        md_sections.append("- Audience: New Service Desk Analysts")
        md_sections.append("- Focus Areas: High-volume departments, common low-quality patterns, high-complexity resolution steps")
        md_sections.append("- Artifacts: Playbooks, checklists, SOPs, FAQ entries")
        md_sections.append("- Metrics: Adoption rate, ticket quality uplift, resolution time reduction")
        md_sections.append("")

        # Context echo (truncated safely to keep output bounded by input size)
        md_sections.append("## Context (Echo)")
        # Echo only the first N context lines deterministically
        N = 50
        echo_lines = lines[:N]
        if echo_lines:
            md_sections.extend([f"> {ln}" for ln in echo_lines])
        else:
            md_sections.append("> (no context lines)")

        return "\n".join(md_sections).strip()
