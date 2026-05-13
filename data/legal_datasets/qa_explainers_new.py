import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


DATASET_PATH = Path(r"D:\AI-Chatbot\data\legal_datasets\explainers_new.json")
QA_REPORT_PATH = DATASET_PATH.with_name("qa_report.json")
ERRORS_REPORT_PATH = DATASET_PATH.with_name("errors_report.json")

REQUIRED_FIELDS = [
    "topic_id",
    "title",
    "short_explanation",
    "simple_explanation",
    "legal_position",
    "important_points",
    "common_situations",
    "user_questions",
    "step_by_step_process",
    "documents_needed",
    "authority_to_contact",
    "possible_remedies",
    "important_law_references",
    "court_or_forum",
    "estimated_timeframe",
    "estimated_cost_range",
    "risk_level",
    "when_to_contact_lawyer",
    "common_mistakes",
    "related_topics",
    "keywords",
    "aliases",
    "practical_tips",
    "state_variation_possible",
    "disclaimer",
    "jurisdiction",
    "last_updated",
    "primary_sources",
    "legal_accuracy_notes",
    "jurisdiction_notes",
    "enforcement_practicality",
    "severity_indicators",
    "ai_response_guidelines",
]

ARRAY_FIELDS = [
    "important_points",
    "common_situations",
    "user_questions",
    "step_by_step_process",
    "documents_needed",
    "authority_to_contact",
    "possible_remedies",
    "important_law_references",
    "when_to_contact_lawyer",
    "common_mistakes",
    "related_topics",
    "keywords",
    "aliases",
    "practical_tips",
    "primary_sources",
    "legal_accuracy_notes",
    "jurisdiction_notes",
]

STRING_FIELDS = [
    "topic_id",
    "title",
    "short_explanation",
    "simple_explanation",
    "legal_position",
    "court_or_forum",
    "estimated_timeframe",
    "estimated_cost_range",
    "risk_level",
    "disclaimer",
    "jurisdiction",
    "last_updated",
]

PRIMARY_SOURCE_REQUIRED_FIELDS = [
    "type",
    "name",
    "citation",
    "official_source_url",
    "relevance",
]

PRIMARY_SOURCE_TYPES = {
    "statute",
    "constitution",
    "rule",
    "government_portal",
    "court_case",
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
MIN_LENGTHS = {
    "short_explanation": 40,
    "simple_explanation": 120,
    "legal_position": 120,
    "disclaimer": 80,
}


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"


def enable_windows_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def c(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def is_valid_url(value: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def add_issue(issues: list, severity: str, topic_key: str, field: str, message: str) -> None:
    issues.append(
        {
            "severity": severity,
            "topic_key": topic_key,
            "field": field,
            "message": message,
        }
    )


def load_dataset(path: Path) -> tuple[dict | None, list]:
    issues = []
    if not path.exists():
        add_issue(issues, "error", "__dataset__", "file", f"Dataset file not found: {path}")
        return None, issues

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        add_issue(issues, "error", "__dataset__", "encoding", f"Dataset is not valid UTF-8: {exc}")
        return None, issues

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        add_issue(
            issues,
            "error",
            "__dataset__",
            "json",
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        )
        return None, issues

    if not isinstance(data, dict):
        add_issue(issues, "error", "__dataset__", "root", "Root JSON value must be an object.")
        return None, issues

    return data, issues


def check_required_fields(topic_key: str, topic: object, issues: list) -> None:
    if not isinstance(topic, dict):
        add_issue(issues, "error", topic_key, "__topic__", "Topic value must be an object.")
        return

    for field in REQUIRED_FIELDS:
        if field not in topic:
            add_issue(issues, "error", topic_key, field, "Missing required field.")
            continue
        value = topic[field]
        if value is None:
            add_issue(issues, "error", topic_key, field, "Required field is null.")

    for field in STRING_FIELDS:
        if field in topic and not isinstance(topic[field], str):
            add_issue(issues, "error", topic_key, field, "Expected a string.")
        elif field in topic and not topic[field].strip():
            add_issue(issues, "error", topic_key, field, "String field is empty.")

    for field in ARRAY_FIELDS:
        if field in topic and not isinstance(topic[field], list):
            add_issue(issues, "error", topic_key, field, "Expected an array.")
        elif field in topic and len(topic[field]) == 0:
            add_issue(issues, "error", topic_key, field, "Array field is empty.")

    if "state_variation_possible" in topic and not isinstance(topic["state_variation_possible"], bool):
        add_issue(issues, "error", topic_key, "state_variation_possible", "Expected a boolean.")


def check_key_and_topic_id(topic_key: str, topic: dict, issues: list) -> None:
    topic_id = topic.get("topic_id")
    if isinstance(topic_id, str) and topic_key != topic_id:
        add_issue(
            issues,
            "error",
            topic_key,
            "topic_id",
            f"Topic key must equal topic_id. Found topic_id={topic_id!r}.",
        )


def check_date(topic_key: str, topic: dict, issues: list) -> None:
    value = topic.get("last_updated")
    if not isinstance(value, str):
        return
    if not DATE_RE.match(value):
        add_issue(issues, "error", topic_key, "last_updated", "Date must use YYYY-MM-DD format.")
        return
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        add_issue(issues, "error", topic_key, "last_updated", "Date is not a real calendar date.")


def check_short_explanations(topic_key: str, topic: dict, issues: list) -> None:
    for field, minimum in MIN_LENGTHS.items():
        value = topic.get(field)
        if isinstance(value, str) and len(value.strip()) < minimum:
            add_issue(
                issues,
                "warning",
                topic_key,
                field,
                f"Very short text: {len(value.strip())} characters; expected at least {minimum}.",
            )


def check_primary_sources(topic_key: str, topic: dict, issues: list) -> None:
    sources = topic.get("primary_sources")
    if not isinstance(sources, list) or not sources:
        add_issue(issues, "error", topic_key, "primary_sources", "Missing primary_sources.")
        return

    for index, source in enumerate(sources):
        field_prefix = f"primary_sources[{index}]"
        if not isinstance(source, dict):
            add_issue(issues, "error", topic_key, field_prefix, "Primary source entry must be an object.")
            continue

        for field in PRIMARY_SOURCE_REQUIRED_FIELDS:
            if field not in source:
                add_issue(issues, "error", topic_key, f"{field_prefix}.{field}", "Missing source field.")
            elif not isinstance(source[field], str):
                add_issue(issues, "error", topic_key, f"{field_prefix}.{field}", "Source field must be a string.")
            elif not source[field].strip() and field != "official_source_url":
                add_issue(issues, "error", topic_key, f"{field_prefix}.{field}", "Source field is empty.")

        source_type = source.get("type")
        if isinstance(source_type, str) and source_type not in PRIMARY_SOURCE_TYPES:
            add_issue(
                issues,
                "error",
                topic_key,
                f"{field_prefix}.type",
                f"Invalid source type: {source_type!r}.",
            )

        url = source.get("official_source_url")
        if isinstance(url, str) and url.strip() and not is_valid_url(url):
            add_issue(issues, "error", topic_key, f"{field_prefix}.official_source_url", f"Invalid URL: {url}")


def walk_strings(value: object) -> list[str]:
    found = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, list):
        for item in value:
            found.extend(walk_strings(item))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(walk_strings(item))
    return found


def check_urls(topic_key: str, topic: dict, issues: list) -> None:
    for text in walk_strings(topic):
        for raw_url in URL_RE.findall(text):
            url = raw_url.rstrip(".,);]")
            if not is_valid_url(url):
                add_issue(issues, "error", topic_key, "url", f"Invalid URL found: {url}")


def check_nested_objects(topic_key: str, topic: dict, issues: list) -> None:
    enforcement = topic.get("enforcement_practicality")
    if isinstance(enforcement, dict):
        for field in [
            "easy_to_use_for_common_person",
            "usually_requires_lawyer",
            "police_involvement_possible",
            "court_process_likely",
        ]:
            if field not in enforcement:
                add_issue(issues, "error", topic_key, f"enforcement_practicality.{field}", "Missing field.")
            elif not isinstance(enforcement[field], bool):
                add_issue(issues, "error", topic_key, f"enforcement_practicality.{field}", "Expected boolean.")
    elif "enforcement_practicality" in topic:
        add_issue(issues, "error", topic_key, "enforcement_practicality", "Expected an object.")

    severity = topic.get("severity_indicators")
    if isinstance(severity, dict):
        if "possible_arrest" not in severity:
            add_issue(issues, "error", topic_key, "severity_indicators.possible_arrest", "Missing field.")
        elif not isinstance(severity["possible_arrest"], bool):
            add_issue(issues, "error", topic_key, "severity_indicators.possible_arrest", "Expected boolean.")
        for field in ["financial_risk", "reputation_risk"]:
            if field not in severity:
                add_issue(issues, "error", topic_key, f"severity_indicators.{field}", "Missing field.")
            elif severity[field] not in {"low", "medium", "high"}:
                add_issue(issues, "error", topic_key, f"severity_indicators.{field}", "Expected low, medium, or high.")
        if "urgent_action_needed" not in severity:
            add_issue(issues, "error", topic_key, "severity_indicators.urgent_action_needed", "Missing field.")
        elif not isinstance(severity["urgent_action_needed"], bool):
            add_issue(issues, "error", topic_key, "severity_indicators.urgent_action_needed", "Expected boolean.")
    elif "severity_indicators" in topic:
        add_issue(issues, "error", topic_key, "severity_indicators", "Expected an object.")

    guidelines = topic.get("ai_response_guidelines")
    if isinstance(guidelines, dict):
        if not isinstance(guidelines.get("safe_response_style"), str) or not guidelines.get("safe_response_style", "").strip():
            add_issue(issues, "error", topic_key, "ai_response_guidelines.safe_response_style", "Missing or empty string.")
        if not isinstance(guidelines.get("avoid_definitive_outcomes"), bool):
            add_issue(issues, "error", topic_key, "ai_response_guidelines.avoid_definitive_outcomes", "Expected boolean.")
        for field in ["must_recommend_lawyer_for", "should_ask_followup_questions_for"]:
            if not isinstance(guidelines.get(field), list) or not guidelines.get(field):
                add_issue(issues, "error", topic_key, f"ai_response_guidelines.{field}", "Expected non-empty array.")
    elif "ai_response_guidelines" in topic:
        add_issue(issues, "error", topic_key, "ai_response_guidelines", "Expected an object.")


def collect_duplicate_topic_ids(data: dict, issues: list) -> dict:
    topic_id_map = defaultdict(list)
    for topic_key, topic in data.items():
        if isinstance(topic, dict):
            topic_id = topic.get("topic_id")
            if isinstance(topic_id, str):
                topic_id_map[normalize_text(topic_id)].append(topic_key)

    duplicates = {
        topic_id: keys
        for topic_id, keys in topic_id_map.items()
        if len(keys) > 1
    }
    for topic_id, keys in duplicates.items():
        add_issue(
            issues,
            "error",
            "__dataset__",
            "topic_id",
            f"Duplicate topic_id {topic_id!r} used by keys: {', '.join(keys)}",
        )
    return duplicates


def collect_duplicate_aliases(data: dict, issues: list) -> dict:
    alias_map = defaultdict(list)
    for topic_key, topic in data.items():
        if not isinstance(topic, dict):
            continue
        aliases = topic.get("aliases", [])
        if not isinstance(aliases, list):
            continue
        seen_in_topic = set()
        for alias in aliases:
            if not isinstance(alias, str):
                add_issue(issues, "error", topic_key, "aliases", "Alias must be a string.")
                continue
            normalized = normalize_text(alias)
            if not normalized:
                add_issue(issues, "error", topic_key, "aliases", "Alias is empty.")
                continue
            if normalized in seen_in_topic:
                add_issue(issues, "error", topic_key, "aliases", f"Duplicate alias inside topic: {alias!r}")
            seen_in_topic.add(normalized)
            alias_map[normalized].append({"topic_key": topic_key, "alias": alias})

    duplicates = {
        alias: entries
        for alias, entries in alias_map.items()
        if len({entry["topic_key"] for entry in entries}) > 1
    }
    for alias, entries in duplicates.items():
        topics = sorted({entry["topic_key"] for entry in entries})
        add_issue(
            issues,
            "error",
            "__dataset__",
            "aliases",
            f"Duplicate alias across topics {alias!r}: {', '.join(topics)}",
        )
    return duplicates


def generate_statistics(data: dict, issues: list, duplicate_aliases: dict, duplicate_topic_ids: dict) -> dict:
    array_lengths = defaultdict(list)
    source_type_counts = Counter()
    risk_levels = Counter()
    jurisdictions = Counter()
    last_updated_values = Counter()
    source_url_count = 0
    blank_source_url_count = 0

    for topic in data.values():
        if not isinstance(topic, dict):
            continue
        for field in ARRAY_FIELDS:
            value = topic.get(field)
            if isinstance(value, list):
                array_lengths[field].append(len(value))

        risk = topic.get("risk_level")
        if isinstance(risk, str):
            risk_levels[risk] += 1

        jurisdiction = topic.get("jurisdiction")
        if isinstance(jurisdiction, str):
            jurisdictions[jurisdiction] += 1

        last_updated = topic.get("last_updated")
        if isinstance(last_updated, str):
            last_updated_values[last_updated] += 1

        for source in topic.get("primary_sources", []) if isinstance(topic.get("primary_sources"), list) else []:
            if not isinstance(source, dict):
                continue
            source_type = source.get("type")
            if isinstance(source_type, str):
                source_type_counts[source_type] += 1
            url = source.get("official_source_url")
            if isinstance(url, str) and url.strip():
                source_url_count += 1
            else:
                blank_source_url_count += 1

    return {
        "topic_count": len(data),
        "error_count": sum(1 for issue in issues if issue["severity"] == "error"),
        "warning_count": sum(1 for issue in issues if issue["severity"] == "warning"),
        "duplicate_topic_id_count": len(duplicate_topic_ids),
        "duplicate_alias_count": len(duplicate_aliases),
        "risk_level_counts": dict(risk_levels),
        "jurisdiction_counts": dict(jurisdictions),
        "last_updated_counts": dict(last_updated_values),
        "primary_source_type_counts": dict(source_type_counts),
        "primary_source_url_count": source_url_count,
        "primary_source_blank_url_count": blank_source_url_count,
        "array_field_stats": {
            field: {
                "min": min(lengths),
                "max": max(lengths),
                "avg": round(sum(lengths) / len(lengths), 2),
            }
            for field, lengths in sorted(array_lengths.items())
            if lengths
        },
    }


def run_qa() -> dict:
    data, issues = load_dataset(DATASET_PATH)
    if data is None:
        stats = {
            "topic_count": 0,
            "error_count": sum(1 for issue in issues if issue["severity"] == "error"),
            "warning_count": sum(1 for issue in issues if issue["severity"] == "warning"),
        }
        return {
            "dataset_path": str(DATASET_PATH),
            "valid_json": False,
            "statistics": stats,
            "issues": issues,
            "duplicate_aliases": {},
            "duplicate_topic_ids": {},
        }

    for topic_key, topic in data.items():
        check_required_fields(topic_key, topic, issues)
        if not isinstance(topic, dict):
            continue
        check_key_and_topic_id(topic_key, topic, issues)
        check_date(topic_key, topic, issues)
        check_short_explanations(topic_key, topic, issues)
        check_primary_sources(topic_key, topic, issues)
        check_urls(topic_key, topic, issues)
        check_nested_objects(topic_key, topic, issues)

    duplicate_topic_ids = collect_duplicate_topic_ids(data, issues)
    duplicate_aliases = collect_duplicate_aliases(data, issues)
    stats = generate_statistics(data, issues, duplicate_aliases, duplicate_topic_ids)

    return {
        "dataset_path": str(DATASET_PATH),
        "valid_json": True,
        "statistics": stats,
        "issues": issues,
        "duplicate_aliases": duplicate_aliases,
        "duplicate_topic_ids": duplicate_topic_ids,
    }


def write_reports(report: dict) -> None:
    errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
    errors_report = {
        "dataset_path": report["dataset_path"],
        "valid_json": report["valid_json"],
        "error_count": len(errors),
        "errors": errors,
    }
    QA_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ERRORS_REPORT_PATH.write_text(json.dumps(errors_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_report(report: dict) -> None:
    stats = report["statistics"]
    issues = report["issues"]
    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]

    print(c("\nIndian Legal Explainers QA", Colors.BOLD + Colors.CYAN))
    print(c("=" * 28, Colors.CYAN))
    print(f"Dataset: {DATASET_PATH}")
    print(f"QA report: {QA_REPORT_PATH}")
    print(f"Errors report: {ERRORS_REPORT_PATH}")
    print()

    json_status = c("PASS", Colors.GREEN) if report["valid_json"] else c("FAIL", Colors.RED)
    overall_status = c("PASS", Colors.GREEN) if not errors else c("FAIL", Colors.RED)
    print(f"Valid JSON: {json_status}")
    print(f"Overall QA: {overall_status}")
    print(f"Topics: {c(str(stats.get('topic_count', 0)), Colors.BLUE)}")
    print(f"Errors: {c(str(len(errors)), Colors.RED if errors else Colors.GREEN)}")
    print(f"Warnings: {c(str(len(warnings)), Colors.YELLOW if warnings else Colors.GREEN)}")
    print(f"Duplicate topic_ids: {stats.get('duplicate_topic_id_count', 0)}")
    print(f"Duplicate aliases: {stats.get('duplicate_alias_count', 0)}")
    print()

    print(c("Dataset Statistics", Colors.BOLD + Colors.BLUE))
    for label, value in [
        ("Risk levels", stats.get("risk_level_counts", {})),
        ("Jurisdictions", stats.get("jurisdiction_counts", {})),
        ("Last updated", stats.get("last_updated_counts", {})),
        ("Primary source types", stats.get("primary_source_type_counts", {})),
        ("Primary source URLs", stats.get("primary_source_url_count", 0)),
        ("Blank source URLs", stats.get("primary_source_blank_url_count", 0)),
    ]:
        print(f"- {label}: {value}")

    print()
    print(c("Array Field Stats", Colors.BOLD + Colors.BLUE))
    for field, field_stats in stats.get("array_field_stats", {}).items():
        print(f"- {field}: min={field_stats['min']} max={field_stats['max']} avg={field_stats['avg']}")

    if errors:
        print()
        print(c("Errors", Colors.BOLD + Colors.RED))
        for issue in errors[:50]:
            print(f"- [{issue['topic_key']}] {issue['field']}: {issue['message']}")
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more errors written to {ERRORS_REPORT_PATH}")

    if warnings:
        print()
        print(c("Warnings", Colors.BOLD + Colors.YELLOW))
        for issue in warnings[:50]:
            print(f"- [{issue['topic_key']}] {issue['field']}: {issue['message']}")
        if len(warnings) > 50:
            print(f"... {len(warnings) - 50} more warnings written to {QA_REPORT_PATH}")

    print()


def main() -> int:
    enable_windows_ansi()
    report = run_qa()
    write_reports(report)
    print_report(report)
    return 1 if report["statistics"].get("error_count", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
