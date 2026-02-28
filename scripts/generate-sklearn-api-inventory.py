#!/usr/bin/env python3
"""Generate a machine-readable sklearn public API inventory."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn


def classify_symbol(value: Any) -> str:
    if inspect.isclass(value):
        return "class"
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return "function"
    if isinstance(value, types.ModuleType):
        return "module"
    return "other"


def should_include(name: str, include_private: bool) -> bool:
    return include_private or not name.startswith("_")


def collect_module_symbols(
    module_name: str,
    include_private: bool,
) -> tuple[list[dict[str, str]], str | None]:
    qualified_module_name = f"sklearn.{module_name}"
    try:
        module = importlib.import_module(qualified_module_name)
    except Exception as exc:  # pragma: no cover - defensive inventory capture
        return [], str(exc)

    public_names = getattr(module, "__all__", [])
    symbols: list[dict[str, str]] = []
    for symbol_name in public_names:
        if not should_include(symbol_name, include_private):
            continue
        try:
            value = getattr(module, symbol_name)
            kind = classify_symbol(value)
        except Exception:
            kind = "unresolved"
        symbols.append(
            {
                "name": symbol_name,
                "qualifiedName": f"{qualified_module_name}.{symbol_name}",
                "module": qualified_module_name,
                "kind": kind,
            }
        )
    return symbols, None


def collect_top_level_symbols(
    include_private: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    symbols: list[dict[str, str]] = []
    module_names: list[str] = []
    for entry in getattr(sklearn, "__all__", []):
        if not should_include(entry, include_private):
            continue

        # sklearn lazily exposes many public modules; import modules directly first.
        try:
            importlib.import_module(f"sklearn.{entry}")
            module_names.append(entry)
            continue
        except Exception:
            pass

        try:
            value = getattr(sklearn, entry)
        except Exception:
            continue
        if isinstance(value, types.ModuleType):
            module_names.append(entry)
            continue
        symbols.append(
            {
                "name": entry,
                "qualifiedName": f"sklearn.{entry}",
                "module": "sklearn",
                "kind": classify_symbol(value),
            }
        )
    return symbols, module_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sklearn public API inventory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/sklearn-public-api.json"),
        help="Output inventory JSON path",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include symbols that start with underscores",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_private = bool(args.include_private)

    top_level_symbols, module_names = collect_top_level_symbols(include_private)
    symbol_records = list(top_level_symbols)
    skipped_modules: dict[str, str] = {}

    for module_name in module_names:
        records, error = collect_module_symbols(module_name, include_private)
        if error is not None:
            skipped_modules[module_name] = error
            continue
        symbol_records.extend(records)

    # Keep entries deterministic and deduplicated by fully-qualified name.
    deduped: dict[str, dict[str, str]] = {}
    for record in symbol_records:
        deduped[record["qualifiedName"]] = record
    sorted_records = sorted(
        deduped.values(),
        key=lambda item: (item["module"], item["name"]),
    )

    symbol_name_to_qualified: dict[str, list[str]] = defaultdict(list)
    for record in sorted_records:
        symbol_name_to_qualified[record["name"]].append(record["qualifiedName"])

    duplicate_short_names = {
        name: qualified
        for name, qualified in symbol_name_to_qualified.items()
        if len(qualified) > 1
    }

    unique_short_names = sorted(symbol_name_to_qualified.keys())
    module_counts: dict[str, int] = defaultdict(int)
    kind_counts: dict[str, int] = defaultdict(int)
    for record in sorted_records:
        module_counts[record["module"]] += 1
        kind_counts[record["kind"]] += 1

    inventory = {
        "metadata": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "sklearnVersion": sklearn.__version__,
            "includePrivate": include_private,
            "moduleCount": len(module_names),
            "skippedModuleCount": len(skipped_modules),
            "symbolCountQualified": len(sorted_records),
            "symbolCountUniqueShortName": len(unique_short_names),
            "duplicateShortNameCount": len(duplicate_short_names),
        },
        "moduleNames": sorted(module_names),
        "skippedModules": skipped_modules,
        "moduleCounts": dict(sorted(module_counts.items())),
        "kindCounts": dict(sorted(kind_counts.items())),
        "uniqueShortNames": unique_short_names,
        "duplicateShortNames": dict(sorted(duplicate_short_names.items())),
        "symbols": sorted_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(
        f"Wrote {args.output} ({len(sorted_records)} qualified symbols, "
        f"{len(unique_short_names)} unique short names) for sklearn {sklearn.__version__}"
    )


if __name__ == "__main__":
    main()
