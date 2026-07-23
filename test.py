from pathlib import Path

replacements = {
    "list[": "List[",
    "dict[": "Dict[",
    "tuple[": "Tuple[",
    "set[": "Set[",
}

for file in Path("src").rglob("*.py"):
    text = file.read_text(encoding="utf-8")

    changed = False
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
            changed = True

    if changed:
        file.write_text(text, encoding="utf-8")
        print("Updated", file)