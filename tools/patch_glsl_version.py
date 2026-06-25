from pathlib import Path


RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"


def main():
    for path in sorted(RESOURCE_DIR.glob("*.[vf]s")):
        text = path.read_text(encoding="utf-8")
        if text.startswith("#version 300 es"):
            text = text.replace("#version 300 es", "#version 410", 1)
            text = text.replace("precision highp float;\n\n", "", 1)
            path.write_text(text, encoding="utf-8")
            print(f"patched {path.name}")


if __name__ == "__main__":
    main()
