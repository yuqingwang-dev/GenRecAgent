"""Placeholder for a future explicit data-preparation command."""

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    print(f"Data preparation is not implemented in Phase 0: {project_root / 'data'}")


if __name__ == "__main__":
    main()

