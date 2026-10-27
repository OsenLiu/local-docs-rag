import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


DEFAULT_SOURCE_DIR = Path.home() / "Downloads/doc"
DEFAULT_DATABASE_PATH = Path.home() / "Library/Application Support/local-docs-rag/index.sqlite3"
DEFAULT_CONFIG_PATH = Path.home() / "Library/Application Support/local-docs-rag/config.json"
DEFAULT_PROJECT = "default"
_PROJECT_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def validate_project_name(name: str) -> str:
    if not _PROJECT_NAME_PATTERN.fullmatch(name):
        raise ValueError("project name must be a lowercase slug using letters, numbers, and hyphens")
    return name


@dataclass(frozen=True)
class Settings:
    source_dir: Path
    database_path: Path
    base_url: str | None
    embedding_model: str | None
    embedding_dimensions: int | None
    answer_model: str | None
    projects: dict[str, list[Path]] = field(default_factory=dict)

    def save(self, config_path: Path | None = None) -> None:
        path = DEFAULT_CONFIG_PATH if config_path is None else config_path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {
            "source_dir": str(self.source_dir),
            "database_path": str(self.database_path),
            "base_url": self.base_url,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "answer_model": self.answer_model,
            "projects": {
                name: [str(source_dir) for source_dir in source_dirs]
                for name, source_dirs in self.projects.items()
            },
        }
        path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(
        cls,
        environ: Mapping[str, str] | None = None,
        config_path: Path | None = None,
    ) -> "Settings":
        env = os.environ if environ is None else environ
        path = DEFAULT_CONFIG_PATH if config_path is None else config_path.expanduser()
        values = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

        source_dir = env.get("RAG_SOURCE_DIR", values.get("source_dir", str(DEFAULT_SOURCE_DIR)))
        database_path = env.get("RAG_DATABASE_PATH", values.get("database_path", str(DEFAULT_DATABASE_PATH)))
        dimensions_value = env.get("RAG_EMBEDDING_DIMENSIONS", values.get("embedding_dimensions"))
        embedding_dimensions = None if dimensions_value in (None, "") else int(dimensions_value)
        if embedding_dimensions is not None and embedding_dimensions <= 0:
            raise ValueError("embedding dimensions must be a positive integer")
        raw_projects = values.get("projects", {})
        if not isinstance(raw_projects, dict):
            raise ValueError("projects setting must be an object mapping project names to source directories")
        projects: dict[str, list[Path]] = {}
        for name, raw_source_dirs in raw_projects.items():
            validate_project_name(name)
            source_dirs = [raw_source_dirs] if isinstance(raw_source_dirs, str) else raw_source_dirs
            if not isinstance(source_dirs, list) or not all(isinstance(item, str) for item in source_dirs):
                raise ValueError(f"source directories for project {name!r} must be a list of strings")
            if name.casefold() in {existing.casefold() for existing in projects}:
                raise ValueError(f"duplicate project name: {name}")
            projects[name] = [Path(source_dir).expanduser() for source_dir in source_dirs]

        return cls(
            source_dir=Path(source_dir).expanduser(),
            database_path=Path(database_path).expanduser(),
            base_url=env.get("RAG_BASE_URL", values.get("base_url")),
            embedding_model=env.get("RAG_EMBEDDING_MODEL", values.get("embedding_model")),
            embedding_dimensions=embedding_dimensions,
            answer_model=env.get("RAG_ANSWER_MODEL", values.get("answer_model")),
            projects=projects,
        )