import argparse
import getpass
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from local_docs_rag.config import DEFAULT_CONFIG_PATH, Settings, validate_project_name
from local_docs_rag.credentials import CredentialStore
from local_docs_rag.evaluation import RetrievalQuestion, evaluate_retrieval
from local_docs_rag.indexing import Indexer
from local_docs_rag.providers import AnswerProvider, EmbeddingProvider
from local_docs_rag.service import RagService, SearchService
from local_docs_rag.store import IndexStore


def _prompt_value(label: str, current: str | None) -> str | None:
    default = "" if current is None else current
    response = input(f"{label} [{default}]: ").strip()
    return response or current


def _setup(config_path: Path, use_defaults: bool = False) -> int:
    current = Settings.load(environ={}, config_path=config_path)
    if use_defaults:
        current.save(config_path)
        print(f"Settings saved to {config_path.expanduser()}")
        return 0

    source_dir_value = _prompt_value("Legacy document directory for index adoption", str(current.source_dir))
    database_path_value = _prompt_value("Index database", str(current.database_path))
    base_url = _prompt_value("OpenAI-compatible API base URL", current.base_url)
    embedding_model = _prompt_value("Embedding model", current.embedding_model)
    dimensions_value = _prompt_value(
        "Embedding dimensions",
        None if current.embedding_dimensions is None else str(current.embedding_dimensions),
    )
    answer_model = _prompt_value("Answer model", current.answer_model)

    dimensions = None if dimensions_value is None else int(dimensions_value)
    if dimensions is not None and dimensions <= 0:
        raise ValueError("embedding dimensions must be a positive integer")

    settings = Settings(
        source_dir=Path(source_dir_value or current.source_dir).expanduser(),
        database_path=Path(database_path_value or current.database_path).expanduser(),
        base_url=base_url,
        embedding_model=embedding_model,
        embedding_dimensions=dimensions,
        answer_model=answer_model,
        projects=current.projects,
    )
    settings.save(config_path)
    print(f"Settings saved to {config_path.expanduser()}")
    return 0


def _set_credentials(credentials: CredentialStore) -> int:
    api_key = getpass.getpass("Model API key: ").strip()
    if not api_key:
        raise ValueError("API key cannot be empty")
    credentials.set_api_key(api_key)
    print("API key saved to the system credential store.")
    return 0


def _create_embedding_provider(settings: Settings) -> EmbeddingProvider | None:
    if settings.embedding_model is None and settings.embedding_dimensions is None:
        return None
    return EmbeddingProvider.create(settings)


def _load_project(config_path: Path, project: str) -> Settings:
    validate_project_name(project)
    settings = Settings.load(config_path=config_path)
    if project not in settings.projects:
        raise ValueError(f"unknown project {project!r}; register it with 'local-docs-rag project add'")
    return settings


def _index(config_path: Path, project: str) -> int:
    settings = _load_project(config_path, project)
    source_dir = Path.cwd().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"document directory does not exist: {source_dir}")
    projects = dict(settings.projects)
    source_dirs = list(projects[project])
    if source_dir not in {source.resolve() for source in source_dirs}:
        source_dirs.append(source_dir)
        projects[project] = source_dirs
        settings = replace(settings, projects=projects)
        settings.save(config_path)
    store = IndexStore(settings.database_path, settings.embedding_dimensions)
    report = Indexer(
        source_dir,
        store,
        _create_embedding_provider(settings),
        project=project,
    ).run()
    print(
        f"Index complete: added={report.added} updated={report.updated} "
        f"unchanged={report.unchanged} removed={report.removed} failed={report.failed}"
    )
    for issue in report.issues:
        print(f"Warning: {issue}")
    return 0 if report.failed == 0 else 1


def _search(config_path: Path, project: str, query: str, limit: int) -> int:
    settings = _load_project(config_path, project)
    store = IndexStore(settings.database_path, settings.embedding_dimensions)
    search_service = SearchService(store, _create_embedding_provider(settings))
    results = search_service.search(query, limit, project=project)
    if not results:
        print("No matching documents.")
        return 0
    for rank, result in enumerate(results, start=1):
        print(f"[{rank}] {result.chunk.source}#{result.chunk.locator} (score={result.score:.4f})")
        print(result.chunk.text)
    return 0


def _evaluate(
    config_path: Path,
    project: str,
    questions_path: Path,
    output_path: Path | None,
) -> int:
    questions: list[RetrievalQuestion] = []
    for line_number, line in enumerate(questions_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            questions.append(
                RetrievalQuestion(
                    query=entry["query"],
                    expected_source=entry["expected_source"],
                    expected_locator=entry["expected_locator"],
                    language=entry["language"],
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid evaluation question on line {line_number}") from error

    settings = _load_project(config_path, project)
    source_dirs = settings.projects[project]
    if not source_dirs:
        raise ValueError(f"project {project!r} has no indexed source directories")
    store = IndexStore(settings.database_path, settings.embedding_dimensions)
    search_service = SearchService(store, _create_embedding_provider(settings))
    metrics = evaluate_retrieval(
        questions,
        lambda query, limit: search_service.search(query, limit, project=project),
        source_dirs,
    )
    serialized_metrics = json.dumps(asdict(metrics), indent=2) + "\n"
    if output_path is None:
        output_path = Path(__file__).resolve().parents[2] / "tests/evaluation" / f"results-{project}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized_metrics, encoding="utf-8")
    print(serialized_metrics, end="")
    return 0


def _ask(config_path: Path, project: str, query: str, limit: int) -> int:
    settings = _load_project(config_path, project)
    store = IndexStore(settings.database_path, settings.embedding_dimensions)
    search_service = SearchService(store, _create_embedding_provider(settings))
    answer_provider = AnswerProvider.create(settings)
    answer = RagService(search_service, answer_provider).answer(query, limit, project=project)
    print(answer.answer)
    for chunk in store.get_chunks(answer.citations, project=project):
        print(f"Source: {chunk.source}#{chunk.locator}")
    return 0


def _mcp_serve(config_path: Path) -> int:
    from local_docs_rag.mcp_server import serve

    serve(config_path)
    return 0


def _project_add(config_path: Path, name: str, adopt_existing: bool) -> int:
    validate_project_name(name)
    settings = Settings.load(config_path=config_path)
    if name in settings.projects:
        raise ValueError(f"project {name!r} is already registered")
    source_dirs: list[Path] = []
    if adopt_existing:
        if not settings.database_path.expanduser().is_file():
            raise ValueError("there is no existing index database to adopt")
        store = IndexStore(settings.database_path, settings.embedding_dimensions)
        store.adopt_legacy_documents(name)
        source_dirs.append(settings.source_dir.expanduser().resolve())
    projects = dict(settings.projects)
    projects[name] = source_dirs
    replace(settings, projects=projects).save(config_path)
    print(f"Project registered: {name}")
    return 0


def _project_list(config_path: Path) -> int:
    settings = Settings.load(config_path=config_path)
    if not settings.projects:
        print("No projects registered.")
        return 0
    for name, source_dirs in sorted(settings.projects.items()):
        print(name)
        for source_dir in source_dirs:
            print(f"  {source_dir}")
    return 0


def _project_remove(config_path: Path, name: str, assume_yes: bool) -> int:
    validate_project_name(name)
    settings = Settings.load(config_path=config_path)
    if name not in settings.projects:
        raise ValueError(f"unknown project {name!r}")
    if not assume_yes:
        if not sys.stdin.isatty():
            raise ValueError("project removal requires confirmation; pass --yes in non-interactive mode")
        confirmation = input(f"Remove project {name!r} and its indexed data? [y/N] ")
        if confirmation.strip().casefold() not in {"y", "yes"}:
            print("Project removal cancelled.")
            return 0
    database_path = settings.database_path.expanduser()
    if database_path.is_file():
        IndexStore(database_path, settings.embedding_dimensions).remove_project(name)
    projects = dict(settings.projects)
    del projects[name]
    replace(settings, projects=projects).save(config_path)
    print(f"Project removed: {name}")
    return 0


def _project_remove_dir(config_path: Path, name: str, source_dir: Path, assume_yes: bool) -> int:
    validate_project_name(name)
    settings = Settings.load(config_path=config_path)
    if name not in settings.projects:
        raise ValueError(f"unknown project {name!r}")
    root = source_dir.expanduser().resolve()
    source_dirs = [source.resolve() for source in settings.projects[name]]
    if root not in source_dirs:
        raise ValueError(f"source directory {root} is not registered for project {name!r}")
    if not assume_yes:
        if not sys.stdin.isatty():
            raise ValueError("directory removal requires confirmation; pass --yes in non-interactive mode")
        confirmation = input(
            f"Remove {root} and its indexed data from project {name!r}? [y/N] "
        )
        if confirmation.strip().casefold() not in {"y", "yes"}:
            print("Source directory removal cancelled.")
            return 0
    database_path = settings.database_path.expanduser()
    remaining_roots = [source for source in source_dirs if source != root]
    if database_path.is_file():
        IndexStore(database_path, settings.embedding_dimensions).remove_project_source_root(
            name,
            root,
            remaining_roots,
        )
    projects = dict(settings.projects)
    projects[name] = remaining_roots
    replace(settings, projects=projects).save(config_path)
    print(f"Source directory removed from project {name}: {root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-docs-rag")
    commands = parser.add_subparsers(dest="command", required=True)

    setup_parser = commands.add_parser("setup", help="configure document and model settings")
    setup_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    setup_parser.add_argument("--defaults", action="store_true", help="write local defaults without prompting")
    setup_parser.set_defaults(handler=lambda args: _setup(args.config_path, args.defaults))

    credentials_parser = commands.add_parser("credentials", help="manage provider credentials")
    credential_commands = credentials_parser.add_subparsers(dest="credentials_command", required=True)
    set_parser = credential_commands.add_parser("set", help="save the model API key")
    set_parser.set_defaults(handler=lambda args: _set_credentials(CredentialStore.create()))

    project_parser = commands.add_parser("project", help="manage indexed document projects")
    project_commands = project_parser.add_subparsers(dest="project_command", required=True)
    project_add_parser = project_commands.add_parser("add", help="register a project without a source directory")
    project_add_parser.add_argument("name")
    project_add_parser.add_argument("--adopt-existing", action="store_true")
    project_add_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    project_add_parser.set_defaults(
        handler=lambda args: _project_add(
            args.config_path,
            args.name,
            args.adopt_existing,
        )
    )

    project_list_parser = project_commands.add_parser("list", help="list registered projects")
    project_list_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    project_list_parser.set_defaults(handler=lambda args: _project_list(args.config_path))

    project_remove_parser = project_commands.add_parser("remove", help="remove a project and its indexed data")
    project_remove_parser.add_argument("name")
    project_remove_parser.add_argument("--yes", action="store_true", help="confirm removal without prompting")
    project_remove_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    project_remove_parser.set_defaults(
        handler=lambda args: _project_remove(args.config_path, args.name, args.yes)
    )

    project_remove_dir_parser = project_commands.add_parser(
        "remove-dir",
        help="remove one source directory and its indexed data from a project",
    )
    project_remove_dir_parser.add_argument("name")
    project_remove_dir_parser.add_argument("source_dir", type=Path)
    project_remove_dir_parser.add_argument("--yes", action="store_true", help="confirm removal without prompting")
    project_remove_dir_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    project_remove_dir_parser.set_defaults(
        handler=lambda args: _project_remove_dir(
            args.config_path,
            args.name,
            args.source_dir,
            args.yes,
        )
    )

    index_parser = commands.add_parser("index", help="index PDF and Markdown files in the current directory")
    index_parser.add_argument("project")
    index_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    index_parser.set_defaults(handler=lambda args: _index(args.config_path, args.project))

    search_parser = commands.add_parser("search", help="search indexed documents")
    search_parser.add_argument("project")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=8)
    search_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    search_parser.set_defaults(
        handler=lambda args: _search(args.config_path, args.project, args.query, args.limit)
    )

    project_root = Path(__file__).resolve().parents[2]
    evaluate_parser = commands.add_parser("evaluate", help="measure retrieval against local evaluation questions")
    evaluate_parser.add_argument("project")
    evaluate_parser.add_argument("--questions-path", type=Path, default=project_root / "tests/evaluation/questions.jsonl")
    evaluate_parser.add_argument("--output-path", type=Path)
    evaluate_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    evaluate_parser.set_defaults(
        handler=lambda args: _evaluate(
            args.config_path,
            args.project,
            args.questions_path,
            args.output_path,
        )
    )

    ask_parser = commands.add_parser("ask", help="answer a question with document citations")
    ask_parser.add_argument("project")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--limit", type=int, default=8)
    ask_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    ask_parser.set_defaults(handler=lambda args: _ask(args.config_path, args.project, args.query, args.limit))

    mcp_parser = commands.add_parser("mcp", help="run the MCP server")
    mcp_commands = mcp_parser.add_subparsers(dest="mcp_command", required=True)
    serve_parser = mcp_commands.add_parser("serve", help="serve RAG tools over stdio")
    serve_parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    serve_parser.set_defaults(handler=lambda args: _mcp_serve(args.config_path))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())