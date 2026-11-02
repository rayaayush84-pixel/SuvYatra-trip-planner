"""Terminal tool: python manage.py [ingest | add "text" | add-url URL | chat]"""
import asyncio
import sys

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.rag import RAGService
from app.schemas.chat import Message

USAGE = __doc__


def chat_loop(rag: RAGService) -> None:
    if rag.is_empty():
        sys.exit("Database is empty. Run: python manage.py ingest")
    history: list[Message] = []
    print("Nepal Travel Bot ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer, _ = asyncio.run(rag.chat(question, history))
        print(f"\nBot: {answer}\n")
        history += [
            Message(role="user", content=question),
            Message(role="assistant", content=answer),
        ]


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(USAGE)
        return

    settings = get_settings()
    setup_logging(settings.log_level)
    rag = RAGService(settings)

    cmd = args[0]
    if cmd == "ingest":
        print(f"Added {rag.ingest_directory()} new chunks. Total: {rag.count()}")
    elif cmd == "add" and len(args) > 1:
        print(f"Added {rag.add_text(' '.join(args[1:]), 'terminal')} chunks")
    elif cmd == "add-url" and len(args) > 1:
        print(f"Added {rag.add_url(args[1])} chunks")
    elif cmd == "chat":
        chat_loop(rag)
    else:
        print(USAGE)


if __name__ == "__main__":
    main()
