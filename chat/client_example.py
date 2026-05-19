"""
chat/client_example.py
───────────────────────
Example client showing REST + WebSocket usage.
Run after the API is started: uvicorn chat.api:app --port 8000
"""

import asyncio
import json

import httpx
import websockets

BASE_URL = "http://localhost:8000"
WS_URL   = "ws://localhost:8000"


# ── REST: blocking chat ───────────────────────────────────────────────────────

def ask(message: str, subject: str = None, grade: str = None):
    payload = {"message": message, "subject": subject, "grade": grade}
    resp = httpx.post(f"{BASE_URL}/chat", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print("\n── Answer ──────────────────────────────────────")
    print(data["answer"])
    print("\n── Sources ─────────────────────────────────────")
    for s in data["sources"]:
        print(f"  • [{s['subject']}|{s['grade']}] {s['topic']} › {s['section']}")
        print(f"    {s['url']}")


# ── WebSocket: streaming chat ─────────────────────────────────────────────────

async def ask_stream(message: str, subject: str = None, grade: str = None):
    uri = f"{WS_URL}/chat/stream"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "message": message,
            "subject": subject,
            "grade":   grade,
        }))
        print("\n── Streaming Answer ─────────────────────────────")
        async for raw in ws:
            try:
                frame = json.loads(raw)
                if frame.get("done"):
                    print("\n\n── Sources ─────────────────────────────────────")
                    for s in frame.get("sources", []):
                        print(f"  • {s['topic']} › {s['section']}")
                    break
                if "error" in frame:
                    print(f"Error: {frame['error']}")
                    break
            except json.JSONDecodeError:
                # Plain text chunk
                print(raw, end="", flush=True)


# ── Health check ──────────────────────────────────────────────────────────────

def health():
    resp = httpx.get(f"{BASE_URL}/health")
    print(json.dumps(resp.json(), indent=2))


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Health Check ===")
    health()

    print("\n── Biology question (English) ──────────────────────────────────────")
    ask(
        message="Explain kreb's cycle in simple terms",
        subject="biology",
        grade="M4",
    )

    print("\n── Chemistry question ──────────────────────────────────────")
    ask(
        message="Explain titration in simple terms",
        subject="chemistry",
        grade="M5",
    )

    print("\n── Physics question ──────────────────────────────────────")
    ask(
        message="Explain what does golgi complex does in a cell",
        subject="physics",
        grade="M4",
    )