#!/usr/bin/env python3
import sys, json, requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7895"

print("Health:", requests.get(f"{BASE}/health").json())

# Step 1: Search
payload = {"query": "Can we grow plants in space??"}
res = requests.post(f"{BASE}/search", json=payload).json()
print("\n=== SUMMARY ===\n", res["summary"])
print("\n=== TOP 3 ===")
for i, p in enumerate(res["top3"], 1):
    print(f"{i}. {p['title']} (dist={p['distance']:.3f}) -> {p['link']}")
sid = res["session_id"]

# Step 2a: Chat with paper #1
chat = {
    "session_id": sid,
    "paper_idx": 1,
    "message": "What growth challenges are discussed?"
}
ans = requests.post(f"{BASE}/chat", json=chat).json()
print("\n=== CHAT ===\n", ans["answer"])

# Step 2b: Related to paper #2
rel = requests.post(f"{BASE}/related", json={"session_id": sid, "paper_idx": 2}).json()
print("\n=== RELATED ===")
for i, p in enumerate(rel["related"], 1):
    print(f"{i}. {p['title']} (dist={p['distance']:.3f}) -> {p['link']}")
