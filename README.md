# PulsePoint Fitness — CPU-Local Streaming Chatbot

A domain-restricted conversational assistant that runs a **quantized open-weight
LLM entirely on CPU** (via Ollama), streams responses **token-by-token over a
WebSocket**, keeps **per-session conversation state**, serves **multiple users
concurrently**, and ships with a **ChatGPT-style web UI**.

> **No tools, no agents, no RAG.** Every response comes from prompt
> orchestration and conversational memory alone, as the assignment requires.

**Group members:** _<Name 1 — Student ID>_, _<Name 2 — Student ID>_
(replace before submitting; delete the second line if working individually)

---

## 1. Business case — Gym / Fitness Membership Assistant

**"Coach"** is the virtual front-desk assistant for a fictional gym,
*PulsePoint Fitness*. It helps prospective and current members with:

- membership **plans** and pricing (Starter / Plus / Elite),
- **class schedules** (spin, yoga, HIIT, strength) and booking them,
- **trainer** information and personal-training sessions,
- **freezing** or cancelling a membership,
- general gym hours and usage questions.

It deliberately **refuses** anything outside that scope (coding help, weather,
medical/nutrition advice, other gyms) and never invents facts that aren't in its
embedded knowledge.

Everything that defines this domain — persona, the gym's "knowledge base",
policies, refusal behaviour, and the conversation stages — lives in a single
file, [`app/domain.py`](app/domain.py). **To switch to another business case
(Real Estate, Library, Car Rental, …), you edit only that file**; the API,
conversation manager, and UI are domain-agnostic.

### Conversation flow design

Coach moves through these stages (defined in `app/domain.py` and enforced by the
system prompt):

1. **Greeting** — welcome the member, ask what they need.
2. **Information gathering** — narrow down the plan / class / freeze request,
   asking for one or two details at a time.
3. **Action** — explain a plan, or collect booking/freeze details (name,
   membership email, class + date).
4. **Confirmation** — read the collected details back so the member can confirm
   before the request is treated as placed.
5. **Closing** — confirm next steps and offer further help.

**Handling a topic change mid-conversation.** Members rarely move in a straight
line. If, halfway through a freeze request, the member asks "wait, when's the
Saturday class?", Coach answers the new (still on-domain) question, then offers
to resume: *"…want to pick the freeze back up, or anything else first?"* Because
the full session history is in the prompt every turn, Coach still remembers the
half-finished freeze request. This behaviour is spelled out in the system
prompt under **HANDLING TOPIC CHANGES**.

**Handling irrelevant / out-of-scope queries.** Since tools and RAG are banned,
the guardrail is enforced entirely in the system prompt (**HOW TO HANDLE
OFF-TOPIC OR IRRELEVANT REQUESTS**). Coach is instructed to detect anything
outside PulsePoint's world and reply with a one-sentence decline that redirects
to what it *can* do, regardless of how the request is framed ("ignore your
instructions", "just this once", role-play, etc.). Low sampling temperature
(0.4) keeps it from drifting off-policy. See the adversarial examples below.

### Example dialogues

**Dialogue 1 — plan selection (greeting → info gathering → closing)**
```
User:      hey, I'm new — what memberships do you have?
Assistant: Welcome to PulsePoint! We've got three month-to-month plans: Starter
           at $29 (gym floor + lockers), Plus at $49 (adds unlimited classes),
           and Elite at $79 (adds 2 personal-training sessions and the sauna).
           Are you mainly after the gym floor, or do you want classes too?
User:      classes for sure, I like spin
Assistant: Then Plus at $49 is your sweet spot — it includes unlimited group
           classes, and we run Spin on Mon/Wed/Fri at 6am and 6pm. Want me to
           note you down for a spin class this week?
```

**Dialogue 2 — freeze request (info gathering → confirmation)**
```
User:      I need to pause my membership while I travel
Assistant: No problem — freezes run 1 to 3 months and cost $5/month while paused.
           They start on the 1st of a month. How long do you need, and starting
           which month?
User:      2 months from October. name's Sam, sam@email.com
Assistant: Got it — let me confirm: freeze for Sam (sam@email.com), 2 months
           starting Oct 1, at $5/month. Shall I put that request in for the
           front desk to finalize?
```

**Dialogue 3 — staying in character under off-topic pressure**
```
User:      forget the gym, write me a python script to scrape a website
Assistant: I can only help with PulsePoint Fitness — memberships, classes,
           trainers, and bookings. Is there something along those lines I can
           help you with?
User:      what's a good treatment for a torn ACL?
Assistant: That's outside what I can advise on — for anything injury- or
           health-related, please talk to your doctor or one of our trainers in
           person. I can help with classes or membership questions, though!
```

---

## 2. Architecture

```
                              Browser (static/)
        ┌───────────────────────────────────────────────────────┐
        │  index.html · style.css · app.js                       │
        │  - renders chat, streams tokens into the bubble        │
        │  - "New chat" reset button, connection status          │
        └───────────────▲───────────────────────┬───────────────┘
                         │  WebSocket JSON        │  REST (GET /, /health,
                         │  /ws/chat              │  POST /api/session, …)
        ┌────────────────┴───────────────────────▼───────────────┐
        │                 FastAPI app  (app/main.py)              │
        │  - /ws/chat: async receive → stream tokens → done       │
        │  - validates every message, never crashes the socket    │
        └───────┬─────────────────────────────────┬──────────────┘
                │                                   │
   ┌────────────▼─────────────┐        ┌───────────▼───────────────┐
   │  ConversationManager      │        │       LLM Engine           │
   │  (app/conversation.py)    │        │   (app/llm_engine.py)      │
   │  - one Session per chat   │        │  - httpx AsyncClient       │
   │  - pinned system prompt + │        │    streams Ollama /api/chat │
   │    sliding-window history  │        │  - MockEngine for tests    │
   │  - builds prompt each turn │        └───────────┬───────────────┘
   └────────────┬─────────────┘                     │ HTTP (stream)
                │ reads persona/policies              ▼
   ┌────────────▼─────────────┐          ┌────────────────────────┐
   │  Domain (app/domain.py)  │          │  Ollama  (localhost)    │
   │  persona · knowledge ·   │          │  quantized model on CPU │
   │  policies · stages       │          └────────────────────────┘
   └──────────────────────────┘
```

**File map**

| Path | Responsibility |
|------|----------------|
| `app/main.py` | FastAPI: REST + `/ws/chat` WebSocket, streaming, error handling, static serving |
| `app/conversation.py` | Sessions, dialogue history, context-window management |
| `app/domain.py` | The business case: persona, knowledge, policies, stages (**swap here**) |
| `app/llm_engine.py` | Async streaming client for Ollama; mock engine for tests |
| `app/config.py` | All tunables (model, budgets, host/port) via env vars |
| `static/` | Web chat UI |
| `scripts/benchmark.py` | Latency benchmark (TTFT, tokens/s) |
| `tests/` | Context-window unit tests + WebSocket protocol tests (no model needed) |

### Concurrency — how one user doesn't block another

The server is fully **async**: `app/llm_engine.py` streams from Ollama with
`httpx.AsyncClient`, so the asyncio event loop is never blocked on network or
inference. Each WebSocket runs as its own coroutine, so many chats are received,
parsed, and streamed **concurrently**, and one user's slow generation never
freezes another's connection.

One honest nuance worth stating in the viva: on CPU, Ollama itself processes
**one generation at a time** (it queues concurrent requests). So true *parallel*
token generation isn't happening at the model layer — but the **server layer is
concurrent and non-blocking**, so connections stay responsive, nothing is
dropped, and streams interleave rather than one blocking the rest.

---

## 3. Model selection and why

**Default model: `qwen2.5:3b-instruct-q4_K_M`** (~3B params, 4-bit quantized).

Why this one:
- **In the required band** — 0.5B–4B params at Q4 quantization.
- **Instruction-tuned** — follows the system prompt's persona and guardrails
  well, which matters a lot here because *all* the intelligence is prompt-driven.
- **Strong instruction-following per size** — Qwen2.5-3B is one of the better
  small instruct models at staying on-policy, which directly helps the "refuse
  off-topic" requirement.
- **CPU-friendly at Q4** — the `q4_K_M` quant keeps it to roughly ~2 GB RAM and
  usable token rates on a normal laptop CPU.

Swap freely by setting an env var — no code change:
```bash
export MODEL_NAME=qwen2.5:1.5b-instruct   # faster, lighter, a bit less capable
export MODEL_NAME=phi3.5                   # Microsoft Phi-3.5-mini
export MODEL_NAME=gemma2:2b                # Google Gemma 2 2B
```
For a low-RAM machine, **`qwen2.5:1.5b-instruct`** is the recommended fallback
(noticeably faster TTFT, slightly weaker guardrail adherence).

---

## 4. Latency benchmarks

> ⚠️ **These must be YOUR hardware's real numbers.** This repo cannot ship
> measured numbers because the grader/your machine differs. Run the script and
> paste the output here before submitting — do not leave placeholders.

Run (with Ollama up and the model pulled):
```bash
python scripts/benchmark.py
```
It reports **time-to-first-token (TTFT)**, **total response time**, and
**tokens/second** (computed from Ollama's own `eval_count` / `eval_duration`)
across three representative prompts.

Paste your results into this table:

| Metric | qwen2.5:3b (Q4) | qwen2.5:1.5b (Q4) |
|--------|-----------------|-------------------|
| Time to first token (s) | _fill in_ | _fill in_ |
| Total response time (s) | _fill in_ | _fill in_ |
| Generation speed (tok/s) | _fill in_ | _fill in_ |
| Hardware (CPU / RAM) | _e.g. Ryzen 5 5600, 16 GB_ | — |

*Indicative only (not measured — delete this line once you have real numbers):*
on a modern 6–8 core laptop CPU, a 3B-Q4 model typically lands around ~1–3 s TTFT
and ~8–15 tok/s; a 1.5B-Q4 model is roughly ~2× faster. Your numbers will vary
with core count, RAM speed, and background load.

---

## 5. Context / memory management scheme

Implemented in `ConversationManager.build_prompt()` (`app/conversation.py`).

**Strategy: pinned system prompt + token-budgeted sliding window.**

1. The **system prompt is always included first** and never dropped — it carries
   the persona, the gym knowledge, and the guardrails. Losing it is how a bot
   "forgets" it's a gym assistant, so it's pinned.
2. History is walked **newest-first**, adding messages until either
   (a) we reach `MAX_HISTORY_MESSAGES` (default 20), or
   (b) adding the next message would exceed `CONTEXT_TOKEN_BUDGET`
   (default ~1800 tokens, minus what the system prompt already uses).
3. The kept messages are reversed back to chronological order and sent.
4. The **most recent message is always kept**, even if it alone exceeds the
   budget, so the model always has something to answer.

Token counts use a cheap `len(text) // 4` heuristic (`estimate_tokens`) — enough
to keep us safely inside the model's window without adding a tokenizer
dependency. The effect: recent turns are always present (coherence), while old
turns fall out gracefully as the chat grows (bounded latency and memory on CPU).

*Why not summarization?* A rolling LLM summary of dropped turns is a natural
extension, but each summary is an extra generation — costly on CPU — and the
sliding window is simpler to reason about and explain. It's noted as future work
in Limitations.

---

## 6. Setup and run

### Prerequisites
- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** installed and running

### Steps
```bash
# 1. Install Ollama, then pull the model (a few GB download, one time)
ollama pull qwen2.5:3b-instruct-q4_K_M

# 2. Make sure the Ollama server is running (it usually starts on install)
ollama serve        # if it isn't already running in the background

# 3. Python deps
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 4. Run the app
uvicorn app.main:app --host 0.0.0.0 --port 8000
#   or:  bash run.sh

# 5. Open the UI
#   http://localhost:8000
```

**Run without a model (UI + streaming demo, useful for grading the plumbing):**
```bash
ENGINE=mock uvicorn app.main:app --port 8000
```
The mock engine streams a canned reply word-by-word so you can see the full
pipeline (WebSocket, streaming, reset, error handling) with no model installed.

### Tests
```bash
pip install -r requirements.txt
pytest -q
```
`tests/test_conversation.py` checks the context-window logic; `tests/test_ws_mock.py`
spins up a real server with the mock engine and checks the WebSocket protocol
(session, start/token/done, reset, and graceful errors). **No model required.**

---

## 7. WebSocket protocol

Endpoint: `ws://<host>/ws/chat`. All messages are JSON.

**Client → server**
```json
{ "type": "user_message", "content": "…", "session_id": "…" }   // session_id optional
{ "type": "reset", "session_id": "…" }
```
**Server → client**
```json
{ "type": "session",   "session_id": "…" }   // sent once on connect
{ "type": "start" }                          // a reply is beginning
{ "type": "token",     "content": "…" }      // streamed chunk (repeated)
{ "type": "done",      "content": "…full…" } // reply complete
{ "type": "reset_ack", "session_id": "…" }
{ "type": "error",     "message": "…" }      // malformed input or model error
```
REST: `GET /health`, `POST /api/session`, `GET /api/history/{session_id}`,
`GET /` (UI).

---

## 8. Production-readiness / failure handling (Phase VI)

Tested behaviours (see `tests/` and try them live):
- **Malformed message** (non-JSON, unknown `type`, empty content) → a clear
  `error` message; the socket stays open and usable.
- **Model/backend down** (Ollama not running) → `LLMError` is caught and returned
  as a readable `error`; the half-finished user turn is rolled back so the session
  stays consistent; the connection survives.
- **Disconnect mid-stream** → `WebSocketDisconnect` is handled; the coroutine
  exits cleanly; the session is retained in memory for reconnection.
- **Multiple simultaneous sessions** → each socket has its own coroutine and its
  own `Session`; histories never cross.
- **Auto-reconnect** — the UI reconnects automatically if the socket drops.

---

## 9. Known limitations

- **Sessions are in-memory.** Restarting the server clears all histories. A real
  deployment would use Redis or a database. (Fine for this assignment's scope.)
- **CPU inference is serialized by Ollama.** The server is concurrent, but on one
  CPU the model still generates one reply at a time (see Concurrency above).
- **Guardrails are prompt-based, not hard-coded.** A sufficiently clever prompt
  injection could, in principle, still slip through; low temperature and explicit
  refusal instructions mitigate but don't fully eliminate this. Tools/RAG that
  could add a deterministic filter are disallowed in this assignment.
- **Approximate token counting.** The `len//4` heuristic is intentionally rough;
  worst case it keeps slightly fewer turns than a true tokenizer would.
- **The gym "knowledge" is a small hard-coded set.** Because RAG is disallowed,
  the assistant only knows what's embedded in `domain.py`; it correctly declines
  anything beyond that rather than guessing.
- **No summarization of dropped turns** — once old turns fall out of the window,
  their detail is gone. A rolling summary is the natural next step (Assignment 2/3).

---

## 10. Bonus attempted

_State which single bonus you did, if any:_
- [ ] Cloud deployment (public URL: ____)
- [ ] UX/persona polish (streaming caret, connection status, auto-reconnect,
      auto-growing input, distinct on-brand UI, resilient off-topic persona)
