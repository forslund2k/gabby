#!/usr/bin/env python3
"""
Gabby MVP — AI chat agents for small businesses.

A business owner signs up, creates a chat agent (name + plain-English
instructions + knowledge base), and embeds one <script> tag on their website.
Visitors chat with the agent; the owner watches conversations roll in.

Run:
    pip install -r requirements.txt
    python app.py            # http://localhost:5000

Set GABBY_API_KEY to use a real OpenAI-compatible LLM; without it the app
runs in DEMO MODE (keyword-based replies, zero cost).
"""
import os
import re
import json
import math
import secrets
import threading
from datetime import datetime
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from bs4 import BeautifulSoup
from flask import (Flask, render_template, request, redirect, url_for, flash,
                   session, jsonify, Response, abort)
import sqlalchemy as sa
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------------------------------------------------------- config

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
_db_url = os.environ.get("DATABASE_URL", "").strip()
if _db_url.startswith("postgres://"):  # Render/Heroku style -> SQLAlchemy wants postgresql://
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = _db_url or "sqlite:///" + os.path.join(BASE_DIR, "gabby.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["GABBY_MODEL_HINT"] = "OpenAI"
db = SQLAlchemy(app)

# Chat-brain (LLM) configuration. Demo mode when no key is present.
GABBY_API_KEY = os.environ.get("GABBY_API_KEY", "").strip()
GABBY_API_BASE = os.environ.get("GABBY_API_BASE", "https://api.openai.com/v1").rstrip("/")
GABBY_MODEL = os.environ.get("GABBY_MODEL", "gpt-4o-mini")
GABBY_EMBED_MODEL = os.environ.get("GABBY_EMBED_MODEL", "text-embedding-3-small")
DEMO_MODE = not GABBY_API_KEY

GABBY_VERSION = "0.23.0"

CRAWL_MAX_PAGES = 100
CRAWL_MAX_CHARS = 50000

# Browser-like UA: some WAFs challenge or block obvious bot user-agents.
# The GabbyBot token stays in for honest identification.
CRAWL_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 "
            "GabbyBot/1.0 (+https://trygabby.com)")
# Full browser header set: some WAFs also inspect these.
CRAWL_HEADERS = {
    "User-Agent": CRAWL_UA,
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

DEFAULT_GREETING = "Hi there! How can I help?"

# Built-in assistant that lives on the TryGabby marketing site and answers
# visitor questions about the product itself. Seeded at startup; owned by an
# internal system user so it never appears in anyone's dashboard.
SITE_AGENT_PUBLIC_KEY = "trygabby-site-assistant"
SITE_AGENT_EMAIL = "system@trygabby.com"
SITE_AGENT_SUGGESTIONS = [
    "How much does TryGabby cost?",
    "How does it work?",
    "Will it capture leads for my business?",
]

# Support inbox: client messages from /support are emailed here via Resend.
# Replies go straight back to the client via the Reply-To header.
SUPPORT_EMAIL = os.environ.get("SUPPORT_EMAIL", "tforslund@gmail.com")
SUPPORT_FROM = "TryGabby Support <support@trygabby.com>"


@app.context_processor
def inject_version():
    return {"gabby_version": GABBY_VERSION}

# ---------------------------------------------------------------- models


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    agents = db.relationship("Agent", backref="owner", cascade="all, delete-orphan")


class Agent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    instructions = db.Column(db.Text, default="")          # system prompt text
    color = db.Column(db.String(7), default="#4F46E5")     # widget accent color
    timezone = db.Column(db.String(64), default="America/New_York")  # business tz
    suggestions = db.Column(db.Text, default="[]")  # JSON list of suggested questions
    greeting = db.Column(db.Text, default="Hi there! How can I help?")  # widget greeting
    hours = db.Column(db.Text, default="")  # JSON {monday: {open, close} | null, ...}
    hours_source = db.Column(db.String(20), default="")  # "" | "auto" | "confirmed"
    crawl_status = db.Column(db.String(20), default="")  # "" | "running" | "done" | "failed"
    crawl_error = db.Column(db.String(200), default="")  # why a crawl returned zero pages
    public_key = db.Column(db.String(32), unique=True, nullable=False)
    lead_allow_call = db.Column(db.Boolean, default=True)   # owner lets visitors request a phone call
    lead_allow_sms = db.Column(db.Boolean, default=True)    # ... a text message
    lead_allow_email = db.Column(db.Boolean, default=True)  # ... an email
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    chunks = db.relationship("KnowledgeChunk", backref="agent", cascade="all, delete-orphan")
    conversations = db.relationship("Conversation", backref="agent", cascade="all, delete-orphan")
    leads = db.relationship("Lead", backref="agent", cascade="all, delete-orphan")


class Lead(db.Model):
    """A visitor asked for a callback / to be contacted. Shown in the owner's inbox."""
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey("agent.id"), nullable=False)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversation.id"), nullable=True)
    name = db.Column(db.String(120), default="")
    phone = db.Column(db.String(40), default="")
    email = db.Column(db.String(120), default="")
    message = db.Column(db.Text, default="")  # one-line summary of what they wanted
    contact_method = db.Column(db.String(10), default="")  # "call" | "sms" | "email"
    sms_consent = db.Column(db.Boolean, default=False)  # visitor agreed to receive texts
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class KnowledgeChunk(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey("agent.id"), nullable=False)
    source_type = db.Column(db.String(20), nullable=False)  # paste | upload | crawl
    source_label = db.Column(db.String(255), default="")
    content = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.Text, default="")  # JSON list[float], semantic search


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey("agent.id"), nullable=False)
    session_token = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship("Message", backref="conversation", cascade="all, delete-orphan")


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversation.id"), nullable=False)
    role = db.Column(db.String(16), nullable=False)  # user | assistant
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


def lead_methods(agent):
    """Contact methods the owner allows, as (key, label) tuples."""
    methods = []
    if getattr(agent, "lead_allow_call", True):
        methods.append(("call", "a phone call"))
    if getattr(agent, "lead_allow_sms", True):
        methods.append(("sms", "a text message"))
    if getattr(agent, "lead_allow_email", True):
        methods.append(("email", "email"))
    return methods


def _lead_instruction(agent):
    methods = lead_methods(agent)
    if not methods:
        return ("If the visitor asks to be contacted, politely explain the team "
                "isn't taking callback requests right now and offer to help another way.")
    if len(methods) == 1:
        detail = {"call": "their phone number for a callback",
                  "sms": "their phone number for a text message",
                  "email": "their email address"}[methods[0][0]]
        return ("If the visitor asks to be contacted or wants a callback, warmly ask "
                f"for their name and {detail}, and confirm you'll pass it along "
                "to the team right away.")
    opts = ", ".join(m[1] for m in methods[:-1]) + f", or {methods[-1][1]}"
    return ("If the visitor asks to be contacted or wants a callback, warmly ask how "
            f"they'd like to be reached — {opts}. Then collect their name and the "
            "matching contact detail (phone number for a call or text, email address "
            "for email), and confirm you'll pass it along to the team right away. "
            "After that, if they haven't shared one of the other allowed methods, "
            "offer it once as a backup — e.g. 'Want to add an email too, in case we "
            "can't reach you at that number?' If they decline, thank them and move on; "
            "never ask twice. If they gave a phone number for a call (not a text) and "
            "texting is an allowed method, ask once whether it's also okay to text them "
            "at that number.")


# ---------------------------------------------------------------- helpers


def current_user():
    """Return the logged-in User or None (simple session-based auth)."""
    uid = session.get("user_id")
    return db.session.get(User, uid) if uid else None


def login_required(view):
    from functools import wraps

    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user():
            flash("Please log in first.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapper


def get_agent_or_404(agent_id, user):
    agent = Agent.query.filter_by(id=agent_id, user_id=user.id).first()
    if not agent:
        abort(404)
    return agent


_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text):
    """Lowercase word tokens, dropping very short words."""
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2}


def embed_texts(texts):
    """Return embedding vectors for texts via the configured API.

    Returns a list of lists of floats (or Nones on failure), same length as
    the input. Empty list in -> empty list out.
    """
    texts = [t[:8000] for t in texts]
    if not texts or DEMO_MODE:
        return [None] * len(texts)
    try:
        r = requests.post(
            f"{GABBY_API_BASE}/embeddings",
            headers={"Authorization": f"Bearer {GABBY_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": GABBY_EMBED_MODEL, "input": texts},
            timeout=30,
        )
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]
    except Exception as e:
        app.logger.warning("Embedding call failed: %s", e)
        return [None] * len(texts)


def _cosine(a, b):
    """Cosine similarity between two vectors (pure python)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _chunk_vector(chunk):
    try:
        v = json.loads(chunk.embedding or "")
        return v if isinstance(v, list) and v else None
    except (ValueError, TypeError):
        return None


def _ensure_embeddings(agent):
    """Backfill missing chunk embeddings (lazy, so old chunks just work)."""
    if DEMO_MODE:
        return
    missing = [c for c in agent.chunks if not _chunk_vector(c)]
    if not missing:
        return
    vecs = embed_texts([c.content for c in missing])
    changed = False
    for chunk, vec in zip(missing, vecs):
        if vec:
            chunk.embedding = json.dumps(vec)
            changed = True
    if changed:
        db.session.commit()


def retrieve_chunks(agent, query, top_n=3):
    """Semantic retrieval when an API key is set, keyword fallback in demo mode."""
    if not DEMO_MODE:
        _ensure_embeddings(agent)
        qvec = embed_texts([query])[0]
        if qvec:
            scored = []
            for chunk in agent.chunks:
                vec = _chunk_vector(chunk)
                if vec:
                    scored.append((_cosine(qvec, vec), chunk))
            scored.sort(key=lambda s: s[0], reverse=True)
            return [c for _, c in scored[:top_n]]
        # embedding failed -> fall through to keyword matching
    q = _tokens(query)
    scored = []
    for chunk in agent.chunks:
        overlap = len(q & _tokens(chunk.content))
        if overlap:
            scored.append((overlap, chunk))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [c for _, c in scored[:top_n]]


# Common timezones offered in the agent form (business location).
# Stored as IANA values; shown with friendly US names.
TIMEZONES = [
    ("America/New_York", "New York (Eastern Time Zone)"),
    ("America/Chicago", "Chicago (Central Time Zone)"),
    ("America/Denver", "Denver (Mountain Time Zone)"),
    ("America/Los_Angeles", "Los Angeles (Pacific Time Zone)"),
    ("America/Anchorage", "Anchorage (Alaska Time Zone)"),
    ("Pacific/Honolulu", "Honolulu (Hawaii Time Zone)"),
    ("UTC", "UTC (Coordinated Universal Time)"),
]
TIMEZONE_VALUES = [v for v, _ in TIMEZONES]


def _clean_timezone(value):
    value = (value or "").strip()
    return value if value in TIMEZONE_VALUES else "America/New_York"


def parse_suggestions(text):
    """One suggested question per line, max 4, max 80 chars each."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line and len(out) < 4:
            out.append(line[:80])
    return out


def get_greeting(agent):
    return (agent.greeting or "").strip() or DEFAULT_GREETING


DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday"]


def extract_hours(text):
    """Ask the LLM to pull structured weekly hours out of crawled text.

    Returns {day: {"open": "HH:MM", "close": "HH:MM"} | None} or None.
    Skipped in demo mode (no API key).
    """
    if DEMO_MODE or not (text or "").strip():
        return None
    prompt = (
        "Extract this business's weekly opening hours from the text below. "
        'Return ONLY a JSON object with keys monday through sunday. Each value is '
        'either {"open": "HH:MM", "close": "HH:MM"} in 24-hour format, or null if '
        "that day is not listed or is closed. If a range like \"Mon-Fri\" is given, "
        'apply it to each of those days. "Open 24 hours" means '
        '{"open": "00:00", "close": "23:59"}. Output JSON only, no explanation.\n\n'
        "TEXT:\n" + text[:6000]
    )
    try:
        r = requests.post(
            f"{GABBY_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GABBY_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": GABBY_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 500},
            timeout=30,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(m.group(0)) if m else {}
        hours, found = {}, False
        for d in DAYS:
            v = data.get(d)
            if (isinstance(v, dict)
                    and re.fullmatch(r"\d{2}:\d{2}", str(v.get("open", "")))
                    and re.fullmatch(r"\d{2}:\d{2}", str(v.get("close", "")))):
                hours[d] = {"open": v["open"], "close": v["close"]}
                found = True
            else:
                hours[d] = None
        return hours if found else None
    except Exception as e:
        app.logger.warning("Hours extraction failed: %s", e)
        return None


def suggest_questions(text):
    """Ask the LLM to draft customer questions from the crawled site text.

    Returns a list of up to 4 short questions, or []. Skipped in demo mode.
    """
    if DEMO_MODE or not (text or "").strip():
        return []
    prompt = (
        "You are helping a small business set up their website chat agent. "
        "Based on the website text below, write 4 questions a customer would "
        "most likely ask this business. Make them specific to what the business "
        "actually does (not generic). Keep each under 60 characters. "
        "Return ONLY a JSON array of strings, no explanation.\n\n"
        "TEXT:\n" + text[:6000]
    )
    try:
        r = requests.post(
            f"{GABBY_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GABBY_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": GABBY_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7, "max_tokens": 300},
            timeout=30,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\[[\s\S]*\]", content)
        data = json.loads(m.group(0)) if m else []
        out = [str(q).strip() for q in data if str(q).strip()][:4]
        return out
    except Exception as e:
        app.logger.warning("Question suggestion failed: %s", e)
        return []


def _fmt_time(hhmm):
    h, m = int(hhmm[:2]), int(hhmm[3:5])
    suffix = "AM" if h < 12 else "PM"
    return f"{h % 12 or 12}:{m:02d} {suffix}"


def get_hours(agent):
    try:
        hours = json.loads(agent.hours or "")
        return hours if isinstance(hours, dict) else {}
    except (ValueError, TypeError):
        return {}


def hours_prompt_block(agent):
    """Deterministic hours summary + today's open/closed status. No LLM guessing."""
    hours = get_hours(agent)
    if not any(hours.get(d) for d in DAYS):
        return ""
    tzname = (agent.timezone or "America/New_York").strip() or "America/New_York"
    try:
        now = datetime.now(ZoneInfo(tzname))
    except (ZoneInfoNotFoundError, ValueError):
        now = datetime.now(ZoneInfo("America/New_York"))
    today = now.strftime("%A").lower()
    lines = []
    for d in DAYS:
        v = hours.get(d)
        lines.append(f"- {d.capitalize()}: {_fmt_time(v['open'])} – {_fmt_time(v['close'])}"
                     if v else f"- {d.capitalize()}: closed")
    today_v = hours.get(today)
    now_hm = now.strftime("%H:%M")
    if today_v and today_v["open"] <= now_hm < today_v["close"]:
        status = (f"Today is {today.capitalize()}: currently OPEN "
                  f"(closes at {_fmt_time(today_v['close'])})")
    elif today_v:
        status = (f"Today is {today.capitalize()}: currently CLOSED "
                  f"(today's hours {_fmt_time(today_v['open'])} – {_fmt_time(today_v['close'])})")
    else:
        idx = DAYS.index(today)
        nxt = None
        for i in range(1, 8):
            d = DAYS[(idx + i) % 7]
            if hours.get(d):
                nxt = (d, hours[d], i)
                break
        if nxt:
            when = "tomorrow" if nxt[2] == 1 else nxt[0].capitalize()
            status = (f"Today is {today.capitalize()}: CLOSED today. "
                      f"Next open {when} at {_fmt_time(nxt[1]['open'])}")
        else:
            status = f"Today is {today.capitalize()}: CLOSED today."
    trust = ("confirmed by the business owner" if agent.hours_source == "confirmed"
             else "auto-detected from the website (not yet confirmed by the owner)")
    return ("Business hours (" + trust + " — trust these over any other text):\n"
            + "\n".join(lines) + "\n" + status)


def get_suggestions(agent):
    try:
        items = json.loads(agent.suggestions or "[]")
        return [s for s in items if isinstance(s, str)][:4]
    except (ValueError, TypeError):
        return []


def business_now_str(agent):
    """Human-readable current date/time in the business's timezone."""
    tzname = (agent.timezone or "America/New_York").strip() or "America/New_York"
    try:
        now = datetime.now(ZoneInfo(tzname))
    except (ZoneInfoNotFoundError, ValueError):
        now = datetime.now(ZoneInfo("America/New_York"))
        tzname = "America/New_York"
    return now.strftime("%A, %B %d, %Y at %I:%M %p") + f" ({tzname})"


def build_prompt(agent, chunks, history):
    """Assemble the system prompt + knowledge context for the LLM."""
    parts = [
        f"You are {agent.name}, the AI chat agent for this business. "
        "Speak as the business itself (use 'we' and 'our'). "
        "You already know which business you represent — never ask the user "
        "which business or restaurant they mean.",
        f"Current date and time at the business: {business_now_str(agent)}. "
        "Use this for time-sensitive questions: figure out what day it is, "
        "whether the business is open right now, and when it opens/closes today, "
        "based on the hours in the knowledge below. "
        "IMPORTANT: if the hours list some days but NOT today, the business is "
        "CLOSED today — say so plainly ('We're closed today, back Monday at 10am'). "
        "Never borrow another day's hours for today, and never claim to be open "
        "when the listed hours don't cover today.",
    ]
    if agent.instructions.strip():
        parts.append(agent.instructions.strip())
    hours_block = hours_prompt_block(agent)
    if hours_block:
        parts.append(hours_block)
    if chunks:
        parts.append("Relevant knowledge from the business (use this to answer):")
        for c in chunks:
            parts.append(f"- {c.content[:1500]}")
    parts.append(_lead_instruction(agent))
    parts.append(
        "Answer concisely and helpfully. CRITICAL: never invent specific facts "
        "— hours, prices, addresses, phone numbers, menu items. If the knowledge "
        "above doesn't contain the answer, say you don't know that yet and offer "
        "to connect them with the business. A wrong guess about hours or prices "
        "is worse than no answer.")
    return "\n\n".join(parts)


def llm_reply(agent, user_message, history, chunks):
    """Get a reply from the configured LLM (OpenAI-compatible API)."""
    system = build_prompt(agent, chunks, history)
    messages = [{"role": "system", "content": system}]
    for m in history[-10:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": user_message})
    try:
        r = requests.post(
            f"{GABBY_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GABBY_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": GABBY_MODEL, "messages": messages,
                  "temperature": 0.2, "max_tokens": 400},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        app.logger.warning("LLM call failed: %s", e)
        return ("Sorry, I'm having trouble thinking right now — "
                "please try again in a moment.")


def demo_reply(agent, user_message, chunks):
    """Zero-cost demo-mode responder using keyword-matched knowledge."""
    chunks = chunks[:1]
    if chunks:
        excerpt = chunks[0].content.strip().replace("\n", " ")
        if len(excerpt) > 400:
            excerpt = excerpt[:400].rsplit(" ", 1)[0] + "…"
        return f"Here's what I know: {excerpt}"
    name = agent.name
    return (f"Hi! I'm {name}. I don't have information on that yet — "
            f"my owner can add more to my knowledge base and I'll be able "
            f"to help with questions like that.")


def crawl_site(start_url, max_pages=CRAWL_MAX_PAGES, max_chars=CRAWL_MAX_CHARS):
    """Fetch visible text from up to max_pages same-domain pages (BFS).

    Returns (pages, site_title, warnings, error_note): pages is a list of
    {url, text} dicts, site_title is the homepage <title> (used to name the
    business), warnings flags pages that look like scanned images with no
    readable text, and error_note explains a zero-page result ("" when pages
    were found).
    """
    parsed = urlparse(start_url)
    domain = parsed.netloc
    seen, queue, pages, warnings = set(), [start_url], [], []
    site_title = None
    fetch_error = ""
    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, timeout=15, headers=CRAWL_HEADERS)
            if r.status_code != 200:
                if not fetch_error:
                    if r.status_code == 403:
                        fetch_error = "the site's bot protection blocked our crawler"
                    elif r.status_code == 404:
                        fetch_error = "that page wasn't found (HTTP 404)"
                    else:
                        fetch_error = f"the site returned an error (HTTP {r.status_code})"
                continue
            if "text/html" not in r.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            if site_title is None and soup.title and soup.title.string:
                site_title = soup.title.string.strip()
            # Collect links BEFORE stripping anything: nav menus live in
            # header/nav/footer tags, and that's where subpage links are.
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"]).split("#")[0]
                p = urlparse(link)
                if (p.scheme in ("http", "https") and p.netloc == domain
                        and link not in seen and len(queue) < 60
                        and not re.search(r"\.(pdf|jpe?g|png|gif|webp|zip|css|js|mp4)(\?|$)",
                                          link, re.I)):
                    queue.append(link)
            # Strip only scripts/styles: footers/headers often hold the
            # business info a chat agent needs most (hours, address, phone).
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))
            img_count = len(soup.find_all("img"))
            if len(text) > 200:
                pages.append({"url": url, "text": text[:max_chars]})
            # Image-heavy page with almost no text (e.g. a scanned menu):
            # nothing here for the agent to learn from.
            if len(text) < 800 and img_count >= 3:
                warnings.append({"url": url, "reason": "image-heavy"})
        except requests.Timeout:
            if not fetch_error:
                fetch_error = "the site took too long to respond"
            continue  # skip failed pages, keep crawling
        except Exception:
            if not fetch_error:
                fetch_error = "couldn't connect to the site"
            continue  # skip failed pages, keep crawling
    if not pages and not fetch_error:
        fetch_error = "no readable text found on the site's pages"
    return pages, site_title, warnings, fetch_error


def clean_site_title(title):
    """Turn a <title> like "Jeffrey's | Family Restaurant" into a business name."""
    if not title:
        return ""
    for sep in [" | ", " - ", " – ", " — ", " :: ", " \u00b7 "]:
        if sep in title:
            title = title.split(sep)[0]
            break
    return title.strip()[:80]


# ---------------------------------------------------------------- public routes


@app.route("/")
def index():
    return render_template("index.html", site_agent_key=SITE_AGENT_PUBLIC_KEY)


@app.route("/pricing")
def pricing():
    return render_template("pricing.html", site_agent_key=SITE_AGENT_PUBLIC_KEY)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Email and password are required.", "error")
        elif User.query.filter_by(email=email).first():
            flash("That email is already registered — try logging in.", "error")
        else:
            user = User(email=email, password_hash=generate_password_hash(password))
            db.session.add(user)
            db.session.commit()
            session["user_id"] = user.id
            flash("Welcome to Gabby! Create your first agent to get started.", "ok")
            return redirect(url_for("dashboard"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            session["user_id"] = user.id
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("index"))


# ---------------------------------------------------------------- dashboard / agents


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    agents = Agent.query.filter_by(user_id=user.id).order_by(Agent.created_at.desc()).all()
    base = request.host_url.rstrip("/")
    snippets = {a.id: f'<script src="{base}/embed/{a.public_key}.js"></script>' for a in agents}
    unread = {a.id: Lead.query.filter_by(agent_id=a.id, is_read=False).count() for a in agents}
    return render_template("dashboard.html", agents=agents, demo_mode=DEMO_MODE,
                           snippets=snippets, unread=unread)



def flash_image_warnings(warnings):
    """One calm amber banner for image-heavy pages, instead of a red one each."""
    if not warnings:
        return
    paths = []
    for w in warnings:
        try:
            paths.append(urlparse(w["url"]).path or w["url"])
        except Exception:
            paths.append(str(w.get("url", "")))
    shown = ", ".join(paths[:4])
    more = f" (+{len(paths) - 4} more)" if len(paths) > 4 else ""
    n = len(paths)
    flash(f"Heads up: {n} page{'s' if n != 1 else ''} looked like scanned images "
          f"({shown}{more}) — I couldn't read any text from them. "
          f"Paste that content as text and I'll learn it instantly.", "warning")


def _crawl_agent_site(agent_id, url):
    """Crawl in a background thread so the create request returns instantly."""
    with app.app_context():
        agent = db.session.get(Agent, agent_id)
        if not agent:
            return
        try:
            pages, site_title, warnings, fetch_error = crawl_site(url)
            new_chunks = []
            for pg in pages:
                chunk = KnowledgeChunk(agent_id=agent.id, source_type="crawl",
                                       source_label=pg["url"], content=pg["text"])
                db.session.add(chunk)
                new_chunks.append(chunk)
            db.session.commit()
            _embed_new_chunks(new_chunks)
            if pages and not any(get_hours(agent).get(d) for d in DAYS):
                extracted = extract_hours("\n\n".join(pg["text"] for pg in pages))
                if extracted:
                    agent.hours = json.dumps(extracted)
                    agent.hours_source = "auto"
            if pages and not get_suggestions(agent):
                drafted = suggest_questions("\n\n".join(pg["text"] for pg in pages))
                if drafted:
                    agent.suggestions = json.dumps(drafted)
            agent.crawl_status = "done" if pages else "failed"
            agent.crawl_error = "" if pages else (fetch_error or "")[:200]
            db.session.commit()
        except Exception:
            app.logger.exception("Background crawl failed for agent %s", agent_id)
            try:
                agent.crawl_status = "failed"
                agent.crawl_error = "an unexpected error stopped the crawl"
                db.session.commit()
            except Exception:
                pass


@app.route("/agents/<int:agent_id>/crawl-status")
@login_required
def agent_crawl_status(agent_id):
    agent = get_agent_or_404(agent_id, current_user())
    return jsonify({"status": agent.crawl_status or ""})


@app.route("/agents/new", methods=["GET", "POST"])
@login_required
def agent_new():
    user = current_user()
    if request.method == "POST":
        name = request.form.get("name", "").strip() or "Untitled Agent"
        color = request.form.get("color", "#4F46E5").strip() or "#4F46E5"
        timezone = _clean_timezone(request.form.get("timezone"))
        greeting = request.form.get("greeting", "").strip() or DEFAULT_GREETING
        website_url = request.form.get("website_url", "").strip()
        agent = Agent(user_id=user.id, name=name, instructions="",
                      color=color, timezone=timezone, suggestions="[]",
                      greeting=greeting, public_key=secrets.token_urlsafe(16),
                      hours="{}", hours_source="")
        db.session.add(agent)
        db.session.commit()
        # Crawl in the background so this request returns instantly.
        # Knowledge, hours, and suggestions land on the agent when it finishes.
        if website_url:
            p = urlparse(website_url)
            if p.scheme in ("http", "https") and p.netloc:
                agent.crawl_status = "running"
                db.session.commit()
                threading.Thread(target=_crawl_agent_site,
                                 args=(agent.id, website_url),
                                 daemon=True).start()
                flash(f"Agent '{name}' created! We're reading your website now — "
                      f"your knowledge, hours, and conversation starters will appear "
                      f"here when it's done.", "ok")
            else:
                flash(f"Agent '{name}' created! That website URL didn't look right — "
                      f"you can crawl it later from the Knowledge page.", "error")
        else:
            flash(f"Agent '{name}' created! Add your website on the Knowledge page "
                  f"so it can learn your business.", "ok")
        return redirect(url_for("agent_edit", agent_id=agent.id))
    return render_template("agent_form.html", agent=None, timezones=TIMEZONES)


@app.route("/agents/<int:agent_id>/edit", methods=["GET", "POST"])
@login_required
def agent_edit(agent_id):
    user = current_user()
    agent = get_agent_or_404(agent_id, user)
    if request.method == "POST":
        agent.name = request.form.get("name", "").strip() or agent.name
        agent.instructions = request.form.get("instructions", "").strip()
        agent.color = request.form.get("color", "#4F46E5").strip() or "#4F46E5"
        agent.timezone = _clean_timezone(request.form.get("timezone"))
        agent.suggestions = json.dumps(parse_suggestions(request.form.get("suggestions")))
        agent.greeting = request.form.get("greeting", "").strip() or DEFAULT_GREETING
        hours = parse_hours_form(request.form)
        agent.hours = json.dumps(hours)
        agent.hours_source = "confirmed" if any(hours.values()) else ""
        agent.lead_allow_call = request.form.get("lead_allow_call") == "on"
        agent.lead_allow_sms = request.form.get("lead_allow_sms") == "on"
        agent.lead_allow_email = request.form.get("lead_allow_email") == "on"
        db.session.commit()
        flash("Agent updated.", "ok")
        return redirect(url_for("dashboard"))
    snippet = f'<script src="{request.host_url.rstrip("/")}/embed/{agent.public_key}.js"></script>'
    return render_template("agent_form.html", agent=agent, snippet=snippet, timezones=TIMEZONES, suggestions=get_suggestions(agent), hours=get_hours(agent), days=DAYS)


@app.route("/agents/<int:agent_id>/leads")
@login_required
def agent_leads(agent_id):
    agent = get_agent_or_404(agent_id, current_user())
    leads = Lead.query.filter_by(agent_id=agent.id).order_by(Lead.created_at.desc()).all()
    return render_template("leads.html", agent=agent, leads=leads)


@app.route("/agents/<int:agent_id>/leads/read", methods=["POST"])
@login_required
def agent_leads_read(agent_id):
    agent = get_agent_or_404(agent_id, current_user())
    Lead.query.filter_by(agent_id=agent.id, is_read=False).update({"is_read": True})
    db.session.commit()
    return redirect(url_for("agent_leads", agent_id=agent.id))


@app.route("/agents/<int:agent_id>/delete", methods=["POST"])
@login_required
def agent_delete(agent_id):
    user = current_user()
    agent = get_agent_or_404(agent_id, user)
    db.session.delete(agent)
    db.session.commit()
    flash(f"Agent '{agent.name}' deleted.", "ok")
    return redirect(url_for("dashboard"))


# ---------------------------------------------------------------- knowledge base


def parse_hours_form(form):
    """Read the per-day hours fields from the agent form. Returns {day: {...}|None}."""
    hours = {}
    for d in DAYS:
        if form.get(f"hours_{d}_closed"):
            hours[d] = None
        else:
            o = (form.get(f"hours_{d}_open") or "").strip()
            c = (form.get(f"hours_{d}_close") or "").strip()
            if re.fullmatch(r"\d{2}:\d{2}", o) and re.fullmatch(r"\d{2}:\d{2}", c):
                hours[d] = {"open": o, "close": c}
            else:
                hours[d] = None
    return hours


def _embed_new_chunks(chunks):
    """Embed freshly added chunks right away (no-op in demo mode)."""
    if DEMO_MODE or not chunks:
        return
    vecs = embed_texts([c.content for c in chunks])
    changed = False
    for chunk, vec in zip(chunks, vecs):
        if vec:
            chunk.embedding = json.dumps(vec)
            changed = True
    if changed:
        db.session.commit()


@app.route("/agents/<int:agent_id>/knowledge", methods=["GET", "POST"])
@login_required
def knowledge(agent_id):
    user = current_user()
    agent = get_agent_or_404(agent_id, user)

    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "paste":
            text = request.form.get("text", "").strip()
            label = request.form.get("label", "").strip() or "Pasted text"
            if text:
                chunk = KnowledgeChunk(agent_id=agent.id, source_type="paste",
                                       source_label=label, content=text)
                db.session.add(chunk)
                db.session.commit()
                _embed_new_chunks([chunk])
                flash("Knowledge added.", "ok")
            else:
                flash("Paste some text first.", "error")

        elif action == "upload":
            f = request.files.get("file")
            if f and f.filename:
                name = f.filename.lower()
                if name.endswith((".txt", ".md")):
                    try:
                        text = f.read().decode("utf-8", errors="replace").strip()
                    except Exception:
                        text = ""
                    if text:
                        chunk = KnowledgeChunk(
                            agent_id=agent.id, source_type="upload",
                            source_label=f.filename, content=text[:20000])
                        db.session.add(chunk)
                        db.session.commit()
                        _embed_new_chunks([chunk])
                        flash(f"Uploaded '{f.filename}'.", "ok")
                    else:
                        flash("That file looks empty.", "error")
                else:
                    flash("Only .txt and .md files are supported in the MVP.", "error")
            else:
                flash("Choose a file to upload.", "error")

        elif action == "crawl":
            url = request.form.get("url", "").strip()
            p = urlparse(url)
            if p.scheme not in ("http", "https") or not p.netloc:
                flash("Enter a full URL starting with http:// or https://", "error")
            else:
                pages, site_title, warnings, fetch_error = crawl_site(url)
                if pages:
                    new_chunks = []
                    for pg in pages:
                        chunk = KnowledgeChunk(
                            agent_id=agent.id, source_type="crawl",
                            source_label=pg["url"], content=pg["text"])
                        db.session.add(chunk)
                        new_chunks.append(chunk)
                    # If the agent still has a placeholder name, name it after
                    # the business we just crawled so it knows who it is.
                    biz = clean_site_title(site_title)
                    if biz and agent.name.strip().lower() in (
                            "", "my agent", "new agent", "test", "test agent",
                            "demo", "demo agent"):
                        agent.name = biz
                    db.session.commit()
                    _embed_new_chunks(new_chunks)
                    flash(f"Crawled {len(pages)} page(s) from {p.netloc}.", "ok")
                    # Auto-detect business hours from the crawled text so the
                    # owner only has to confirm or correct them.
                    if not any(get_hours(agent).get(d) for d in DAYS):
                        extracted = extract_hours(
                            "\n\n".join(pg["text"] for pg in pages))
                        if extracted:
                            agent.hours = json.dumps(extracted)
                            agent.hours_source = "auto"
                            db.session.commit()
                            flash("We detected business hours on your site — "
                                  "please review them under Edit agent → "
                                  "Business hours.", "ok")
                    # Draft conversation starters from the site too,
                    # if none are set yet.
                    if not get_suggestions(agent):
                        drafted = suggest_questions(
                            "\n\n".join(pg["text"] for pg in pages))
                        if drafted:
                            agent.suggestions = json.dumps(drafted)
                            db.session.commit()
                            flash("We drafted conversation starters from your "
                                  "site — review them under Edit agent.", "ok")
                    flash_image_warnings(warnings)
                else:
                    reason = f" ({fetch_error})" if fetch_error else ""
                    flash(f"Couldn't extract any readable text from that URL{reason}.", "error")

        elif action == "delete":
            chunk_id = request.form.get("chunk_id", type=int)
            chunk = KnowledgeChunk.query.filter_by(id=chunk_id, agent_id=agent.id).first()
            if chunk:
                db.session.delete(chunk)
                db.session.commit()
                flash("Knowledge removed.", "ok")

        return redirect(url_for("knowledge", agent_id=agent.id))

    chunks = KnowledgeChunk.query.filter_by(agent_id=agent.id).order_by(
        KnowledgeChunk.id.desc()).all()
    return render_template("knowledge.html", agent=agent, chunks=chunks,
                           crawl_max_pages=CRAWL_MAX_PAGES,
                           crawl_max_chars=CRAWL_MAX_CHARS)


# ---------------------------------------------------------------- test chat page


@app.route("/agents/<int:agent_id>/chat")
@login_required
def chat_page(agent_id):
    user = current_user()
    agent = get_agent_or_404(agent_id, user)
    return render_template("chat.html", agent=agent, demo_mode=DEMO_MODE,
                           suggestions=get_suggestions(agent),
                           greeting=get_greeting(agent))


# ---------------------------------------------------------------- chat API + widget


PHONE_RE = re.compile(r"(\+?1[\s.\-]?)?(\(?\d{3}\)?[\s.\-]?)?\d{3}[\s.\-]\d{4}|\b\d{10}\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def _contact_share_signals(user_message, history):
    """Cheap pre-filter: does this turn look like the visitor sharing contact info?"""
    if PHONE_RE.search(user_message) or EMAIL_RE.search(user_message):
        return True
    if history:
        last_assistant = next((m.content for m in reversed(history) if m.role == "assistant"), "")
        asked = any(w in last_assistant.lower() for w in (
            "phone number", "phone", "email", "call you", "reach you", "contact you",
            "get back to you"))
        if asked and len(user_message.strip()) >= 2:
            return True
    return False


def _parse_lead_json(text):
    """Parse the extraction model's reply, tolerating markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except Exception:
            pass
    return None


def extract_lead(history):
    """Pull name/phone/email from recent conversation via the LLM. Returns dict or None."""
    convo = "\n".join(f"{m.role}: {m.content}" for m in history[-8:])
    prompt = (
        "Extract visitor contact info from this chat transcript for a callback request. "
        'Return ONLY valid JSON like {"name": "...", "phone": "...", "email": "...", '
        '"message": "...", "contact_method": "call", "sms_consent": true}. Use empty '
        'strings for anything not provided. "message" is a one-line summary of what they '
        'wanted (e.g. "callback about catering prices"). "contact_method" is "call", '
        '"sms", or "email" based on what they chose. "sms_consent" is true only if they '
        "agreed to receive text messages. If no contact info was shared, return {}. "
        "Transcript:\n" + convo)
    try:
        r = requests.post(
            f"{GABBY_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GABBY_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": GABBY_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 200},
            timeout=30,
        )
        r.raise_for_status()
        data = _parse_lead_json(r.json()["choices"][0]["message"]["content"])
        if data and (data.get("phone") or data.get("email")):
            return data
    except Exception as e:
        app.logger.warning("Lead extraction failed: %s", e)
    return None


def capture_lead(agent, conv, user_msg_obj, assistant_msg_obj, history):
    """Create or update a Lead for this conversation if contact info was shared.
    Returns the Lead if newly created (for the new-lead email), else None."""
    try:
        if DEMO_MODE:
            return None
        if not _contact_share_signals(user_msg_obj.content, history):
            return None
        data = extract_lead(history + [user_msg_obj, assistant_msg_obj])
        if not data:
            return None
        lead = Lead.query.filter_by(conversation_id=conv.id).first()
        is_new = lead is None
        if is_new:
            lead = Lead(agent_id=agent.id, conversation_id=conv.id)
            db.session.add(lead)
        # Fill in whatever we got; never blank out existing values.
        if data.get("name"):
            lead.name = data["name"][:120]
        if data.get("phone"):
            lead.phone = data["phone"][:40]
        if data.get("email"):
            lead.email = data["email"][:120]
        if data.get("message"):
            lead.message = data["message"]
        if data.get("contact_method") in ("call", "sms", "email"):
            lead.contact_method = data["contact_method"]
        if data.get("sms_consent") is True:
            lead.sms_consent = True
        elif data.get("phone"):
            lead.contact_method = "call"
        elif data.get("email"):
            lead.contact_method = "email"
        lead.is_read = False
        return lead if is_new else None
    except Exception as e:
        app.logger.warning("Lead capture failed: %s", e)
    return None


def send_lead_email(owner_email, agent_name, lead, leads_url):
    """Email the owner about a new lead via Resend. Skips silently if unconfigured."""
    import html as _html
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key or not owner_email:
        return
    from_addr = os.environ.get("LEAD_EMAIL_FROM", "TryGabby <leads@trygabby.com>").strip()

    def esc(v):
        return _html.escape(str(v or "-"))

    name = esc(lead["name"])
    phone = esc(lead["phone"])
    email = esc(lead["email"])
    contact = esc(lead["contact_method"])
    ok_text = "Yes" if (lead["sms_consent"] or lead["contact_method"] == "sms") else "No"
    wanted = esc(lead["message"])
    agent_safe = esc(agent_name)

    phone_row = phone
    if lead["phone"]:
        phone_row = f'<a href="tel:{_html.escape(lead["phone"])}" style="color:#0d9488;font-weight:700;text-decoration:none;">{phone}</a>'
    email_row = email
    if lead["email"]:
        email_row = f'<a href="mailto:{_html.escape(lead["email"])}" style="color:#0d9488;text-decoration:none;">{email}</a>'

    html_body = f"""\
<div style="background:#f1f5f9;padding:24px 12px;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <div style="max-width:560px;margin:0 auto;background:#ffffff;border-radius:14px;overflow:hidden;box-shadow:0 2px 10px rgba(15,23,42,0.08);">
    <div style="background:#0d9488;padding:20px 24px;">
      <div style="font-size:22px;font-weight:800;color:#ffffff;letter-spacing:0.5px;">Try<span style="opacity:0.85;">Gabby</span></div>
      <div style="color:#ccfbf1;font-size:13px;margin-top:4px;">New lead for {agent_safe}</div>
    </div>
    <div style="padding:24px;">
      <div style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;">
        <div style="padding:14px 18px;border-bottom:1px solid #f1f5f9;">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#94a3b8;">Name</div>
          <div style="font-size:18px;font-weight:700;color:#0f172a;margin-top:2px;">{name}</div>
        </div>
        <div style="padding:14px 18px;border-bottom:1px solid #f1f5f9;">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#94a3b8;">Phone</div>
          <div style="font-size:16px;color:#0f172a;margin-top:2px;">{phone_row}</div>
        </div>
        <div style="padding:14px 18px;border-bottom:1px solid #f1f5f9;">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#94a3b8;">Email</div>
          <div style="font-size:16px;color:#0f172a;margin-top:2px;">{email_row}</div>
        </div>
        <div style="padding:14px 18px;border-bottom:1px solid #f1f5f9;">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#94a3b8;">Preferred contact</div>
          <div style="font-size:16px;color:#0f172a;margin-top:2px;">{contact} &nbsp;·&nbsp; OK to text: {ok_text}</div>
        </div>
        <div style="padding:14px 18px;">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#94a3b8;">Wanted</div>
          <div style="font-size:16px;color:#0f172a;margin-top:2px;">{wanted}</div>
        </div>
      </div>
      <div style="text-align:center;margin-top:22px;">
        <a href="{_html.escape(leads_url)}" style="display:inline-block;background:#0d9488;color:#ffffff;font-weight:700;font-size:16px;padding:13px 34px;border-radius:10px;text-decoration:none;">View all leads</a>
      </div>
    </div>
    <div style="padding:16px 24px;border-top:1px solid #f1f5f9;text-align:center;">
      <div style="font-size:12px;color:#94a3b8;">Sent by TryGabby — your AI chat agent</div>
    </div>
  </div>
</div>"""

    text_body = (
        f"You have a new lead from {agent_name}.\n\n"
        f"Name: {lead['name'] or '-'}\n"
        f"Phone: {lead['phone'] or '-'}\n"
        f"Email: {lead['email'] or '-'}\n"
        f"Preferred contact: {lead['contact_method'] or '-'}\n"
        f"OK to text: {'yes' if lead['sms_consent'] or lead['contact_method'] == 'sms' else 'no'}\n"
        f"Wanted: {lead['message'] or '-'}\n\n"
        f"View all leads: {leads_url}")
    try:
        requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from": from_addr,
                "to": [owner_email],
                "subject": f"New lead for {agent_name}: {lead['name'] or 'a visitor'}",
                "html": html_body,
                "text": text_body,
            },
            timeout=15,
        )
    except Exception as e:
        app.logger.warning("Lead email failed: %s", e)


def send_support_email(user_email, subject, message):
    """Email a client's support request to the support inbox via Resend.

    Reply-To is the client's email, so replying in Gmail goes straight back
    to them. Skips silently if unconfigured.
    """
    import html as _html
    api_key = os.environ.get("RESEND_API_KEY", "")
    if not api_key or not SUPPORT_EMAIL:
        return False
    esc = lambda v: _html.escape(str(v or ""))
    subj = esc(subject)
    body = esc(message).replace("\n", "<br>")
    html_body = f"""\
<div style="background:#f1f5f9;padding:24px 12px;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <div style="max-width:560px;margin:0 auto;background:#ffffff;border-radius:14px;overflow:hidden;box-shadow:0 2px 10px rgba(15,23,42,0.08);">
    <div style="background:#0d9488;padding:20px 24px;">
      <div style="font-size:22px;font-weight:800;color:#ffffff;letter-spacing:0.5px;">Try<span style="opacity:0.85;">Gabby</span></div>
      <div style="color:#ccfbf1;font-size:13px;margin-top:4px;">Support request from {esc(user_email)}</div>
    </div>
    <div style="padding:24px;">
      <div style="font-size:18px;font-weight:700;color:#0f172a;margin-bottom:12px;">{subj}</div>
      <div style="font-size:15px;color:#334155;line-height:1.6;">{body}</div>
    </div>
  </div>
</div>"""
    try:
        requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}",
                      "Content-Type": "application/json"},
            json={
                "from": SUPPORT_FROM,
                "reply_to": user_email,
                "to": [SUPPORT_EMAIL],
                "subject": f"[Support] {subject}",
                "html": html_body,
                "text": f"Support request from {user_email}\n\nSubject: {subject}\n\n{message}",
            },
            timeout=15,
        )
        return True
    except Exception as e:
        app.logger.warning("Support email failed: %s", e)
        return False


@app.route("/support", methods=["GET", "POST"])
@login_required
def support():
    user = current_user()
    if request.method == "POST":
        subject = (request.form.get("subject") or "").strip()[:120]
        message = (request.form.get("message") or "").strip()[:4000]
        if not subject or not message:
            flash("Please add a subject and a message.", "error")
            return render_template("support.html")
        if send_support_email(user.email, subject, message):
            flash("Message sent — we'll reply to the email on your account.", "ok")
        else:
            flash("Couldn't send that just now — please email support@trygabby.com directly.", "error")
        return redirect(url_for("support"))
    return render_template("support.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    public_key = data.get("public_key", "")
    session_token = data.get("session_token", "")[:64]
    user_message = (data.get("message") or "").strip()

    agent = Agent.query.filter_by(public_key=public_key).first()
    if not agent:
        return jsonify({"error": "Unknown agent key."}), 404
    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    conv = Conversation.query.filter_by(agent_id=agent.id,
                                        session_token=session_token).first()
    if not conv:
        conv = Conversation(agent_id=agent.id, session_token=session_token or
                            secrets.token_hex(8))
        db.session.add(conv)
        db.session.flush()

    history = Message.query.filter_by(conversation_id=conv.id).order_by(Message.id).all()
    chunks = retrieve_chunks(agent, user_message)
    reply = demo_reply(agent, user_message, chunks) if DEMO_MODE else \
        llm_reply(agent, user_message, history, chunks)

    user_msg = Message(conversation_id=conv.id, role="user", content=user_message)
    asst_msg = Message(conversation_id=conv.id, role="assistant", content=reply)
    db.session.add(user_msg)
    db.session.add(asst_msg)
    new_lead = None
    if agent.public_key != SITE_AGENT_PUBLIC_KEY:
        new_lead = capture_lead(agent, conv, user_msg, asst_msg, history)
    lead_info = None
    if new_lead:
        lead_info = {"name": new_lead.name, "phone": new_lead.phone,
                     "email": new_lead.email, "message": new_lead.message,
                     "contact_method": new_lead.contact_method,
                     "sms_consent": new_lead.sms_consent}
    db.session.commit()
    if lead_info:
        owner = db.session.get(User, agent.user_id)
        leads_url = f"{request.host_url.rstrip('/')}/agents/{agent.id}/leads"
        threading.Thread(target=send_lead_email,
                         args=(owner.email if owner else "", agent.name, lead_info, leads_url),
                         daemon=True).start()
    return jsonify({"reply": reply, "demo_mode": DEMO_MODE,
                    "session_token": conv.session_token,
                    "sources": [c.source_label for c in chunks]})


@app.route("/embed/<public_key>.js")
def embed_js(public_key):
    """Serve the chat widget loader with the agent key + API base injected."""
    agent = Agent.query.filter_by(public_key=public_key).first_or_404()
    with open(os.path.join(BASE_DIR, "static", "widget.js")) as f:
        js = f.read()
    js = js.replace("__PUBLIC_KEY__", agent.public_key)
    js = js.replace("__API_BASE__", request.host_url.rstrip("/"))
    js = js.replace("__AGENT_NAME__", agent.name.replace('"', ""))
    js = js.replace("__AGENT_COLOR__", agent.color)
    js = js.replace("__SUGGESTIONS__", json.dumps(get_suggestions(agent)))
    js = js.replace("__GREETING__", json.dumps(get_greeting(agent)))
    return Response(js, mimetype="application/javascript")


# ---------------------------------------------------------------- main


def _table_columns(table):
    insp = sa.inspect(db.engine)
    return [c["name"] for c in insp.get_columns(table)]


def run_migrations():
    """Lightweight migrations for databases created by older versions."""
    cols = _table_columns("agent")
    if "timezone" not in cols:
        db.session.execute(db.text(
            "ALTER TABLE agent ADD COLUMN timezone VARCHAR(64) DEFAULT 'America/New_York'"))
        db.session.commit()
    if "suggestions" not in cols:
        db.session.execute(db.text(
            "ALTER TABLE agent ADD COLUMN suggestions TEXT DEFAULT '[]'"))
        db.session.commit()
    if "hours" not in cols:
        db.session.execute(db.text(
            "ALTER TABLE agent ADD COLUMN hours TEXT DEFAULT ''"))
        db.session.commit()
    if "hours_source" not in cols:
        db.session.execute(db.text(
            "ALTER TABLE agent ADD COLUMN hours_source VARCHAR(20) DEFAULT ''"))
        db.session.commit()
    if "greeting" not in cols:
        db.session.execute(db.text(
            "ALTER TABLE agent ADD COLUMN greeting TEXT DEFAULT 'Hi there! How can I help?'"))
        db.session.commit()
    if "crawl_status" not in cols:
        db.session.execute(db.text(
            "ALTER TABLE agent ADD COLUMN crawl_status VARCHAR(20) DEFAULT ''"))
        db.session.commit()
    if "crawl_error" not in cols:
        db.session.execute(db.text(
            "ALTER TABLE agent ADD COLUMN crawl_error VARCHAR(200) DEFAULT ''"))
        db.session.commit()
    for _col in ("lead_allow_call", "lead_allow_sms", "lead_allow_email"):
        if _col not in cols:
            db.session.execute(db.text(
                f"ALTER TABLE agent ADD COLUMN {_col} BOOLEAN DEFAULT TRUE"))
            db.session.commit()
    _lcols = _table_columns("lead")
    if "contact_method" not in _lcols:
        db.session.execute(db.text(
            "ALTER TABLE lead ADD COLUMN contact_method VARCHAR(10) DEFAULT ''"))
        db.session.commit()
    if "sms_consent" not in _lcols:
        db.session.execute(db.text(
            "ALTER TABLE lead ADD COLUMN sms_consent BOOLEAN DEFAULT FALSE"))
        db.session.commit()
    kcols = _table_columns("knowledge_chunk")
    if "embedding" not in kcols:
        db.session.execute(db.text(
            "ALTER TABLE knowledge_chunk ADD COLUMN embedding TEXT DEFAULT ''"))
        db.session.commit()


# Run at import time too — production servers (gunicorn) never hit __main__.
def ensure_site_agent():
    """Create the built-in TryGabby assistant for the marketing site, once.

    Owned by an internal system user so it never shows in a customer
    dashboard. Knowledge comes from site_knowledge.md; embeddings backfill
    lazily on first chat via _ensure_embeddings.
    """
    agent = Agent.query.filter_by(public_key=SITE_AGENT_PUBLIC_KEY).first()
    if agent and agent.chunks:
        return agent
    user = User.query.filter_by(email=SITE_AGENT_EMAIL).first()
    if not user:
        user = User(email=SITE_AGENT_EMAIL,
                    password_hash="!")
        db.session.add(user)
        db.session.flush()
    if not agent:
        agent = Agent(
            user_id=user.id,
            name="TryGabby Assistant",
            greeting="Hi! I'm the TryGabby assistant. Ask me about pricing, features, or how it works.",
            color="#0d9488",
            timezone="America/Detroit",
            public_key=SITE_AGENT_PUBLIC_KEY,
            instructions=(
                "You are the friendly assistant for TryGabby (trygabby.com), "
                "an AI chat agent service for businesses of every size. Answer "
                "visitor questions about TryGabby using the provided knowledge. "
                "Keep answers short and conversational (1-3 sentences). If asked "
                "about something not in your knowledge, say you don't know and "
                "suggest signing up for the free beta. Never invent pricing, "
                "features, or customer stories."
            ),
            suggestions=json.dumps(SITE_AGENT_SUGGESTIONS),
        )
        db.session.add(agent)
        db.session.flush()
    if not agent.chunks:
        here = os.path.dirname(os.path.abspath(__file__))
        md_path = os.path.join(here, "site_knowledge.md")
        try:
            with open(md_path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            text = ""
        sections = [s.strip() for s in re.split(r"\n## ", text) if s.strip()]
        for i, section in enumerate(sections):
            db.session.add(KnowledgeChunk(
                agent_id=agent.id,
                source_type="paste",
                source_label="About TryGabby",
                content=section[:2000],
            ))
    db.session.commit()
    return agent


with app.app_context():
    db.create_all()
    run_migrations()
    ensure_site_agent()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
