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
db = SQLAlchemy(app)

# Chat-brain (LLM) configuration. Demo mode when no key is present.
GABBY_API_KEY = os.environ.get("GABBY_API_KEY", "").strip()
GABBY_API_BASE = os.environ.get("GABBY_API_BASE", "https://api.openai.com/v1").rstrip("/")
GABBY_MODEL = os.environ.get("GABBY_MODEL", "gpt-4o-mini")
GABBY_EMBED_MODEL = os.environ.get("GABBY_EMBED_MODEL", "text-embedding-3-small")
DEMO_MODE = not GABBY_API_KEY

GABBY_VERSION = "0.18.0"

CRAWL_MAX_PAGES = 100
CRAWL_MAX_CHARS = 30000

DEFAULT_GREETING = "Hi there! How can I help?"


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
    public_key = db.Column(db.String(32), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    chunks = db.relationship("KnowledgeChunk", backref="agent", cascade="all, delete-orphan")
    conversations = db.relationship("Conversation", backref="agent", cascade="all, delete-orphan")


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

    Returns (pages, site_title, warnings): pages is a list of {url, text} dicts,
    site_title is the homepage <title> (used to name the business), and warnings
    flags pages that look like scanned images with no readable text.
    """
    parsed = urlparse(start_url)
    domain = parsed.netloc
    seen, queue, pages, warnings = set(), [start_url], [], []
    site_title = None
    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, timeout=10,
                             headers={"User-Agent": "GabbyBot/1.0 (+https://trygabby.com)"})
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
        except Exception:
            continue  # skip failed pages, keep crawling
    return pages, site_title, warnings


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
    return render_template("index.html")


@app.route("/pricing")
def pricing():
    return render_template("pricing.html")


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
    return render_template("dashboard.html", agents=agents, demo_mode=DEMO_MODE)



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
            pages, site_title, warnings = crawl_site(url)
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
            db.session.commit()
        except Exception:
            app.logger.exception("Background crawl failed for agent %s", agent_id)
            try:
                agent.crawl_status = "failed"
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
        db.session.commit()
        flash("Agent updated.", "ok")
        return redirect(url_for("dashboard"))
    snippet = f'<script src="{request.host_url.rstrip("/")}/embed/{agent.public_key}.js"></script>'
    return render_template("agent_form.html", agent=agent, snippet=snippet, timezones=TIMEZONES, suggestions=get_suggestions(agent), hours=get_hours(agent), days=DAYS)


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
                pages, site_title, warnings = crawl_site(url)
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
                    flash("Couldn't extract any readable text from that URL.", "error")

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

    db.session.add(Message(conversation_id=conv.id, role="user", content=user_message))
    db.session.add(Message(conversation_id=conv.id, role="assistant", content=reply))
    db.session.commit()
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
    kcols = _table_columns("knowledge_chunk")
    if "embedding" not in kcols:
        db.session.execute(db.text(
            "ALTER TABLE knowledge_chunk ADD COLUMN embedding TEXT DEFAULT ''"))
        db.session.commit()


# Run at import time too — production servers (gunicorn) never hit __main__.
with app.app_context():
    db.create_all()
    run_migrations()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
