# Legacy entrypoint retained only for historical reference.
# Active runtime entrypoint: backend/app/main.py
# from backend.app import *  # noqa: F401,F403
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os
import secrets
import string
import logging
import hashlib
import requests
from cachetools import TTLCache
from requests import exceptions as requests_exceptions
from dotenv import load_dotenv
# from openai import OpenAI
import re
from pathlib import Path
from urllib.parse import urlencode

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=5000)
    raise SystemExit(0)

app = FastAPI(title="Integrated Legal Chatbot", version="1.0")
sys.path.append(os.path.dirname(__file__))
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chatbot.app")

from backend.RAG_Bot.doc.pdfRP import generate_answer as pdf_rag_answer
from backend.RAG_Bot.txt.txt_rag import txt_rag_answer
from backend.RAG_Bot.txt.txt_rag import model, tokenizer
from backend.RAG_Bot.case.case_rag import case_rag_answer
from backend.RAG_Bot.state.state_rag import state_rag_answer
from backend.Intent_Bot.Intent_bot import IntentBot
from database import (
    create_session,
    create_user,
    create_google_user,
    delete_session,
    get_user_by_email,
    get_user_by_google_sub,
    get_user_by_token,
    init_db,
    link_google_account,
    list_chat_messages,
    store_chat_message,
    authenticate_user,
)
init_db()

origins = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]

# Development workaround for origin mismatch (avoids 405 on CORS preflight from different dev hosts)
# TODO: lock this down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/frontend", StaticFiles(directory="Frontend"), name="frontend")

intent_bot = None

def get_intent_bot():
    global intent_bot
    if intent_bot is None:
        logger.info("Initializing Intent Bot...")
        intent_bot = IntentBot()
        logger.info("Intent Bot initialized")
    return intent_bot

INTENT_THRESHOLD = 0.7
RAG_THRESHOLD = 0.7
MIN_ANSWER_LENGTH = 20  
MAX_CONTEXT_CHUNKS = 5
MAX_CONTEXT_CHARS_PER_CHUNK = 900
LLM_SYSTEM_PROMPT = """
You are a professional Indian legal assistant who speaks like a real advocate.

GOAL:
- Talk like a human lawyer, not a robot
- Guide the user step-by-step like a consultation
- Ask relevant questions gradually (not all at once)
- Give practical, clear legal guidance

BEHAVIOR:
- Start naturally (like a human, not scripted)
- Understand user's problem first
- Then ask ONE relevant follow-up question at a time
- Based on answers, guide them forward

TONE:
- Calm, helpful, professional
- Simple language (no heavy legal jargon unless needed)
- Slight conversational tone (like a real advocate talking)

RULES:
- Do NOT ask generic questions like "what is your issue?"
- Do NOT dump long paragraphs
- Do NOT act like FAQ bot
- ALWAYS use previous conversation context (state/memory)

FLOW:
1. Acknowledge user problem
2. Clarify situation with 1 question
3. Give small actionable guidance
4. Continue step-by-step

OUTPUT STYLE:
- 2–4 lines max per response
- If needed, give options (1,2,3)
- Ask next question at end

EXAMPLE STYLE:

User: I got e challan

Assistant:
Alright, no problem — this happens quite often.

First, tell me: do you want to check it, pay it, or challenge it?

User: check

Assistant:
Okay. You can check it online using your vehicle number.

Have you received an SMS or do you have the challan number?

IMPORTANT:
- Maintain conversational continuity
- Adapt based on user's replies
- Act like a real legal advisor, not a chatbot
""".strip()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
# OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_REQUEST_TIMEOUT_SECONDS = 20
LLM_CACHE = TTLCache(maxsize=256, ttl=900)
def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()
# OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GOOGLE_OAUTH_STATE_CACHE = TTLCache(maxsize=128, ttl=600)
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "APP_BASE_URL=https://cheating-uncover-resubmit.ngrok-free.dev").strip()
FRONTEND_AUTH_URL = os.environ.get("FRONTEND_AUTH_URL", "http://127.0.0.1:5000/frontend/auth.html").strip()
FRONTEND_APP_URL = os.environ.get("FRONTEND_APP_URL", "http://127.0.0.1:5000/frontend/Index.html").strip()
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_SCOPES = "openid email profile"

SUPPORTED_MATTER_TYPES = {
    "civil",
    "criminal",
    "constitutional",
    "family",
    "property",
    "labour",
    "consumer",
    "cyber",
    "general",
    "unsure",
}

SUPPORTED_URGENCY = {"low", "medium", "high", "urgent"}
SUPPORTED_MATTER_STAGES = {
    "pre_fir",
    "fir_registered",
    "arrest",
    "bail",
    "notice_received",
    "pre_suit",
    "suit_filed",
    "trial",
    "appeal",
    "execution",
    "not_sure",
}

INDIAN_STATES_AND_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]

LEGAL_TERM_EXPANSIONS = {
    "cpc": "civil procedure code CPC 1908 procedure",
    "Article": "Rule in Constitution (Constitution of India)",
    "Law": "General term for all legal rules",
    "Act": "Law passed by Parliament (e.g. Indian Penal Code)",
    "Bill": "Draft before becoming Act",
    "Section": "Part of an Act",
    "Clause": "Part of a Section/Article",
    "Rule": "How to implement an Act",
    "Regulation": "Detailed rules by authorities",
    "Ordinance": "Temporary law by President of India",
    "Amendment": "Change in law",           
    "Judgment": "Court decision (Supreme Court of India)",
    "ipc": "indian penal code IPC criminal law",
    "constitution": "constitution of india article",
    "fir": "first information report FIR police",
    "adr": "alternative dispute resolution arbitration mediation conciliation",
    "tort": "tort law torts wrongful act",
    "contract": "contract law agreement",
    "property": "property law immovable movable estate",
    "family law": "marriage divorce inheritance succession",
    "evidence": "indian evidence act testimony proof",
}

FAQ = {
    "what is fir": "FIR stands for First Information Report. It is a written document prepared by police when they receive information about a cognizable offence.",
    "fir": "FIR stands for First Information Report. It is a written document prepared by police when they receive information about a cognizable offence.",
    "what is ipc": "The Indian Penal Code (IPC) is the official criminal code of India. It covers all substantive aspects of criminal law.",
    "ipc": "The Indian Penal Code (IPC) is the official criminal code of India. It covers all substantive aspects of criminal law.",
    "what is posh act": "The POSH Act (Prevention of Sexual Harassment at Workplace) is an Indian law that protects women from sexual harassment at work.",
    "fundamental rights": "Fundamental Rights are enshrined in Part III of the Constitution (Articles 12-35). They include Right to Equality, Freedom, against Exploitation, Freedom of Religion, Cultural & Educational Rights, and Right to Constitutional Remedies.",
    "what is adr": "ADR (Alternative Dispute Resolution) refers to methods like arbitration, mediation, and conciliation used to resolve disputes without going to court.",
    "adr": "ADR (Alternative Dispute Resolution) refers to methods like arbitration, mediation, and conciliation used to resolve disputes without going to court.",
    "what is article 21": "Article 21 of the Constitution guarantees the right to life and personal liberty. It states: 'No person shall be deprived of his life or personal liberty except according to procedure established by law.'",
    "what are fundamental duties": "Fundamental Duties are listed in Article 51A. They include respecting the Constitution, national flag, anthem; upholding sovereignty; promoting harmony; protecting environment; and striving for excellence.",
}

def expand_query_with_synonyms(query: str) -> str:  
    expanded = query.lower()
    for term, expansion in LEGAL_TERM_EXPANSIONS.items():
        if term in expanded:
            expanded = expanded.replace(term, expansion)

    section_match = re.search(r"\bsection\s+([0-9]+[A-Z]?)\b", query, re.IGNORECASE)
    article_match = re.search(r"\barticle\s+([0-9]+[A-Z]?)\b", query, re.IGNORECASE)

    if section_match:
        num = section_match.group(1)
        expanded += f" section {num} law code"  

    if article_match:
        num = article_match.group(1)
        expanded += f" article {num} constitution fundamental rights"

    if "ipc" in query.lower():
        expanded += " indian penal code section" 
    if "cpc" in query.lower():
        expanded += " civil procedure code section"

    return expanded

class ChatRequest(BaseModel):
    message: str
    state: str | None = None
    district: str | None = None
    matter_type: str | None = None
    matter_stage: str | None = None
    is_own_matter: bool | None = None
    urgency: str | None = None

class ChatResponse(BaseModel):
    response: str
    intent: str | None = None
    confidence: float | None = None
    source: str | None = None
    state: str | None = None
    district: str | None = None
    matter_type: str | None = None
    matter_stage: str | None = None
    is_own_matter: bool | None = None
    urgency: str | None = None
    issue_category: str | None = None
    next_steps: list[str] = []
    suggested_questions: list[str] = []
    citations: list[str] = []
    source_snippets: list[str] = []
    forum_hint: str | None = None
    intake_missing_fields: list[str] = []
    disclaimer: str | None = None
    intake_complete: bool = False


class UserSignupRequest(BaseModel):
    full_name: str
    email: str
    password: str
    state: str | None = None


class UserLoginRequest(BaseModel):
    email: str
    password: str


class AuthUser(BaseModel):
    id: int
    full_name: str
    email: str
    state: str | None = None
    created_at: str


class AuthResponse(BaseModel):
    token: str
    user: AuthUser


class HistoryMessage(BaseModel):
    id: int
    role: str
    message: str
    source: str | None = None
    confidence: float | None = None
    metadata: dict = {}
    created_at: str


class ChatHistoryResponse(BaseModel):
    items: list[HistoryMessage]


class PasswordResetRequest(BaseModel):
    email: str


def normalize_query(text: str) -> str:
    text = text.strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def normalize_state_name(state: str | None) -> str | None:
    if not state:
        return None

    cleaned = " ".join(state.strip().split())
    if not cleaned:
        return None

    lowered = cleaned.lower()
    for known_state in INDIAN_STATES_AND_UTS:
        if known_state.lower() == lowered:
            return known_state
    return cleaned.title()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def build_frontend_auth_redirect(error: str | None = None, token: str | None = None) -> str:
    query_params: dict[str, str] = {}
    if error:
        query_params["error"] = error
    if token:
        query_params["token"] = token
    if not query_params:
        return FRONTEND_AUTH_URL
    return f"{FRONTEND_AUTH_URL}?{urlencode(query_params)}"


def google_auth_configured() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REDIRECT_URI)


def exchange_google_code_for_tokens(code: str) -> dict:
    response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def fetch_google_userinfo(access_token: str) -> dict:
    response = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    profile = response.json()
    if not profile.get("sub") or not profile.get("email"):
        raise ValueError("Google profile missing required fields")
    if profile.get("email_verified") is not True:
        raise ValueError("Google email is not verified")
    return profile


def get_or_create_google_auth_user(profile: dict) -> dict:
    google_sub = profile["sub"]
    email = normalize_email(profile["email"])
    full_name = (profile.get("name") or email.split("@")[0]).strip()

    existing_google_user = get_user_by_google_sub(google_sub)
    if existing_google_user:
        return existing_google_user

    existing_email_user = get_user_by_email(email)
    if existing_email_user:
        linked_user = link_google_account(existing_email_user["id"], google_sub)
        if linked_user:
            return linked_user
        raise ValueError("Failed to link Google account")

    return create_google_user(full_name=full_name, email=email, google_sub=google_sub, state=None)


def get_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization.strip()


def require_current_user(authorization: str | None) -> dict:
    token = get_bearer_token(authorization)
    user = get_user_by_token(token or "")
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def save_chat_exchange(user_id: int, user_message: str, response: ChatResponse) -> None:
    store_chat_message(user_id=user_id, role="user", message=user_message, source="user_input")
    store_chat_message(
        user_id=user_id,
        role="assistant",
        message=response.response,
        source=response.source,
        confidence=response.confidence,
        metadata={
            "intent": response.intent,
            "state": response.state,
            "district": response.district,
            "matter_type": response.matter_type,
            "matter_stage": response.matter_stage,
            "issue_category": response.issue_category,
            "citations": response.citations,
            "source_snippets": response.source_snippets,
            "forum_hint": response.forum_hint,
            "suggested_questions": response.suggested_questions,
            "next_steps": response.next_steps,
            "disclaimer": response.disclaimer,
        },
    )

def normalize_matter_type(matter_type: str | None) -> str | None:
    if not matter_type:
        return None
    cleaned = normalize_query(matter_type)
    return cleaned if cleaned in SUPPORTED_MATTER_TYPES else "unsure"

def normalize_urgency(urgency: str | None) -> str | None:
    if not urgency:
        return None
    cleaned = normalize_query(urgency)
    return cleaned if cleaned in SUPPORTED_URGENCY else "medium"

def normalize_district_name(district: str | None) -> str | None:
    if not district:
        return None
    cleaned = " ".join(district.strip().split())
    return cleaned.title() if cleaned else None

def normalize_matter_stage(matter_stage: str | None) -> str | None:
    if not matter_stage:
        return None
    cleaned = normalize_query(matter_stage)
    return cleaned if cleaned in SUPPORTED_MATTER_STAGES else "not_sure"


def detect_simple_query(query: str) -> bool:
    normalized = normalize_query(query)
    
    # Simple keywords that indicate basic questions
    simple_keywords = [
        "what is", "define", "meaning of", "explain", "traffic", "violation", 
        "fine", "penalty", "basic", "simple", "definition", "how to", "procedure for",
        "difference between", "types of", "list of", "examples of", "rules", "driving",
        "vehicle", "license", "signal", "speed", "parking", "accident", "minor",
        "small", "quick", "easy", "straightforward"
    ]
    
    # Check if query starts with or contains simple indicators
    for keyword in simple_keywords:
        if keyword in normalized:
            return True
    
    # Check if it's a short query (less than 10 words)
    words = normalized.split()
    if len(words) <= 10:
        # Additional check for common simple legal terms
        simple_terms = [
            "fir", "ipc", "constitution", "article", "section", "law", "act", 
            "court", "judge", "advocate", "lawyer", "police", "complaint",
            "traffic", "violation", "fine", "penalty", "rules", "driving", "vehicle"
        ]
        if any(term in normalized for term in simple_terms):
            return True
    
    return False

def infer_matter_type(message: str) -> str:
    normalized = normalize_query(message)
    keyword_groups = {
        "criminal": ["fir", "arrest", "bail", "police", "charge", "crime", "complaint"],
        "family": ["divorce", "maintenance", "custody", "marriage", "domestic violence"],
        "property": ["property", "partition", "rent", "tenant", "land", "registry", "sale deed"],
        "labour": ["salary", "termination", "wages", "employer", "employee", "gratuity", "pf"],
        "consumer": ["consumer", "refund", "deficiency", "product", "service", "invoice"],
        "cyber": ["cyber", "online fraud", "upi", "scam", "hacking", "otp"],
        "constitutional": ["article", "constitution", "fundamental right", "writ", "high court", "supreme court"],
        "civil": ["notice", "agreement", "contract", "injunction", "suit", "damages"],
    }

    for matter_type, keywords in keyword_groups.items():
        if any(keyword in normalized for keyword in keywords):
            return matter_type
    return "general"

def build_profile_summary(state: str | None, district: str | None, matter_type: str | None, matter_stage: str | None, is_own_matter: bool | None, urgency: str | None) -> str:
    own_matter_text = "their own matter" if is_own_matter else "someone else's matter"
    if is_own_matter is None:
        own_matter_text = "an unspecified matter"
    return (
        f"State: {state or 'not provided'}; "
        f"District: {district or 'not provided'}; "
        f"Matter type: {matter_type or 'not provided'}; "
        f"Matter stage: {matter_stage or 'not provided'}; "
        f"Context: {own_matter_text}; "
        f"Urgency: {urgency or 'not provided'}"
    )

def detect_issue_category(message: str, matter_type: str | None) -> str:
    normalized = normalize_query(message)
    keyword_map = {
        "arrest": "criminal procedure",
        "bail": "criminal procedure",
        "fir": "criminal procedure",
        "police": "criminal procedure",
        "divorce": "family law",
        "maintenance": "family law",
        "custody": "family law",
        "property": "property dispute",
        "partition": "property dispute",
        "rent": "property dispute",
        "salary": "labour dispute",
        "termination": "labour dispute",
        "consumer": "consumer dispute",
        "fraud": "cyber or criminal issue",
        "article": "constitutional issue",
        "constitution": "constitutional issue",
        "writ": "constitutional issue",
    }
    for keyword, category in keyword_map.items():
        if keyword in normalized:
            return category
    if matter_type and matter_type != "general":
        return f"{matter_type} matter"
    return "general legal issue"

def build_next_steps(issue_category: str, state: str | None, district: str | None, matter_type: str | None, matter_stage: str | None, is_own_matter: bool | None, urgency: str | None) -> list[str]:
    steps = [
        "Write down the facts in date order and keep copies of notices, contracts, IDs, chats, photos, and other records.",
        f"Check the authority, court, or police process that applies in {district + ', ' if district else ''}{state or 'your state'} before taking the next procedural step.",
    ]

    workflow_map = {
        "criminal": [
            "Preserve evidence immediately and avoid deleting chats, call logs, CCTV, bank statements, or device records.",
            "Check whether an FIR, notice under law, arrest risk, bail question, or charge-sheet stage is already involved.",
            "If police action or arrest risk is present, contact a criminal lawyer or legal aid without delay.",
        ],
        "civil": [
            "Check whether limitation, territorial jurisdiction, valuation, and a legal notice requirement apply before filing.",
            "Organize all agreements, emails, invoices, notices, and proof of breach or loss.",
            "Identify the exact relief needed, such as recovery, injunction, declaration, possession, or damages.",
        ],
        "family": [
            "Collect marriage, residence, income, child, medical, and communication records before seeking family-court relief.",
            "Identify whether the matter concerns divorce, maintenance, custody, domestic violence, residence, or succession.",
            "Check whether immediate protection, maintenance, or child-related interim relief is needed.",
        ],
        "property": [
            "Collect title, possession, mutation, tax, rent, registration, encumbrance, and revenue records before acting.",
            "Identify whether the issue concerns ownership, partition, eviction, tenancy, possession, boundary, or specific performance.",
            "Verify whether civil court, revenue authority, registrar records, or local municipal records are central to the dispute.",
        ],
        "constitutional": [
            "Identify whether the issue concerns state action, a public authority, or a fundamental-right violation.",
            "Check whether the right remedy is a representation, statutory appeal, tribunal proceeding, or writ in the High Court.",
        ],
    }

    for step in workflow_map.get(matter_type or "", []):
        steps.append(step)

    stage_workflow_map = {
        "pre_fir": [
            "Prepare a concise written complaint with dates, witnesses, and supporting records before approaching police or another authority.",
        ],
        "fir_registered": [
            "Collect the FIR number, sections invoked, police station details, and copies of all complaint papers.",
        ],
        "arrest": [
            "Record the exact time, place, and grounds of arrest and contact a lawyer or legal aid immediately.",
        ],
        "bail": [
            "Keep the FIR, remand papers, grounds for bail, and identity/address documents ready for the bail hearing.",
        ],
        "notice_received": [
            "Do not ignore the notice; check deadline, issuing authority, and whether a reply with documents is needed.",
        ],
        "pre_suit": [
            "Check limitation, legal notice requirements, and whether negotiation, mediation, or document collection should happen first.",
        ],
        "suit_filed": [
            "Track case number, next date, pleadings filed, interim applications, and service status carefully.",
        ],
        "trial": [
            "Organize witnesses, exhibits, affidavits, and the sequence of evidence before the next hearing.",
        ],
        "appeal": [
            "Obtain the impugned order/judgment, limitation calculation, certified copies, and grounds of challenge.",
        ],
        "execution": [
            "Keep the decree/order copy, compliance history, and asset/property details ready for execution proceedings.",
        ],
    }
    for step in stage_workflow_map.get(matter_stage or "", []):
        steps.append(step)

    if issue_category == "family law" and matter_type != "family":
        steps.append("Collect marriage, residence, income, and child-related documents before seeking family-court remedies.")
    if issue_category == "property dispute" and matter_type != "property":
        steps.append("Collect title, possession, mutation, tax, rent, and registration records before acting.")
    if urgency in {"high", "urgent"}:
        steps.append("Because the matter is urgent, do not rely only on chat guidance; speak to a local advocate or legal aid authority quickly.")
    if is_own_matter:
        steps.append("Since this is your own matter, verify deadlines, forum, and documents with a local practitioner before filing or replying.")

    return steps

def build_suggested_questions(issue_category: str, matter_type: str | None) -> list[str]:
    if matter_type == "criminal":
        return [
            "Has any FIR, notice, or arrest-related action already happened?",
            "Which police station or district is involved?",
            "What exact sections or allegations are being mentioned?",
        ]
    if matter_type == "civil":
        return [
            "What is the timeline of events and when did the dispute start?",
            "Do you have any contract, notice, or written communication?",
            "Which city or court jurisdiction is connected to the dispute?",
        ]
    if issue_category == "family law":
        return [
            "Is there any existing case, notice, or protection order already filed?",
            "Are maintenance, custody, residence, or domestic violence issues involved?",
            "Which state and district are relevant for the family court process?",
        ]
    if issue_category == "property dispute":
        return [
            "Who is in possession of the property right now?",
            "Do you have title papers, sale deed, mutation, or rent documents?",
            "Is the issue about ownership, partition, eviction, or possession?",
        ]
    return [
        "What happened first, and on what date?",
        "Which authority, court, or police office is already involved, if any?",
        "What documents or notices do you already have?",
    ]

def build_forum_hint(state: str | None, district: str | None, matter_type: str | None, matter_stage: str | None, issue_category: str) -> str:
    place_text = f"{district}, {state}" if district and state else state or "your state"
    if matter_type == "criminal":
        return f"For criminal procedure in {place_text}, the relevant forums may include the police station, Magistrate court, Sessions court, or High Court depending on whether the matter is at FIR, arrest, bail, trial, or appeal stage."
    if matter_type == "civil":
        return f"For civil disputes in {place_text}, forum choice usually depends on subject matter, valuation, territorial jurisdiction, and whether the matter is pre-suit, pending suit, trial, appeal, or execution."
    if issue_category == "family law":
        return f"For family issues in {place_text}, the relevant forum may be the Family Court, Magistrate court, or a protection officer depending on whether the issue concerns maintenance, custody, domestic violence, or interim relief."
    if issue_category == "property dispute":
        return f"For property issues in {place_text}, forum choice may involve civil court, revenue authority, registrar records, municipal records, or rent-control mechanisms depending on the dispute stage."
    if matter_type == "constitutional":
        return f"For constitutional remedies in {place_text}, the High Court is often the first forum for writ relief, depending on the facts and available statutory remedies."
    return f"The correct forum in {place_text} depends on the nature of the dispute, the relief sought, the stage of proceedings, and territorial jurisdiction."

def extract_citations(source: str, metadata_list: list[dict] | None) -> list[str]:
    citations: list[str] = []
    if not metadata_list:
        return citations

    for metadata in metadata_list[:3]:
        file_name = metadata.get("file_name") or metadata.get("source") or source
        act_name = metadata.get("act_name") or file_name
        article_number = metadata.get("article_number")
        section_number = metadata.get("section_number")
        title = metadata.get("title")
        court = metadata.get("court")
        case_name = metadata.get("case_name")
        citation_text = metadata.get("citation")
        year = metadata.get("year")

        if case_name:
            case_bits = [case_name]
            if citation_text:
                case_bits.append(citation_text)
            if court:
                case_bits.append(court)
            if year:
                case_bits.append(str(year))
            citations.append(" | ".join(case_bits))
        elif article_number:
            citations.append(f"{act_name} - Article {article_number}" + (f" ({title})" if title else ""))
        elif section_number:
            citations.append(f"{act_name} - Section {section_number}" + (f" ({title})" if title else ""))
        else:
            citations.append(str(act_name))

    return citations

def extract_source_snippets(metadata_list: list[dict] | None, chunks: list[str] | None) -> list[str]:
    snippets: list[str] = []
    if not chunks:
        return snippets

    for index, chunk in enumerate(chunks[:3]):
        metadata = metadata_list[index] if metadata_list and index < len(metadata_list) else {}
        act_name = metadata.get("act_name") or metadata.get("file_name") or "Source"
        label = act_name
        if metadata.get("article_number"):
            label = f"{act_name} Article {metadata['article_number']}"
        elif metadata.get("section_number"):
            label = f"{act_name} Section {metadata['section_number']}"
        snippet_text = " ".join(chunk.split())[:220]
        snippets.append(f"{label}: {snippet_text}")
    return snippets

def build_disclaimer(matter_type: str | None, is_own_matter: bool | None, state: str | None) -> str:
    base = "This is general legal information, not a substitute for representation by a qualified advocate."
    if is_own_matter:
        base += f" Because this concerns your own matter in {state or 'India'}, local facts, state rules, and deadlines can change the legal position."
    if matter_type == "criminal":
        base += " Criminal matters can become urgent quickly, especially where FIR, arrest, bail, or police procedure is involved."
    return base

def enrich_query_with_profile(
    message: str,
    state: str | None,
    district: str | None,
    matter_type: str | None,
    matter_stage: str | None,
    is_own_matter: bool | None,
    urgency: str | None,
) -> str:
    profile_summary = build_profile_summary(state, district, matter_type, matter_stage, is_own_matter, urgency)
    return f"{message}\n\nLegal matter profile: {profile_summary}"

def exact_faq_match(query: str):
    normalized = normalize_query(query)
    if normalized in FAQ:
        return FAQ[normalized], 1.0

    for key, value in FAQ.items():
        if key and (key in normalized or normalized in key):
            return value, 1.0

    return None, 0.0

def is_valid_answer(answer: str, query: str = None) -> bool:
    if not answer or not answer.strip():
        return False

    if query and re.search(r"\b(section|article)\b", query, re.IGNORECASE):
        return len(answer.strip()) >= 6

    return len(answer.strip()) > MIN_ANSWER_LENGTH

def is_weak_answer(answer: str, query: str = None) -> bool:
    if not answer:
        return True

    normalized_answer = normalize_query(answer)
    if query and re.search(r"\b(section|article)\b", query, re.IGNORECASE):
        if re.match(r"^(section|article)\s+[0-9]+", normalized_answer):
            return False

    if len(answer.strip()) < 8:
        return True

    weak_patterns = [
        "i dont have enough information",
        "not enough information provided",
        "cannot determine",
        "not clear from the context",
        "insufficient information",
        "no information",
    ]
    return any(pattern in normalized_answer for pattern in weak_patterns)

def quick_keyword_check(query: str, answer: str) -> bool:
    if not query or not answer:
        return False
    q_words = set(normalize_query(query).split())
    a_words = set(normalize_query(answer).split())
    return len(q_words & a_words) > 0

def validate_and_clean(query: str, answer: str) -> tuple[bool, str]:
    if not answer or not answer.strip():
        logger.info("Answer empty or blank")
        return False, ""

    answer_len = len(answer.strip())
    if query and re.search(r"\b(section|article)\b", query, re.IGNORECASE):
        min_length = 6
    else:
        min_length = MIN_ANSWER_LENGTH

    if answer_len < min_length:
        logger.info("Answer too short: %d chars (min=%d)", answer_len, min_length)
        return False, ""

    if not quick_keyword_check(query, answer):
        logger.info("Keyword check failed for query=%r", query)
        return False, ""

    return True, answer.strip()


def clean_context_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    weak_phrases = [
        "I don't have enough information in the provided context.",
        "I dont have enough information in the provided context.",
        "I don't have enough reliable legal information on that yet.",
        "not enough information provided",
        "insufficient information",
        "no information",
    ]
    for phrase in weak_phrases:
        cleaned = cleaned.replace(phrase, "").strip()
    return cleaned[:MAX_CONTEXT_CHARS_PER_CHUNK].strip()


def select_top_context_chunks(chunks: list[str] | None, limit: int = MAX_CONTEXT_CHUNKS) -> list[str]:
    if not chunks:
        return []

    selected: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        cleaned = clean_context_text(chunk)
        normalized = normalize_query(cleaned)
        if not cleaned or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(cleaned)
        if len(selected) >= limit:
            break
    return selected


def format_context_chunks(chunks: list[str] | None, citations: list[str] | None = None) -> str:
    top_chunks = select_top_context_chunks(chunks)
    context_parts: list[str] = []

    for index, chunk in enumerate(top_chunks, start=1):
        context_parts.append(f"Evidence {index}:\n{chunk}")

    if citations:
        cleaned_citations = [clean_context_text(item) for item in citations if clean_context_text(item)]
        if cleaned_citations:
            context_parts.append("Relevant legal references:\n" + "\n".join(cleaned_citations[:MAX_CONTEXT_CHUNKS]))

    return "\n\n".join(context_parts).strip()

def build_llm_context(
    query: str,
    context: str | None = None,
    state: str | None = None,
    district: str | None = None,
    matter_type: str | None = None,
    matter_stage: str | None = None,
    is_own_matter: bool | None = None,
    urgency: str | None = None,
) -> str:
    profile_summary = build_profile_summary(state, district, matter_type, matter_stage, is_own_matter, urgency)
    context_text = clean_context_text(context or "")
    return f"Question: {query}\n\n" \
           f"legal matter profile:{profile_summary}\n\n" \
           "write a natural, helpful answer in simple language using short paragraphs." \
           "if the context is incomplete, say what is reasonably clear and avoid making up facts.\n\n" \
           'reply in valid JSON only with this shape:{"answer":"..."}'

def extract_llm_answer(payload: dict) -> str:
    answer = payload.get("answer", "")
    if isinstance(answer, str):
        return answer.strip()
    return ""


def parse_llm_json(content: str) -> dict:
    cleaned = (content or "").strip()
    if not cleaned:
        return {"answer": ""}
    try:
        return requests.models.complexjson.loads(cleaned)
    except ValueError:
        logger.warning("LLM did not return valid JSON; using raw text fallback")
        return {"answer": cleaned}


def call_groq_llm(prompt: str) -> dict:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning("GROQ_API_KEY is not configured")
        return {"answer": ""}

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = parse_llm_json(content)
        return {"answer": extract_llm_answer(parsed)}
    except Exception as e:
        logger.error("Groq answer error: %s", e)
        return {"answer": ""}


# def call_openai_llm(query: str, context: str) -> dict:
#     openai_api_key = os.environ.get("OPENAI_API_KEY")
#     if not openai_api_key:
#         logger.warning("OPENAI_API_KEY is not configured")
#         return {"answer": ""}
#
#     payload = {
#         "model": OPENAI_MODEL,
#         "messages": [
#             {"role": "system", "content": LLM_SYSTEM_PROMPT},
#             {"role": "user", "content": build_llm_context(query, context)},
#         ],
#         "temperature": 0.2,
#         "max_tokens": 300,
#         "response_format": {"type": "json_object"},
#     }
#     headers = {
#         "Authorization": f"Bearer {openai_api_key}",
#         "Content-Type": "application/json",
#     }
#
#     try:
#         response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=20)
#         response.raise_for_status()
#         data = response.json()
#         content = data["choices"][0]["message"]["content"]
#         parsed = parse_llm_json(content)
#         return {"answer": extract_llm_answer(parsed)}
#     except Exception as e:
#         logger.error("OpenAI answer error: %s", e)
#         return {"answer": ""}


def generate_llm_answer(query: str, context: str) -> dict:
    return call_groq_llm(f"Question: {query}\n\nContext: {context}")


def build_answer_generation_context(
    rag_chunks: list[str] | None = None,
    citations: list[str] | None = None,
    source_snippets: list[str] | None = None,
    faq_answer: str | None = None,
) -> str:
    context_parts: list[str] = []

    legal_context = format_context_chunks(rag_chunks, citations)
    if legal_context:
        context_parts.append(legal_context)

    if source_snippets:
        cleaned_snippets = [clean_context_text(item) for item in source_snippets if clean_context_text(item)]
        if cleaned_snippets:
            context_parts.append("Supporting excerpts:\n" + "\n\n".join(cleaned_snippets[:MAX_CONTEXT_CHUNKS]))

    if faq_answer:
        cleaned_faq = clean_context_text(faq_answer)
        if cleaned_faq:
            context_parts.append(f"Known legal reference:\n{cleaned_faq}")

    return "\n\n".join(context_parts).strip()

def finalize_response(
    query: str,
    answer: str,
    state: str | None = None,
    district: str | None = None,
    matter_type: str | None = None,
    matter_stage: str | None = None,
    is_own_matter: bool | None = None,
    urgency: str | None = None,
) -> str:
    if not answer or not answer.strip():
        return answer
    return answer.strip()

def build_chat_response(
    *,
    message: str,
    response_text: str,
    source: str,
    confidence: float,
    intent: str | None,
    state: str | None,
    district: str | None,
    matter_type: str | None,
    matter_stage: str | None,
    is_own_matter: bool | None,
    urgency: str | None,
    citations: list[str] | None = None,
    source_snippets: list[str] | None = None,
    intake_missing_fields: list[str] | None = None,
) -> ChatResponse:
    return ChatResponse(
        response=finalize_response(message, response_text, state, district, matter_type, matter_stage, is_own_matter, urgency),
        intent=intent,
        confidence=confidence,
        source=source,
        state=state,
        district=district,
        matter_type=matter_type,
        matter_stage=matter_stage,
        is_own_matter=is_own_matter,
        urgency=urgency,
        issue_category=None,
        next_steps=[],
        suggested_questions=[],
        citations=citations or [],
        source_snippets=source_snippets or [],
        forum_hint=None,
        intake_missing_fields=intake_missing_fields or [],
        disclaimer=None,
        intake_complete=bool(state and matter_type and is_own_matter is not None),
    )

def pick_rag_response(
    query: str,
    state: str | None = None,
    district: str | None = None,
    matter_type: str | None = None,
    matter_stage: str | None = None,
    is_own_matter: bool | None = None,
    urgency: str | None = None,
) -> tuple[str, float, str, list[str], list[str]]:
    best_answer = ""
    best_conf = 0.0
    best_source = ""
    best_citations: list[str] = []
    best_source_snippets: list[str] = []
    
    expanded_query = expand_query_with_synonyms(
        enrich_query_with_profile(query, state, district, matter_type, matter_stage, is_own_matter, urgency)
    )
    logger.info("Original query: %r, Expanded: %r", query, expanded_query)

    state_result = state_rag_answer(expanded_query, state=state) or {}
    state_answer = state_result.get("answer", "")
    state_conf = state_result.get("confidence", 0.0)
    state_chunks = state_result.get("chunks", [])
    state_matches = state_result.get("matches", [])
    if state_conf >= RAG_THRESHOLD and is_valid_answer(state_answer, query) and not is_weak_answer(state_answer, query):
        valid, cleaned = validate_and_clean(query, state_answer)
        if valid:
            best_answer = cleaned
            best_conf = state_conf
            best_source = "state_rag"
            best_citations = extract_citations(best_source, state_matches)
            best_source_snippets = extract_source_snippets(state_matches, state_chunks)
            logger.info("State RAG accepted: conf=%.3f", state_conf)

    case_result = case_rag_answer(expanded_query, state=state) or {}
    case_answer = case_result.get("answer", "")
    case_conf = case_result.get("confidence", 0.0)
    case_chunks = case_result.get("chunks", [])
    case_matches = case_result.get("matches", [])
    if case_conf >= RAG_THRESHOLD and is_valid_answer(case_answer, query) and not is_weak_answer(case_answer, query):
        valid, cleaned = validate_and_clean(query, case_answer)
        if valid and case_conf > best_conf:
            best_answer = cleaned
            best_conf = case_conf
            best_source = "case_rag"
            best_citations = extract_citations(best_source, case_matches)
            best_source_snippets = extract_source_snippets(case_matches, case_chunks)
            logger.info("Case-law RAG accepted: conf=%.3f", case_conf)

    txt_result = txt_rag_answer(expanded_query, state=state) or {}
    txt_answer = txt_result.get("answer", "")
    txt_conf = txt_result.get("confidence", 0.0)
    txt_chunks = txt_result.get("chunks", [])
    txt_matches = txt_result.get("matches", [])
    logger.info("TXT RAG candidate: conf=%.3f chunks=%s answer_len=%d", txt_conf, len(txt_chunks), len(txt_answer))
    
    if txt_conf >= RAG_THRESHOLD and is_valid_answer(txt_answer, query) and not is_weak_answer(txt_answer, query):
        valid, cleaned = validate_and_clean(query, txt_answer)
        if valid and txt_conf > best_conf:
            best_answer = cleaned
            best_conf = txt_conf
            best_source = "txt_rag"
            best_citations = extract_citations(best_source, txt_matches)
            best_source_snippets = extract_source_snippets(txt_matches, txt_chunks)
            logger.info("✓ TXT RAG ACCEPTED: conf=%.3f source=%s", txt_conf, best_source)
        else:
            logger.info("✗ TXT RAG rejected after validation")
    else:
        logger.info("✗ TXT RAG rejected: conf=%.3f < threshold=%.3f, valid=%s, weak=%s", 
                   txt_conf, RAG_THRESHOLD, is_valid_answer(txt_answer), is_weak_answer(txt_answer))

    pdf_result = pdf_rag_answer(expanded_query, state=state) or {}
    pdf_answer = pdf_result.get("answer", "")
    pdf_conf = pdf_result.get("confidence", 0.0)
    pdf_chunks = pdf_result.get("chunks", [])
    pdf_matches = pdf_result.get("matches", [])
    logger.info("PDF RAG candidate: conf=%.3f chunks=%s answer_len=%d", pdf_conf, len(pdf_chunks), len(pdf_answer))
    
    if pdf_conf >= RAG_THRESHOLD and is_valid_answer(pdf_answer, query) and not is_weak_answer(pdf_answer, query):
        valid, cleaned = validate_and_clean(query, pdf_answer)
        if valid and pdf_conf > best_conf:
            best_answer = cleaned
            best_conf = pdf_conf
            best_source = "pdf_rag"
            best_citations = extract_citations(best_source, pdf_matches)
            best_source_snippets = extract_source_snippets(pdf_matches, pdf_chunks)
            logger.info("✓ PDF RAG ACCEPTED: conf=%.3f source=%s", pdf_conf, best_source)
        else:
            logger.info("✗ PDF RAG rejected after validation or lower conf")
    else:
        logger.info("✗ PDF RAG rejected: conf=%.3f < threshold=%.3f, valid=%s, weak=%s",
                   pdf_conf, RAG_THRESHOLD, is_valid_answer(pdf_answer), is_weak_answer(pdf_answer))

    return best_answer, best_conf, best_source, best_citations, best_source_snippets


def pick_rag_context(
    query: str,
    state: str | None = None,
    district: str | None = None,
    matter_type: str | None = None,
    matter_stage: str | None = None,
    is_own_matter: bool | None = None,
    urgency: str | None = None,
) -> tuple[str, float, str, list[str], list[str], list[str]]:
    expanded_query = expand_query_with_synonyms(
        enrich_query_with_profile(query, state, district, matter_type, matter_stage, is_own_matter, urgency)
    )
    logger.info("RAG query prepared query_hash=%s expanded_hash=%s", hash_text(query), hash_text(expanded_query))

    candidates = [
        ("state_rag", state_rag_answer(expanded_query, state=state) or {}),
        ("case_rag", case_rag_answer(expanded_query, state=state) or {}),
        ("txt_rag", txt_rag_answer(expanded_query, state=state) or {}),
        ("pdf_rag", pdf_rag_answer(expanded_query, state=state) or {}),
    ]

    best_answer = ""
    best_conf = 0.0
    best_source = ""
    best_citations: list[str] = []
    best_source_snippets: list[str] = []
    best_chunks: list[str] = []

    for source_name, result in candidates:
        answer = result.get("answer", "")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        chunks = result.get("chunks", []) or []
        matches = result.get("matches", []) or []
        selected_chunks = select_top_context_chunks(chunks)
        if not selected_chunks:
            continue

        cleaned_answer = ""
        if is_valid_answer(answer, query) and not is_weak_answer(answer, query):
            valid, normalized_answer = validate_and_clean(query, answer)
            if valid:
                cleaned_answer = normalized_answer

        if confidence > best_conf or (confidence == best_conf and len(selected_chunks) > len(best_chunks)):
            best_answer = cleaned_answer
            best_conf = confidence
            best_source = source_name
            best_citations = extract_citations(source_name, matches)
            best_source_snippets = extract_source_snippets(matches, selected_chunks)
            best_chunks = selected_chunks
            logger.info("RAG context selected source=%s conf=%.3f chunks=%d", source_name, confidence, len(selected_chunks))

    return best_answer, best_conf, best_source, best_citations, best_source_snippets, best_chunks

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/frontend/auth.html")


@app.get("/auth/google/login")
def google_login():
    if not google_auth_configured():
        return RedirectResponse(url=build_frontend_auth_redirect(error="google_auth_not_configured"))

    state = secrets.token_urlsafe(24)
    GOOGLE_OAUTH_STATE_CACHE[state] = True
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "state": state,
        "access_type": "offline",
        "prompt": "select_account",
    }
    return RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{urlencode(params)}")

@app.get("/auth/google/status")
def google_auth_status():
    return {"configured": google_auth_configured()}


@app.get("/auth/google/callback")
def google_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        logger.warning("Google auth returned error=%s", error)
        return RedirectResponse(url=build_frontend_auth_redirect(error="google_auth_failed"))
    if not code or not state:
        return RedirectResponse(url=build_frontend_auth_redirect(error="google_auth_incomplete"))
    if state not in GOOGLE_OAUTH_STATE_CACHE:
        return RedirectResponse(url=build_frontend_auth_redirect(error="google_auth_state_invalid"))

    GOOGLE_OAUTH_STATE_CACHE.pop(state, None)

    try:
        token_payload = exchange_google_code_for_tokens(code)
        access_token = token_payload.get("access_token", "")
        if not access_token:
            raise ValueError("Missing Google access token")
        google_profile = fetch_google_userinfo(access_token)
        user = get_or_create_google_auth_user(google_profile)
        app_token = create_session(user["id"])
        return RedirectResponse(url=build_frontend_auth_redirect(token=app_token))
    except requests_exceptions.Timeout:
        logger.error("Google auth timeout during callback")
        return RedirectResponse(url=build_frontend_auth_redirect(error="google_auth_timeout"))
    except requests_exceptions.RequestException as e:
        logger.error("Google auth request failure type=%s", type(e).__name__)
        return RedirectResponse(url=build_frontend_auth_redirect(error="google_auth_request_failed"))
    except Exception as e:
        logger.error("Google auth callback failed type=%s", type(e).__name__)
        return RedirectResponse(url=build_frontend_auth_redirect(error="google_auth_failed"))


@app.post("/auth/signup", response_model=AuthResponse)
def signup(request: UserSignupRequest):
    full_name = request.full_name.strip()
    email = normalize_email(request.email)
    password = request.password.strip()
    state = normalize_state_name(request.state)

    if len(full_name) < 2:
        raise HTTPException(status_code=400, detail="Full name must be at least 2 characters")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = create_user(full_name=full_name, email=email, password=password, state=state)
    token = create_session(user["id"])
    return AuthResponse(token=token, user=AuthUser(**user))


@app.post("/signup", include_in_schema=False, response_model=AuthResponse)
def signup_alias(request: UserSignupRequest):
    return signup(request)


@app.post("/signup/", include_in_schema=False, response_model=AuthResponse)
def signup_alias_slash(request: UserSignupRequest):
    return signup(request)


@app.options("/auth/signup", include_in_schema=False)
def signup_options():
    # CORS preflight for POST /auth/signup
    return {"detail": "OPTIONS OK"}


@app.options("/signup", include_in_schema=False)
def signup_alias_options():
    return {"detail": "OPTIONS OK"}


@app.options("/signup/", include_in_schema=False)
def signup_alias_slash_options():
    return {"detail": "OPTIONS OK"}


@app.get("/auth/signup", include_in_schema=False)
def signup_get_hint():
    raise HTTPException(status_code=405, detail="Method GET not allowed; use POST /auth/signup")


@app.get("/signup", include_in_schema=False)
def signup_get_hint_alias():
    raise HTTPException(status_code=405, detail="Method GET not allowed; use POST /auth/signup")


@app.get("/signup/", include_in_schema=False)
def signup_get_hint_alias_slash():
    raise HTTPException(status_code=405, detail="Method GET not allowed; use POST /auth/signup")


@app.post("/auth/login", response_model=AuthResponse)
def login(request: UserLoginRequest):
    user = authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_session(user["id"])
    return AuthResponse(token=token, user=AuthUser(**user))


@app.get("/auth/me", response_model=AuthUser)
def me(authorization: str | None = Header(default=None)):
    user = require_current_user(authorization)
    return AuthUser(**user)


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    token = get_bearer_token(authorization)
    if token:
        delete_session(token)
    return {"status": "ok"}


@app.post("/auth/password-reset")
def password_reset(request: PasswordResetRequest):
    email = normalize_email(request.email)
    user = get_user_by_email(email)
    if not user:
        # Don't reveal if email exists or not for security
        return {"message": "If an account with this email exists, a password reset link has been sent."}

    # In a real application, you would:
    # 1. Generate a secure reset token
    # 2. Store it in the database with expiration
    # 3. Send an email with the reset link
    # For now, we'll just log it
    logger.info("Password reset requested for email: %s", email)

    return {"message": "If an account with this email exists, a password reset link has been sent."}


@app.get("/chat/history", response_model=ChatHistoryResponse)
def chat_history(authorization: str | None = Header(default=None)):
    user = require_current_user(authorization)
    items = [HistoryMessage(**item) for item in list_chat_messages(user["id"])]
    return ChatHistoryResponse(items=items)

@app.get("/debug/status")
def debug_status():
    try:
        store_summary = {}
        store_configs = [
            ("txt_vectorstore", Path("db/doc_db"), "RAG_Bot.txt.txt_rag", "get_vectorstore"),
            ("pdf_vectorstore", Path("db/pdf_db"), "RAG_Bot.doc.pdfRP", "get_vectorstore"),
            ("state_vectorstore", Path("db/state_db"), "RAG_Bot.state.state_rag", "get_vectorstore"),
            ("case_vectorstore", Path("db/case_db"), "RAG_Bot.case.case_rag", "get_vectorstore"),
        ]

        for name, path, module_name, getter_name in store_configs:
            exists = path.exists()
            count = 0
            if exists:
                try:
                    module = __import__(module_name, fromlist=[getter_name])
                    getter = getattr(module, getter_name)
                    count = getter()._collection.count()
                except Exception as e:
                    logger.error("Error accessing %s: %s", name, e)
                    count = -1

            store_summary[name] = {
                "path": str(path),
                "exists": exists,
                "document_count": count,
            }

        return {
            "status": "OK" if any(item["exists"] for item in store_summary.values()) else "ERROR",
            **store_summary,
            "message": "Vector stores initialized" if any(item["exists"] for item in store_summary.values()) else "Vector stores NOT found - run ingestion"
        }
    except Exception as e:
        logger.error(f"Debug status error: {e}")
        return {"status": "ERROR", "error": str(e)}

@app.get("/debug/ingest")
def debug_ingest():
    
    try:
        from backend.RAG_Bot.txt.Txt_Ingestion_Pipeline import main as ingest_documents
        from backend.RAG_Bot.state.State_Ingestion_Pipeline import main as ingest_state_documents
        from backend.RAG_Bot.case.Case_Ingestion_Pipeline import main as ingest_case_documents
        logger.info("Starting manual document ingestion...")
        ingest_documents()
        ingest_state_documents()
        ingest_case_documents()
        return {
            "status": "SUCCESS",
            "message": "Document ingestion completed successfully"
        }
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return {
            "status": "ERROR",
            "error": str(e),
            "message": "Failed to ingest documents"
        }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, authorization: str | None = Header(default=None)):
    user = require_current_user(authorization)
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")
    state = normalize_state_name(request.state) or normalize_state_name(user.get("state"))
    district = normalize_district_name(request.district)
    matter_type = normalize_matter_type(request.matter_type)
    matter_stage = normalize_matter_stage(request.matter_stage) or "not_sure"
    urgency = normalize_urgency(request.urgency)
    is_own_matter = request.is_own_matter
    if not matter_type or matter_type == "unsure":
        matter_type = infer_matter_type(message)
    logger.info(
        "Matter profile: state=%r district=%r matter_type=%r matter_stage=%r is_own_matter=%r urgency=%r",
        state,
        district,
        matter_type,
        matter_stage,
        is_own_matter,
        urgency,
    )
    
    logger.info("=" * 80)
    logger.info("📨 CHAT REQUEST: %r", message)

    try:
        if not state or is_own_matter is None:
            missing_fields = []
            if not state:
                missing_fields.append("your state")
            if is_own_matter is None:
                missing_fields.append("whether this is your own matter")

            response_payload = ChatResponse(
                response=(
                    "Before I guide you properly, please share "
                    + ", ".join(missing_fields)
                    + ". This helps me tailor the legal information to the correct Indian state and type of case."
                ),
                intent="intake_required",
                confidence=1.0,
                source="intake",
                state=state,
                district=district,
                matter_type=matter_type,
                matter_stage=matter_stage,
                is_own_matter=is_own_matter,
                urgency=urgency,
                issue_category=detect_issue_category(message, matter_type),
                next_steps=[
                    "Select your state.",
                    "Confirm whether the matter is your own.",
                    "Add your district and current case stage if you want more precise forum guidance.",
                ],
                intake_missing_fields=missing_fields,
                disclaimer=build_disclaimer(matter_type, is_own_matter, state),
                intake_complete=False,
            )
            save_chat_exchange(user["id"], message, response_payload)
            return response_payload

        faq_answer, faq_conf = exact_faq_match(message)
        if faq_answer is not None:
            logger.info("✓ FAQ MATCH - source=faq confidence=1.0")
            response_payload = build_chat_response(
                message=message,
                response_text=faq_answer,
                intent="faq",
                confidence=float(faq_conf),
                source="faq",
                state=state,
                district=district,
                matter_type=matter_type,
                matter_stage=matter_stage,
                is_own_matter=is_own_matter,
                urgency=urgency,
            )
            save_chat_exchange(user["id"], message, response_payload)
            return response_payload

        try:
            bot = get_intent_bot()
            intents = bot.predict_class(message)
            if intents:
                tag = intents[0]["intent"]
                intent_conf = intents[0]["probability"]
            else:
                tag = "unknown"
                intent_conf = 0.0
        except Exception as e:
            logger.error("Intent bot error: %s", e)
            tag = "unknown"
            intent_conf = 0.0
        
        logger.info("Intent Bot: tag=%s confidence=%.3f", tag, intent_conf)

        if intent_conf >= INTENT_THRESHOLD:
            try:
                bot = get_intent_bot()
                intent_answer = bot.get_response(message)
                logger.info(" INTENT BOT SELECTED - source=intent_bot confidence=%.3f", intent_conf)
                response_payload = build_chat_response(
                    message=message,
                    response_text=intent_answer,
                    intent=tag,
                    confidence=float(intent_conf),
                    source="intent_bot",
                    state=state,
                    district=district,
                    matter_type=matter_type,
                    matter_stage=matter_stage,
                    is_own_matter=is_own_matter,
                    urgency=urgency,
                )
                save_chat_exchange(user["id"], message, response_payload)
                return response_payload
            except Exception as e:
                logger.error("Intent bot response error: %s", e)

        try:
            rag_answer, rag_conf, rag_source, rag_citations, rag_source_snippets = pick_rag_response(
                message,
                state,
                district,
                matter_type,
                matter_stage,
                is_own_matter,
                urgency,
            )
            if rag_answer and rag_conf >= RAG_THRESHOLD:
                logger.info(" RAG SELECTED - source=%s confidence=%.3f", rag_source, rag_conf)
                response_payload = build_chat_response(
                    message=message,
                    response_text=rag_answer,
                    intent=tag,
                    confidence=float(rag_conf),
                    source=rag_source,
                    state=state,
                    district=district,
                    matter_type=matter_type,
                    matter_stage=matter_stage,
                    is_own_matter=is_own_matter,
                    urgency=urgency,
                    citations=rag_citations,
                    source_snippets=rag_source_snippets,
                )
                save_chat_exchange(user["id"], message, response_payload)
                return response_payload
        except Exception as e:
            logger.error("RAG error: %s", e)

        try:
            # Get recent conversation context (last 5 messages)
            recent_messages = list_chat_messages(user["id"])[-10:]  # Get last 10, but we'll take recent 5 exchanges
            conversation_context = ""
            if recent_messages:
                # Take last 5 exchanges (user + assistant pairs)
                recent_exchanges = recent_messages[-5:]
                context_parts = []
                for msg in recent_exchanges:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    context_parts.append(f"{role}: {msg['message']}")
                conversation_context = "\n".join(context_parts)
            
            llm_context = build_answer_generation_context(
                rag_chunks=[rag_answer] if rag_answer else None,
                citations=rag_citations if 'rag_citations' in locals() else [],
                source_snippets=rag_source_snippets if 'rag_source_snippets' in locals() else [],
            )
            # Include conversation context in the prompt
            full_context = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {message}\n\nLegal context: {llm_context}"
            llm_response = call_groq_llm(full_context)
            llm_answer = extract_llm_answer(llm_response)
            if llm_answer and len(llm_answer.strip()) > MIN_ANSWER_LENGTH:
                logger.info("LLM FALLBACK - source=groq confidence=0.6")
                response_payload = build_chat_response(
                    message=message,
                    response_text=llm_answer,
                    intent=tag,
                    confidence=0.6,
                    source="groq",
                    state=state,
                    district=district,
                    matter_type=matter_type,
                    matter_stage=matter_stage,
                    is_own_matter=is_own_matter,
                    urgency=urgency,
                )
                save_chat_exchange(user["id"], message, response_payload)
                return response_payload
        except Exception as e:
            logger.error("LLM fallback error: %s", e)

        highest_conf = 0.0
        if 'rag_conf' in locals() and rag_conf > intent_conf:
            highest_conf = rag_conf
        elif intent_conf >= 0.7:
            highest_conf = intent_conf

        logger.info("✗ NO MATCH - Returning fallback response")
        logger.info("=" * 80)
        response_payload = build_chat_response(
            message=message,
            response_text="I don't have enough reliable legal information on that yet. Try asking with dates, state, documents, and the exact legal issue so I can guide you better.",
            intent=tag,
            confidence=float(highest_conf),
            source="fallback",
            state=state,
            district=district,
            matter_type=matter_type,
            matter_stage=matter_stage,
            is_own_matter=is_own_matter,
            urgency=urgency,
        )
        save_chat_exchange(user["id"], message, response_payload)
        return response_payload
        
    except Exception as e:
        logger.error("Unexpected error in chat endpoint: %s", e)
        response_payload = build_chat_response(
            message=message,
            response_text="Sorry, I encountered an error processing your request. Please try again.",
            intent="error",
            confidence=0.0,
            source="error",
            state=state,
            district=district,
            matter_type=matter_type,
            matter_stage=matter_stage,
            is_own_matter=is_own_matter,
            urgency=urgency,
        )
        save_chat_exchange(user["id"], message, response_payload)
        return response_payload

