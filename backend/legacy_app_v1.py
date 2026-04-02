from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os

import secrets
import sqlite3
import string
import logging
import hashlib

import json
import importlib
import requests

from cachetools import TTLCache
from requests import exceptions as requests_exceptions
from dotenv import load_dotenv
# from openai import OpenAI

import re
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from backend.legal_research import (
    DOMAIN_REFERENCE_LIBRARY,
    build_research_brief,
    citation_from_metadata,
    normalize_legal_citation,
)
from backend.paths import CONFIG_DIR, FRONTEND_DIR, ROOT_DIR, UPLOADS_DIR, VECTOR_DB_DIR

app = FastAPI(title="Integrated Legal Chatbot", version="1.0")
load_dotenv(CONFIG_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chatbot.app")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MULTIPART_AVAILABLE = importlib.util.find_spec("multipart") is not None

from .database import (
    cleanup_expired_password_reset_tokens,
    cleanup_expired_sessions,
    close_active_matter_intake,
    consume_password_reset_token,
    create_session,
    create_password_reset_token,
    create_user,
    create_google_user,
    check_db_health,
    delete_session,
    get_active_matter_intake,
    get_user_by_email,
    get_user_by_google_sub,
    get_user_by_token,
    init_db,
    link_google_account,
    list_chat_sessions,
    get_chat_session,
    create_chat_session,
    clear_chat_messages,
    clear_all_chat_history,
    list_chat_messages,
    list_uploaded_documents,
    store_uploaded_document,
    store_chat_message,
    update_chat_session_title,
    upsert_active_matter_intake,
    authenticate_user,
)
init_db()

CORS_ALLOW_ORIGINS = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000,http://127.0.0.1:8000",
)

def parse_cors_origins(value: str) -> list[str]:
    return [origin.strip() for origin in value.split(",") if origin.strip()]

origins = parse_cors_origins(CORS_ALLOW_ORIGINS)
allow_credentials = "*" not in origins

# TODO: lock this down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["http://127.0.0.1:8000"],
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
cleanup_expired_sessions()
cleanup_expired_password_reset_tokens()

intent_bot = None
_txt_rag_module = None
_pdf_rag_module = None
_case_rag_module = None
_state_rag_module = None
_legalparam_model = None
_legalparam_tokenizer = None

def get_txt_rag_module():
    global _txt_rag_module

    if _txt_rag_module is None:
        logger.info("Loading TXT RAG module...")
        _txt_rag_module = importlib.import_module("backend.RAG_Bot.txt.txt_rag")
    return _txt_rag_module

def get_pdf_rag_module():
    global _pdf_rag_module

    if _pdf_rag_module is None:
        logger.info("Loading PDF RAG module...")
        _pdf_rag_module = importlib.import_module("backend.RAG_Bot.doc.pdfRP")
    return _pdf_rag_module

def get_case_rag_module():
    global _case_rag_module

    if _case_rag_module is None:
        logger.info("Loading case-law RAG module...")
        _case_rag_module = importlib.import_module("backend.RAG_Bot.case.case_rag")
    return _case_rag_module

def get_state_rag_module():
    global _state_rag_module

    if _state_rag_module is None:
        logger.info("Loading state RAG module...")
        _state_rag_module = importlib.import_module("backend.RAG_Bot.state.state_rag")
    return _state_rag_module

def get_intent_bot():
    global intent_bot

    if intent_bot is None:
        logger.info("Initializing Intent Bot...")
        intent_bot_module = importlib.import_module("backend.Intent_Bot.Intent_bot")
        intent_bot = intent_bot_module.IntentBot()
        logger.info("Intent Bot initialized")
    return intent_bot

INTENT_THRESHOLD = 0.7
RAG_THRESHOLD = 0.7
MIN_ANSWER_LENGTH = 20  
MAX_CONTEXT_CHUNKS = 5
MAX_CONTEXT_CHARS_PER_CHUNK = 900
LLM_SYSTEM_PROMPT = """
You are a professional Indian legal advocate assisting a client.

RULES:
1. First give a clear, practical answer based on available information.
2. Do NOT ask unnecessary questions.
3. Ask follow-up questions ONLY if absolutely required.
4. Avoid repeating questions already answered.
5. Speak like a real lawyer guiding a client (natural tone, not robotic).
6. Keep responses short and actionable.

If user reports fraud:
- Immediately guide steps (bank, cyber complaint, etc.)
- Then ask 1–2 relevant follow-up questions.

Tone requirements:
- Be strictly formal, respectful, and professional.
- Do not use casual words, slang, emojis, or friendly phrases such as "Hi", "Hey", "buddy", or "no problem".
- Do not sound robotic, generic, or theatrical.

Answer structure:
1. Give a brief acknowledgement of the user's issue.
2. State the legal understanding in one clear line.
3. State one practical next step.
4. End with one precise clarification question.

Output rules:
- Keep the full response within 4 to 5 short lines.
- Use simple, clear language.
- Avoid long explanations, bullet lists, and unnecessary background.
- Write in short paragraphs or single-sentence lines.
- Do not mention prompts, retrieval, chunks, metadata, or internal logic.
- Base the answer primarily on the provided legal context and profile.
- If the context is limited, answer cautiously and professionally without inventing facts.
""".strip()

# GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
# OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

LEGALPARAM_MODEL_NAME = "bharatgenai/LegalParam"
LLM_REQUEST_TIMEOUT_SECONDS = 20
LLM_CACHE = TTLCache(maxsize=256, ttl=900)

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()
# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

GOOGLE_OAUTH_STATE_CACHE = TTLCache(maxsize=128, ttl=600)
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/auth/google/callback").strip()
FRONTEND_AUTH_URL = os.environ.get("FRONTEND_AUTH_URL", "http://127.0.0.1:8000/frontend/auth.html").strip()
FRONTEND_APP_URL = os.environ.get("FRONTEND_APP_URL", "http://127.0.0.1:8000/frontend/Index.html").strip()
EXPOSE_RESET_TOKEN_IN_RESPONSE = os.environ.get("EXPOSE_RESET_TOKEN_IN_RESPONSE", "true").strip().lower() == "true"
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
MATTER_TYPE_TO_LEGAL_DOMAINS = {
    "constitutional": {"constitutional", "human_rights"},
    "criminal": {"criminal"},
    "civil": {"civil", "contract", "tort", "evidence"},
    "family": {"family"},
    "property": {"property", "civil"},
    "labour": {"labour"},
    "consumer": {"consumer", "civil"},
    "cyber": {"cyber", "criminal"},
    "general": set(),
    "unsure": set(),
}
SOURCE_PRIORITY_BY_MATTER = {
    "constitutional": {"txt_rag": 3, "case_rag": 3, "state_rag": 1, "pdf_rag": 2},
    "criminal": {"state_rag": 3, "txt_rag": 2, "case_rag": 2, "pdf_rag": 1},
    "civil": {"txt_rag": 3, "case_rag": 2, "state_rag": 2, "pdf_rag": 1},
    "family": {"txt_rag": 3, "state_rag": 2, "case_rag": 1, "pdf_rag": 1},
    "property": {"state_rag": 3, "txt_rag": 2, "case_rag": 2, "pdf_rag": 1},
    "labour": {"txt_rag": 3, "state_rag": 2, "case_rag": 1, "pdf_rag": 1},
    "consumer": {"txt_rag": 3, "case_rag": 2, "state_rag": 1, "pdf_rag": 1},
    "cyber": {"txt_rag": 3, "case_rag": 2, "state_rag": 1, "pdf_rag": 1},
}
DOMAIN_WORKFLOWS = {
    "criminal": {
        "intake_questions": [
            "Has an FIR, NCR, police complaint, notice, arrest, or bail issue already arisen?",
            "Which police station, district, or court is presently involved?",
            "What exact sections, allegations, or incident details are being mentioned?",
        ],
        
        "next_steps": [
            "Prepare a clear chronology of the incident and preserve CCTV, chats, call records, and documents immediately.",
            "Identify whether the matter is pre-complaint, FIR, arrest, bail, investigation, trial, or appeal stage.",
            "If arrest risk, custody, or coercive police action is involved, contact a criminal lawyer or legal aid authority without delay.",
        ],
        
        "forum_hint": "The relevant forums may include the police station, Magistrate court, Sessions court, Juvenile Justice Board, or High Court depending on the procedural stage.",
    },
    
    "family": {
        "intake_questions": [
            "Does the matter concern divorce, maintenance, custody, domestic violence, residence, or succession?",
            "Is any case, notice, protection order, mediation, or counselling process already pending?",
            "Which district is connected to the matrimonial home, current residence, or child custody issue?",
        ],
    
        "next_steps": [
            "Collect marriage, residence, income, child, medical, and communication records before taking the next step.",
            "Identify whether urgent relief is required for maintenance, custody, residence, protection, or visitation.",
            "Check whether the issue belongs before the Family Court, Magistrate, protection officer, or mediation forum.",
        ],
    
        "forum_hint": "The relevant forum may include the Family Court, Magistrate court, protection officer, mediation centre, or civil court depending on the relief sought.",
    },
    
    "property": {
        "intake_questions": [
            "Is the dispute about title, possession, partition, tenancy, eviction, mutation, boundary, or registration?",
            "Who is presently in possession of the property?",
            "Do you have title papers, sale deed, mutation entry, rent documents, or revenue records?",
        ],
    
        "next_steps": [
            "Collect title, possession, mutation, tax, rent, registration, encumbrance, and revenue records.",
            "Identify whether the issue concerns ownership, possession, tenancy, partition, eviction, or boundary dispute.",
            "Verify whether civil court, revenue authority, registrar records, municipal authority, or rent forum is central to the dispute.",
        ],
    
        "forum_hint": "The relevant forum may include the civil court, revenue authority, sub-registrar records process, municipal authority, or rent-control forum depending on the dispute type.",
    },
    
    "consumer": {
        "intake_questions": [
            "Does the matter concern refund, defective goods, deficiency in service, overcharging, insurance claim, or unfair trade practice?",
            "Do you have invoice, warranty, receipt, service record, email, or complaint correspondence?",
            "Has any written complaint or legal notice already been sent to the seller or service provider?",
        ],
    
        "next_steps": [
            "Organize the invoice, payment proof, complaint emails, warranty documents, and screenshots of the deficiency.",
            "Check whether a written complaint or legal notice should be sent first with a reasonable response period.",
            "Assess whether the dispute belongs before the District Consumer Commission or another regulatory forum.",
        ],
    
        "forum_hint": "The relevant forum may include the District Consumer Disputes Redressal Commission, State Commission, regulator, ombudsman, or civil forum depending on the service and claim value.",
    },
    
    "labour": {
        "intake_questions": [
            "Does the matter concern salary, termination, resignation, gratuity, PF, harassment, service conditions, or unlawful deduction?",
            "What is your role and whether you are still in service, suspended, resigned, or terminated?",
            "Do you have appointment letter, payslips, termination letter, policy documents, or HR correspondence?",
        ],
    
        "next_steps": [
            "Collect appointment letter, salary records, termination or warning letters, attendance records, and HR communication.",
            "Identify whether the issue concerns wages, termination, misconduct, gratuity, PF, sexual harassment, or service benefits.",
            "Check whether the matter belongs before the labour authority, labour court, industrial tribunal, PF office, gratuity authority, or internal complaints committee.",
        ],
    
        "forum_hint": "The relevant forum may include the labour commissioner, labour court, industrial tribunal, gratuity authority, EPFO process, or internal complaints committee depending on the issue.",
    },
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
    chat_id: int | None = None


class ChatSessionCreate(BaseModel):
    title: str | None = None

class ChatResponse(BaseModel):
    response: str
    chat_id: int | None = None
    intent: str | None = None
    confidence: float | None = None
    source: str | None = None
    intake_step: str | None = None
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
    response_language: str | None = None
    lawyer_workflow: dict = {}
    research_brief: dict = {}
    uploaded_documents: list[dict] = []

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

class ChatSession(BaseModel):
    id: int
    user_id: int
    title: str | None = None
    created_at: str
    last_activity: str
    last_user_message: str | None = None

class ChatSessionsResponse(BaseModel):
    items: list[ChatSession]

class PasswordResetRequest(BaseModel):
    email: str

class PasswordResetConfirmRequest(BaseModel):
    token: str
    new_password: str

class MessageResponse(BaseModel):
    message: str
    reset_token: str | None = None
    reset_url: str | None = None

class UploadedDocumentResponse(BaseModel):
    id: int
    original_name: str
    content_type: str | None = None
    file_size: int | None = None
    document_kind: str | None = None
    description: str | None = None
    created_at: str

class UploadedDocumentListResponse(BaseModel):
    items: list[UploadedDocumentResponse]

ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".doc", ".docx"}

def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (filename or "").strip())
    return cleaned[:120] or "document"

def infer_document_kind(filename: str, extracted_text: str | None = None) -> str:
    haystack = f"{filename or ''}\n{extracted_text or ''}".lower()
    kind_patterns = {
        "fir": ["fir", "first information report", "crime no", "police station"],
        "legal_notice": ["legal notice", "notice under", "you are hereby called upon", "cease and desist"],
        "court_order": ["order", "judgment", "decree", "passed by the court", "interim order"],
        "agreement": ["agreement", "terms and conditions", "party of the first part", "party of the second part"],
        "salary_record": ["salary", "payslip", "pay slip", "ctc", "earnings", "deductions"],
        "property_record": ["sale deed", "property", "mutation", "khata", "patta", "encumbrance", "tenancy"],
        "complaint": ["complaint", "representation", "grievance", "petition"],
        "identity_or_bank_record": ["account", "bank", "statement", "ifsc", "upi", "transaction id"],
    }
    for kind, patterns in kind_patterns.items():
        if any(pattern in haystack for pattern in patterns):
            return kind
    return "legal_document"

def extract_upload_text(filename: str, content_type: str | None, content: bytes) -> str | None:
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":

        try:
            return content.decode("utf-8", errors="ignore")[:4000]

        except Exception:
            return None

    if content_type and content_type.startswith("text/"):

        try:
            return content.decode("utf-8", errors="ignore")[:4000]

        except Exception:
            return None
    return None

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

def build_frontend_password_reset_url(reset_token: str) -> str:
    return f"{FRONTEND_AUTH_URL}?{urlencode({'mode': 'reset', 'reset_token': reset_token})}"

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

def generate_chat_title(message: str) -> str:
    cleaned = re.sub(r"\s+", " ", (message or "").strip())
    if not cleaned:
        return "New Chat"
    cleaned = cleaned.rstrip("?.!,;: ")
    words = cleaned.split()
    short_title = " ".join(words[:7]).strip()
    if len(words) > 7:
        short_title += "..."
    return short_title[:80] or "New Chat"

def maybe_autotitle_chat_session(chat_id: int | None, user_message: str) -> None:
    if not chat_id:
        return
    session = get_chat_session(chat_id)
    if not session:
        return
    current_title = (session.get("title") or "").strip().lower()
    if current_title not in {"", "new chat", "untitled chat"}:
        return
    update_chat_session_title(chat_id, generate_chat_title(user_message))

def save_chat_exchange(user_id: int, user_message: str, response: ChatResponse, chat_id: int | None = None) -> None:
    store_chat_message(user_id=user_id, role="user", message=user_message, chat_id=chat_id, source="user_input")
    maybe_autotitle_chat_session(chat_id, user_message)

    store_chat_message(
        user_id=user_id,
        chat_id=chat_id,
        role="assistant",
        message=response.response,
        source=response.source,
        confidence=response.confidence,

        metadata={
            "intent": response.intent,
            "intake_step": response.intake_step,
            "state": response.state,
            "district": response.district,
            "matter_type": response.matter_type,
            "matter_stage": response.matter_stage,
            "is_own_matter": response.is_own_matter,
            "urgency": response.urgency,
            "issue_category": response.issue_category,
            "citations": response.citations,
            "source_snippets": response.source_snippets,
            "forum_hint": response.forum_hint,
            "suggested_questions": response.suggested_questions,
            "next_steps": response.next_steps,
            "intake_missing_fields": response.intake_missing_fields,
            "disclaimer": response.disclaimer,
            "response_language": response.response_language,
            "lawyer_workflow": response.lawyer_workflow,
            "research_brief": response.research_brief,
            "uploaded_documents": response.uploaded_documents,
        },
    )

def get_or_create_active_chat_session(user_id: int) -> dict:
    sessions = list_chat_sessions(user_id)

    if sessions:
        session = get_chat_session(sessions[0]["id"])

        if session:
            return session
    return create_chat_session(user_id)

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

def detect_response_language(message: str) -> str:
    text = message or ""

    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    normalized = normalize_query(text)
    hindi_markers = [
        "kya", "kaise", "mera", "meri", "mujhe", "kyu", "kab", "kaun", "hai", "nahi", "fraud hua",
        "madad", "kanooni", "shikayat", "police", "court", "notice mila",
    ]

    if any(marker in normalized for marker in hindi_markers):
        return "hi"
    return "en"

def validate_citations(citations: list[str] | None) -> list[str]:

    if not citations:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()

    for item in citations:
        normalized = normalize_legal_citation(item)

        if not normalized:
            continue
        lower = normalized.lower()

        if lower in seen:
            continue
        seen.add(lower)
        cleaned.append(normalized[:220])
    return cleaned[:5]

def get_default_domain_citations(matter_type: str | None) -> list[str]:
    profile = DOMAIN_REFERENCE_LIBRARY.get((matter_type or "").lower(), {})
    defaults = profile.get("authorities")
    if defaults:
        return defaults
    return ["Applicable statute, rules, and case law require verification from the retrieved record"]

def detect_urgency_flags(message: str, matter_type: str | None, matter_stage: str | None, details: dict | None = None) -> list[str]:
    normalized = normalize_query(message)
    details_text = normalize_query(json.dumps(details or {}, ensure_ascii=False))
    combined = f"{normalized} {details_text}".strip()

    flag_patterns = {
        "arrest_risk": ["arrest", "detained", "custody", "police picked", "picked up by police"],
        "domestic_violence_risk": ["domestic violence", "beating", "assault at home", "threat by husband", "threat by spouse"],
        "eviction_risk": ["eviction", "vacate", "locked out", "forcibly removed", "tenant removed"],
        "account_fraud_risk": ["upi", "otp", "bank account", "debited", "unauthorized transaction", "wallet fraud"],
        "limitation_risk": ["deadline", "limitation", "last date", "time barred", "expiry date"],
        "child_custody_risk": ["child custody", "child taken", "visitation denied", "minor child"],
    }

    flags = [flag for flag, patterns in flag_patterns.items() if any(pattern in combined for pattern in patterns)]
    if matter_type == "criminal" and matter_stage in {"arrest", "bail"} and "arrest_risk" not in flags:
        flags.append("arrest_risk")
    return flags

def determine_effective_urgency(message: str, urgency: str | None, matter_type: str | None, matter_stage: str | None, details: dict | None = None) -> str:
    flags = detect_urgency_flags(message, matter_type, matter_stage, details)

    if flags and urgency not in {"high", "urgent"}:
        return "urgent"
    return urgency or "medium"

def infer_stage_from_message(message: str) -> str | None:
    normalized = normalize_query(message)

    keyword_map = {
        "pre_fir": ["before fir", "before complaint", "not filed yet", "not complained yet"],
        "fir_registered": ["fir registered", "fir filed", "complaint registered", "police complaint filed"],
        "arrest": ["arrest", "detained", "custody"],
        "bail": ["bail"],
        "notice_received": ["notice", "legal notice", "received notice"],
        "pre_suit": ["before filing", "not filed", "pre suit", "pre-suit"],
        "suit_filed": ["suit filed", "case filed", "petition filed"],
        "trial": ["trial", "evidence stage", "cross examination", "hearing going on"],
        "appeal": ["appeal", "appealed", "challenging order"],
        "execution": ["execution", "decree execution", "order execution"],
    }

    for stage, keywords in keyword_map.items():
        if any(keyword in normalized for keyword in keywords):
            return stage
    return None

def summarize_issue_from_message(message: str) -> str:
    cleaned = " ".join((message or "").split()).strip()
    return cleaned[:500]

def normalize_yes_no_answer(message: str) -> bool | None:
    normalized = normalize_query(message)

    positive_answers = {
        "yes", "yes it is", "yes this is my matter", "yes this is mine", "mine", "my matter", "own matter",
    }

    negative_answers = {
        "no", "no it is not", "not my matter", "someone else", "someone elses", "not mine",
    }

    if normalized in positive_answers:
        return True

    if normalized in negative_answers:
        return False
    return None

def apply_conversational_intake_updates(
    message: str,
    recent_messages: list[dict],
    state: str | None,
    district: str | None,
    matter_type: str | None,
    matter_stage: str | None,
    is_own_matter: bool | None,
) -> tuple[str | None, str | None, str | None, str | None, bool | None]:

    if not recent_messages:
        return state, district, matter_type, matter_stage, is_own_matter

    assistant_messages = [item for item in recent_messages if item.get("role") == "assistant"]

    if not assistant_messages:
        return state, district, matter_type, matter_stage, is_own_matter

    last_assistant = assistant_messages[-1]
    assistant_text = normalize_query(last_assistant.get("message", ""))
    metadata = last_assistant.get("metadata") or {}
    intake_missing_fields = [normalize_query(item) for item in (metadata.get("intake_missing_fields") or [])]

    parsed_yes_no = normalize_yes_no_answer(message)

    if is_own_matter is None and parsed_yes_no is not None:
        if "your own matter" in assistant_text or any("own matter" in field for field in intake_missing_fields):
            is_own_matter = parsed_yes_no

    if not state:
        parsed_state = normalize_state_name(message)
        if parsed_state and ("share your state" in assistant_text or any("state" in field for field in intake_missing_fields)):
            state = parsed_state

    if (not matter_type or matter_type in {"general", "unsure"}) and last_assistant.get("source") == "clarification":
        inferred_type = infer_matter_type(message)
        if inferred_type not in {"general", "unsure"}:
            matter_type = inferred_type

    return state, district, matter_type, matter_stage, is_own_matter

def merge_profile_from_recent_history(
    recent_messages: list[dict],
    state: str | None,
    district: str | None,
    matter_type: str | None,
    matter_stage: str | None,
    is_own_matter: bool | None,
    urgency: str | None,
) -> tuple[str | None, str | None, str | None, str | None, bool | None, str | None]:

    if not recent_messages:
        return state, district, matter_type, matter_stage, is_own_matter, urgency

    for item in reversed(recent_messages):
        if item.get("role") != "assistant":
            continue
        metadata = item.get("metadata") or {}

        if not state and metadata.get("state"):
            state = normalize_state_name(metadata.get("state"))

        if not district and metadata.get("district"):
            district = normalize_district_name(metadata.get("district"))

        if (not matter_type or matter_type in {"general", "unsure"}) and metadata.get("matter_type"):
            inferred_matter_type = normalize_matter_type(metadata.get("matter_type"))
            if inferred_matter_type and inferred_matter_type != "unsure":
                matter_type = inferred_matter_type

        if (not matter_stage or matter_stage == "not_sure") and metadata.get("matter_stage"):
            inferred_stage = normalize_matter_stage(metadata.get("matter_stage"))
            if inferred_stage:
                matter_stage = inferred_stage

        if is_own_matter is None and isinstance(metadata.get("is_own_matter"), bool):
            is_own_matter = metadata.get("is_own_matter")

        if not urgency and metadata.get("urgency"):
            urgency = normalize_urgency(metadata.get("urgency"))

    return state, district, matter_type, matter_stage, is_own_matter, urgency

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
    words = [word for word in normalized.split() if word]

    scenario_indicators = [
        "i ", "my ", "me ", "we ", "our ", "us ",
        "received", "got", "happened", "issue", "problem", "dispute",
        "notice", "summons", "arrest", "termination",
        "landlord", "tenant", "employer", "wife", "husband", "police",
        "filed", "pending", "case", "agreement", "rent",
        "deposit", "refund", "salary", "property", "custody", "divorce",
        "because", "against", "after", "before", "since", "when",
    ]

    if any(indicator in normalized for indicator in scenario_indicators):
        return False

    if re.search(r"\b\d{1,4}\b", normalized):
        return False

    definition_starters = (
        "what is ",
        "what are ",
        "define ",
        "meaning of ",
        "explain ",
        "difference between ",
        "types of ",
        "list of ",
    )

    if not normalized.startswith(definition_starters):
        return False

    if len(words) > 8:
        return False

    simple_terms = [
        "fir", "ipc", "constitution", "article", "section", "law", "act",
        "court", "judge", "advocate", "lawyer", "writ", "bail", "complaint",
        "consumer", "tenant", "lease", "mutation", "gratuity", "pf", "challan",
    ]

    return any(term in normalized for term in simple_terms)

def looks_like_real_scenario(message: str) -> bool:
    normalized = normalize_query(message)
    words = [word for word in normalized.split() if word]

    if len(words) < 5:
        return False

    incident_terms = [
        "i", "my", "me", "we", "our",
        "received", "got", "happened", "happening", "called", "messaged",
        "paid", "transferred", "debited", "credited", "blocked", "threatened",
        "scammed", "cheated", "fraud", "notice", "transaction", "account", "order",
        "wallet", "upi", "bank", "email", "whatsapp", "telegram", "instagram", "loan app",
        "otp", "link", "call", "message", "profile", "post", "upload",
        "website", "loan app", "refund", "order", "screenshot", "recording",
    ]

    incident_matches = sum(1 for term in incident_terms if term in normalized)

    cyber_keywords = [
        "upi", "otp", "bank", "wallet", "account", "telegram", "whatsapp",
        "instagram", "facebook", "email", "link", "website", "app", "cyber",
        "hack", "hacking", "unauthorized", "transaction", "online",
    ]
    has_cyber_keyword = any(term in normalized for term in cyber_keywords)

    has_time_detail = bool(re.search(r"\b(today|yesterday|last|ago|date|time|am|pm)\b", normalized))
    has_number_detail = bool(re.search(r"\b\d{1,6}\b", normalized))

    if has_cyber_keyword and incident_matches >= 2:
        return True

    return incident_matches >= 3 or (incident_matches >= 2 and (has_time_detail or has_number_detail))


def has_concrete_issue_details(message: str) -> bool:
    normalized = normalize_query(message)

    if not normalized:
        return False

    if looks_like_real_scenario(message):
        return True

    actor_terms = [
        "landlord", "tenant", "employer", "employee", "husband", "wife", "bank",
        "police", "company", "seller", "builder", "buyer", "neighbour", "neighbor",
        "brother", "sister", "partner", "ex employer", "loan app", "agent",
    ]

    issue_terms = [
        "fraud", "scam", "debited", "termination", "dismissed", "salary", "rent",
        "deposit", "notice", "threat", "harassment", "custody", "maintenance",
        "assault", "cheating", "theft", "possession", "eviction", "refund",
        "defective", "breach", "violence", "blackmail", "hacked", "impersonation",
        "unauthorized", "chargeback", "upi", "otp", "account", "fir", "summons",
    ]

    action_terms = [
        "received", "sent", "filed", "taken", "debited", "refused", "stopped",
        "withheld", "locked", "blocked", "shared", "signed", "transferred",
        "cancelled", "terminated", "evicted", "threatened", "posted", "used",
        "deducted", "called", "messaged", "registered",
    ]

    document_terms = [
        "notice", "agreement", "order", "message", "email", "screenshot",
        "recording", "receipt", "invoice", "contract", "letter", "fir",
    ]

    remedy_terms = [
        "what can i do", "what should i do", "legal action", "options", "remedy",
        "file a complaint", "how can i recover", "how do i respond", "what is the next step",
    ]

    actor_matches = sum(1 for term in actor_terms if term in normalized)
    issue_matches = sum(1 for term in issue_terms if term in normalized)
    action_matches = sum(1 for term in action_terms if term in normalized)
    has_document = any(term in normalized for term in document_terms)
    has_remedy_request = any(term in normalized for term in remedy_terms)
    has_number_detail = bool(re.search(r"\b\d{1,8}\b", normalized))

    has_time_detail = bool(
        re.search(
            r"\b(today|yesterday|tomorrow|last|ago|morning|evening|night|week|month|year|am|pm)\b",
            normalized,
        )
    )

    return (
        issue_matches >= 2
        or (issue_matches >= 1 and action_matches >= 1)
        or (actor_matches >= 1 and issue_matches >= 1)
        or (issue_matches >= 1 and has_document)
        or (issue_matches >= 1 and (has_number_detail or has_time_detail))
        or (issue_matches >= 1 and has_remedy_request)
    )

def is_category_already_clear(message: str, matter_type: str | None = None) -> bool:
    normalized = normalize_query(message)

    explicit_subtypes = {
        "criminal": ["fir stage", "arrest stage", "bail stage", "police complaint", "charge sheet", "chargesheet"],
        "family": ["domestic violence", "child custody", "mutual divorce", "maintenance case", "succession dispute"],
        "property": ["tenant eviction", "title dispute", "partition suit", "rent deposit", "property registration"],
        "consumer": ["defective goods", "deficient service", "unfair trade practice", "refund dispute", "overcharging complaint"],
        "labour": ["wrongful termination", "salary non payment", "gratuity claim", "pf withdrawal", "workplace harassment"],
        "cyber": ["online fraud", "upi fraud", "otp fraud", "unauthorized transaction", "account hacking", "identity theft", "impersonation", "abusive content"],
        "constitutional": ["article 14", "article 19", "article 21", "writ petition", "fundamental right"],
    }

    if matter_type in explicit_subtypes:
        return any(term in normalized for term in explicit_subtypes[matter_type])
    return False

def should_skip_category_clarification(
    message: str,
    matter_type: str | None,
    recent_messages: list[dict] | None = None,
) -> bool:

    if has_concrete_issue_details(message) or is_category_already_clear(message, matter_type):
        return True

    if not recent_messages:
        return False

    assistant_messages = [item for item in recent_messages if item.get("role") == "assistant"]

    if not assistant_messages:
        return False

    last_assistant = assistant_messages[-1]
    if last_assistant.get("source") != "clarification":
        return False

    metadata = last_assistant.get("metadata") or {}
    last_issue_category = normalize_query(metadata.get("issue_category", ""))
    current_issue_category = normalize_query(detect_issue_category(message, matter_type))

    if current_issue_category and current_issue_category == last_issue_category and has_concrete_issue_details(message):
        return True

    return False

def is_query_vague(
    message: str,
    state: str | None = None,
    matter_stage: str | None = None,
    matter_type: str | None = None,
) -> bool:

    normalized = normalize_query(message)
    words = [word for word in normalized.split() if word]

    if has_concrete_issue_details(message) or is_category_already_clear(message, matter_type):
        return False

    if len(words) < 7:
        return True

    generic_queries = {
        "echallan",
        "e challan",
        "challan",
        "case problem",
        "legal help",
        "legal issue",
        "help me",
        "court case",
        "problem",
    }

    if normalized in generic_queries:
        return True

    generic_patterns = [
        "i have a case",
        "i have case problem",
        "i need legal help",
        "i need help",
        "there is a problem",
        "i got challan",
        "i got e challan",
        "case issue",
    ]

    if any(pattern in normalized for pattern in generic_patterns):
        return True

    has_location = any(state_name.lower() in normalized for state_name in INDIAN_STATES_AND_UTS)

    if not has_location:
        has_location = any(term in normalized for term in ["city", "district", "state", "police station", "court"])

    stage_keywords = [
        "notice", "fir", "charge sheet", "chargesheet", "bail", "arrest", "summons",
        "trial", "appeal", "execution", "registered", "filed", "hearing",
    ]

    violation_keywords = [
        "speed", "signal", "parking", "accident", "fraud", "refund", "termination",
        "divorce", "maintenance", "custody", "rent", "property", "assault",
        "cheating", "theft", "breach", "injury", "harassment", "upi", "otp",
        "bank", "account", "debited", "hacked", "unauthorized", "impersonation",
    ]

    has_stage = any(keyword in normalized for keyword in stage_keywords)
    has_violation_type = any(keyword in normalized for keyword in violation_keywords)

    if not state and not has_location:
        return True

    if matter_stage in {None, "not_sure"} and not has_stage and not has_violation_type:
        return True

    if matter_type in {None, "general", "unsure"} and not has_violation_type:
        return True

    if not has_stage and not has_violation_type and not has_location:
        return True

    return False

def build_clarification_question(
    message: str,
    state: str | None = None,
    matter_stage: str | None = None,
    matter_type: str | None = None,
) -> str:

    normalized = normalize_query(message)
    has_issue_details = has_concrete_issue_details(message)

    if "challan" in normalized or "traffic" in normalized:
        return (
            "You have referred to a traffic enforcement issue.\n"
            "To guide you accurately, please confirm the city or state where the challan was issued."
        )

    if not has_issue_details and (matter_type == "criminal" or any(word in normalized for word in ["fir", "police", "arrest", "bail"])):
        return (
            "Your query appears to concern a criminal law issue.\n"
            "Please confirm whether the matter is at the complaint stage, FIR stage, arrest stage, or bail stage."
        )

    if not has_issue_details and (matter_type == "property" or any(word in normalized for word in ["property", "land", "tenant", "rent"])):
        return (
            "Your query appears to concern a property-related dispute.\n"
            "Please specify whether the issue concerns ownership, possession, tenancy, registration, or partition."
        )

    if not has_issue_details and (matter_type == "family" or any(word in normalized for word in ["divorce", "maintenance", "custody", "marriage"])):
        return (
            "Your query appears to concern a family law issue.\n"
            "Please specify whether the matter relates to divorce, maintenance, custody, domestic violence, or succession."
        )

    if not has_issue_details and (matter_type == "labour" or any(word in normalized for word in ["salary", "termination", "wages", "employer", "employee"])):
        return (
            "Your query appears to concern a labour or employment issue.\n"
            "Please specify whether the matter concerns salary, termination, gratuity, PF, harassment, or service conditions."
        )

    if not has_issue_details and (matter_type == "consumer" or any(word in normalized for word in ["consumer", "refund", "deficiency", "product", "service"])):
        return (
            "Your query appears to concern a consumer dispute.\n"
            "Please specify whether the issue concerns refund, defective goods, deficient service, overcharging, or unfair trade practice."
        )

    if not has_issue_details and (matter_type == "cyber" or any(word in normalized for word in ["cyber", "fraud", "upi", "otp", "hacking", "online"])):
        return (
            "Your query appears to concern a cyber-related issue.\n"
            "Please specify whether the matter concerns online fraud, unauthorized transaction, hacking, impersonation, or abusive content."
        )

    if not has_issue_details and (matter_type == "constitutional" or any(word in normalized for word in ["article", "constitution", "writ", "fundamental right"])):
        return (
            "Your query appears to concern a constitutional issue.\n"
            "Please specify the public authority action or fundamental right that you believe has been affected."
        )

    if not state:
        return (
            "Your issue requires further factual context before legal guidance can be given.\n"
            "Please confirm the State or Union Territory in which this matter has arisen."
        )

    if matter_stage in {None, "not_sure"}:
        return (
            "Your issue requires one procedural detail before advice can be refined.\n"
            "Please state whether the matter is at the notice stage, filing stage, hearing stage, or appeal stage."
        )

    return (
        "Your issue requires one further factual detail before legal guidance can be given.\n"
        "Please state the exact legal problem or violation involved."
    )

def choose_missing_detail_question(
    *,
    state: str | None,
    is_own_matter: bool | None,
    matter_stage: str | None,
    matter_type: str | None,
    district: str | None,
) -> str | None:

    if not state:
        return "Please confirm the State or Union Territory where this matter arose."

    if is_own_matter is None:
        return "Please confirm whether this matter concerns you personally or someone else."

    if matter_type in {"property", "family", "consumer", "labour", "criminal"} and not district:
        return "Please confirm the district connected with this matter so the forum guidance can be narrowed."

    if matter_stage in {None, "not_sure"} and matter_type in {"criminal", "civil", "property", "family"}:
        return "Please confirm the present stage, such as notice, complaint, filing, hearing, or appeal."
    return None

def should_require_intake_before_answer(
    message: str,
    *,
    state: str | None,
    is_own_matter: bool | None,
) -> bool:

    if state and is_own_matter is not None:
        return False
    return not has_concrete_issue_details(message)

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

MATTER_COLLECTION_STEPS = {
    "is_own_matter": "Please confirm whether this matter concerns you personally or someone else.",
    "state": "Please state the State or Union Territory where this matter arose.",
    "district": "Please state the district or city connected with this matter.",
    "matter_type": "Please state the broad area of law, such as cyber, criminal, property, family, labour, or consumer.",
    "matter_stage": "Please state the present stage, such as notice, complaint, FIR, filing, hearing, trial, or appeal.",
    "relief_goal": "Please state the main outcome you want, such as refund, complaint, FIR, account freeze, legal notice, bail, or court relief.",
}

DOMAIN_COLLECTION_STEPS = {

    "cyber": [
        ("is_own_matter", "Please confirm whether this cyber incident concerns you personally or someone else."),
        ("state", "Please state the State or Union Territory where the cyber incident occurred or where you are filing the complaint."),
        ("district", "Please state the district or city linked to the bank account, police complaint, or cyber cell jurisdiction."),
        ("cyber_issue_type", "Please specify whether this concerns UPI fraud, card or bank fraud, account hacking, impersonation, abusive content, or another cyber issue."),
        ("financial_loss_amount", "Please state the approximate amount lost or placed at risk, if any."),
        ("institution_name", "Please state the bank, wallet, platform, or app involved."),
        ("matter_stage", "Please state whether you are before complaint, after complaint, after FIR, or already before a court or cyber cell."),
        ("relief_goal", "Please state the main result you want, such as account freeze, money recovery, FIR, chargeback, or takedown of content."),
    ],

    "family": [
        ("is_own_matter", "Please confirm whether this family matter concerns you personally or someone else."),
        ("state", "Please state the State or Union Territory connected to the matrimonial home or current family dispute."),
        ("district", "Please state the district connected with the marriage, residence, child, or court jurisdiction."),
        ("family_issue_type", "Please specify whether this concerns divorce, maintenance, custody, domestic violence, residence, or succession."),
        ("relationship_context", "Please briefly state the relationship involved, such as spouse, child, parents, or in-laws."),
        ("matter_stage", "Please state whether this is before notice, after notice, mediation, pending case, or appeal."),
        ("relief_goal", "Please state the main relief you want, such as divorce, maintenance, custody, residence protection, or visitation."),
    ],

    "property": [
        ("is_own_matter", "Please confirm whether this property matter concerns you personally or someone else."),
        ("state", "Please state the State or Union Territory where the property is located."),
        ("district", "Please state the district, city, or taluka where the property is situated."),
        ("property_issue_type", "Please specify whether this concerns title, possession, partition, tenancy, eviction, boundary, mutation, or registration."),
        ("possession_status", "Please state who is presently in possession of the property."),
        ("property_documents", "Please state whether you have sale deed, title papers, mutation records, rent documents, or other property records."),
        ("matter_stage", "Please state whether this is before notice, after notice, before filing, pending suit, trial, or appeal."),
        ("relief_goal", "Please state the main relief you want, such as possession, injunction, eviction, partition, mutation, or declaration."),
    ],

    "labour": [
        ("is_own_matter", "Please confirm whether this employment matter concerns you personally or someone else."),
        ("state", "Please state the State or Union Territory connected with the workplace."),
        ("district", "Please state the district or city where the employer or workplace is located."),
        ("labour_issue_type", "Please specify whether this concerns salary, termination, resignation, gratuity, PF, workplace harassment, or service conditions."),
        ("employment_status", "Please state whether you are still employed, suspended, resigned, terminated, or serving notice period."),
        ("employment_documents", "Please state whether you have appointment letter, payslips, termination letter, HR emails, or policy records."),
        ("matter_stage", "Please state whether this is before complaint, internal grievance, labour authority complaint, pending labour case, or appeal."),
        ("relief_goal", "Please state the main relief you want, such as unpaid salary, reinstatement, dues recovery, gratuity, PF, or formal complaint."),
    ],
}

def merge_matter_profile(
    state: str | None,
    district: str | None,
    matter_type: str | None,
    matter_stage: str | None,
    is_own_matter: bool | None,
    urgency: str | None,
    matter_record: dict | None,
) -> tuple[str | None, str | None, str | None, str | None, bool | None, str | None]:

    if not matter_record:
        return state, district, matter_type, matter_stage, is_own_matter, urgency

    state = state or normalize_state_name(matter_record.get("state"))
    district = district or normalize_district_name(matter_record.get("district"))

    if not matter_type or matter_type in {"general", "unsure"}:
        matter_type = normalize_matter_type(matter_record.get("matter_type")) or matter_type
    matter_stage = normalize_matter_stage(matter_stage) or normalize_matter_stage(matter_record.get("matter_stage")) or "not_sure"

    if is_own_matter is None:
        is_own_matter = matter_record.get("is_own_matter")
    urgency = urgency or normalize_urgency(matter_record.get("urgency"))
    return state, district, matter_type, matter_stage, is_own_matter, urgency

def normalize_detail_value(message: str) -> str | None:
    cleaned = " ".join((message or "").split()).strip()
    return cleaned[:300] if cleaned else None

def infer_domain_detail_defaults(message: str, matter_type: str | None, details: dict) -> dict:
    normalized = normalize_query(message)
    updated = dict(details)

    if matter_type == "cyber":
        if not updated.get("cyber_issue_type"):

            if "upi" in normalized:
                updated["cyber_issue_type"] = "upi fraud"

            elif "otp" in normalized or "bank" in normalized or "account" in normalized:
                updated["cyber_issue_type"] = "bank or unauthorized transaction fraud"

            elif "hack" in normalized:
                updated["cyber_issue_type"] = "account hacking"

        if not updated.get("financial_loss_amount"):
            amount_match = re.search(r"(?:rs\.?|inr)?\s*([0-9][0-9,]{2,})", message, re.IGNORECASE)

            if amount_match:
                updated["financial_loss_amount"] = amount_match.group(1).replace(",", "")

        if not updated.get("institution_name"):
            institutions = ["sbi", "hdfc", "icici", "axis", "paytm", "phonepe", "gpay", "google pay", "whatsapp", "telegram", "instagram", "facebook"]

            for item in institutions:
                if item in normalized:
                    updated["institution_name"] = item
                    break
                
    if matter_type == "family":
        if not updated.get("family_issue_type"):
            for phrase in ["divorce", "maintenance", "custody", "domestic violence", "succession", "residence"]:
                
                if phrase in normalized:
                    updated["family_issue_type"] = phrase
                    break

    if matter_type == "property":
        if not updated.get("property_issue_type"):
            for phrase in ["title", "possession", "partition", "tenancy", "eviction", "boundary", "mutation", "registration"]:
                
                if phrase in normalized:
                    updated["property_issue_type"] = phrase
                    break

    if matter_type == "labour":
        if not updated.get("labour_issue_type"):
            for phrase in ["salary", "termination", "resignation", "gratuity", "pf", "harassment", "service conditions"]:
                
                if phrase in normalized:
                    updated["labour_issue_type"] = phrase
                    break
        
        if not updated.get("employment_status"):
            for phrase in ["employed", "suspended", "resigned", "terminated", "notice period"]:
        
                if phrase in normalized:
                    updated["employment_status"] = phrase
                    break

    return updated

def apply_message_to_matter_record(
    message: str,
    matter_record: dict | None,
    *,
    state: str | None,
    district: str | None,
    matter_type: str | None,
    matter_stage: str | None,
    is_own_matter: bool | None,
    urgency: str | None,
) -> dict:
    record = dict(matter_record or {})
    details = dict(record.get("details") or {})
    issue_summary = record.get("issue_summary")
    initial_query = record.get("initial_query")
    relief_goal = record.get("relief_goal")
    last_question_key = record.get("last_question_key")
    parsed_yes_no = normalize_yes_no_answer(message)

    if not initial_query and message:
        initial_query = summarize_issue_from_message(message)

    if not issue_summary and message and parsed_yes_no is None:
        issue_summary = summarize_issue_from_message(message)

    if not state:
        state = normalize_state_name(message)

    if not district and last_question_key == "district":
        district = normalize_district_name(message)

    if (not matter_type or matter_type in {"general", "unsure"}) and message:
        inferred_type = infer_matter_type(message)

        if inferred_type not in {"general", "unsure"}:
            matter_type = inferred_type

    if (not matter_stage or matter_stage == "not_sure") and message:
        inferred_stage = infer_stage_from_message(message)

        if inferred_stage:
            matter_stage = inferred_stage

    if is_own_matter is None and parsed_yes_no is not None:
        is_own_matter = parsed_yes_no

    if not urgency:
        urgency = normalize_urgency(message)

    if not relief_goal and last_question_key == "relief_goal" and parsed_yes_no is None:
        relief_goal = summarize_issue_from_message(message)

    if has_concrete_issue_details(message):
        details["latest_incident_details"] = summarize_issue_from_message(message)

    detail_field_keys = {
        "cyber_issue_type",
        "financial_loss_amount",
        "institution_name",
        "family_issue_type",
        "relationship_context",
        "property_issue_type",
        "possession_status",
        "property_documents",
        "labour_issue_type",
        "employment_status",
        "employment_documents",
    }

    if last_question_key in detail_field_keys:
        normalized_value = normalize_detail_value(message)

        if normalized_value:
            details[last_question_key] = normalized_value

    details = infer_domain_detail_defaults(message, matter_type, details)

    record.update(
        {
            "initial_query": initial_query,
            "issue_summary": issue_summary,
            "state": state,
            "district": district,
            "matter_type": matter_type,
            "matter_stage": matter_stage,
            "is_own_matter": is_own_matter,
            "urgency": urgency,
            "relief_goal": relief_goal,
            "details": details,
        }
    )
    return record

def get_next_matter_question(record: dict) -> tuple[str | None, str | None]:

    if not record.get("issue_summary"):
        return "issue_summary", "Please briefly describe the legal problem you want guidance on."
    matter_type = record.get("matter_type")

    if not matter_type or matter_type in {"general", "unsure"}:
        return "matter_type", MATTER_COLLECTION_STEPS["matter_type"]

    details = record.get("details") or {}
    domain_steps = DOMAIN_COLLECTION_STEPS.get(matter_type, [])

    for step_key, question in domain_steps:
        if step_key in {"is_own_matter", "state", "district", "matter_stage", "relief_goal"}:
            value = record.get(step_key)

            if step_key == "matter_stage" and value == "not_sure":
                value = None

        else:
            value = details.get(step_key)

        if value in {None, "", "not_sure"}:
            return step_key, question

    if record.get("is_own_matter") is None:
        return "is_own_matter", MATTER_COLLECTION_STEPS["is_own_matter"]

    if not record.get("state"):
        return "state", MATTER_COLLECTION_STEPS["state"]

    if not record.get("district"):
        return "district", MATTER_COLLECTION_STEPS["district"]

    if not record.get("matter_stage") or record.get("matter_stage") == "not_sure":
        return "matter_stage", MATTER_COLLECTION_STEPS["matter_stage"]

    if not record.get("relief_goal"):
        return "relief_goal", MATTER_COLLECTION_STEPS["relief_goal"]
    return None, None

def is_ready_for_solution(record: dict) -> bool:
    matter_type = record.get("matter_type")
    details = record.get("details") or {}

    required_fields = [
        record.get("issue_summary"),
        record.get("state"),
        record.get("district"),
        record.get("matter_stage"),
        record.get("relief_goal"),
    ]

    if not matter_type or matter_type in {"general", "unsure"}:
        return False

    required_detail_fields = {
        "cyber": ["cyber_issue_type", "financial_loss_amount", "institution_name"],
        "family": ["family_issue_type", "relationship_context"],
        "property": ["property_issue_type", "possession_status", "property_documents"],
        "labour": ["labour_issue_type", "employment_status", "employment_documents"],
    }

    domain_required = required_detail_fields.get(matter_type, [])
    return (
        all(required_fields)
        and record.get("is_own_matter") is not None
        and all(details.get(field) for field in domain_required)
    )

def get_collection_suggestions(step_key: str, matter_type: str | None) -> list[str]:

    suggestions_map = {
        "is_own_matter": ["It concerns me personally", "It concerns someone else"],
        "state": ["Gujarat", "Maharashtra", "Delhi"],
        "district": ["Ahmedabad", "Surat", "Vadodara"],
        "matter_stage": ["Before complaint", "After complaint", "Notice received", "Case already filed"],
        "relief_goal": ["I want legal action", "I want money recovery", "I want a formal complaint"],
        "cyber_issue_type": ["UPI fraud", "Unauthorized bank transaction", "Account hacking", "Impersonation"],
        "financial_loss_amount": ["Rs 5,000", "Rs 25,000", "No money loss, only account misuse"],
        "institution_name": ["SBI", "HDFC", "PhonePe", "Google Pay", "WhatsApp"],
        "family_issue_type": ["Divorce", "Maintenance", "Custody", "Domestic violence", "Succession"],
        "relationship_context": ["Spouse", "Child", "Parents or in-laws"],
        "property_issue_type": ["Title dispute", "Possession issue", "Partition", "Tenancy or eviction"],
        "possession_status": ["I am in possession", "The other side is in possession", "Property is locked or disputed"],
        "property_documents": ["I have sale deed", "I have mutation and tax records", "I have rent agreement", "I have no documents yet"],
        "labour_issue_type": ["Salary not paid", "Termination", "Gratuity or PF", "Workplace harassment"],
        "employment_status": ["Still employed", "Resigned", "Terminated", "Serving notice period"],
        "employment_documents": ["I have appointment letter and payslips", "I have HR emails", "I have termination letter"],
    }
    return suggestions_map.get(step_key, [])

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
    workflow = get_domain_workflow(matter_type)

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

    for step in workflow.get("next_steps", []):
        if step not in steps:
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
    workflow = get_domain_workflow(matter_type)

    if workflow.get("intake_questions"):
        return workflow["intake_questions"]

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

def build_domain_guidance(matter_type: str | None, matter_stage: str | None, issue_category: str | None) -> str:
    domain_lines: list[str] = []
    
    if matter_type == "criminal":
        domain_lines.append("Focus on criminal procedure, police action, FIR stage, arrest risk, bail, and immediate procedural safeguards.")
    
    elif matter_type == "family":
        domain_lines.append("Focus on family-court relief, maintenance, custody, domestic violence, residence, and succession context as applicable.")
    
    elif matter_type == "property":
        domain_lines.append("Focus on title, possession, tenancy, registration, mutation, partition, or civil-property remedies as applicable.")
    
    elif matter_type == "labour":
        domain_lines.append("Focus on employer-employee rights, wages, termination, gratuity, PF, and labour forum options where relevant.")
    
    elif matter_type == "consumer":
        domain_lines.append("Focus on deficiency in service, refund, compensation, notice strategy, and consumer forum process.")
    
    elif matter_type == "cyber":
        domain_lines.append("Focus on digital fraud, online abuse, platform evidence, complaint route, and urgent preservation of records.")
    
    elif matter_type == "constitutional":
        domain_lines.append("Focus on constitutional rights, state action, public authority duties, and writ or public-law remedies where relevant.")
    
    elif matter_type == "civil":
        domain_lines.append("Focus on notice, documentation, limitation, jurisdiction, filing strategy, and civil remedies.")

    if matter_stage and matter_stage != "not_sure":
        domain_lines.append(f"Take the current procedural stage into account: {matter_stage}.")
    
    if issue_category:
        domain_lines.append(f"Primary issue category: {issue_category}.")
    return " ".join(domain_lines).strip()

def get_domain_workflow(matter_type: str | None) -> dict:
    return DOMAIN_WORKFLOWS.get(matter_type or "", {})

def build_forum_hint(state: str | None, district: str | None, matter_type: str | None, matter_stage: str | None, issue_category: str) -> str:
    place_text = f"{district}, {state}" if district and state else state or "your state"
    workflow = get_domain_workflow(matter_type)
    workflow_forum_hint = workflow.get("forum_hint")
    if workflow_forum_hint:
        return f"For matters in {place_text}, {workflow_forum_hint}"
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


def build_forum_options(state: str | None, district: str | None, matter_type: str | None, urgency_flags: list[str] | None = None) -> list[str]:
    place_text = f"{district}, {state}" if district and state else state or "your local jurisdiction"
    urgency_flags = urgency_flags or []
    options: list[str] = []
    if matter_type == "cyber":
        options.extend([
            f"Cyber crime portal or local cyber cell for {place_text}",
            f"Bank grievance channel and nodal officer for {place_text}",
            f"Local police station or Magistrate if criminal complaint escalation becomes necessary in {place_text}",
        ])
    elif matter_type == "criminal":
        options.extend([
            f"Local police station for {place_text}",
            f"Jurisdictional Magistrate court for {place_text}",
            f"Sessions Court or High Court depending on bail, quashing, or appellate relief in {place_text}",
        ])
    elif matter_type == "family":
        options.extend([
            f"Family Court for {place_text}",
            f"Jurisdictional Magistrate or Protection Officer where domestic violence relief is involved in {place_text}",
            f"Mediation centre attached to the Family Court for {place_text}",
        ])
    elif matter_type == "property":
        options.extend([
            f"Civil court for title, possession, partition, or injunction relief in {place_text}",
            f"Revenue authority or mutation office for land and revenue entries in {place_text}",
            f"Rent-control or tenancy forum where landlord-tenant relief applies in {place_text}",
        ])
    elif matter_type == "labour":
        options.extend([
            f"Labour Commissioner or labour authority for {place_text}",
            f"Labour Court or Industrial Tribunal for {place_text}",
            f"EPFO or gratuity authority where statutory dues are involved in {place_text}",
        ])
    elif matter_type == "consumer":
        options.extend([
            f"District Consumer Commission for {place_text}",
            f"Sector regulator or ombudsman if the service is regulated in {place_text}",
            f"Civil forum where the dispute falls outside the consumer pathway in {place_text}",
        ])
    elif matter_type == "constitutional":
        options.extend([
            f"High Court for writ relief connected with {place_text}",
            f"Relevant statutory authority or tribunal before writ, where an alternate remedy exists in {place_text}",
        ])
    if "arrest_risk" in urgency_flags:
        options.insert(0, f"Immediate criminal defence consultation and nearest Magistrate or Sessions forum for urgent relief in {place_text}")
    if "domestic_violence_risk" in urgency_flags:
        options.insert(0, f"Protection Officer, Magistrate, and immediate police assistance for safety in {place_text}")
    return options[:4]


def build_domain_solution_template(
    matter_type: str | None,
    *,
    state: str | None,
    district: str | None,
    matter_stage: str | None,
    relief_goal: str | None,
    urgency_flags: list[str],
    details: dict | None = None,
) -> str:
    details = details or {}
    place_text = f"{district}, {state}" if district and state else state or "the relevant local jurisdiction"
    templates = {
        "cyber": (
            f"This appears to be a cyber-fraud matter connected with {place_text}. "
            f"The immediate focus should be preservation of transaction evidence, urgent complaint routing, and bank/platform escalation. "
            f"The present objective appears to be: {relief_goal or 'recovery and complaint action'}."
        ),
        "criminal": (
            f"This appears to be a criminal-law matter connected with {place_text}. "
            f"The next steps should be shaped around the present stage ({matter_stage or 'not confirmed'}) and the need for complaint, FIR, bail, or court protection. "
            f"The present objective appears to be: {relief_goal or 'procedural legal relief'}."
        ),
        "family": (
            f"This appears to be a family-law matter connected with {place_text}. "
            f"The immediate strategy should be to identify the exact family relief, preserve communications and records, and move before the correct forum. "
            f"The present objective appears to be: {relief_goal or 'family-court relief'}."
        ),
        "property": (
            f"This appears to be a property dispute connected with {place_text}. "
            f"The immediate strategy should be to verify title and possession records, assess notice or filing stage, and choose the correct civil or revenue forum. "
            f"The present objective appears to be: {relief_goal or 'property relief'}."
        ),
        "labour": (
            f"This appears to be a labour or employment matter connected with {place_text}. "
            f"The immediate strategy should be to preserve employment records, identify the statutory route, and frame the dues or service relief clearly. "
            f"The present objective appears to be: {relief_goal or 'employment-related relief'}."
        ),
        "consumer": (
            f"This appears to be a consumer dispute connected with {place_text}. "
            f"The immediate strategy should be to preserve invoices and complaint records, define the deficiency clearly, and assess the correct complaint forum. "
            f"The present objective appears to be: {relief_goal or 'consumer relief'}."
        ),
    }
    template = templates.get(
        matter_type or "",
        f"This appears to be a legal dispute connected with {place_text}. The next steps should be aligned to the relief sought and the correct forum."
    )
    if urgency_flags:
        template += " Urgent flags identified: " + ", ".join(flag.replace("_", " ") for flag in urgency_flags) + "."
    if details.get("latest_incident_details"):
        template += f" Core facts noted: {details['latest_incident_details']}."
    return template


def build_lawyer_workflow(
    matter_record: dict | None,
    *,
    matter_type: str | None,
    state: str | None,
    district: str | None,
    matter_stage: str | None,
    urgency_flags: list[str],
    citations: list[str],
    uploaded_documents: list[dict] | None = None,
) -> dict:
    record = matter_record or {}
    details = record.get("details") or {}
    chronology = [
        record.get("initial_query") or record.get("issue_summary"),
        details.get("latest_incident_details"),
        f"Current stage: {matter_stage}" if matter_stage and matter_stage != "not_sure" else None,
    ]
    chronology = [item for item in chronology if item]
    issue_frame = [
        f"Matter type: {matter_type or 'general'}",
        f"Jurisdiction focus: {district + ', ' if district else ''}{state or 'not confirmed'}",
        f"Relief sought: {record.get('relief_goal') or 'not fully stated'}",
    ]
    checklist = build_next_steps(detect_issue_category(record.get("issue_summary") or "", matter_type), state, district, matter_type, matter_stage, record.get("is_own_matter"), determine_effective_urgency(record.get("issue_summary") or "", record.get("urgency"), matter_type, matter_stage, details))[:5]
    draft_notice_points = [
        "Set out the parties, dates, and core facts in chronological order.",
        "State the exact grievance, loss, or violation clearly and without unnecessary emotion.",
        "State the immediate relief sought and a clear time for compliance where applicable.",
    ]
    if matter_type == "cyber":
        draft_notice_points.append("Attach transaction proof, screenshots, bank complaint references, and platform complaint details.")
    elif matter_type == "property":
        draft_notice_points.append("Attach title, possession, rent, mutation, and notice records relevant to the claim.")
    elif matter_type == "labour":
        draft_notice_points.append("Attach appointment letter, payslips, HR emails, attendance records, and termination or warning documents.")
    elif matter_type == "family":
        draft_notice_points.append("Attach marriage, residence, income, child, and relevant communication records where applicable.")
    readiness_notes = [
        "Chronology prepared" if chronology else "Chronology still incomplete",
        "Supporting citations identified" if citations else "Source citations need strengthening",
        "Documents uploaded" if uploaded_documents else "No supporting documents uploaded yet",
        "Urgent action required" if urgency_flags else "No emergency escalation signal identified",
    ]
    return {
        "chronology": chronology,
        "issue_framing": issue_frame,
        "draft_notice_points": draft_notice_points,
        "complaint_checklist": checklist,
        "filing_readiness_review": readiness_notes,
    }

def extract_citations(source: str, metadata_list: list[dict] | None) -> list[str]:
    citations: list[str] = []
    if not metadata_list:
        return citations

    for metadata in metadata_list[:3]:
        citation = citation_from_metadata(metadata, fallback_source=source)
        if citation:
            citations.append(citation)

    return validate_citations(citations)

def extract_source_snippets(metadata_list: list[dict] | None, chunks: list[str] | None) -> list[str]:
    snippets: list[str] = []
    if not chunks:
        return snippets

    for index, chunk in enumerate(chunks[:3]):
        metadata = metadata_list[index] if metadata_list and index < len(metadata_list) else {}
        label = citation_from_metadata(metadata, fallback_source="Source") or "Source"
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


def build_retrieval_query(
    message: str,
    state: str | None,
    district: str | None,
    matter_type: str | None,
    matter_stage: str | None,
) -> str:
    query_parts = [message.strip()]
    filters: list[str] = []
    if state:
        filters.append(f"State {state}")
    if district:
        filters.append(f"District {district}")
    if matter_type and matter_type not in {"general", "unsure"}:
        filters.append(f"Matter type {matter_type}")
    if matter_stage and matter_stage != "not_sure":
        filters.append(f"Stage {matter_stage}")
    if filters:
        query_parts.append("Jurisdiction and case filters: " + "; ".join(filters))
    return "\n".join(query_parts)


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
    missing_detail_question: str | None = None,
    response_language: str | None = None,
) -> str:
    profile_summary = build_profile_summary(state, district, matter_type, matter_stage, is_own_matter, urgency)
    context_text = clean_context_text(context or "")
    language_instruction = (
        "Respond in clear Hindi written in Devanagari script. " if response_language == "hi"
        else "Respond in clear professional English. "
    )
    return (
        f"Question: {query}\n\n"
        f"Legal matter profile: {profile_summary}\n\n"
        + (f"Legal context:\n{context_text}\n\n" if context_text else "")
        + (
            "Answer the user's exact factual question first. "
            "Use the legal matter profile only as supporting context and jurisdiction filtering, not as a replacement for the question. "
            "Respond in a strictly formal and professional manner, but sound natural and advocate-like rather than robotic. "
            "Keep the reply concise, clear, and respectful. "
            + language_instruction
            + "Ground the answer in the supplied legal references, sections, articles, rules, or case citations where available. "
            "Give the practical answer first. "
            + (
                f"After answering, ask only this one follow-up question if still needed: {missing_detail_question} "
                if missing_detail_question
                else "Do not ask a follow-up question unless one specific missing detail is genuinely required. "
            )
            + "Avoid slang, greetings, emojis, generic filler, and long explanations.\n\n"
        )
        + 'Reply in valid JSON only with this shape: {"answer": "..."}'
    )

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


def get_legalparam_model():
    global _legalparam_model, _legalparam_tokenizer
    if _legalparam_model is None or _legalparam_tokenizer is None:
        logger.info("Loading Hugging Face model %s", LEGALPARAM_MODEL_NAME)
        _legalparam_tokenizer = AutoTokenizer.from_pretrained(
            LEGALPARAM_MODEL_NAME,
            trust_remote_code=False,
        )
        _legalparam_model = AutoModelForCausalLM.from_pretrained(
            LEGALPARAM_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
    return _legalparam_tokenizer, _legalparam_model


def generate_response(prompt: str) -> str:
    try:
        tokenizer, model = get_legalparam_model()
        formatted_prompt = f"<system>\n{LLM_SYSTEM_PROMPT}\n</system>\n<user>\n{prompt}\n<assistant>\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        model_device = getattr(model, "device", None) or next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.4,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        logger.error("LegalParam generation error: %s", e)
        return ""


# def call_groq_llm(prompt: str) -> dict:
#     groq_api_key = os.environ.get("GROQ_API_KEY")
#     ...
#
# def call_openai_llm(query: str, context: str) -> dict:
#     openai_api_key = os.environ.get("OPENAI_API_KEY")
#     ...


def generate_llm_answer(
    query: str,
    context: str,
    *,
    state: str | None = None,
    district: str | None = None,
    matter_type: str | None = None,
    matter_stage: str | None = None,
    is_own_matter: bool | None = None,
    urgency: str | None = None,
    missing_detail_question: str | None = None,
    response_language: str | None = None,
) -> dict:
    prompt = build_llm_context(
        query=query,
        context=context,
        state=state,
        district=district,
        matter_type=matter_type,
        matter_stage=matter_stage,
        is_own_matter=is_own_matter,
        urgency=urgency,
        missing_detail_question=missing_detail_question,
        response_language=response_language,
    )
    raw_output = generate_response(prompt)
    parsed = parse_llm_json(raw_output)
    answer = extract_llm_answer(parsed)
    if answer:
        return {"answer": answer}
    return {"answer": raw_output.strip()}


def maybe_translate_label(label: str, language: str) -> str:
    if language != "hi":
        return label
    translations = {
        "Assessment": "प्रारंभिक मूल्यांकन",
        "Action": "तत्काल कदम",
        "Forum": "उपयुक्त मंच",
        "References": "कानूनी संदर्भ",
        "Urgent": "तत्काल सावधानी",
        "Documents": "दस्तावेज़",
    }
    return translations.get(label, label)


def translate_prompt(text: str, language: str) -> str:
    if language != "hi":
        return text
    translations = {
        "I have noted the details shared so far.\n": "अब तक साझा की गई जानकारी मैंने दर्ज कर ली है।\n",
        "Please confirm whether this cyber incident concerns you personally or someone else.": "कृपया बताइए कि यह साइबर घटना आपके अपने मामले से जुड़ी है या किसी अन्य व्यक्ति से।",
        "Please state the State or Union Territory where the cyber incident occurred or where you are filing the complaint.": "कृपया वह राज्य या केंद्र शासित प्रदेश बताइए जहाँ यह साइबर घटना हुई या जहाँ आप शिकायत दर्ज कर रहे हैं।",
        "Please state the district or city linked to the bank account, police complaint, or cyber cell jurisdiction.": "कृपया वह जिला या शहर बताइए जो बैंक खाते, पुलिस शिकायत, या साइबर सेल क्षेत्राधिकार से जुड़ा है।",
        "Please specify whether this concerns UPI fraud, card or bank fraud, account hacking, impersonation, abusive content, or another cyber issue.": "कृपया बताइए कि यह यूपीआई धोखाधड़ी, बैंक या कार्ड फ्रॉड, अकाउंट हैकिंग, प्रतिरूपण, आपत्तिजनक सामग्री, या किसी अन्य साइबर समस्या से जुड़ा है।",
        "Please state the approximate amount lost or placed at risk, if any.": "यदि कोई राशि गई है या जोखिम में है, तो कृपया उसका लगभग विवरण बताइए।",
        "Please state the bank, wallet, platform, or app involved.": "कृपया संबंधित बैंक, वॉलेट, प्लेटफ़ॉर्म, या ऐप का नाम बताइए।",
        "Please state whether you are before complaint, after complaint, after FIR, or already before a court or cyber cell.": "कृपया बताइए कि मामला शिकायत से पहले का है, शिकायत के बाद का है, एफआईआर के बाद का है, या पहले से अदालत/साइबर सेल के समक्ष है।",
        "Please state the main result you want, such as account freeze, money recovery, FIR, chargeback, or takedown of content.": "कृपया बताइए कि आप मुख्य रूप से क्या राहत चाहते हैं, जैसे अकाउंट फ्रीज़, धन-वसूली, एफआईआर, चार्जबैक, या सामग्री हटवाना।",
    }
    return translations.get(text, text)


def build_solution_sections(
    *,
    llm_answer: str,
    matter_type: str | None,
    state: str | None,
    district: str | None,
    matter_stage: str | None,
    relief_goal: str | None,
    urgency_flags: list[str],
    citations: list[str],
    forum_options: list[str],
    uploaded_documents: list[dict],
    details: dict | None,
    language: str,
) -> str:
    sections: list[str] = []
    research_brief = build_research_brief(
        matter_type=matter_type,
        state=state,
        district=district,
        urgency_flags=urgency_flags,
        citations=citations,
        uploaded_documents=uploaded_documents,
    )
    template_text = build_domain_solution_template(
        matter_type,
        state=state,
        district=district,
        matter_stage=matter_stage,
        relief_goal=relief_goal,
        urgency_flags=urgency_flags,
        details=details,
    )
    sections.append(f"{maybe_translate_label('Assessment', language)}: {template_text}")
    if llm_answer:
        sections.append(f"{maybe_translate_label('Action', language)}: {llm_answer}")
    if forum_options:
        sections.append(f"{maybe_translate_label('Forum', language)}: " + "; ".join(forum_options[:3]))
    if citations:
        sections.append(f"{maybe_translate_label('References', language)}: " + "; ".join(validate_citations(citations)))
    elif research_brief.get("primary_authorities"):
        sections.append(f"{maybe_translate_label('References', language)}: " + "; ".join(validate_citations(research_brief["primary_authorities"][:3])))
    if uploaded_documents:
        document_names = ", ".join(doc.get("original_name", "document") for doc in uploaded_documents[:3])
        sections.append(f"{maybe_translate_label('Documents', language)}: {document_names}")
    if urgency_flags:
        sections.append(f"{maybe_translate_label('Urgent', language)}: " + ", ".join(flag.replace("_", " ") for flag in urgency_flags))
    return "\n\n".join(sections)
    # return generate_response(prompt)


def should_use_direct_shortcut(message: str) -> bool:
    normalized = normalize_query(message)
    words = [word for word in normalized.split() if word]
    return detect_simple_query(message) and len(words) <= 8

def build_answer_generation_context(
    rag_chunks: list[str] | None = None,
    citations: list[str] | None = None,
    source_snippets: list[str] | None = None,
    faq_answer: str | None = None,
    matter_type: str | None = None,
    matter_stage: str | None = None,
    issue_category: str | None = None,
) -> str:
    context_parts: list[str] = []

    legal_context = format_context_chunks(rag_chunks, citations)
    if legal_context:
        context_parts.append(legal_context)

    domain_guidance = build_domain_guidance(matter_type, matter_stage, issue_category)
    if domain_guidance:
        context_parts.append(f"Domain guidance:\n{domain_guidance}")

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
    cleaned = re.sub(r"\r\n?", "\n", answer).strip()
    cleaned = re.sub(r"(?im)^(hi|hello|hey|buddy|dear)\b[:,\s-]*", "", cleaned).strip()
    lines = [line.strip(" -") for line in cleaned.split("\n") if line.strip()]
    if not lines:
        return ""

    if len(lines) > 5:
        lines = lines[:5]

    return "\n".join(lines)

def build_chat_response(
    *,
    message: str,
    response_text: str,
    source: str,
    intake_step: str | None = None,
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
    response_language: str | None = None,
    lawyer_workflow: dict | None = None,
    research_brief: dict | None = None,
    uploaded_documents: list[dict] | None = None,
    chat_id: int | None = None,
) -> ChatResponse:
    issue_category = detect_issue_category(message, matter_type)
    return ChatResponse(
        response=finalize_response(message, response_text, state, district, matter_type, matter_stage, is_own_matter, urgency),
        chat_id=chat_id,
        intent=intent,
        confidence=confidence,
        source=source,
        intake_step=intake_step,
        state=state,
        district=district,
        matter_type=matter_type,
        matter_stage=matter_stage,
        is_own_matter=is_own_matter,
        urgency=urgency,
        issue_category=issue_category,
        next_steps=build_next_steps(issue_category, state, district, matter_type, matter_stage, is_own_matter, urgency),
        suggested_questions=build_suggested_questions(issue_category, matter_type),
        citations=citations or [],
        source_snippets=source_snippets or [],
        forum_hint=build_forum_hint(state, district, matter_type, matter_stage, issue_category),
        intake_missing_fields=intake_missing_fields or [],
        disclaimer=build_disclaimer(matter_type, is_own_matter, state),
        intake_complete=bool(state and matter_type and is_own_matter is not None),
        response_language=response_language,
        lawyer_workflow=lawyer_workflow or {},
        research_brief=research_brief or {},
        uploaded_documents=uploaded_documents or [],
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
    
    retrieval_query = build_retrieval_query(query, state, district, matter_type, matter_stage)
    expanded_query = expand_query_with_synonyms(retrieval_query)
    logger.info("RAG retrieval query prepared query_hash=%s expanded_hash=%s", hash_text(query), hash_text(expanded_query))

    state_result = get_state_rag_module().state_rag_answer(expanded_query, state=state, matter_type=matter_type) or {}
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

    case_result = get_case_rag_module().case_rag_answer(expanded_query, state=state, matter_type=matter_type) or {}
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

    txt_result = get_txt_rag_module().txt_rag_answer(expanded_query, state=state, matter_type=matter_type) or {}
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

    pdf_result = get_pdf_rag_module().generate_answer(expanded_query, state=state, matter_type=matter_type) or {}
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
    retrieval_query = build_retrieval_query(query, state, district, matter_type, matter_stage)
    expanded_query = expand_query_with_synonyms(retrieval_query)
    logger.info("RAG query prepared query_hash=%s expanded_hash=%s", hash_text(query), hash_text(expanded_query))

    candidates = [
        ("state_rag", get_state_rag_module().state_rag_answer(expanded_query, state=state, matter_type=matter_type) or {}),
        ("case_rag", get_case_rag_module().case_rag_answer(expanded_query, state=state, matter_type=matter_type) or {}),
        ("txt_rag", get_txt_rag_module().txt_rag_answer(expanded_query, state=state, matter_type=matter_type) or {}),
        ("pdf_rag", get_pdf_rag_module().generate_answer(expanded_query, state=state, matter_type=matter_type) or {}),
    ]

    best_answer = ""
    best_conf = 0.0
    best_source = ""
    best_citations: list[str] = []
    best_source_snippets: list[str] = []
    best_chunks: list[str] = []
    best_candidate_score = (-1.0, -1, -1, -1)

    preferred_domains = MATTER_TYPE_TO_LEGAL_DOMAINS.get(matter_type or "general", set())
    source_priority_map = SOURCE_PRIORITY_BY_MATTER.get(matter_type or "general", {})

    for source_name, result in candidates:
        answer = result.get("answer", "")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        chunks = result.get("chunks", []) or []
        matches = result.get("matches", []) or []
        selected_chunks = select_top_context_chunks(chunks)
        if not selected_chunks:
            continue

        domain_match_count = 0
        for metadata in matches:
            legal_domain = (metadata or {}).get("legal_domain", "")
            if legal_domain in preferred_domains:
                domain_match_count += 1

        cleaned_answer = ""
        if is_valid_answer(answer, query) and not is_weak_answer(answer, query):
            valid, normalized_answer = validate_and_clean(query, answer)
            if valid:
                cleaned_answer = normalized_answer

        candidate_score = (
            round(confidence, 6),
            domain_match_count,
            source_priority_map.get(source_name, 0),
            len(selected_chunks),
        )
        if candidate_score > best_candidate_score:
            best_answer = cleaned_answer
            best_conf = confidence
            best_source = source_name
            best_citations = extract_citations(source_name, matches)
            best_source_snippets = extract_source_snippets(matches, selected_chunks)
            best_chunks = selected_chunks
            best_candidate_score = candidate_score
            logger.info(
                "RAG context selected source=%s conf=%.3f chunks=%d domain_matches=%d",
                source_name,
                confidence,
                len(selected_chunks),
                domain_match_count,
            )

    return best_answer, best_conf, best_source, best_citations, best_source_snippets, best_chunks

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/frontend/auth.html")


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.get("/auth/ready", include_in_schema=False)
def auth_ready():
    db_health = check_db_health()
    if not db_health["ok"]:
        logger.warning("Auth readiness check failed error=%s", db_health["error"])
        raise HTTPException(status_code=503, detail="Authentication database is not ready")
    return {"status": "ready"}


@app.get("/debug/auth-status", include_in_schema=False)
def debug_auth_status():
    db_health = check_db_health()
    return {
        "status": "ok" if db_health["ok"] else "degraded",
        "db_ready": db_health["ok"],
        "db_error": db_health["error"],
        "google_configured": google_auth_configured(),
        "frontend_auth_url": FRONTEND_AUTH_URL,
        "frontend_app_url": FRONTEND_APP_URL,
        "cors_allow_origins": origins,
        "cors_allow_credentials": allow_credentials,
        "password_reset_exposed_in_response": EXPOSE_RESET_TOKEN_IN_RESPONSE,
    }


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
    logger.info("Signup attempt email_hash=%s state=%s", hash_text(email), state or "not_provided")

    if len(full_name) < 2:
        raise HTTPException(status_code=400, detail="Full name must be at least 2 characters")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    try:
        if get_user_by_email(email):
            raise HTTPException(status_code=409, detail="An account with this email already exists")
        user = create_user(full_name=full_name, email=email, password=password, state=state)
        token = create_session(user["id"])
    except sqlite3.OperationalError as exc:
        logger.error("Signup database failure email_hash=%s error=%s", hash_text(email), exc)
        raise HTTPException(status_code=503, detail="Authentication database is temporarily unavailable. Please restart the server and try again.")
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
    email = normalize_email(request.email)
    logger.info("Login attempt email_hash=%s", hash_text(email))
    try:
        user = authenticate_user(request.email, request.password)
    except sqlite3.OperationalError as exc:
        logger.error("Login lookup failure email_hash=%s error=%s", hash_text(email), exc)
        raise HTTPException(status_code=503, detail="Authentication database is temporarily unavailable. Please restart the server and try again.")
    if not user:
        logger.warning("Login failed email_hash=%s reason=invalid_credentials", hash_text(email))
        raise HTTPException(status_code=401, detail="Invalid email or password")
    try:
        token = create_session(user["id"])
    except sqlite3.OperationalError as exc:
        logger.error("Login session creation failure email_hash=%s user_id=%s error=%s", hash_text(email), user["id"], exc)
        raise HTTPException(status_code=503, detail="Session service is temporarily unavailable. Please restart the server and try again.")
    logger.info("Login success email_hash=%s user_id=%s", hash_text(email), user["id"])
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


@app.post("/auth/password-reset", response_model=MessageResponse)
def password_reset(request: PasswordResetRequest):
    email = normalize_email(request.email)
    logger.info("Password reset requested email_hash=%s", hash_text(email))
    try:
        user = get_user_by_email(email)
    except sqlite3.OperationalError as exc:
        logger.error("Password reset lookup failed email_hash=%s error=%s", hash_text(email), exc)
        raise HTTPException(status_code=503, detail="Authentication database is temporarily unavailable. Please try again shortly.")

    response_payload = {
        "message": "If an account with this email exists, a password reset link has been generated.",
        "reset_token": None,
        "reset_url": None,
    }
    if not user:
        return MessageResponse(**response_payload)

    try:
        reset_token = create_password_reset_token(user["id"])
    except sqlite3.OperationalError as exc:
        logger.error("Password reset token creation failed email_hash=%s error=%s", hash_text(email), exc)
        raise HTTPException(status_code=503, detail="Password reset service is temporarily unavailable. Please try again shortly.")

    reset_url = build_frontend_password_reset_url(reset_token)
    logger.info("Password reset token issued email_hash=%s token_hash=%s", hash_text(email), hash_text(reset_token))

    if EXPOSE_RESET_TOKEN_IN_RESPONSE:
        response_payload["reset_token"] = reset_token
        response_payload["reset_url"] = reset_url
    return MessageResponse(**response_payload)


@app.post("/auth/password-reset/confirm", response_model=MessageResponse)
def password_reset_confirm(request: PasswordResetConfirmRequest):
    token = request.token.strip()
    new_password = request.new_password.strip()
    if len(token) < 20:
        raise HTTPException(status_code=400, detail="Enter a valid password reset token")
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    try:
        user = consume_password_reset_token(token, new_password)
    except sqlite3.OperationalError as exc:
        logger.error("Password reset confirmation failed token_hash=%s error=%s", hash_text(token), exc)
        raise HTTPException(status_code=503, detail="Password reset service is temporarily unavailable. Please try again shortly.")

    if not user:
        raise HTTPException(status_code=400, detail="Password reset token is invalid or has expired")

    logger.info("Password reset completed email_hash=%s", hash_text(user["email"]))
    return MessageResponse(message="Password updated successfully. Please sign in with your new password.")


@app.get("/chat/history", response_model=ChatSessionsResponse)
def chat_history(authorization: str | None = Header(default=None)):
    user = require_current_user(authorization)
    sessions = list_chat_sessions(user["id"])
    return ChatSessionsResponse(items=[ChatSession(**session) for session in sessions])


@app.post("/chat/session", response_model=ChatSession)
def create_chat_session_endpoint(session: ChatSessionCreate, authorization: str | None = Header(default=None)):
    user = require_current_user(authorization)
    created = create_chat_session(user["id"], title=session.title)
    return ChatSession(**created)


@app.get("/chat/{chat_id}/messages", response_model=ChatHistoryResponse)
def chat_messages(chat_id: int, authorization: str | None = Header(default=None)):
    user = require_current_user(authorization)
    session = get_chat_session(chat_id)
    if not session or session["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Chat session not found")

    items = [HistoryMessage(**item) for item in list_chat_messages(user["id"], chat_id=chat_id)]
    return ChatHistoryResponse(items=items)


@app.delete("/chat/{chat_id}/messages")
def clear_chat_session_messages(chat_id: int, authorization: str | None = Header(default=None)):
    user = require_current_user(authorization)
    session = get_chat_session(chat_id)
    if not session or session["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Chat session not found")

    clear_chat_messages(user["id"], chat_id)
    if get_active_matter_intake(user["id"], chat_id=chat_id):
        close_active_matter_intake(user["id"], chat_id=chat_id)
    return {"status": "ok", "chat_id": chat_id}


@app.delete("/chat/history")
def clear_full_chat_history(authorization: str | None = Header(default=None)):
    user = require_current_user(authorization)
    clear_all_chat_history(user["id"])
    return {"status": "ok"}


@app.post("/chat/matter/reset")
def reset_chat_matter(chat_id: int | None = None, authorization: str | None = Header(default=None)):
    user = require_current_user(authorization)
    close_active_matter_intake(user["id"], chat_id=chat_id)
    return {"status": "ok", "chat_id": chat_id}


@app.get("/chat/documents", response_model=UploadedDocumentListResponse)
def chat_documents(chat_id: int | None = None, authorization: str | None = Header(default=None)):
    user = require_current_user(authorization)
    matter_record = get_active_matter_intake(user["id"], chat_id=chat_id)
    items = list_uploaded_documents(user["id"], matter_record["id"] if matter_record else None)
    return UploadedDocumentListResponse(
        items=[
            UploadedDocumentResponse(
                id=item["id"],
                original_name=item["original_name"],
                content_type=item["content_type"],
                file_size=item["file_size"],
                document_kind=item["document_kind"],
                description=item["description"],
                created_at=item["created_at"],
            )
            for item in items
        ]
    )


if MULTIPART_AVAILABLE:
    @app.post("/chat/documents", response_model=UploadedDocumentResponse)
    async def upload_chat_document(
        file: UploadFile = File(...),
        chat_id: int | None = Form(default=None),
        document_kind: str | None = Form(default=None),
        description: str | None = Form(default=None),
        authorization: str | None = Header(default=None),
    ):
        user = require_current_user(authorization)
        matter_record = get_active_matter_intake(user["id"], chat_id=chat_id)
        if matter_record is None:
            matter_record = upsert_active_matter_intake(
                user["id"],
                chat_id=chat_id,
                initial_query=None,
                issue_summary=None,
                state=normalize_state_name(user.get("state")),
                last_question_key="issue_summary",
                details={},
            )

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_DOCUMENT_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file type for document intake")

        content = await file.read()
        user_upload_dir = UPLOADS_DIR / f"user_{user['id']}"
        user_upload_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid4().hex}_{sanitize_filename(file.filename or 'document')}"
        stored_path = user_upload_dir / stored_name
        stored_path.write_bytes(content)

        extracted_text = extract_upload_text(file.filename or stored_name, file.content_type, content)
        resolved_document_kind = document_kind or infer_document_kind(file.filename or stored_name, extracted_text)
        saved = store_uploaded_document(
            user["id"],
            matter_intake_id=matter_record["id"],
            original_name=file.filename or stored_name,
            stored_name=stored_name,
            file_path=str(stored_path),
            content_type=file.content_type,
            file_size=len(content),
            document_kind=resolved_document_kind,
            description=description,
            extracted_text=extracted_text,
        )
        return UploadedDocumentResponse(
            id=saved["id"],
            original_name=saved["original_name"],
            content_type=saved["content_type"],
            file_size=saved["file_size"],
            document_kind=saved["document_kind"],
            description=saved["description"],
            created_at=saved["created_at"],
        )
else:
    @app.post("/chat/documents", response_model=UploadedDocumentResponse)
    async def upload_chat_document_unavailable(authorization: str | None = Header(default=None)):
        require_current_user(authorization)
        raise HTTPException(status_code=503, detail="Document upload requires python-multipart to be installed on the server")

@app.get("/debug/status")
def debug_status():
    try:
        store_summary = {}
        store_configs = [
            ("txt_vectorstore", VECTOR_DB_DIR / "doc_db", "backend.RAG_Bot.txt.txt_rag", "get_vectorstore"),
            ("pdf_vectorstore", VECTOR_DB_DIR / "pdf_db", "backend.RAG_Bot.doc.pdfRP", "get_vectorstore"),
            ("state_vectorstore", VECTOR_DB_DIR / "state_db", "backend.RAG_Bot.state.state_rag", "get_vectorstore"),
            ("case_vectorstore", VECTOR_DB_DIR / "case_db", "backend.RAG_Bot.case.case_rag", "get_vectorstore"),
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

@app.get("/debug/chat-db-check")
def debug_chat_database(authorization: str | None = Header(default=None)):
    """Check chat database status and session integrity"""
    try:
        user = require_current_user(authorization)
        user_id = user["id"]
        
        # Check if tables exist
        import sqlite3
        from contextlib import closing
        from .database import DB_PATH
        
        with closing(sqlite3.connect(str(DB_PATH))) as conn:
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chat_sessions', 'chat_messages')")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get user's sessions
            cursor.execute("SELECT id, title, created_at, last_activity FROM chat_sessions WHERE user_id = ? ORDER BY id DESC LIMIT 5", (user_id,))
            sessions_rows = cursor.fetchall()
            sessions = [{'id': row[0], 'title': row[1], 'created_at': row[2], 'last_activity': row[3]} for row in sessions_rows]
            
            # Get message count per session
            cursor.execute("""
                SELECT chat_id, COUNT(*) as count 
                FROM chat_messages 
                WHERE user_id = ? 
                GROUP BY chat_id 
                ORDER BY chat_id DESC
            """, (user_id,))
            message_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "status": "OK",
                "database": str(DB_PATH),
                "tables_exist": tables,
                "user_sessions": sessions,
                "message_counts": message_counts,
                "total_sessions": len(sessions),
            }
    except Exception as e:
        logger.error(f"Chat database check error: {e}")
        return {
            "status": "ERROR",
            "error": str(e),
            "message": "Failed to check database"
        }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, authorization: str | None = Header(default=None)):
    user = require_current_user(authorization)
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")
    response_language = detect_response_language(message)

    if request.chat_id is not None:
        session = get_chat_session(request.chat_id)
        if not session or session["user_id"] != user["id"]:
            raise HTTPException(status_code=404, detail="Chat session not found")
        chat_id = request.chat_id
    else:
        session = create_chat_session(user["id"])
        chat_id = session["id"]
    matter_record = get_active_matter_intake(user["id"], chat_id=chat_id)

    state = normalize_state_name(request.state) or normalize_state_name(user.get("state"))
    district = normalize_district_name(request.district)
    matter_type = normalize_matter_type(request.matter_type)
    matter_stage = normalize_matter_stage(request.matter_stage) or "not_sure"
    urgency = normalize_urgency(request.urgency)
    is_own_matter = request.is_own_matter
    recent_messages = list_chat_messages(user["id"], chat_id=chat_id, limit=12)
    state, district, matter_type, matter_stage, is_own_matter, urgency = merge_matter_profile(
        state,
        district,
        matter_type,
        matter_stage,
        is_own_matter,
        urgency,
        matter_record,
    )
    state, district, matter_type, matter_stage, is_own_matter, urgency = merge_profile_from_recent_history(
        recent_messages,
        state,
        district,
        matter_type,
        matter_stage,
        is_own_matter,
        urgency,
    )
    state, district, matter_type, matter_stage, is_own_matter = apply_conversational_intake_updates(
        message,
        recent_messages,
        state,
        district,
        matter_type,
        matter_stage,
        is_own_matter,
    )
    updated_matter = apply_message_to_matter_record(
        message,
        matter_record,
        state=state,
        district=district,
        matter_type=matter_type,
        matter_stage=matter_stage,
        is_own_matter=is_own_matter,
        urgency=urgency,
    )
    state = updated_matter.get("state")
    district = updated_matter.get("district")
    matter_type = updated_matter.get("matter_type")
    matter_stage = updated_matter.get("matter_stage") or "not_sure"
    is_own_matter = updated_matter.get("is_own_matter")
    urgency = updated_matter.get("urgency")
    urgency = determine_effective_urgency(message, urgency, matter_type, matter_stage, updated_matter.get("details"))
    updated_matter["urgency"] = urgency
    if not matter_type or matter_type == "unsure":
        matter_type = infer_matter_type(message)
    direct_shortcut_ok = should_use_direct_shortcut(message)
    skip_category_clarification = should_skip_category_clarification(
        message,
        matter_type,
        recent_messages=recent_messages,
    )
    missing_detail_question = choose_missing_detail_question(
        state=state,
        is_own_matter=is_own_matter,
        matter_stage=matter_stage,
        matter_type=matter_type,
        district=district,
    )
    logger.info(
        "Routing signals: direct_shortcut_ok=%r skip_category_clarification=%r concrete_issue_details=%r missing_detail_question=%r",
        direct_shortcut_ok,
        skip_category_clarification,
        has_concrete_issue_details(message),
        missing_detail_question,
    )
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
        next_step, next_question = get_next_matter_question(updated_matter)
        if next_step:
            stored_matter = upsert_active_matter_intake(
                user["id"],
                chat_id=chat_id,
                initial_query=updated_matter.get("initial_query"),
                issue_summary=updated_matter.get("issue_summary"),
                state=state,
                district=district,
                matter_type=matter_type,
                matter_stage=matter_stage,
                is_own_matter=is_own_matter,
                urgency=urgency,
                relief_goal=updated_matter.get("relief_goal"),
                last_question_key=next_step,
                details=updated_matter.get("details"),
            )
            uploaded_docs = list_uploaded_documents(user["id"], stored_matter["id"])
            response_text = (
                translate_prompt("I have noted the details shared so far.\n", response_language)
                + translate_prompt((next_question or "Please provide the next relevant detail."), response_language)
            )
            response_payload = ChatResponse(
                response=response_text,
                chat_id=chat_id,
                intent="matter_collection",
                confidence=1.0,
                source="collection",
                intake_step=next_step,
                state=state,
                district=district,
                matter_type=matter_type,
                matter_stage=matter_stage,
                is_own_matter=is_own_matter,
                urgency=urgency,
                issue_category=None,
                next_steps=[],
                suggested_questions=get_collection_suggestions(next_step, matter_type),
                citations=[],
                source_snippets=[],
                forum_hint=None,
                intake_missing_fields=[next_step],
                disclaimer=None,
                intake_complete=False,
                response_language=response_language,
                lawyer_workflow={},
                uploaded_documents=[
                    {
                        "id": item["id"],
                        "original_name": item["original_name"],
                        "document_kind": item["document_kind"],
                    }
                    for item in uploaded_docs
                ],
            )
            save_chat_exchange(user["id"], message, response_payload, chat_id=chat_id)
            return response_payload

        stored_matter = upsert_active_matter_intake(
            user["id"],
            chat_id=chat_id,
            initial_query=updated_matter.get("initial_query"),
            issue_summary=updated_matter.get("issue_summary"),
            state=state,
            district=district,
            matter_type=matter_type,
            matter_stage=matter_stage,
            is_own_matter=is_own_matter,
            urgency=urgency,
            relief_goal=updated_matter.get("relief_goal"),
            last_question_key=None,
            details=updated_matter.get("details"),
        )
        uploaded_docs = list_uploaded_documents(user["id"], stored_matter["id"])
        urgency_flags = detect_urgency_flags(message, matter_type, matter_stage, stored_matter.get("details"))
        validated_citations = get_default_domain_citations(matter_type)
        forum_options = build_forum_options(state, district, matter_type, urgency_flags)
        research_brief = build_research_brief(
            matter_type=matter_type,
            state=state,
            district=district,
            urgency_flags=urgency_flags,
            citations=validated_citations,
            uploaded_documents=uploaded_docs,
        )

        faq_answer, faq_conf = exact_faq_match(message)
        if faq_answer is not None and direct_shortcut_ok:
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
                response_language=response_language,
                lawyer_workflow=build_lawyer_workflow(
                    stored_matter,
                    matter_type=matter_type,
                    state=state,
                    district=district,
                    matter_stage=matter_stage,
                    urgency_flags=urgency_flags,
                    citations=validated_citations,
                    uploaded_documents=uploaded_docs,
                ),
                uploaded_documents=uploaded_docs,
                research_brief=research_brief,
                chat_id=chat_id,
            )
            save_chat_exchange(user["id"], message, response_payload, chat_id=chat_id)
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

        if intent_conf >= INTENT_THRESHOLD and direct_shortcut_ok:
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
                    response_language=response_language,
                    lawyer_workflow=build_lawyer_workflow(
                        stored_matter,
                        matter_type=matter_type,
                        state=state,
                        district=district,
                        matter_stage=matter_stage,
                        urgency_flags=urgency_flags,
                        citations=validated_citations,
                        uploaded_documents=uploaded_docs,
                    ),
                    uploaded_documents=uploaded_docs,
                    research_brief=research_brief,
                    chat_id=chat_id,
                )
                save_chat_exchange(user["id"], message, response_payload, chat_id=chat_id)
                return response_payload
            except Exception as e:
                logger.error("Intent bot response error: %s", e)

        rag_answer = ""
        rag_conf = 0.0
        rag_source = ""
        rag_citations: list[str] = []
        rag_source_snippets: list[str] = []
        rag_chunks: list[str] = []
        try:
            rag_answer, rag_conf, rag_source, rag_citations, rag_source_snippets, rag_chunks = pick_rag_context(
                message,
                state,
                district,
                matter_type,
                matter_stage,
                is_own_matter,
                urgency,
            )
            logger.info(
                "RAG context prepared source=%s confidence=%.3f chunks=%d",
                rag_source,
                rag_conf,
                len(rag_chunks),
            )
        except Exception as e:
            logger.error("RAG error: %s", e)

        try:
            # Get recent conversation context (last 5 messages)
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
                rag_chunks=rag_chunks or ([rag_answer] if rag_answer else None),
                citations=rag_citations,
                source_snippets=rag_source_snippets,
                faq_answer=faq_answer if direct_shortcut_ok else None,
                matter_type=matter_type,
                matter_stage=matter_stage,
                issue_category=detect_issue_category(message, matter_type),
            )
            # Include conversation context in the prompt
            full_context = (
                f"Previous conversation:\n{conversation_context}\n\n"
                f"Matter summary: {stored_matter.get('issue_summary') or message}\n"
                f"Desired outcome: {stored_matter.get('relief_goal') or 'not yet stated'}\n"
                f"Uploaded documents: {', '.join(doc.get('original_name', 'document') for doc in uploaded_docs[:5]) or 'none'}\n"
                f"Uploaded document excerpts: {' | '.join((doc.get('extracted_text') or '')[:200] for doc in uploaded_docs[:3] if doc.get('extracted_text')) or 'none'}\n"
                f"Current question: {message}\n\n"
                f"Legal context: {llm_context}"
            )
            llm_response = generate_llm_answer(
                stored_matter.get("issue_summary") or message,
                full_context,
                state=state,
                district=district,
                matter_type=matter_type,
                matter_stage=matter_stage,
                is_own_matter=is_own_matter,
                urgency=urgency,
                missing_detail_question=missing_detail_question,
                response_language=response_language,
            )
            llm_answer = extract_llm_answer(llm_response)
            if llm_answer and len(llm_answer.strip()) > MIN_ANSWER_LENGTH:
                validated_citations = validate_citations(rag_citations) or get_default_domain_citations(matter_type)
                forum_options = build_forum_options(state, district, matter_type, urgency_flags)
                workflow = build_lawyer_workflow(
                    stored_matter,
                    matter_type=matter_type,
                    state=state,
                    district=district,
                    matter_stage=matter_stage,
                    urgency_flags=urgency_flags,
                    citations=validated_citations,
                    uploaded_documents=uploaded_docs,
                    chat_id=chat_id,
                )
                response_text = build_solution_sections(
                    llm_answer=llm_answer,
                    matter_type=matter_type,
                    state=state,
                    district=district,
                    matter_stage=matter_stage,
                    relief_goal=stored_matter.get("relief_goal"),
                    urgency_flags=urgency_flags,
                    citations=validated_citations,
                    forum_options=forum_options,
                    uploaded_documents=uploaded_docs,
                    details=stored_matter.get("details"),
                    language=response_language,
                )
                logger.info("LLM ANSWER GENERATED - source=legalparam confidence=%.3f", max(rag_conf, 0.6))
                response_payload = build_chat_response(
                    message=message,
                    response_text=response_text,
                    intent=tag,
                    confidence=max(rag_conf, 0.6),
                    source="legalparam" if rag_chunks else "legalparam_direct",
                    state=state,
                    district=district,
                    matter_type=matter_type,
                    matter_stage=matter_stage,
                    is_own_matter=is_own_matter,
                    urgency=urgency,
                    citations=validated_citations,
                    source_snippets=rag_source_snippets,
                    response_language=response_language,
                    lawyer_workflow=workflow,
                    research_brief=build_research_brief(
                        matter_type=matter_type,
                        state=state,
                        district=district,
                        urgency_flags=urgency_flags,
                        citations=validated_citations,
                        uploaded_documents=uploaded_docs,
                    ),
                    uploaded_documents=uploaded_docs,
                    chat_id=chat_id,
                )
                save_chat_exchange(user["id"], message, response_payload, chat_id=chat_id)
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
            response_language=response_language,
            lawyer_workflow=build_lawyer_workflow(
                stored_matter,
                matter_type=matter_type,
                state=state,
                district=district,
                matter_stage=matter_stage,
                urgency_flags=urgency_flags,
                citations=validated_citations,
                uploaded_documents=uploaded_docs,
            ),
            research_brief=research_brief,
            uploaded_documents=uploaded_docs,
            chat_id=chat_id,
        )
        save_chat_exchange(user["id"], message, response_payload, chat_id=chat_id)
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
            response_language=response_language,
            lawyer_workflow={},
            uploaded_documents=[],
            chat_id=chat_id,
        )
        save_chat_exchange(user["id"], message, response_payload, chat_id=chat_id)
        return response_payload

if __name__ == "__main__":
    try:
        print("Starting server...")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
