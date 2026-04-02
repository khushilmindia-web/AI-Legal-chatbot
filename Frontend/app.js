const apiBase = window.APP_CONFIG?.apiBaseUrl ?? "";
const frontendBase = window.APP_CONFIG?.frontendBaseUrl || "/frontend";
const TOKEN_KEY = 'legalAuthToken';
const USER_KEY = 'legalAuthUser';
const ME_URL = `${apiBase}/auth/me`;
const LOGOUT_URL = `${apiBase}/auth/logout`;

const state = {
  sessions: [],
  activeChatId: null,
  draftMessages: [],
  liveMessages: [],
  isBusy: false,
  currentUser: null,
};

const elements = {
  historyList: document.getElementById("historyList"),
  emptyHistory: document.getElementById("emptyHistory"),
  chatThread: document.getElementById("chatThread"),
  composerForm: document.getElementById("composerForm"),
  messageInput: document.getElementById("messageInput"),
  sendButton: document.getElementById("sendButton"),
  typingIndicator: document.getElementById("typingIndicator"),
  statusText: document.getElementById("statusText"),
  errorBanner: document.getElementById("errorBanner"),
  newChatButton: document.getElementById("newChatButton"),
  clearHistoryButton: document.getElementById("clearHistoryButton"),
  detailsToggle: document.getElementById("detailsToggle"),
  detailsPanel: document.getElementById("detailsPanel"),
  stateInput: document.getElementById("stateInput"),
  districtInput: document.getElementById("districtInput"),
  caseStageInput: document.getElementById("caseStageInput"),
  ownMatterInput: document.getElementById("ownMatterInput"),
  imageUrlsInput: document.getElementById("imageUrlsInput"),
  fileInput: document.getElementById("fileInput"),
  currentUserName: document.getElementById("currentUserName"),
  currentUserMeta: document.getElementById("currentUserMeta"),
  logoutButton: document.getElementById("logoutButton"),
};

function formatTime(timestamp) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function nlToBr(text) {
  return escapeHtml(text).replace(/\n/g, "<br>");
}

function setBusy(isBusy) {
  state.isBusy = isBusy;
  elements.sendButton.disabled = isBusy;
  elements.messageInput.disabled = isBusy;
  elements.fileInput.disabled = isBusy;
  elements.typingIndicator.hidden = !isBusy;
  elements.statusText.textContent = isBusy ? "Lawyer AI is preparing a response..." : "Ready";
}

function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
}

function clearError() {
  elements.errorBanner.hidden = true;
  elements.errorBanner.textContent = "";
}

function getAuthToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setStoredUser(user) {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  state.currentUser = null;
}

function redirectToAuth() {
  window.location.href = `${frontendBase}/auth.html`;
}

async function requestJson(path, options = {}) {
  const token = getAuthToken();
  const headers = {
    ...(options.headers || {}),
  };
  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const response = await fetch(`${apiBase}${path}`, {
    ...options,
    credentials: options.credentials || "include",
    headers,
  });
  const payload = await response.json().catch(() => ({}));
  if (response.status === 401) {
    clearSession();
    redirectToAuth();
    throw new Error("Authentication required");
  }
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || `Request failed with status ${response.status}`);
  }
  return payload;
}

async function fetchCurrentUser() {
  return requestJson("/auth/me", {
    method: "GET",
    credentials: "include",
  });
}

function renderUserCard(user) {
  if (!elements.currentUserName || !elements.currentUserMeta) {
    return;
  }
  const displayName = user?.full_name || "Secure workspace";
  const meta = user?.state ? `${user.email} | ${user.state}` : (user?.email || "Signed in");
  elements.currentUserName.textContent = displayName;
  elements.currentUserMeta.textContent = meta;
}

async function ensureAuthenticated() {
  try {
    const user = await fetchCurrentUser();
    state.currentUser = user;
    setStoredUser(user);
    renderUserCard(user);
    return user;
  } catch (error) {
    clearSession();
    redirectToAuth();
    throw error;
  }
}

function buildMatterDetails() {
  return {
    state: elements.stateInput.value.trim() || null,
    district: elements.districtInput.value.trim() || null,
    case_stage: elements.caseStageInput.value.trim() || null,
    is_own_matter:
      elements.ownMatterInput.value === ""
        ? null
        : elements.ownMatterInput.value === "true",
  };
}

function setBlankDraft() {
  state.activeChatId = null;
  state.draftMessages = [];
  state.liveMessages = [];
  elements.messageInput.value = "";
  elements.fileInput.value = "";
  elements.imageUrlsInput.value = "";
  renderHistory();
  renderMessages();
}

function getRenderedMessages() {
  return state.activeChatId === null ? state.draftMessages : state.liveMessages;
}

function renderHistory() {
  elements.historyList.innerHTML = "";
  elements.emptyHistory.hidden = state.sessions.length > 0;

  state.sessions.forEach((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `history-item${state.activeChatId === session.id ? " active" : ""}`;
    button.innerHTML = `
      <span class="history-item-title">${escapeHtml(session.title)}</span>
      <span class="history-item-time">${formatTime(session.updated_at)}</span>
      <span class="history-item-preview">${escapeHtml(session.last_message_preview || "No preview yet")}</span>
    `;
    button.addEventListener("click", () => openChat(session.id));
    elements.historyList.appendChild(button);
  });
}

function buildMetaList(label, items) {
  if (!items || !items.length) {
    return "";
  }
  return `
    <div class="meta-block">
      <div class="meta-label">${escapeHtml(label)}</div>
      <ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </div>
  `;
}

function buildAssistantMeta(metadata) {
  if (!metadata || Object.keys(metadata).length === 0) {
    return "";
  }
  const parts = [
    metadata.follow_up_question
      ? `<div class="meta-block"><div class="meta-label">Next question</div><p>${escapeHtml(metadata.follow_up_question)}</p></div>`
      : "",
    metadata.likely_forum
      ? `<div class="meta-block"><div class="meta-label">Likely forum</div><p>${escapeHtml(metadata.likely_forum)}</p></div>`
      : "",
    metadata.caution
      ? `<div class="meta-block"><div class="meta-label">Caution</div><p>${escapeHtml(metadata.caution)}</p></div>`
      : "",
    buildMetaList("Authorities", metadata.authorities || []),
    buildMetaList("Documents to keep ready", metadata.documents_to_keep || []),
    buildMetaList("Citations", metadata.citations || []),
    buildMetaList("Warnings", metadata.warnings || []),
  ].filter(Boolean);

  if (!parts.length) {
    return "";
  }
  return `<div class="message-meta">${parts.join("")}</div>`;
}

function renderMessages() {
  const messages = getRenderedMessages();
  elements.chatThread.innerHTML = "";

  if (!messages.length) {
    elements.chatThread.innerHTML = `
      <section class="blank-state">
        <h2>Start a new legal guidance chat</h2>
        <p>Describe the issue in plain language. Lawyer AI will help first, then ask only the most useful next question if needed.</p>
        <div class="blank-state-examples">
          <button type="button" class="example-chip">I clicked a fake UPI link and money got debited.</button>
          <button type="button" class="example-chip">How do I file an FIR?</button>
          <button type="button" class="example-chip">My landlord is harassing me. What can I do?</button>
        </div>
      </section>
    `;
    elements.chatThread.querySelectorAll(".example-chip").forEach((button) => {
      button.addEventListener("click", () => {
        elements.messageInput.value = button.textContent || "";
        elements.messageInput.focus();
      });
    });
    return;
  }

  messages.forEach((message) => {
    const article = document.createElement("article");
    article.className = `message-bubble ${message.role === "user" ? "user" : "assistant"}`;
    article.innerHTML = `
      <div class="message-content">${nlToBr(message.content)}</div>
      ${message.role === "assistant" ? buildAssistantMeta(message.metadata || {}) : ""}
      <div class="message-footer">${formatTime(message.created_at)}</div>
    `;
    elements.chatThread.appendChild(article);
  });

  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
}

async function loadHistory() {
  const payload = await requestJson("/chat/history", { method: "GET", credentials: "include" });
  state.sessions = payload.items || [];
  renderHistory();
}

async function openChat(chatId) {
  clearError();
  const payload = await requestJson(`/chat/${chatId}/messages`, { method: "GET", credentials: "include" });
  state.activeChatId = chatId;
  state.liveMessages = (payload.items || []).map((item) => ({
    role: item.role,
    content: item.content,
    created_at: item.created_at,
    metadata: item.metadata || {},
  }));
  state.draftMessages = [];
  renderHistory();
  renderMessages();
}

function appendOptimisticUserMessage(message) {
  const entry = {
    role: "user",
    content: message,
    created_at: new Date().toISOString(),
    metadata: {},
  };
  if (state.activeChatId === null) {
    state.draftMessages = [...state.draftMessages, entry];
  } else {
    state.liveMessages = [...state.liveMessages, entry];
  }
  renderMessages();
}

function appendFailureMessage(errorMessage) {
  const entry = {
    role: "assistant",
    content: errorMessage,
    created_at: new Date().toISOString(),
    metadata: {},
  };
  if (state.activeChatId === null) {
    state.draftMessages = [...state.draftMessages, entry];
  } else {
    state.liveMessages = [...state.liveMessages, entry];
  }
  renderMessages();
}

async function submitMessage(event) {
  event.preventDefault();
  clearError();

  const message = elements.messageInput.value.trim();
  const hasFiles = elements.fileInput.files && elements.fileInput.files.length > 0;
  const imageUrls = elements.imageUrlsInput.value.trim();
  if (!message) {
    return;
  }

  appendOptimisticUserMessage(message);
  elements.messageInput.value = "";
  setBusy(true);

  try {
    const details = buildMatterDetails();
    let payload;

    if (hasFiles || imageUrls) {
      const formData = new FormData();
      formData.append("message", message);
      if (state.activeChatId !== null) {
        formData.append("chat_id", String(state.activeChatId));
      }
      Object.entries(details).forEach(([key, value]) => {
        if (value !== null && value !== undefined) {
          formData.append(key, String(value));
        }
      });
      if (imageUrls) {
        formData.append("image_urls", imageUrls);
      }
      Array.from(elements.fileInput.files).forEach((file) => formData.append("files", file));
      payload = await requestJson("/chat/upload", { method: "POST", body: formData, credentials: "include" });
    } else {
      payload = await requestJson("/chat", {
        method: "POST",
        body: JSON.stringify({
          message,
          chat_id: state.activeChatId,
          ...details,
        }),
        credentials: "include",
      });
    }

    const assistantMessage = {
      role: "assistant",
      content: payload.answer,
      created_at: payload.created_at,
      metadata: {
        follow_up_question: payload.follow_up_question,
        citations: payload.citations,
        warnings: payload.warnings,
        authorities: payload.authorities,
        documents_to_keep: payload.documents_to_keep,
        likely_forum: payload.likely_forum,
        caution: payload.caution,
      },
    };

    if (state.activeChatId === null) {
      state.activeChatId = payload.chat_id;
      state.liveMessages = [...state.draftMessages, assistantMessage];
      state.draftMessages = [];
    } else {
      state.liveMessages = [...state.liveMessages, assistantMessage];
    }

    elements.fileInput.value = "";
    elements.imageUrlsInput.value = "";
    await loadHistory();
    renderHistory();
    renderMessages();
  } catch (error) {
    const messageText = error instanceof Error ? error.message : "Something went wrong while sending the message.";
    showError(messageText);
    appendFailureMessage("I could not complete that request just now. Please try again in a moment.");
  } finally {
    setBusy(false);
  }
}

async function clearHistory() {
  const confirmed = window.confirm("Delete all chats and messages from local history?");
  if (!confirmed) {
    return;
  }
  clearError();
  await requestJson("/chat/history", { method: "DELETE", credentials: "include" });
  state.sessions = [];
  setBlankDraft();
}

async function logout() {
  try {
    await requestJson("/auth/logout", { method: "POST", credentials: "include" });
  } catch (error) {
    // Even if the backend session is already gone, finish local logout.
  } finally {
    clearSession();
    redirectToAuth();
  }
}

function bindEvents() {
  elements.composerForm.addEventListener("submit", submitMessage);
  elements.newChatButton.addEventListener("click", () => {
    clearError();
    setBlankDraft();
  });
  elements.clearHistoryButton.addEventListener("click", () => {
    clearHistory().catch((error) => {
      showError(error instanceof Error ? error.message : "Could not clear chat history.");
    });
  });
  elements.detailsToggle.addEventListener("click", () => {
    const expanded = elements.detailsToggle.getAttribute("aria-expanded") === "true";
    elements.detailsToggle.setAttribute("aria-expanded", String(!expanded));
    elements.detailsPanel.hidden = expanded;
  });
  if (elements.logoutButton) {
    elements.logoutButton.addEventListener("click", () => {
      logout().catch(() => {
        clearSession();
        redirectToAuth();
      });
    });
  }
  elements.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      elements.composerForm.requestSubmit();
    }
  });
}

async function boot() {
  bindEvents();
  await ensureAuthenticated();
  setBlankDraft();
  await loadHistory();
}

boot().catch((error) => {
  showError(error instanceof Error ? error.message : "Failed to load Lawyer AI.");
});
