const frontendBase = window.APP_CONFIG?.frontendBaseUrl || "/frontend";
const USER_KEY = 'legalAuthUser';
const TOKEN_KEY = 'legalAuthToken';
const ME_URL = "/auth/me";
const LOGOUT_URL = "/auth/logout";
const DEFAULT_REQUEST_TIMEOUT_MS = 30000;

const state = {
  sessions: [],
  activeChatId: null,
  draftMessages: [],
  liveMessages: [],
  isBusy: false,
  currentUser: null,
  sidebarOpen: true,
  desktopSidebarLayout: true,
};

const elements = {
  sidebar: document.getElementById("sidebar"),
  sidebarBackdrop: document.getElementById("sidebarBackdrop"),
  sidebarToggle: document.getElementById("sidebarToggle"),
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
  adminPanelLink: document.getElementById("adminPanelLink"),
  logoutButton: document.getElementById("logoutButton"),
};

function isDesktopSidebarLayout() {
  return window.innerWidth > 980;
}

function setSidebarOpen(isOpen) {
  state.sidebarOpen = isOpen;
  document.body.classList.toggle("sidebar-open", isOpen);
  document.body.classList.toggle("sidebar-closed", !isOpen);
  if (elements.sidebarToggle) {
    elements.sidebarToggle.setAttribute("aria-expanded", String(isOpen));
  }
  if (elements.sidebarBackdrop) {
    elements.sidebarBackdrop.hidden = !isOpen;
  }
}

function syncSidebarForViewport(force = false) {
  const desktopLayout = isDesktopSidebarLayout();
  if (force || desktopLayout !== state.desktopSidebarLayout) {
    setSidebarOpen(desktopLayout);
  }
  state.desktopSidebarLayout = desktopLayout;
}

function closeSidebarIfOverlayMode() {
  if (!isDesktopSidebarLayout()) {
    setSidebarOpen(false);
  }
}

function setElementText(element, text) {
  if (element) {
    element.textContent = text;
  }
}

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
  if (elements.sendButton) {
    elements.sendButton.disabled = isBusy;
    elements.sendButton.textContent = isBusy ? "Sending..." : "Send";
  }
  if (elements.messageInput) {
    elements.messageInput.disabled = isBusy;
  }
  if (elements.fileInput) {
    elements.fileInput.disabled = isBusy;
  }
  if (elements.typingIndicator) {
    elements.typingIndicator.hidden = !isBusy;
  }
  setElementText(elements.statusText, isBusy ? "Lawyer AI is preparing a response..." : "Ready");
}

function showError(message) {
  if (!elements.errorBanner) {
    return;
  }
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
}

function errorTextFromPayload(payload, fallbackMessage) {
  if (!payload) {
    return fallbackMessage;
  }
  if (typeof payload.detail === "string" && payload.detail.trim()) {
    return payload.detail;
  }
  if (Array.isArray(payload.detail) && payload.detail.length > 0) {
    const first = payload.detail[0];
    if (typeof first === "string" && first.trim()) {
      return first;
    }
    if (first && typeof first.msg === "string" && first.msg.trim()) {
      return first.msg;
    }
  }
  if (typeof payload.message === "string" && payload.message.trim()) {
    return payload.message;
  }
  return fallbackMessage;
}

function clearError() {
  if (!elements.errorBanner) {
    return;
  }
  elements.errorBanner.hidden = true;
  elements.errorBanner.textContent = "";
}

function safeAuthUser(user) {
  if (!user || typeof user !== "object") {
    return null;
  }
  return {
    id: user.id,
    full_name: user.full_name,
    email: user.email,
    role: user.role === "admin" ? "admin" : "user",
    status: user.status === "blocked" ? "blocked" : "active",
    state: user.state || null,
    created_at: user.created_at,
  };
}

function setStoredUser(user) {
  const safeUser = safeAuthUser(user);
  if (safeUser) {
    localStorage.setItem(USER_KEY, JSON.stringify(safeUser));
  }
}

function getStoredUser() {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) {
    return null;
  }
  try {
    return safeAuthUser(JSON.parse(raw));
  } catch (error) {
    localStorage.removeItem(USER_KEY);
    return null;
  }
}

function getStoredToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

function clearSession() {
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(TOKEN_KEY);
  state.currentUser = null;
}

function redirectToAuth() {
  window.location.replace(`${frontendBase}/auth.html`);
}

async function requestJson(path, options = {}) {
  const headers = {
    ...(options.headers || {}),
  };
  const token = getStoredToken();
  const controller = new AbortController();
  const timeoutMs = options.timeoutMs || DEFAULT_REQUEST_TIMEOUT_MS;
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  if (token && !headers.Authorization) {
    headers.Authorization = `Bearer ${token}`;
  }
  try {
    const response = await fetch(path, {
      ...options,
      credentials: options.credentials || "include",
      headers,
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({}));
    if (response.status === 401) {
      clearSession();
      redirectToAuth();
      throw new Error("Authentication required");
    }
    if (!response.ok) {
      throw new Error(errorTextFromPayload(payload, options.timeoutMessage || `Request failed with status ${response.status}`));
    }
    return payload;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(options.timeoutMessage || "Request timed out. Please try again.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
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

function renderAdminNavigation(user) {
  if (!elements.adminPanelLink) {
    return;
  }
  elements.adminPanelLink.hidden = user?.role !== "admin";
}

async function ensureAuthenticated() {
  const storedUser = getStoredUser();
  if (storedUser) {
    state.currentUser = storedUser;
    renderUserCard(storedUser);
    renderAdminNavigation(storedUser);
  }

  try {
    const user = safeAuthUser(await fetchCurrentUser());
    state.currentUser = user;
    setStoredUser(user);
    renderUserCard(user);
    renderAdminNavigation(user);
    return user;
  } catch (error) {
    if (!storedUser) {
      clearSession();
      redirectToAuth();
    }
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

function clearCaseDetails() {
  if (elements.stateInput) {
    elements.stateInput.value = "";
  }
  if (elements.districtInput) {
    elements.districtInput.value = "";
  }
  if (elements.caseStageInput) {
    elements.caseStageInput.value = "";
  }
  if (elements.ownMatterInput) {
    elements.ownMatterInput.value = "";
  }
}

function setBlankDraft() {
  state.activeChatId = null;
  state.draftMessages = [];
  state.liveMessages = [];
  if (elements.messageInput) {
    elements.messageInput.value = "";
  }
  if (elements.fileInput) {
    elements.fileInput.value = "";
  }
  if (elements.imageUrlsInput) {
    elements.imageUrlsInput.value = "";
  }
  clearCaseDetails();
  renderHistory();
  renderMessages();
}

function getRenderedMessages() {
  return state.activeChatId === null ? state.draftMessages : state.liveMessages;
}

function renderHistory() {
  if (!elements.historyList || !elements.emptyHistory) {
    return;
  }
  elements.historyList.innerHTML = "";
  elements.emptyHistory.hidden = state.sessions.length > 0;

  state.sessions.forEach((session) => {
    const item = document.createElement("div");
    item.className = `history-item${state.activeChatId === session.id ? " active" : ""}`;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item-open";
    button.innerHTML = `
      <span class="history-item-title">${escapeHtml(session.title)}</span>
      <span class="history-item-time">${formatTime(session.updated_at)}</span>
      <span class="history-item-preview">${escapeHtml(session.last_message_preview || "No preview yet")}</span>
    `;
    button.addEventListener("click", () => openChat(session.id));

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "history-item-delete";
    deleteButton.setAttribute("aria-label", `Delete chat ${session.title}`);
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteChat(session.id).catch((error) => {
        showError(error instanceof Error ? error.message : "Could not delete this chat.");
      });
    });

    item.appendChild(button);
    item.appendChild(deleteButton);
    elements.historyList.appendChild(item);
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
  if (!elements.chatThread) {
    return;
  }
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
    const feedbackControls = message.role === "assistant" && message.id
      ? `<div class="message-feedback" data-message-id="${message.id}">
          <button type="button" class="feedback-button" data-rating="up">Thumbs up</button>
          <button type="button" class="feedback-button" data-rating="down">Thumbs down</button>
        </div>`
      : "";
    article.innerHTML = `
      <div class="message-content">${nlToBr(message.content)}</div>
      ${message.role === "assistant" ? buildAssistantMeta(message.metadata || {}) : ""}
      ${feedbackControls}
      <div class="message-footer">${formatTime(message.created_at)}</div>
    `;
    elements.chatThread.appendChild(article);
  });

  elements.chatThread.querySelectorAll(".feedback-button").forEach((button) => {
    button.addEventListener("click", submitMessageFeedback);
  });

  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
}

async function submitMessageFeedback(event) {
  const button = event.currentTarget;
  const container = button.closest(".message-feedback");
  const messageId = container?.dataset.messageId;
  const rating = button.dataset.rating;
  if (!messageId || !rating) {
    return;
  }
  let comment = "";
  if (rating === "down") {
    comment = window.prompt("Optional: what felt unhelpful?", "") || "";
  }
  container.querySelectorAll(".feedback-button").forEach((item) => {
    item.disabled = true;
  });
  try {
    await requestJson(`/chat/messages/${encodeURIComponent(messageId)}/feedback`, {
      method: "POST",
      body: JSON.stringify({ rating, comment: comment.trim() || null }),
      credentials: "include",
    });
    const status = document.createElement("span");
    status.className = "feedback-status";
    status.textContent = "Feedback saved";
    container.appendChild(status);
  } catch (error) {
    container.querySelectorAll(".feedback-button").forEach((item) => {
      item.disabled = false;
    });
    showError(error instanceof Error ? error.message : "Could not save feedback.");
  }
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
    id: item.id,
    role: item.role,
    content: item.content,
    created_at: item.created_at,
    metadata: item.metadata || {},
  }));
  state.draftMessages = [];
  renderHistory();
  renderMessages();
  closeSidebarIfOverlayMode();
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
  if (state.isBusy) {
    return;
  }
  clearError();

  const message = elements.messageInput.value.trim();
  const hasFiles = Boolean(elements.fileInput && elements.fileInput.files && elements.fileInput.files.length > 0);
  const imageUrls = elements.imageUrlsInput ? elements.imageUrlsInput.value.trim() : "";
  if (!message) {
    return;
  }

  appendOptimisticUserMessage(message);
  if (elements.messageInput) {
    elements.messageInput.value = "";
  }
  setBusy(true);
  const fallbackAnswer = "I could not complete that request. Please try again in a moment.";
  const timeoutAnswer = "The request took too long. Please try again with a shorter message or fewer uploads.";

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
      Array.from(elements.fileInput?.files || []).forEach((file) => formData.append("files", file));
      payload = await requestJson("/chat/upload", {
        method: "POST",
        body: formData,
        credentials: "include",
        timeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
        timeoutMessage: timeoutAnswer,
      });
    } else {
      payload = await requestJson("/chat", {
        method: "POST",
        body: JSON.stringify({
          message,
          chat_id: state.activeChatId,
          ...details,
        }),
        credentials: "include",
        timeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
        timeoutMessage: timeoutAnswer,
      });
    }

    const assistantMessage = {
      id: payload.assistant_message_id,
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

    if (elements.fileInput) {
      elements.fileInput.value = "";
    }
    if (elements.imageUrlsInput) {
      elements.imageUrlsInput.value = "";
    }
    await loadHistory();
    renderHistory();
    renderMessages();
  } catch (error) {
    const messageText = error instanceof Error ? error.message : fallbackAnswer;
    showError(messageText);
    appendFailureMessage(messageText);
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
  setBlankDraft();
  await loadHistory();
}

async function deleteChat(chatId) {
  const session = state.sessions.find((item) => item.id === chatId);
  const title = session?.title || "this chat";
  const confirmed = window.confirm(`Delete only ${title}?`);
  if (!confirmed) {
    return;
  }
  clearError();
  await requestJson(`/chat/${chatId}`, { method: "DELETE", credentials: "include" });
  state.sessions = state.sessions.filter((item) => item.id !== chatId);
  if (state.activeChatId === chatId) {
    setBlankDraft();
  } else {
    renderHistory();
  }
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
  if (elements.composerForm) {
    elements.composerForm.addEventListener("submit", submitMessage);
  }
  if (elements.newChatButton) {
    elements.newChatButton.addEventListener("click", () => {
      clearError();
      setBlankDraft();
      closeSidebarIfOverlayMode();
    });
  }
  if (elements.sidebarToggle) {
    elements.sidebarToggle.addEventListener("click", () => {
      setSidebarOpen(!state.sidebarOpen);
    });
  }
  if (elements.sidebarBackdrop) {
    elements.sidebarBackdrop.addEventListener("click", () => {
      setSidebarOpen(false);
    });
  }
  if (elements.clearHistoryButton) {
    elements.clearHistoryButton.addEventListener("click", () => {
      clearHistory().catch((error) => {
        showError(error instanceof Error ? error.message : "Could not clear chat history.");
      });
    });
  }
  if (elements.detailsToggle && elements.detailsPanel) {
    elements.detailsToggle.addEventListener("click", () => {
      const expanded = elements.detailsToggle.getAttribute("aria-expanded") === "true";
      elements.detailsToggle.setAttribute("aria-expanded", String(!expanded));
      elements.detailsPanel.hidden = expanded;
    });
  }
  if (elements.logoutButton) {
    elements.logoutButton.addEventListener("click", () => {
      logout().catch(() => {
        clearSession();
        redirectToAuth();
      });
    });
  }
  if (elements.messageInput && elements.composerForm) {
    elements.messageInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        elements.composerForm.requestSubmit();
      }
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.sidebarOpen && !isDesktopSidebarLayout()) {
      setSidebarOpen(false);
    }
  });
  window.addEventListener("resize", syncSidebarForViewport);
}

async function boot() {
  syncSidebarForViewport(true);
  bindEvents();
  await ensureAuthenticated();
  setBlankDraft();
  await loadHistory();
}

boot().catch((error) => {
  showError(error instanceof Error ? error.message : "Failed to load Lawyer AI.");
});
