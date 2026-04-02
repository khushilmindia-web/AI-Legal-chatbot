const chatBox = document.getElementById('chatBox');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const uploadBtn = document.getElementById('uploadBtn');
const documentInput = document.getElementById('documentInput');
const historyList = document.getElementById('historyList');
const matterStatus = document.getElementById('matterStatus');
const currentUserName = document.getElementById('currentUserName');
const currentUserMeta = document.getElementById('currentUserMeta');
const logoutBtn = document.getElementById('logoutBtn');

const API_BASE_URL = window.APP_CONFIG?.apiBaseUrl || 'http://127.0.0.1:8000';
const FRONTEND_BASE_URL = window.APP_CONFIG?.frontendBaseUrl || `${API_BASE_URL}/frontend`;
const CHAT_URL = `${API_BASE_URL}/chat`;
const HISTORY_URL = `${API_BASE_URL}/chat/history`;
const CHAT_SESSION_URL = `${API_BASE_URL}/chat/session`;
const RESET_MATTER_URL = `${API_BASE_URL}/chat/matter/reset`;
const DOCUMENTS_URL = `${API_BASE_URL}/chat/documents`;
const ME_URL = `${API_BASE_URL}/auth/me`;
const LOGOUT_URL = `${API_BASE_URL}/auth/logout`;
const TOKEN_KEY = 'legalAuthToken';
const USER_KEY = 'legalAuthUser';

let transcriptItems = [];
let activeChatId = null;
let chatSessions = [];
let activeMatterProfile = {
    state: null,
    district: null,
    matter_type: null,
    matter_stage: null,
    is_own_matter: null,
    urgency: 'medium'
};
let initializationStarted = false;

function getStoredUser() {
    return JSON.parse(localStorage.getItem(USER_KEY) || '{}');
}

function setStoredUser(user) {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
}

function clearSession() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
}

function resetActiveMatterProfile() {
    const storedUser = getStoredUser();
    activeMatterProfile = {
        state: storedUser.state || null,
        district: null,
        matter_type: null,
        matter_stage: null,
        is_own_matter: null,
        urgency: 'medium'
    };
}

function resetDraftChat() {
    transcriptItems = [];
    activeChatId = null;
    resetActiveMatterProfile();
    if (documentInput) {
        documentInput.value = '';
    }
    chatBox.innerHTML = '';
    addWelcomeMessage();
    renderMatterStatus();
    renderHistoryList();
}

function formatSessionDate(value) {
    if (!value || typeof value !== 'string') {
        return 'Recently';
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
        return 'Recently';
    }
    return parsed.toLocaleString();
}

function deriveMatterProfileFromTranscript(items) {
    const storedUser = getStoredUser();
    const profile = {
        state: storedUser.state || null,
        district: null,
        matter_type: null,
        matter_stage: null,
        is_own_matter: null,
        urgency: 'medium'
    };

    for (let index = items.length - 1; index >= 0; index -= 1) {
        const metadata = items[index] && items[index].metadata ? items[index].metadata : {};
        if (!profile.state && typeof metadata.state === 'string' && metadata.state) profile.state = metadata.state;
        if (!profile.district && typeof metadata.district === 'string' && metadata.district) profile.district = metadata.district;
        if (!profile.matter_type && typeof metadata.matter_type === 'string' && metadata.matter_type) profile.matter_type = metadata.matter_type;
        if (!profile.matter_stage && typeof metadata.matter_stage === 'string' && metadata.matter_stage) profile.matter_stage = metadata.matter_stage;
        if (typeof profile.is_own_matter !== 'boolean' && typeof metadata.is_own_matter === 'boolean') profile.is_own_matter = metadata.is_own_matter;
        if ((profile.urgency === 'medium' || !profile.urgency) && typeof metadata.urgency === 'string' && metadata.urgency) profile.urgency = metadata.urgency;
    }
    return profile;
}

window.addEventListener('load', async () => {
    if (initializationStarted) return;
    initializationStarted = true;
    bindEvents();

    const token = localStorage.getItem(TOKEN_KEY);
    if (!token) {
        window.location.href = `${FRONTEND_BASE_URL}/auth.html`;
        return;
    }

    try {
        const user = await fetchCurrentUser();
        renderUserCard(user);
        resetActiveMatterProfile();
        await loadHistory();
        resetDraftChat();
    } catch (error) {
        console.error('Initialization failed:', error);
        clearSession();
        window.location.href = `${FRONTEND_BASE_URL}/auth.html`;
    }
});

function bindEvents() {
    sendBtn.addEventListener('click', sendMessage);
    logoutBtn.addEventListener('click', logout);
    if (uploadBtn) {
        uploadBtn.addEventListener('click', uploadDocuments);
    }
    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendMessage();
        }
    });
}

function getAuthToken() {
    return localStorage.getItem(TOKEN_KEY);
}

async function fetchJson(url, options = {}) {
    const token = getAuthToken();
    const headers = {
        ...(options.headers || {})
    };
    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
    }
    if (token) {
        headers.Authorization = `Bearer ${token}`;
    }

    const response = await fetch(url, { ...options, headers });
    if (response.status === 401) {
        clearSession();
        window.location.href = `${FRONTEND_BASE_URL}/auth.html`;
        throw new Error('Unauthorized');
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.detail || `Request failed with status ${response.status}`);
    }
    return data;
}

async function fetchCurrentUser() {
    const user = await fetchJson(ME_URL, { method: 'GET' });
    setStoredUser(user);
    return user;
}

function renderUserCard(user) {
    currentUserName.textContent = user.full_name;
    currentUserMeta.textContent = user.state ? `${user.email} | ${user.state}` : user.email;
}

async function loadHistory() {
    const data = await fetchJson(HISTORY_URL, { method: 'GET' });
    chatSessions = Array.isArray(data.items) ? data.items : [];
    const activeExists = chatSessions.some((session) => session.id === activeChatId);
    if (!activeExists) activeChatId = null;
    renderHistoryList();
}

async function loadChatMessages(chatId) {
    if (!chatId) return;
    activeChatId = chatId;
    const data = await fetchJson(`${CHAT_URL}/${chatId}/messages`, { method: 'GET' });
    transcriptItems = Array.isArray(data.items) ? data.items : [];
    activeMatterProfile = deriveMatterProfileFromTranscript(transcriptItems);
    renderTranscript();
    renderMatterStatus();
    renderHistoryList();
}

function addWelcomeMessage() {
    const welcome = document.createElement('div');
    welcome.classList.add('message', 'bot');
    welcome.textContent = 'Please describe your legal issue in your own words. I will collect the matter details one step at a time and then provide the legal guidance.';
    chatBox.appendChild(welcome);
}

function createMessageElement(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    messageDiv.textContent = text;
    return messageDiv;
}

function renderTranscript() {
    chatBox.innerHTML = '';
    if (transcriptItems.length === 0) {
        addWelcomeMessage();
        return;
    }
    transcriptItems.forEach((item) => {
        const sender = item.role === 'user' ? 'user' : 'bot';
        const text = sender === 'bot' ? formatStoredAssistantReply(item) : item.message;
        chatBox.appendChild(createMessageElement(text, sender));
    });
    chatBox.scrollTop = chatBox.scrollHeight;
}

function renderHistoryList() {
    historyList.innerHTML = '';
    if (!chatSessions || chatSessions.length === 0) {
        historyList.innerHTML = '<div class="empty-history">No previous chats. Click New Chat to start.</div>';
        return;
    }
    chatSessions.forEach((session) => {
        const displayTitle = session.title || `Chat ${session.id}`;
        const preview = session.last_user_message
            ? (session.last_user_message.length > 44 ? `${session.last_user_message.slice(0, 44)}...` : session.last_user_message)
            : 'No messages yet';
        const item = document.createElement('div');
        item.className = 'history-item';
        if (activeChatId === session.id) item.classList.add('active');
        item.onclick = () => loadChatMessages(session.id);
        item.innerHTML = `<strong>${displayTitle}</strong> <span class="small-text">(${formatSessionDate(session.last_activity || session.created_at)})</span><br /><span class="history-item-preview">${preview}</span>`;
        historyList.appendChild(item);
    });
}

function renderResearchBrief(brief) {
    if (!brief || typeof brief !== 'object') return [];
    const parts = [];
    if (Array.isArray(brief.primary_authorities) && brief.primary_authorities.length > 0) {
        parts.push(`Research authorities:\n- ${brief.primary_authorities.join('\n- ')}`);
    }
    if (Array.isArray(brief.recommended_evidence) && brief.recommended_evidence.length > 0) {
        parts.push(`Key evidence to collect:\n- ${brief.recommended_evidence.join('\n- ')}`);
    }
    if (Array.isArray(brief.forum_strategy) && brief.forum_strategy.length > 0) {
        parts.push(`Forum strategy:\n- ${brief.forum_strategy.join('\n- ')}`);
    }
    if (Array.isArray(brief.uploaded_material) && brief.uploaded_material.length > 0) {
        parts.push(`Uploaded material considered:\n- ${brief.uploaded_material.join('\n- ')}`);
    }
    return parts;
}

function formatBotReply(data) {
    const parts = [data.response || 'I am not sure how to answer that yet.'];
    if (data.issue_category) parts.push(`Issue category: ${data.issue_category}`);
    if (data.forum_hint) parts.push(`Forum hint: ${data.forum_hint}`);
    if (Array.isArray(data.citations) && data.citations.length > 0) parts.push(`Citations:\n- ${data.citations.join('\n- ')}`);
    if (Array.isArray(data.source_snippets) && data.source_snippets.length > 0) parts.push(`Source snippets:\n- ${data.source_snippets.join('\n- ')}`);
    if (data.research_brief) parts.push(...renderResearchBrief(data.research_brief));
    if (data.lawyer_workflow && Object.keys(data.lawyer_workflow).length > 0) {
        const workflow = data.lawyer_workflow;
        if (Array.isArray(workflow.chronology) && workflow.chronology.length > 0) parts.push(`Chronology:\n- ${workflow.chronology.join('\n- ')}`);
        if (Array.isArray(workflow.issue_framing) && workflow.issue_framing.length > 0) parts.push(`Issue framing:\n- ${workflow.issue_framing.join('\n- ')}`);
        if (Array.isArray(workflow.draft_notice_points) && workflow.draft_notice_points.length > 0) parts.push(`Draft notice points:\n- ${workflow.draft_notice_points.join('\n- ')}`);
        if (Array.isArray(workflow.complaint_checklist) && workflow.complaint_checklist.length > 0) parts.push(`Complaint checklist:\n- ${workflow.complaint_checklist.join('\n- ')}`);
        if (Array.isArray(workflow.filing_readiness_review) && workflow.filing_readiness_review.length > 0) parts.push(`Filing readiness review:\n- ${workflow.filing_readiness_review.join('\n- ')}`);
    }
    if (Array.isArray(data.uploaded_documents) && data.uploaded_documents.length > 0) {
        parts.push(`Uploaded documents:\n- ${data.uploaded_documents.map((item) => item.original_name || 'document').join('\n- ')}`);
    }
    if (data.disclaimer) parts.push(`Disclaimer: ${data.disclaimer}`);
    return parts.join('\n\n');
}

function formatStoredAssistantReply(item) {
    const metadata = item.metadata || {};
    return formatBotReply({
        response: item.message,
        issue_category: metadata.issue_category,
        forum_hint: metadata.forum_hint,
        next_steps: metadata.next_steps || [],
        suggested_questions: metadata.suggested_questions || [],
        citations: metadata.citations || [],
        source_snippets: metadata.source_snippets || [],
        disclaimer: metadata.disclaimer,
        lawyer_workflow: metadata.lawyer_workflow || {},
        research_brief: metadata.research_brief || {},
        uploaded_documents: metadata.uploaded_documents || [],
        confidence: item.confidence,
        source: item.source
    });
}

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.classList.add('typing-indicator');
    indicator.id = 'typingIndicator';
    for (let index = 0; index < 3; index += 1) {
        const dot = document.createElement('span');
        indicator.appendChild(dot);
    }
    chatBox.appendChild(indicator);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();
}

function setLoadingState(isLoading) {
    userInput.disabled = isLoading;
    sendBtn.disabled = isLoading;
    if (uploadBtn) uploadBtn.disabled = isLoading;
    sendBtn.textContent = isLoading ? '...' : 'Send';
}

function renderMatterStatus(data = null) {
    if (!matterStatus) return;
    const profile = { ...activeMatterProfile, ...(data || {}) };
    const items = [];
    if (profile.state) items.push(`State: ${profile.state}`);
    if (profile.district) items.push(`District: ${profile.district}`);
    if (profile.matter_type) items.push(`Matter: ${profile.matter_type}`);
    if (profile.matter_stage && profile.matter_stage !== 'not_sure') items.push(`Stage: ${profile.matter_stage}`);
    if (typeof profile.is_own_matter === 'boolean') items.push(profile.is_own_matter ? 'Own matter' : 'Someone else\'s matter');
    matterStatus.textContent = items.length > 0
        ? `Collected details: ${items.join(' | ')}`
        : 'Matter collection is now step-by-step. The assistant will ask only one question at a time.';
}

function applyServerProfileToForm(data) {
    if (!data || typeof data !== 'object') return;
    if (typeof data.state === 'string' && data.state) activeMatterProfile.state = data.state;
    if (typeof data.district === 'string') activeMatterProfile.district = data.district || null;
    if (typeof data.matter_type === 'string' && data.matter_type) activeMatterProfile.matter_type = data.matter_type;
    if (typeof data.matter_stage === 'string' && data.matter_stage) activeMatterProfile.matter_stage = data.matter_stage;
    if (typeof data.is_own_matter === 'boolean') activeMatterProfile.is_own_matter = data.is_own_matter;
    if (typeof data.urgency === 'string' && data.urgency) activeMatterProfile.urgency = data.urgency;
    renderMatterStatus(data);
}

async function uploadDocuments() {
    if (!activeChatId) {
        chatBox.appendChild(createMessageElement('Please send your first message before uploading documents to this chat.', 'bot'));
        chatBox.scrollTop = chatBox.scrollHeight;
        return;
    }
    if (!documentInput || !documentInput.files || documentInput.files.length === 0) return;

    const token = getAuthToken();
    const files = Array.from(documentInput.files);
    setLoadingState(true);
    try {
        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('chat_id', String(activeChatId));
            const response = await fetch(DOCUMENTS_URL, {
                method: 'POST',
                headers: token ? { Authorization: `Bearer ${token}` } : {},
                body: formData
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(data.detail || `Upload failed for ${file.name}`);
            chatBox.appendChild(createMessageElement(`Document uploaded: ${data.original_name}`, 'bot'));
        }
        documentInput.value = '';
        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (error) {
        chatBox.appendChild(createMessageElement(error.message || 'Document upload failed.', 'bot'));
        chatBox.scrollTop = chatBox.scrollHeight;
    } finally {
        setLoadingState(false);
    }
}

async function newChat() {
    resetDraftChat();
}

async function clearChatHistory() {
    const confirmed = window.confirm('This will permanently delete all chats and messages. Do you want to continue?');
    if (!confirmed) return;
    try {
        await fetchJson(HISTORY_URL, { method: 'DELETE' });
        chatSessions = [];
        resetDraftChat();
    } catch (error) {
        chatBox.appendChild(createMessageElement(error.message || 'Failed to clear chat history.', 'bot'));
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

function toggleSidebar() {
    document.body.classList.toggle('sidebar-open');
    const toggleBtn = document.getElementById('toggleBtn');
    toggleBtn.textContent = document.body.classList.contains('sidebar-open') ? 'x' : '=';
}

async function logout() {
    try {
        await fetchJson(LOGOUT_URL, { method: 'POST' });
    } catch (error) {
        console.error('Logout error:', error);
    } finally {
        clearSession();
        window.location.href = `${FRONTEND_BASE_URL}/auth.html`;
    }
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    const payload = {
        message,
        state: activeMatterProfile.state || null,
        district: activeMatterProfile.district || null,
        matter_type: activeMatterProfile.matter_type || null,
        matter_stage: activeMatterProfile.matter_stage || 'not_sure',
        is_own_matter: typeof activeMatterProfile.is_own_matter === 'boolean' ? activeMatterProfile.is_own_matter : null,
        urgency: activeMatterProfile.urgency || 'medium'
    };

    userInput.value = '';
    chatBox.appendChild(createMessageElement(message, 'user'));
    chatBox.scrollTop = chatBox.scrollHeight;
    setLoadingState(true);
    showTypingIndicator();

    try {
        if (activeChatId) payload.chat_id = activeChatId;
        const data = await fetchJson(CHAT_URL, {
            method: 'POST',
            body: JSON.stringify(payload)
        });
        if (data.chat_id) activeChatId = data.chat_id;
        applyServerProfileToForm(data);
        removeTypingIndicator();
        chatBox.appendChild(createMessageElement(formatBotReply(data), 'bot'));
        chatBox.scrollTop = chatBox.scrollHeight;
        await loadHistory();
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator();
        chatBox.appendChild(createMessageElement(error.message || 'Something went wrong.', 'bot'));
        chatBox.scrollTop = chatBox.scrollHeight;
    } finally {
        setLoadingState(false);
        userInput.focus();
    }
}
