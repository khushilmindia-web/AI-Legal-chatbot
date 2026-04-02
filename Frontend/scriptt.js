const chatBox = document.getElementById('chatBox');
const userInput = document.getElementById('userInput');

const sendBtn = document.getElementById('sendBtn');
const historyList = document.getElementById('historyList');

const API_URL = '/chat';

let currentChatMessages = [];
let chatHistory = JSON.parse(localStorage.getItem('chatHistory')) || [];

window.addEventListener('load', () => {
    chatBox.innerHTML = '';
    addWelcomeMessage();
    displayHistory();
    currentChatMessages = [];

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {

        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});

window.addEventListener('beforeunload', () => {
    if (currentChatMessages.length > 0) {
        saveCurrentChatToHistory();
    }
});

function addWelcomeMessage() {
    const welcome = document.createElement('div');

    welcome.classList.add('message', 'bot');
    welcome.textContent = "Hello! I'm your legal assistant. Ask me anything about the Indian Constitution or other legal topics.";
    chatBox.appendChild(welcome);
}

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');

    messageDiv.classList.add('message', sender);
    messageDiv.textContent = text;
    chatBox.appendChild(messageDiv);

    chatBox.scrollTop = chatBox.scrollHeight;
    currentChatMessages.push({ sender, text });
}

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.classList.add('typing-indicator');
    indicator.id = 'typingIndicator';

    for (let i = 0; i < 3; i++) {
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
    sendBtn.textContent = isLoading ? '...' : 'Send';
}

function saveCurrentChatToHistory() {
    if (currentChatMessages.length === 0) return;

    const newConversation = {
        id: Date.now(),
        messages: [...currentChatMessages]
    };

    chatHistory.unshift(newConversation);
    if (chatHistory.length > 50) chatHistory.pop();

    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    displayHistory();
}

function newChat() {

    if (currentChatMessages.length > 0) {
        saveCurrentChatToHistory();
    }

    chatBox.innerHTML = '';
    addWelcomeMessage();
    currentChatMessages = [];
}

function loadChatFromHistory(id) {
    const conversation = chatHistory.find(item => item.id === id);

    if (!conversation) return;

    if (
        currentChatMessages.length > 0 &&
        !chatHistory.some(c =>
            JSON.stringify(c.messages) === JSON.stringify(currentChatMessages)
        )
    ) {
        saveCurrentChatToHistory();
    }
    const index = chatHistory.findIndex(item => item.id === id);

    if (index !== -1) {
        chatHistory.splice(index, 1);
    }

    chatHistory.unshift(conversation);
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    chatBox.innerHTML = '';
    conversation.messages.forEach(msg => {

        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', msg.sender);
        messageDiv.textContent = msg.text;
        chatBox.appendChild(messageDiv);
    });

    currentChatMessages = [...conversation.messages];
    chatBox.scrollTop = chatBox.scrollHeight;

    displayHistory();
}

function deleteHistoryItem(id, event) {
    event.stopPropagation();
    chatHistory = chatHistory.filter(item => item.id !== id);

    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    displayHistory();
}

function clearAllHistory() {

    if (confirm('Are you sure you want to clear all history?')) {
        chatHistory = [];
        localStorage.removeItem('chatHistory');
        displayHistory();
    }
}

function displayHistory() {
    historyList.innerHTML = '';

    if (chatHistory.length === 0) {
        historyList.innerHTML = '<div class="empty-history">No previous chats</div>';
        return;
    }

    chatHistory.forEach(conv => {
        const firstUserMsg = conv.messages.find(m => m.sender === 'user')?.text || 'Chat';
        const preview = firstUserMsg.length > 40 ? firstUserMsg.substring(0, 40) + '…' : firstUserMsg;

        const item = document.createElement('div');
        item.className = 'history-item';
        item.onclick = () => loadChatFromHistory(conv.id);

        item.innerHTML = `<span class="history-item-preview">${preview}</span>
        <span class="delete-history" onclick="deleteHistoryItem(${conv.id}, event)">×</span>`;
        historyList.appendChild(item);
    });
}

function toggleSidebar() {
    document.body.classList.toggle('sidebar-open');
    const toggleBtn = document.getElementById('toggleBtn');

    if (document.body.classList.contains('sidebar-open')) {
        toggleBtn.textContent = '✕';
    } 

    else {
        toggleBtn.textContent = '☰';
    }
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    userInput.value = '';
    addMessage(message, 'user');
    setLoadingState(true);
    showTypingIndicator();

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        const botReply = `${data.response}\n\n(source: ${data.source}, confidence: ${data.confidence.toFixed(2)})`;

        removeTypingIndicator();
        addMessage(botReply, 'bot');
    } 

    catch (error) {
        console.error('Fetch error:', error);
        removeTypingIndicator();

        let errorMessage = 'Sorry, something went wrong. Please try again.';
    
        if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Cannot connect to the Lawyer AI server.';
        }

        addMessage(errorMessage, 'bot');
    } 

    finally {
        setLoadingState(false);
        userInput.focus();
    }
}
