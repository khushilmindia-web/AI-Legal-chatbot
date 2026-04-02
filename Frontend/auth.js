const loginTab = document.getElementById('loginTab');
const signupTab = document.getElementById('signupTab');
const loginForm = document.getElementById('loginForm');
const signupForm = document.getElementById('signupForm');
const authMessage = document.getElementById('authMessage');
const signupState = document.getElementById('signupState');
const googleLoginBtn = document.getElementById('googleLoginBtn');
const forgotPasswordLink = document.getElementById('forgotPasswordLink');
const passwordResetModal = document.getElementById('passwordResetModal');
const passwordResetForm = document.getElementById('passwordResetForm');
const resetMessage = document.getElementById('resetMessage');
const closeModal = document.getElementsByClassName('close')[0];

const BASE_API_URL = window.APP_CONFIG?.apiBaseUrl || 'http://127.0.0.1:8000';
const FRONTEND_BASE_URL = window.APP_CONFIG?.frontendBaseUrl || `${BASE_API_URL}/frontend`;
const APP_URL = `${FRONTEND_BASE_URL}/index.html`;
const LOGIN_URL = `${BASE_API_URL}/auth/login`;
const SIGNUP_URL = `${BASE_API_URL}/auth/signup`;
const GOOGLE_LOGIN_URL = `${BASE_API_URL}/auth/google/login`;
const GOOGLE_STATUS_URL = `${BASE_API_URL}/auth/google/status`;
const PASSWORD_RESET_URL = `${BASE_API_URL}/auth/password-reset`;
const ME_URL = `${BASE_API_URL}/auth/me`;
const TOKEN_KEY = 'legalAuthToken';
const USER_KEY = 'legalAuthUser';

const STATES_AND_UTS = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
    'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
    'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu',
    'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry'
];

window.addEventListener('load', async () => {
    populateStates();
    bindEvents();
    await checkGoogleAuthStatus();
    const handledLegacyCallback = await handleOAuthCallback();
    if (!handledLegacyCallback) {
        await redirectIfAuthenticated();
    }
});

function buildAuthHeaders(token) {
    const headers = {};
    if (token) {
        headers.Authorization = `Bearer ${token}`;
    }
    return headers;
}

function redirectToApp() {
    window.location.href = APP_URL;
}

function clearSession() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
}

function saveSession(data) {
    if (data.token) {
        localStorage.setItem(TOKEN_KEY, data.token);
    }
    if (data.user) {
        localStorage.setItem(USER_KEY, JSON.stringify(data.user));
    }
}

function bindEvents() {
    loginTab.addEventListener('click', () => switchTab('login'));
    signupTab.addEventListener('click', () => switchTab('signup'));
    googleLoginBtn.addEventListener('click', startGoogleLogin);
    forgotPasswordLink.addEventListener('click', openPasswordResetModal);
    closeModal.addEventListener('click', closePasswordResetModal);
    loginForm.addEventListener('submit', handleLogin);
    signupForm.addEventListener('submit', handleSignup);
    passwordResetForm.addEventListener('submit', handlePasswordReset);

    window.addEventListener('click', (event) => {
        if (event.target === passwordResetModal) {
            closePasswordResetModal();
        }
    });
}

function populateStates() {
    STATES_AND_UTS.forEach((state) => {
        const option = document.createElement('option');
        option.value = state;
        option.textContent = state;
        signupState.appendChild(option);
    });
}

function switchTab(mode) {
    const loginMode = mode === 'login';
    loginTab.classList.toggle('active', loginMode);
    signupTab.classList.toggle('active', !loginMode);
    loginForm.classList.toggle('hidden', !loginMode);
    signupForm.classList.toggle('hidden', loginMode);
    showMessage('');
}

function showMessage(text, isError = false) {
    authMessage.textContent = text;
    authMessage.classList.toggle('error', isError);
    authMessage.classList.toggle('visible', Boolean(text));
}

function showResetMessage(text, isError = false) {
    resetMessage.textContent = text;
    resetMessage.classList.toggle('error', isError);
    resetMessage.classList.toggle('visible', Boolean(text));
}

async function submitJson(url, payload) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(payload),
            signal: controller.signal
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || `Request failed with status ${response.status}`);
        }
        return data;
    } catch (error) {
        if (error.name === 'AbortError') {
            throw new Error('Request timed out. Please check your connection and try again.');
        }
        throw error;
    } finally {
        clearTimeout(timeoutId);
    }
}

async function fetchCurrentUser(token) {
    const response = await fetch(ME_URL, {
        method: 'GET',
        credentials: 'include',
        headers: buildAuthHeaders(token)
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.detail || `Request failed with status ${response.status}`);
    }
    return data;
}

async function redirectIfAuthenticated() {
    const storedToken = localStorage.getItem(TOKEN_KEY);
    if (storedToken) {
        try {
            const user = await fetchCurrentUser(storedToken);
            saveSession({ token: storedToken, user });
            redirectToApp();
            return;
        } catch (error) {
            clearSession();
        }
    }

    try {
        const user = await fetchCurrentUser();
        saveSession({ user });
        redirectToApp();
    } catch (error) {
        // No active cookie session. Stay on auth page.
    }
}

async function checkGoogleAuthStatus() {
    try {
        const response = await fetch(GOOGLE_STATUS_URL, { credentials: 'include' });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.configured) {
            googleLoginBtn.disabled = true;
            googleLoginBtn.textContent = data.configured === false ? 'Google Sign-in Unavailable' : 'Google Sign-in Offline';
            googleLoginBtn.title = 'Google sign-in is not configured right now.';
        }
    } catch (error) {
        googleLoginBtn.disabled = true;
        googleLoginBtn.textContent = 'Google Sign-in Offline';
        googleLoginBtn.title = 'Google sign-in could not be reached.';
    }
}

function clearOAuthQueryParams() {
    const cleanUrl = `${window.location.origin}${window.location.pathname}`;
    window.history.replaceState({}, document.title, cleanUrl);
}

async function handleOAuthCallback() {
    const params = new URLSearchParams(window.location.search);
    const token = params.get('token');
    const error = params.get('error');

    if (error) {
        const errorMessages = {
            google_auth_not_configured: 'Google sign-in is not configured. Please use email/password login.',
            google_auth_timeout: 'Google sign-in timed out. Please try again.',
            google_auth_request_failed: 'Network error during Google sign-in. Please try again.',
            google_auth_state_invalid: 'Google sign-in state expired. Please try again.',
            google_auth_incomplete: 'Google sign-in did not complete. Please try again.',
            google_auth_failed: 'Google sign-in failed. Please try again.'
        };
        showMessage(errorMessages[error] || 'Google sign-in failed. Please try again.', true);
        clearOAuthQueryParams();
        return true;
    }

    if (!token) {
        return false;
    }

    try {
        const user = await fetchCurrentUser(token);
        saveSession({ token, user });
        clearOAuthQueryParams();
        redirectToApp();
        return true;
    } catch (oauthError) {
        clearSession();
        showMessage('Google sign-in succeeded but the session could not be restored. Please try again.', true);
        clearOAuthQueryParams();
        return true;
    }
}

function startGoogleLogin() {
    if (googleLoginBtn.disabled) {
        showMessage('Google sign-in is not configured. Please use email/password login.', true);
        return;
    }
    window.location.href = GOOGLE_LOGIN_URL;
}

async function handleLogin(event) {
    event.preventDefault();
    const loginBtn = document.querySelector('#loginForm .auth-submit');
    const originalText = loginBtn.textContent;
    loginBtn.textContent = 'Logging in...';
    loginBtn.disabled = true;
    showMessage('');
    try {
        const data = await submitJson(LOGIN_URL, {
            email: document.getElementById('loginEmail').value.trim(),
            password: document.getElementById('loginPassword').value
        });
        saveSession(data);
        redirectToApp();
    } catch (error) {
        showMessage(error.message, true);
    } finally {
        loginBtn.textContent = originalText;
        loginBtn.disabled = false;
    }
}

async function handleSignup(event) {
    event.preventDefault();
    const signupBtn = document.querySelector('#signupForm .auth-submit');
    const originalText = signupBtn.textContent;
    signupBtn.textContent = 'Creating account...';
    signupBtn.disabled = true;
    showMessage('');
    try {
        const data = await submitJson(SIGNUP_URL, {
            full_name: document.getElementById('signupName').value.trim(),
            email: document.getElementById('signupEmail').value.trim(),
            password: document.getElementById('signupPassword').value,
            state: signupState.value || null
        });
        saveSession(data);
        redirectToApp();
    } catch (error) {
        showMessage(error.message, true);
    } finally {
        signupBtn.textContent = originalText;
        signupBtn.disabled = false;
    }
}

function openPasswordResetModal(event) {
    event.preventDefault();
    passwordResetModal.style.display = 'grid';
    passwordResetModal.setAttribute('aria-hidden', 'false');
    resetMessage.textContent = '';
    resetMessage.classList.remove('visible', 'error');
}

function closePasswordResetModal() {
    passwordResetModal.style.display = 'none';
    passwordResetModal.setAttribute('aria-hidden', 'true');
    passwordResetForm.reset();
}

async function handlePasswordReset(event) {
    event.preventDefault();
    const email = document.getElementById('resetEmail').value.trim();

    try {
        const data = await submitJson(PASSWORD_RESET_URL, { email });
        showResetMessage(data.message || 'Password reset link sent to your email.');
        setTimeout(() => {
            closePasswordResetModal();
        }, 2200);
    } catch (error) {
        showResetMessage(error.message, true);
    }
}
