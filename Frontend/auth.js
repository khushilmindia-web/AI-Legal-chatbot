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
const passwordResetConfirmForm = document.getElementById('passwordResetConfirmForm');
const resetMessage = document.getElementById('resetMessage');
const passwordResetTitle = document.getElementById('passwordResetTitle');
const closeModal = document.getElementsByClassName('close')[0];

const FRONTEND_BASE_URL = window.APP_CONFIG?.frontendBaseUrl || '/frontend';
const LOGIN_URL = '/auth/login';
const SIGNUP_URL = '/auth/signup';
const GOOGLE_LOGIN_URL = '/auth/google/login';
const GOOGLE_STATUS_URL = '/auth/google/status';
const PASSWORD_RESET_URL = '/auth/password-reset';
const PASSWORD_RESET_CONFIRM_URL = '/auth/password-reset/confirm';
const ME_URL = '/auth/me';
const USER_KEY = 'legalAuthUser';
const TOKEN_KEY = 'legalAuthToken';

const GOOGLE_CONFIG = {
    backendConfigured: false,
    clientIdConfigured: false,
    gisScriptLoaded: false
};

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
    initializePasswordResetMode();

    const handledLegacyCallback = await handleOAuthCallback();
    if (!handledLegacyCallback && !getResetTokenFromUrl()) {
        await redirectIfAuthenticated();
    }
});

function redirectToApp() {
    window.location.replace('/frontend/index.html');
}

function clearSession() {
    localStorage.removeItem(USER_KEY);
    localStorage.removeItem(TOKEN_KEY);
}

function safeAuthUser(user) {
    if (!user || typeof user !== 'object') {
        return null;
    }
    return {
        id: user.id,
        full_name: user.full_name,
        email: user.email,
        role: user.role === 'admin' ? 'admin' : 'user',
        status: user.status === 'blocked' ? 'blocked' : 'active',
        state: user.state || null,
        created_at: user.created_at
    };
}

function saveSession(data) {
    if (data.token) {
        localStorage.setItem(TOKEN_KEY, data.token);
    }
    const user = safeAuthUser(data.user);
    if (user) {
        localStorage.setItem(USER_KEY, JSON.stringify(user));
    }
}

function getStoredToken() {
    return localStorage.getItem(TOKEN_KEY) || '';
}

function getStoredUser() {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) {
        return null;
    }
    try {
        return JSON.parse(raw);
    } catch (error) {
        localStorage.removeItem(USER_KEY);
        return null;
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
    passwordResetConfirmForm.addEventListener('submit', handlePasswordResetConfirm);

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
    const token = getStoredToken();

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(token ? { Authorization: `Bearer ${token}` } : {})
            },
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

async function fetchCurrentUser() {
    const token = getStoredToken();
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    try {
        const response = await fetch(ME_URL, {
            method: 'GET',
            credentials: 'include',
            headers: token ? { Authorization: `Bearer ${token}` } : {},
            signal: controller.signal
        });

        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || `Request failed with status ${response.status}`);
        }
        return data;
    } catch (error) {
        if (error.name === 'AbortError') {
            throw new Error('Session check timed out. Please log in again.');
        }
        throw error;
    } finally {
        clearTimeout(timeoutId);
    }
}

async function redirectIfAuthenticated() {
    const token = getStoredToken();
    const storedUser = getStoredUser();
    if (token && storedUser) {
        redirectToApp();
        return;
    }

    try {
        const user = await fetchCurrentUser();
        saveSession({ token, user });
        redirectToApp();
    } catch (error) {
        clearSession();
    }
}

async function checkGoogleAuthStatus() {
    GOOGLE_CONFIG.gisScriptLoaded = typeof window.google !== 'undefined';

    try {
        const response = await fetch(GOOGLE_STATUS_URL, { credentials: 'include' });
        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            googleLoginBtn.disabled = false;
            googleLoginBtn.textContent = 'Continue with Google';
            googleLoginBtn.title = 'Google sign-in status could not be verified, but you can still try the OAuth redirect.';
            return;
        }

        GOOGLE_CONFIG.backendConfigured = Boolean(data.configured);
        GOOGLE_CONFIG.clientIdConfigured = Boolean(data.client_id_configured);

        if (!data.configured) {
            googleLoginBtn.disabled = true;
            googleLoginBtn.textContent = 'Google Sign-in Unavailable';
            googleLoginBtn.title = data.client_id_configured
                ? 'Google sign-in is partially configured, but the backend OAuth setup is incomplete.'
                : 'Google sign-in is not configured because the Google client ID is missing.';
            return;
        }

        googleLoginBtn.disabled = false;
        googleLoginBtn.textContent = 'Continue with Google';
        googleLoginBtn.title = GOOGLE_CONFIG.gisScriptLoaded
            ? 'Google OAuth is configured and ready.'
            : 'Google Identity Services script is not loaded, so the page will use the backend OAuth redirect flow.';
    } catch (error) {
        googleLoginBtn.disabled = false;
        googleLoginBtn.textContent = 'Continue with Google';
        googleLoginBtn.title = 'Google sign-in status could not be reached. The backend redirect flow is still available.';
    }
}

function clearOAuthQueryParams() {
    const cleanUrl = `${window.location.origin}${window.location.pathname}`;
    window.history.replaceState({}, document.title, cleanUrl);
}

async function handleOAuthCallback() {
    const params = new URLSearchParams(window.location.search);
    const error = params.get('error');

    if (error) {
        const errorMessages = {
            google_auth_not_configured: 'Google sign-in is not configured. Please use email/password login.',
            google_auth_timeout: 'Google sign-in timed out. Please try again.',
            google_auth_config_invalid: 'Google sign-in is misconfigured. Check the Google client credentials and authorized redirect URI.',
            google_auth_request_failed: 'Network error during Google sign-in. Please try again.',
            google_auth_state_invalid: 'Google sign-in state expired. Please try again.',
            google_auth_incomplete: 'Google sign-in did not complete. Please try again.',
            google_auth_failed: 'Google sign-in failed. Please try again.'
        };
        showMessage(errorMessages[error] || 'Google sign-in failed. Please try again.', true);
        clearOAuthQueryParams();
        return true;
    }

    return false;
}

function startGoogleLogin(event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    if (googleLoginBtn.disabled) {
        showMessage('Google sign-in is not configured. Please use email/password login.', true);
        return;
    }
    window.location.replace(GOOGLE_LOGIN_URL);
}

async function handleLogin(event) {
    event.preventDefault();
    event.stopPropagation();

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
    event.stopPropagation();

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
    setPasswordResetMode(getResetTokenFromUrl() ? 'confirm' : 'request');
    passwordResetModal.style.display = 'grid';
    passwordResetModal.setAttribute('aria-hidden', 'false');
    resetMessage.textContent = '';
    resetMessage.classList.remove('visible', 'error');
}

function closePasswordResetModal() {
    passwordResetModal.style.display = 'none';
    passwordResetModal.setAttribute('aria-hidden', 'true');
    passwordResetForm.reset();
    passwordResetConfirmForm.reset();

    if (!getResetTokenFromUrl()) {
        setPasswordResetMode('request');
    }
}

function getResetTokenFromUrl() {
    return new URLSearchParams(window.location.search).get('reset_token');
}

function clearResetQueryParams() {
    const url = new URL(window.location.href);
    url.searchParams.delete('reset_token');
    window.history.replaceState({}, document.title, `${url.pathname}${url.search}${url.hash}`);
}

function setPasswordResetMode(mode) {
    const isConfirmMode = mode === 'confirm';
    passwordResetTitle.textContent = isConfirmMode ? 'Choose a New Password' : 'Reset Password';
    passwordResetForm.classList.toggle('hidden', isConfirmMode);
    passwordResetConfirmForm.classList.toggle('hidden', !isConfirmMode);
}

function initializePasswordResetMode() {

    if (getResetTokenFromUrl()) {
        setPasswordResetMode('confirm');
        passwordResetModal.style.display = 'grid';
        passwordResetModal.setAttribute('aria-hidden', 'false');
    } 
    
    else {
        setPasswordResetMode('request');
    }
}

async function handlePasswordReset(event) {
    event.preventDefault();
    event.stopPropagation();
    const email = document.getElementById('resetEmail').value.trim();
    const resetBtn = document.querySelector('#passwordResetForm .auth-submit');
    const originalText = resetBtn.textContent;
    resetBtn.textContent = 'Sending...';
    resetBtn.disabled = true;
    showResetMessage('');

    try {
        const data = await submitJson(PASSWORD_RESET_URL, { email });
        showResetMessage(data.message || 'Password reset link sent to your email.');
        setTimeout(() => {
            closePasswordResetModal();
        }, 2200);
    } 
    
    catch (error) {
        showResetMessage(error.message, true);
    } finally {
        resetBtn.textContent = originalText;
        resetBtn.disabled = false;
    }
}

async function handlePasswordResetConfirm(event) {
    event.preventDefault();
    event.stopPropagation();

    const token = getResetTokenFromUrl();
    const password = document.getElementById('resetNewPassword').value;
    const confirmPassword = document.getElementById('resetConfirmPassword').value;

    if (!token) {
        showResetMessage('Reset link is missing or invalid.', true);
        return;
    }
    
    if (password.length < 8) {
        showResetMessage('Password must be at least 8 characters.', true);
        return;
    }
    
    if (password !== confirmPassword) {
        showResetMessage('Passwords do not match.', true);
        return;
    }

    const confirmBtn = document.querySelector('#passwordResetConfirmForm .auth-submit');
    const originalText = confirmBtn.textContent;
    confirmBtn.textContent = 'Updating...';
    confirmBtn.disabled = true;
    showResetMessage('');

    try {
        const data = await submitJson(PASSWORD_RESET_CONFIRM_URL, { token, password });
        showResetMessage(data.message || 'Your password has been reset.');
        clearResetQueryParams();
        setTimeout(() => {
            closePasswordResetModal();
            setPasswordResetMode('request');
            switchTab('login');
            showMessage('Password updated. Please log in with your new password.');
        }, 1500);
    } 
    
    catch (error) {
        showResetMessage(error.message, true);
    } finally {
        confirmBtn.textContent = originalText;
        confirmBtn.disabled = false;
    }
}
