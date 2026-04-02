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
const LOGIN_URL = `${BASE_API_URL}/auth/login`;
const SIGNUP_URL = `${BASE_API_URL}/auth/signup`;
const GOOGLE_LOGIN_URL = `${BASE_API_URL}/auth/google/login`;
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
    await handleOAuthCallback();

    if (localStorage.getItem(TOKEN_KEY)) {
        window.location.href = `${FRONTEND_BASE_URL}/Index.html`;
    }
});

function bindEvents() {
    loginTab.addEventListener('click', () => switchTab('login'));
    signupTab.addEventListener('click', () => switchTab('signup'));
    googleLoginBtn.addEventListener('click', startGoogleLogin);
    forgotPasswordLink.addEventListener('click', openPasswordResetModal);
    closeModal.addEventListener('click', closePasswordResetModal);
    loginForm.addEventListener('submit', handleLogin);
    signupForm.addEventListener('submit', handleSignup);
    passwordResetForm.addEventListener('submit', handlePasswordReset);

    // Close modal when clicking outside
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
    authMessage.textContent = '';
}

function showMessage(text, isError = false) {
    authMessage.textContent = text;
    authMessage.classList.toggle('error', isError);
}

async function submitJson(url, payload) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || `Request failed with status ${response.status}`);
        }
        return data;
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error('Request timed out. Please check your connection and try again.');
        }
        throw error;
    }
}

function saveSession(data) {
    localStorage.setItem(TOKEN_KEY, data.token);
    localStorage.setItem(USER_KEY, JSON.stringify(data.user));
}

async function checkGoogleAuthStatus() {
    try {
        const response = await fetch(`${BASE_API_URL}/auth/google/status`);
        if (response.ok) {
            const data = await response.json();
            if (!data.configured) {
                googleLoginBtn.disabled = true;
                googleLoginBtn.textContent = 'Google Sign-in (Not Configured)';
                googleLoginBtn.style.opacity = '0.5';
                googleLoginBtn.style.cursor = 'not-allowed';
                googleLoginBtn.title = 'Google sign-in is not configured. Please contact the administrator.';
            }
        } else {
            // If we can't check status, disable the button
            googleLoginBtn.disabled = true;
            googleLoginBtn.textContent = 'Google Sign-in (Unavailable)';
            googleLoginBtn.style.opacity = '0.5';
            googleLoginBtn.style.cursor = 'not-allowed';
        }
    } catch (error) {
        // If there's an error checking status, disable the button
        googleLoginBtn.disabled = true;
        googleLoginBtn.textContent = 'Google Sign-in (Unavailable)';
        googleLoginBtn.style.opacity = '0.5';
        googleLoginBtn.style.cursor = 'not-allowed';
    }
}


async function fetchCurrentUser(token) {
    const response = await fetch(ME_URL, {
        method: 'GET',
        headers: {
            Authorization: `Bearer ${token}`
        }
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.detail || `Request failed with status ${response.status}`);
    }
    return data;
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
        let errorMessage = 'Google sign-in failed. Please try again.';
        if (error === 'google_auth_not_configured') {
            errorMessage = 'Google sign-in is not configured. Please contact the administrator or use email/password login.';
        } else if (error === 'google_auth_timeout') {
            errorMessage = 'Google sign-in timed out. Please try again.';
        } else if (error === 'google_auth_request_failed') {
            errorMessage = 'Network error during Google sign-in. Please check your connection and try again.';
        }
        showMessage(errorMessage, true);
        clearOAuthQueryParams();
        return;
    }

    if (!token) {
        return;
    }

    try {
        localStorage.setItem(TOKEN_KEY, token);
        const user = await fetchCurrentUser(token);
        localStorage.setItem(USER_KEY, JSON.stringify(user));
        clearOAuthQueryParams();
        window.location.href = `${FRONTEND_BASE_URL}/Index.html`;
    } catch (oauthError) {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        showMessage('Google sign-in succeeded but session setup failed. Please try again.', true);
        clearOAuthQueryParams();
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
    const loginBtn = document.querySelector('#loginForm button[type="submit"]');
    const originalText = loginBtn.textContent;
    loginBtn.textContent = 'Logging in...';
    loginBtn.disabled = true;
    try {
        const data = await submitJson(LOGIN_URL, {
            email: document.getElementById('loginEmail').value.trim(),
            password: document.getElementById('loginPassword').value
        });
        saveSession(data);
        window.location.href = `${FRONTEND_BASE_URL}/Index.html`;
    } catch (error) {
        showMessage(error.message, true);
    } finally {
        loginBtn.textContent = originalText;
        loginBtn.disabled = false;
    }
}

async function handleSignup(event) {
    event.preventDefault();
    const signupBtn = document.querySelector('#signupForm button[type="submit"]');
    const originalText = signupBtn.textContent;
    signupBtn.textContent = 'Creating account...';
    signupBtn.disabled = true;
    try {
        const data = await submitJson(SIGNUP_URL, {
            full_name: document.getElementById('signupName').value.trim(),
            email: document.getElementById('signupEmail').value.trim(),
            password: document.getElementById('signupPassword').value,
            state: signupState.value || null
        });
        saveSession(data);
        window.location.href = `${FRONTEND_BASE_URL}/Index.html`;
    } catch (error) {
        showMessage(error.message, true);
    } finally {
        signupBtn.textContent = originalText;
        signupBtn.disabled = false;
    }
}

function openPasswordResetModal(event) {
    event.preventDefault();
    passwordResetModal.style.display = 'block';
    resetMessage.textContent = '';
}

function closePasswordResetModal() {
    passwordResetModal.style.display = 'none';
    passwordResetForm.reset();
}

async function handlePasswordReset(event) {
    event.preventDefault();
    const email = document.getElementById('resetEmail').value.trim();

    try {
        const response = await fetch(PASSWORD_RESET_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email })
        });

        if (response.ok) {
            showResetMessage('Password reset link sent to your email!', false);
            setTimeout(() => {
                closePasswordResetModal();
            }, 3000);
        } else {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || `Request failed with status ${response.status}`);
        }
    } catch (error) {
        showResetMessage(error.message, true);
    }
}

function showResetMessage(text, isError = false) {
    resetMessage.textContent = text;
    resetMessage.classList.toggle('error', isError);
}
