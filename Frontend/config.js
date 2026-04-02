(function initializeAppConfig() {
    const isHttp = window.location.protocol === 'http:' || window.location.protocol === 'https:';
    const isServedByBackend = isHttp && (
        window.location.port === '8000' ||
        window.location.hostname === '127.0.0.1' ||
        window.location.hostname === 'localhost'
    );

    const apiBaseUrl = isServedByBackend
        ? window.location.origin
        : 'http://127.0.0.1:8000';

    window.APP_CONFIG = {
        apiBaseUrl,
        frontendBaseUrl: `${apiBaseUrl}/frontend`,
    };
})();
