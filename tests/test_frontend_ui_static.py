from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_new_chat_resets_optional_case_detail_inputs():
    script = (ROOT / "Frontend" / "app.js").read_text(encoding="utf-8")

    assert "function clearCaseDetails()" in script

    set_blank_draft = script.split("function setBlankDraft()", 1)[1].split(
        "function getRenderedMessages()", 1
    )[0]
    assert "clearCaseDetails();" in set_blank_draft

    for field_name in ["stateInput", "districtInput", "caseStageInput", "ownMatterInput"]:
        assert f"elements.{field_name}" in script
        assert f"elements.{field_name}.value = \"\";" in script


def test_chat_ui_shows_admin_panel_link_only_for_admin_role():
    markup = (ROOT / "Frontend" / "Index.html").read_text(encoding="utf-8")
    app_script = (ROOT / "Frontend" / "app.js").read_text(encoding="utf-8")
    auth_script = (ROOT / "Frontend" / "auth.js").read_text(encoding="utf-8")

    assert 'id="adminPanelLink"' in markup
    assert 'href="/frontend/admin.html"' in markup
    assert "hidden>Admin Panel</a>" in markup
    assert "adminPanelLink: document.getElementById(\"adminPanelLink\")" in app_script
    assert "function renderAdminNavigation(user)" in app_script
    assert "elements.adminPanelLink.hidden = user?.role !== \"admin\";" in app_script
    assert "renderAdminNavigation(storedUser);" in app_script
    assert "renderAdminNavigation(user);" in app_script

    for script in [app_script, auth_script]:
        assert "function safeAuthUser(user)" in script
        assert "role: user.role ===" in script
        for safe_field in ["id", "full_name", "email", "role", "status", "state", "created_at"]:
            assert safe_field in script
        for sensitive_field in ["password_hash", "token_hash", "google_sub", "auth_provider", "auth_session_token"]:
            assert sensitive_field not in script

    assert "window.location.replace('/frontend/index.html')" in auth_script
    assert "admin.html" not in auth_script


def test_admin_frontend_shell_is_read_only_and_loads_stats():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")

    assert "admin.js" in markup
    assert 'id="dashboard"' in markup
    assert 'id="analytics-overview"' in markup
    assert 'id="analytics-trends"' in markup
    assert 'id="query-categories"' in markup
    assert 'id="fallback-reasons"' in markup
    assert 'id="quality-score"' in markup
    assert 'id="low-quality-review"' in markup
    assert 'id="system-status"' in markup
    assert 'id="system-health"' in markup
    assert 'id="config"' in markup
    assert 'id="quality-logs"' in markup
    assert 'id="direct-answer-content"' in markup
    assert 'id="feedback"' in markup
    assert 'id="activity"' in markup
    assert 'id="users"' in markup
    assert 'id="chats"' in markup
    assert "/admin/me" in script
    assert "/admin/stats" in script
    assert 'method: options.method || "GET"' in script
    assert "credentials: \"include\"" in script

    forbidden_actions = ["delete", "edit"]
    combined = f"{markup}\n{script}".lower()
    for action in forbidden_actions:
        assert action not in combined


def test_admin_navigation_and_responsive_layout_are_polished_without_write_actions():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    for section_id in [
        "dashboard",
        "analytics-overview",
        "analytics-trends",
        "query-categories",
        "fallback-reasons",
        "quality-score",
        "low-quality-review",
        "users",
        "chats",
        "quality-logs",
        "system-status",
        "system-health",
        "config",
        "direct-answer-content",
        "feedback",
        "activity",
    ]:
        assert f'href="#{section_id}"' in markup
        assert f'id="{section_id}"' in markup

    assert 'aria-current="page"' in markup
    assert "adminNavLinks" in script
    assert "adminSections" in script
    assert "setActiveAdminSection" in script
    assert "setupAdminNavigation()" in script
    assert "IntersectionObserver" in script
    assert "dataset.label" in script
    assert "setTableCell" in script

    for responsive_hook in [
        ".admin-sidebar",
        ".admin-nav",
        ".admin-nav-link[aria-current=\"page\"]",
        ".admin-table td::before",
        "grid-template-columns: minmax(92px, 34%) minmax(0, 1fr)",
        "overflow-x: auto",
        "white-space: nowrap",
        "@media (max-width: 980px)",
        "@media (max-width: 720px)",
    ]:
        assert responsive_hook in styles

    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["approve", "delete", "edit"]:
        assert forbidden_action not in combined
    for sensitive_field in ["password_hash", "token_hash", "google_sub", "auth_provider", "auth_session_token"]:
        assert sensitive_field not in script


def test_admin_users_list_ui_is_read_only_and_uses_safe_fields():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")

    assert "/admin/users" in script
    assert 'id="userSearchForm"' in markup
    assert 'id="userSearchInput"' in markup
    assert 'id="userLimitSelect"' in markup
    assert 'id="usersTableBody"' in markup
    assert 'id="usersStatus"' in markup
    assert 'id="usersPrevButton"' in markup
    assert 'id="usersNextButton"' in markup
    assert "loadUsers()" in script
    assert "renderUsers(payload)" in script

    assert 'method: "PATCH"' in script
    assert "/admin/users" in script
    assert "/status" in script
    assert "updateUserStatus" in script
    assert "Block" in script
    assert "Unblock" in script
    for safe_field in ["full_name", "email", "role", "status", "created_at", "last_active"]:
        assert safe_field in script

    for sensitive_field in ["password_hash", "token_hash", "google_sub", "auth_provider"]:
        assert sensitive_field not in script


def test_admin_response_audit_ui_is_read_only_and_filters_quality_metadata():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "/admin/users" in script
    assert "/admin/chats" in script
    assert 'id="chatLookupForm"' in markup
    assert 'id="userChatsList"' in markup
    assert 'id="messageAuditFilter"' in markup
    assert 'value="fallback"' in markup
    assert 'value="low_confidence"' in markup
    assert 'value="unsupported_output"' in markup
    assert 'id="chatMessagesList"' in markup

    for quality_field in [
        "route_type",
        "confidence",
        "fallback_reason",
        "validation_flags",
        "disclaimer_mode",
        "source_sufficiency",
    ]:
        assert quality_field in script

    assert "risk-fallback" in script
    assert "risk-low-confidence" in script
    assert "risk-unsupported" in script
    assert ".admin-quality-chip" in styles
    assert ".admin-message-card.risk-fallback" in styles
    assert ".admin-message-card.risk-unsupported" in styles
    assert 'method: options.method || "GET"' in script
    assert "textContent" in script

    for sensitive_field in ["password_hash", "token_hash", "google_sub", "auth_provider", "auth_session_token"]:
        assert sensitive_field not in script


def test_admin_system_status_ui_is_read_only_and_hides_configuration_values():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "/admin/system/status" in script
    assert 'id="system-status"' in markup
    assert 'id="refreshSystemStatusButton"' in markup
    assert 'id="systemStatusState"' in markup
    assert 'id="systemStatusList"' in markup
    assert "loadSystemStatus()" in script
    assert "renderSystemStatus(payload)" in script
    assert "Loading system status..." in script

    for service_label in ["MongoDB", "India Kanoon", "Google Custom Search", "OpenAI", "SMTP"]:
        assert service_label in script

    for indicator in [".admin-status-card.green", ".admin-status-card.yellow", ".admin-status-card.red"]:
        assert indicator in styles

    assert 'method: options.method || "GET"' in script
    assert "textContent" in script
    for sensitive_term in [
        "api_key",
        "secret",
        "password",
        "connection",
        "mongodb_uri",
        "smtp_username",
        "smtp_from_email",
    ]:
        assert sensitive_term not in script.lower()


def test_admin_system_health_ui_is_read_only_and_auto_refreshable():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "/admin/system/health" in script
    assert 'href="#system-health"' in markup
    assert 'id="system-health"' in markup
    assert 'id="refreshSystemHealthButton"' in markup
    assert 'id="systemHealthState"' in markup
    assert 'id="systemHealthSummary"' in markup
    assert 'id="systemHealthAutoRefresh"' in markup
    assert "loadSystemHealth()" in script
    assert "renderSystemHealth(payload)" in script
    assert "setSystemHealthAutoRefresh" in script
    assert "window.setInterval(loadSystemHealth, 30000)" in script
    assert "Loading system health..." in script
    for label in [
        "Overall",
        "API uptime",
        "Database",
        "AI providers",
        "Chat average",
        "Active sessions",
        "Memory",
        "CPU",
    ]:
        assert label in script
    for status in ["healthy", "warning", "critical"]:
        assert status in script
    assert ".admin-status-card.green" in styles
    assert ".admin-status-card.yellow" in styles
    assert ".admin-status-card.red" in styles
    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["restart", "shutdown", "websocket"]:
        assert forbidden_action not in combined
    for sensitive_term in [
        "api_key",
        "secret",
        "password",
        "mongodb_uri",
        "connection",
        "internal path",
        "token_hash",
        "auth_session_token",
    ]:
        assert sensitive_term not in script.lower()


def test_admin_analytics_overview_ui_is_read_only_and_aggregated_only():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")

    assert "/admin/analytics/overview" in script
    assert 'href="#analytics-overview"' in markup
    assert 'id="analytics-overview"' in markup
    assert 'id="analyticsOverviewForm"' in markup
    assert 'id="analyticsPeriodSelect"' in markup
    assert 'id="refreshAnalyticsOverviewButton"' in markup
    assert 'id="analyticsOverviewStatus"' in markup
    assert 'id="analyticsOverviewSummary"' in markup
    for period in ['value="24h"', 'value="7d"', 'value="30d"']:
        assert period in markup
    assert "loadAnalyticsOverview()" in script
    assert "renderAnalyticsOverview(payload)" in script
    assert "Loading analytics overview..." in script
    assert "Analytics overview could not be loaded." in script
    assert "new URLSearchParams({ period: adminState.analyticsPeriod || \"24h\" })" in script
    for label in [
        "Total chats",
        "Total messages",
        "Active users",
        "Fallbacks",
        "Unsupported responses",
        "Average confidence",
        "Average response time",
    ]:
        assert label in script
    for safe_field in [
        "total_chats",
        "total_messages",
        "active_users",
        "fallback_count",
        "unsupported_response_count",
        "average_confidence",
        "average_response_time_ms",
    ]:
        assert safe_field in script
    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["export", "download"]:
        assert forbidden_action not in combined
    for sensitive_field in [
        "password_hash",
        "token_hash",
        "google_sub",
        "auth_provider",
        "auth_session_token",
    ]:
        assert sensitive_field not in script


def test_admin_analytics_trends_ui_is_read_only_and_aggregated_only():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "/admin/analytics/trends" in script
    assert 'href="#analytics-trends"' in markup
    assert 'id="analytics-trends"' in markup
    assert 'id="analyticsTrendsForm"' in markup
    assert 'id="analyticsTrendsPeriodSelect"' in markup
    assert 'id="refreshAnalyticsTrendsButton"' in markup
    assert 'id="analyticsTrendsStatus"' in markup
    assert 'id="analyticsTrendsList"' in markup
    assert 'value="7d"' in markup
    assert 'value="30d"' in markup
    assert 'value="24h"' in markup
    assert "loadAnalyticsTrends()" in script
    assert "renderAnalyticsTrends(payload)" in script
    assert "Loading analytics trends..." in script
    assert "No trend data found for this period." in script
    assert "Analytics trends could not be loaded." in script
    assert "new URLSearchParams({ period: adminState.analyticsTrendsPeriod || \"7d\" })" in script
    for label in ["Chats", "Messages", "Fallbacks", "Avg confidence"]:
        assert label in script
    for safe_field in ["date", "chats", "messages", "fallback_count", "average_confidence"]:
        assert safe_field in script
    assert ".admin-trend-row" in styles
    assert ".admin-trend-bar" in styles
    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["export", "download"]:
        assert forbidden_action not in combined
    for sensitive_field in ["password_hash", "token_hash", "google_sub", "auth_provider", "auth_session_token"]:
        assert sensitive_field not in script


def test_admin_query_categories_ui_is_read_only_and_aggregated_only():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "/admin/analytics/query-categories" in script
    assert 'href="#query-categories"' in markup
    assert 'id="query-categories"' in markup
    assert 'id="queryCategoriesForm"' in markup
    assert 'id="queryCategoriesPeriodSelect"' in markup
    assert 'id="refreshQueryCategoriesButton"' in markup
    assert 'id="queryCategoriesStatus"' in markup
    assert 'id="queryCategoriesList"' in markup
    for period in ['value="24h"', 'value="7d"', 'value="30d"']:
        assert period in markup
    assert "loadQueryCategories()" in script
    assert "renderQueryCategories(payload)" in script
    assert "Loading query categories..." in script
    assert "No query category data found for this period." in script
    assert "Query categories could not be loaded." in script
    assert "new URLSearchParams({ period: adminState.queryCategoriesPeriod || \"24h\" })" in script
    for category in [
        "constitutional",
        "criminal",
        "consumer",
        "cyber fraud",
        "property",
        "family",
        "document/legal notice",
        "other",
    ]:
        assert category in script or category in markup
    for safe_field in ["category", "count"]:
        assert safe_field in script
    assert ".admin-trend-row" in styles
    assert ".admin-trend-bar" in styles
    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["export", "download"]:
        assert forbidden_action not in combined
    for sensitive_field in ["password_hash", "token_hash", "google_sub", "auth_provider", "auth_session_token"]:
        assert sensitive_field not in script


def test_admin_fallback_reasons_ui_is_read_only_and_aggregated_only():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "/admin/analytics/fallback-reasons" in script
    assert 'href="#fallback-reasons"' in markup
    assert 'id="fallback-reasons"' in markup
    assert 'id="fallbackReasonsForm"' in markup
    assert 'id="fallbackReasonsPeriodSelect"' in markup
    assert 'id="refreshFallbackReasonsButton"' in markup
    assert 'id="fallbackReasonsStatus"' in markup
    assert 'id="fallbackReasonsList"' in markup
    for period in ['value="24h"', 'value="7d"', 'value="30d"']:
        assert period in markup
    assert "loadFallbackReasons()" in script
    assert "renderFallbackReasons(payload)" in script
    assert "Loading fallback reasons..." in script
    assert "No fallback reason data found for this period." in script
    assert "Fallback reasons could not be loaded." in script
    assert "new URLSearchParams({ period: adminState.fallbackReasonsPeriod || \"24h\" })" in script
    for reason in [
        "low confidence",
        "no relevant authority",
        "unsupported output",
        "technical failure",
        "source insufficient",
        "validation failed",
        "other",
    ]:
        assert reason in script
    for safe_field in ["reason", "count"]:
        assert safe_field in script
    assert ".admin-trend-row" in styles
    assert ".admin-trend-bar" in styles
    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["export", "download"]:
        assert forbidden_action not in combined
    for sensitive_field in ["password_hash", "token_hash", "google_sub", "auth_provider", "auth_session_token"]:
        assert sensitive_field not in script


def test_admin_quality_score_ui_is_read_only_and_aggregated_only():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")

    assert "/admin/analytics/quality-score" in script
    assert 'href="#quality-score"' in markup
    assert 'id="quality-score"' in markup
    assert 'id="qualityScoreForm"' in markup
    assert 'id="qualityScorePeriodSelect"' in markup
    assert 'id="refreshQualityScoreButton"' in markup
    assert 'id="qualityScoreStatus"' in markup
    assert 'id="qualityScoreSummary"' in markup
    for period in ['value="24h"', 'value="7d"', 'value="30d"']:
        assert period in markup
    assert "loadQualityScore()" in script
    assert "renderQualityScore(payload)" in script
    assert "Loading response quality score..." in script
    assert "Response quality score could not be loaded." in script
    assert "new URLSearchParams({ period: adminState.qualityScorePeriod || \"24h\" })" in script
    for label in [
        "Overall quality score",
        "Average confidence",
        "Fallback rate",
        "Unsupported rate",
        "Thumbs up rate",
        "Thumbs down rate",
        "Assistant responses",
        "Feedback count",
    ]:
        assert label in script
    for safe_field in [
        "average_confidence",
        "fallback_rate",
        "unsupported_response_rate",
        "thumbs_up_rate",
        "thumbs_down_rate",
        "overall_quality_score",
        "assistant_response_count",
        "feedback_count",
    ]:
        assert safe_field in script
    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["export", "download"]:
        assert forbidden_action not in combined
    for sensitive_field in ["password_hash", "token_hash", "google_sub", "auth_provider", "auth_session_token"]:
        assert sensitive_field not in script


def test_admin_low_quality_review_queue_ui_is_read_only_and_sanitized():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")

    assert "/admin/review/low-quality" in script
    assert 'href="#low-quality-review"' in markup
    assert 'id="low-quality-review"' in markup
    assert 'id="lowQualityReviewForm"' in markup
    assert 'id="lowQualityReviewFilter"' in markup
    assert 'id="lowQualityReviewLimitSelect"' in markup
    assert 'id="loadLowQualityReviewButton"' in markup
    assert 'id="lowQualityReviewStatus"' in markup
    assert 'id="lowQualityReviewTableBody"' in markup
    assert 'id="lowQualityReviewPrevButton"' in markup
    assert 'id="lowQualityReviewNextButton"' in markup
    assert 'id="lowQualityReviewPageInfo"' in markup
    for option in ['value="low_confidence"', 'value="thumbs_down"', 'value="unsupported_output"', 'value="fallback"']:
        assert option in markup
    assert "loadLowQualityReview()" in script
    assert "renderLowQualityReview(payload)" in script
    assert "Loading low quality review queue..." in script
    assert "No low quality review candidates found." in script
    assert "review_filter" in script
    assert "Inspect chat" in script
    assert "inspectReviewChat" in script
    for safe_field in [
        "message_id",
        "chat_id",
        "timestamp",
        "confidence",
        "fallback_reason",
        "validation_flags",
        "feedback_rating",
    ]:
        assert safe_field in script
    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["approve", "edit", "delete"]:
        assert forbidden_action not in combined
    for sensitive_field in ["password_hash", "token_hash", "google_sub", "auth_provider", "auth_session_token"]:
        assert sensitive_field not in script


def test_admin_config_ui_allows_only_maintenance_mode_write_and_hides_runtime_values():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "/admin/config" in script
    assert "/admin/config/maintenance" in script
    assert 'id="config"' in markup
    assert 'id="refreshConfigButton"' in markup
    assert 'id="configStatus"' in markup
    assert 'id="configToggleList"' in markup
    assert "loadConfig()" in script
    assert "renderConfig(payload)" in script
    assert "updateMaintenanceMode(enabled)" in script
    assert 'method: "PATCH"' in script
    assert "window.confirm" in script
    assert "Turn on maintenance" in script
    assert "Turn off maintenance" in script
    assert "Loading runtime config..." in script
    assert "No runtime config flags found." in script
    for toggle_name in [
        "enable_google_fallback",
        "enable_india_kanoon",
        "enable_direct_answers",
        "enable_feedback",
        "maintenance_mode",
    ]:
        assert toggle_name in script
    assert ".admin-config-grid" in styles
    assert ".admin-config-action" in styles

    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["approve", "delete", "edit", "save"]:
        assert forbidden_action not in combined
    for forbidden_write_control in [
        "updategoogle",
        "updateindi",
        "updateopenai",
        "updatefeedback",
        "googlefallbackbutton",
        "indiakanoonbutton",
        "feedbacktogglebutton",
    ]:
        assert forbidden_write_control not in script.lower()
    for sensitive_term in [
        "api_key",
        "secret",
        "password",
        "connection",
        "mongodb_uri",
        "smtp_username",
        "smtp_from_email",
        "openai_api_key",
        "google_custom_search_api_key",
        "indiankanoon_api_token",
    ]:
        assert sensitive_term not in script.lower()


def test_admin_quality_logs_ui_is_read_only_and_uses_backend_filters():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "/admin/quality/logs" in script
    assert 'id="quality-logs"' in markup
    assert 'id="qualityLogsForm"' in markup
    assert 'id="qualityLogsFilter"' in markup
    assert 'value="fallback"' in markup
    assert 'value="low_confidence"' in markup
    assert 'value="unsupported_output"' in markup
    assert 'id="qualityLogsTableBody"' in markup
    assert 'id="qualityLogsPrevButton"' in markup
    assert 'id="qualityLogsNextButton"' in markup
    assert "fallback_only" in script
    assert "low_confidence_only" in script
    assert "unsupported_output_only" in script
    assert "loadQualityLogs()" in script
    assert "renderQualityLogs(payload)" in script
    assert "Loading quality logs..." in script
    assert "No quality logs match this filter." in script

    for safe_field in [
        "chat_id",
        "timestamp",
        "route_type",
        "confidence",
        "fallback_reason",
        "validation_flags",
        "source_sufficiency",
    ]:
        assert safe_field in script

    assert ".admin-quality-table" in styles
    assert ".admin-quality-row.risk-fallback" in styles
    assert ".admin-quality-row.risk-unsupported" in styles

    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["approve", "delete", "edit"]:
        assert forbidden_action not in combined
    for sensitive_field in ["password_hash", "token_hash", "google_sub", "auth_provider", "auth_session_token"]:
        assert sensitive_field not in script


def test_admin_direct_answer_content_ui_is_read_only_inventory_only():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "/admin/content/direct-answers" in script
    assert 'id="direct-answer-content"' in markup
    assert 'id="refreshDirectContentButton"' in markup
    assert 'id="directContentStatus"' in markup
    assert 'id="directDatasetList"' in markup
    assert 'id="directCatalogList"' in markup
    assert "loadDirectContent()" in script
    assert "renderDirectContent(payload)" in script
    assert "Loading direct-answer content..." in script
    assert "No local datasets found." in script
    assert "No direct explainer catalogs found." in script
    assert "datasets" in script
    assert "direct_explainer_catalogs" in script
    assert "source_names" in script
    assert "file_name" in script
    assert ".admin-content-grid" in styles

    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["approve", "delete", "edit"]:
        assert forbidden_action not in combined
    for sensitive_or_full_content_term in [
        "password_hash",
        "token_hash",
        "google_sub",
        "auth_provider",
        "auth_session_token",
        "description",
        "legal_position",
        "next_steps",
    ]:
        assert sensitive_or_full_content_term not in script


def test_chat_feedback_buttons_post_to_safe_feedback_endpoint():
    script = (ROOT / "Frontend" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "message-feedback" in script
    assert "feedback-button" in script
    assert "Thumbs up" in script
    assert "Thumbs down" in script
    assert "/chat/messages/${encodeURIComponent(messageId)}/feedback" in script
    assert "submitMessageFeedback" in script
    assert "assistant_message_id" in script
    assert "item.id" in script
    assert ".message-feedback" in styles
    assert ".feedback-button" in styles


def test_admin_feedback_section_is_read_only_and_sanitized():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "/admin/feedback" in script
    assert 'id="feedback"' in markup
    assert 'id="refreshFeedbackButton"' in markup
    assert 'id="feedbackFilterForm"' in markup
    assert 'id="feedbackRatingFilter"' in markup
    assert 'id="feedbackRecentLimitSelect"' in markup
    assert 'id="feedbackTotalCount"' in markup
    assert 'id="feedbackUpCount"' in markup
    assert 'id="feedbackDownCount"' in markup
    assert 'id="feedbackUncheckedCount"' in markup
    assert 'id="feedbackStatus"' in markup
    assert 'id="feedbackTableBody"' in markup
    assert 'id="feedbackPrevButton"' in markup
    assert 'id="feedbackNextButton"' in markup
    assert "loadFeedback()" in script
    assert "renderFeedback(payload)" in script
    assert "Loading feedback..." in script
    assert "No feedback found." in script
    assert "rating" in script
    assert "recent_limit" in script
    assert "summary.total_feedback" in script
    assert "summary.thumbs_up" in script
    assert "summary.thumbs_down" in script
    assert "summary.unchecked" in script
    for safe_field in ["chat_id", "message_id", "rating", "comment", "created_at"]:
        assert safe_field in script
    assert ".admin-feedback-table" in styles
    assert ".admin-feedback-summary" in styles
    assert ".admin-feedback-row.positive" in styles
    assert ".admin-feedback-row.negative" in styles

    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["approve", "delete", "edit"]:
        assert forbidden_action not in combined
    for sensitive_field in ["password_hash", "token_hash", "google_sub", "auth_provider", "auth_session_token"]:
        assert sensitive_field not in script


def test_admin_activity_section_is_read_only_and_sanitized():
    markup = (ROOT / "Frontend" / "admin.html").read_text(encoding="utf-8")
    script = (ROOT / "Frontend" / "admin.js").read_text(encoding="utf-8")
    styles = (ROOT / "Frontend" / "style.css").read_text(encoding="utf-8")

    assert "/admin/activity" in script
    assert 'id="activity"' in markup
    assert 'id="refreshActivityButton"' in markup
    assert 'id="activityStatus"' in markup
    assert 'id="activityTableBody"' in markup
    assert 'id="activityPrevButton"' in markup
    assert 'id="activityNextButton"' in markup
    assert 'id="activityPageInfo"' in markup
    assert "loadActivity()" in script
    assert "renderActivity(payload)" in script
    assert "Loading admin activity..." in script
    assert "No admin activity found." in script
    for safe_field in ["admin_user_id", "admin_email", "action", "target_type", "target_id", "created_at"]:
        assert safe_field in script
    assert ".admin-activity-table" in styles

    combined = f"{markup}\n{script}".lower()
    for forbidden_action in ["approve", "delete", "edit"]:
        assert forbidden_action not in combined
    for sensitive_field in [
        "password_hash",
        "token_hash",
        "google_sub",
        "auth_provider",
        "auth_session_token",
        "request_body",
    ]:
        assert sensitive_field not in script
