from __future__ import annotations

import smtplib
from email.message import EmailMessage
from urllib.parse import quote

from backend.app.core.config import Settings


class MailDeliveryError(RuntimeError):
    pass


class SmtpMailer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_password_reset_url(self, token: str) -> str:
        base_url = self.settings.app_base_url.rstrip("/")
        return f"{base_url}{self.settings.frontend_auth_path}?reset_token={quote(token)}"

    def send_password_reset_email(self, recipient_email: str, reset_url: str) -> None:
        subject = "Reset your Lawyer AI password"
        text_body = (
            "We received a request to reset your Lawyer AI password.\n\n"
            f"Open this link to choose a new password:\n{reset_url}\n\n"
            f"This link expires in {self.settings.password_reset_token_ttl_minutes} minutes.\n\n"
            "If you did not request a password reset, you can ignore this email."
        )
        html_body = (
            "<p>We received a request to reset your Lawyer AI password.</p>"
            f"<p><a href=\"{reset_url}\">Open this secure password reset link</a></p>"
            f"<p>This link expires in {self.settings.password_reset_token_ttl_minutes} minutes.</p>"
            "<p>If you did not request a password reset, you can ignore this email.</p>"
        )
        self._send_message(recipient_email=recipient_email, subject=subject, text_body=text_body, html_body=html_body)

    def _send_message(self, recipient_email: str, subject: str, text_body: str, html_body: str) -> None:
        if not self.settings.smtp_configured:
            raise MailDeliveryError("SMTP is not configured")

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = f"{self.settings.smtp_from_name} <{self.settings.smtp_from_email}>"
        message["To"] = recipient_email
        message.set_content(text_body)
        message.add_alternative(html_body, subtype="html")

        try:
            with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=20) as server:
                if self.settings.smtp_use_tls:
                    server.starttls()
                server.login(self.settings.smtp_username, self.settings.smtp_password)
                server.send_message(message)
        except Exception as exc:  # pragma: no cover - exact SMTP failure varies by provider
            raise MailDeliveryError("SMTP delivery failed") from exc
