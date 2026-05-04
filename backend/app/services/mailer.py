from __future__ import annotations

import smtplib
from email.message import EmailMessage
from html import escape
from urllib.parse import quote

from backend.app.core.config import Settings

class MailDeliveryError(RuntimeError):
    pass

PASSWORD_RESET_TEXT_TEMPLATE = """We received a request to reset your Lawyer AI password.

Open this link to choose a new password:
{{RESET_LINK}}

This link expires in {{EXPIRY_MINUTES}} minutes.

If you did not request a password reset, you can ignore this email. Your current password will remain unchanged.
"""

PASSWORD_RESET_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset your Lawyer AI password</title>
  </head>
  <body style="margin:0;padding:0;background:#f4f7fb;color:#1f2937;font-family:Arial,Helvetica,sans-serif;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#f4f7fb;margin:0;padding:32px 16px;">
      <tr>
        <td align="center">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width:560px;background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;">
            <tr>
              <td style="padding:32px 32px 24px;">
                <h1 style="margin:0 0 16px;font-size:24px;line-height:32px;color:#111827;font-weight:700;">Reset your password</h1>
                <p style="margin:0 0 24px;font-size:16px;line-height:24px;color:#4b5563;">We received a request to reset your Lawyer AI password. Use the secure button below to choose a new password.</p>
                <table role="presentation" cellspacing="0" cellpadding="0" style="margin:0 0 24px;">
                  <tr>
                    <td bgcolor="#2563eb" style="border-radius:6px;">
                      <a href="{{RESET_LINK}}" style="display:inline-block;padding:13px 22px;font-size:16px;line-height:20px;color:#ffffff;text-decoration:none;font-weight:700;">Reset password</a>
                    </td>
                  </tr>
                </table>
                <p style="margin:0 0 8px;font-size:14px;line-height:21px;color:#4b5563;">If the button does not work, copy and paste this link into your browser:</p>
                <p style="margin:0 0 24px;font-size:14px;line-height:21px;word-break:break-all;">
                  <a href="{{RESET_LINK}}" style="color:#2563eb;text-decoration:underline;">{{RESET_LINK}}</a>
                </p>
                <p style="margin:0 0 16px;font-size:14px;line-height:21px;color:#4b5563;">This link expires in <strong>{{EXPIRY_MINUTES}} minutes</strong>.</p>
                <p style="margin:0;font-size:13px;line-height:20px;color:#6b7280;">If you did not request a password reset, you can safely ignore this email. Your current password will remain unchanged.</p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

class SmtpMailer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_password_reset_url(self, token: str) -> str:
        base_url = self.settings.app_base_url.rstrip("/")
        return f"{base_url}{self.settings.frontend_auth_path}?reset_token={quote(token)}"

    def send_password_reset_email(self, recipient_email: str, reset_url: str) -> None:
        subject = "Reset your Lawyer AI password"
        text_body = self._build_password_reset_text(reset_url)
        html_body = self._build_password_reset_html(reset_url)
        self._send_message(recipient_email=recipient_email, subject=subject, text_body=text_body, html_body=html_body)

    def _build_password_reset_text(self, reset_url: str) -> str:
        return (
            PASSWORD_RESET_TEXT_TEMPLATE.replace("{{RESET_LINK}}", reset_url)
            .replace("{{EXPIRY_MINUTES}}", str(self.settings.password_reset_token_ttl_minutes))
            .strip()
        )

    def _build_password_reset_html(self, reset_url: str) -> str:
        safe_reset_url = escape(reset_url, quote=True)
        return (
            PASSWORD_RESET_HTML_TEMPLATE.replace("{{RESET_LINK}}", safe_reset_url)
            .replace("{{EXPIRY_MINUTES}}", str(self.settings.password_reset_token_ttl_minutes))
            .strip()
        )

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
