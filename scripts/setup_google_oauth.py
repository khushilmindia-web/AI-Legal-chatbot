import webbrowser
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"

def main():
    print("🚀 Google OAuth Setup Helper for AI Legal Chatbot")
    print("\n📋 Step-by-step setup instructions:")
    print("\n1. 🌐 Open Google Cloud Console:")
    print("   https://console.cloud.google.com/")
    input("\n   Press Enter to open Google Cloud Console in your browser...")

    try:
        webbrowser.open("https://console.cloud.google.com/")
    except:
        print("   Please manually open: https://console.cloud.google.com/")

    print("\n2. 📁 Create or select a project:")
    print("   - Click 'Select a project' (top left)")
    print("   - Click 'New Project'")
    print("   - Name: 'AI Legal Chatbot' (or any name you prefer)")
    print("   - Click 'Create'")

    input("\n   Press Enter when you've created/selected a project...")
    print("\n3. 🔧 Enable Google APIs:")
    print("   - Go to 'APIs & Services' > 'Library'")
    print("   - Search for 'Google+ API' and enable it")
    print("   - Search for 'People API' and enable it")

    input("\n   Press Enter when APIs are enabled...")

    print("\n4. 🔐 Create OAuth 2.0 credentials:")
    print("   - Go to 'APIs & Services' > 'Credentials'")
    print("   - Click '+ CREATE CREDENTIALS' > 'OAuth 2.0 Client IDs'")
    print("   - Application type: 'Web application'")
    print("   - Name: 'AI Legal Chatbot'")

    print("\n5. 🌐 Add authorized redirect URIs:")
    print("   - Authorized redirect URIs: Add this URL:")
    print("     http://127.0.0.1:5000/auth/google/callback")
    print("   - Click 'Create'")

    input("\n   Press Enter when credentials are created...")

    print("\n6. 📝 Copy your credentials:")
    print("   - Find your new OAuth 2.0 Client ID")
    print("   - Click the download button or copy:")
    print("     • Client ID")
    print("     • Client Secret")

    client_id = input("\n   Paste your Client ID here: ").strip()
    client_secret = input("   Paste your Client Secret here: ").strip()

    if not client_id or not client_secret:
        print("❌ Error: Both Client ID and Client Secret are required!")
        return

    # Update .env file
    env_path = CONFIG_DIR / ".env"

    try:
        with open(env_path, 'r') as f:
            content = f.read()

        # Replace the placeholder values
        content = content.replace(
            'GOOGLE_CLIENT_ID=your_google_client_id_here',
            f'GOOGLE_CLIENT_ID={client_id}'
        )
        content = content.replace(
            'GOOGLE_CLIENT_SECRET=your_google_client_secret_here',
            f'GOOGLE_CLIENT_SECRET={client_secret}'
        )

        with open(env_path, 'w') as f:
            f.write(content)

        print("\n✅ Success! Google OAuth credentials updated in .env file")
        print("\n🔄 Restart your server:")
        print("   python -m uvicorn backend.main:app --host 127.0.0.1 --port 5000 --reload")
        print("\n🎉 Google sign-in should now work!")

    except Exception as e:
        print(f"\n❌ Error updating .env file: {e}")
        print("Please manually update your .env file with:")
        print(f"GOOGLE_CLIENT_ID={client_id}")
        print(f"GOOGLE_CLIENT_SECRET={client_secret}")

if __name__ == "__main__":
    main()
