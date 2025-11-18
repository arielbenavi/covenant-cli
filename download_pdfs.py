"""
Download PDFs from Google Drive to data/raw/
Uses service account (no browser needed).

Usage:
    python download_pdfs.py
"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io
except ImportError:
    print("❌ Google API client not installed")
    print("Run: pip install google-auth google-api-python-client")
    exit(1)


def download_pdfs():
    """Download all PDFs from Google Drive folder to data/raw/"""
    
    # Get folder ID from .env
    folder_id = os.getenv('GDRIVE_FOLDER_ID')
    
    if not folder_id:
        print("❌ GDRIVE_FOLDER_ID not set in .env")
        print("\nAdd this to your .env file:")
        print("GDRIVE_FOLDER_ID=your_folder_id_here")
        exit(1)
    
    # Check for service account credentials
    if not Path('service_account.json').exists():
        print("❌ service_account.json not found")
        print("\nYou need to:")
        print("1. Create a service account in Google Cloud Console")
        print("2. Download the JSON key as 'service_account.json'")
        print("3. Share your Google Drive folder with the service account email")
        exit(1)
    
    # Authenticate with service account (no browser!)
    print("🔐 Authenticating with service account...\n")
    
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    credentials = service_account.Credentials.from_service_account_file(
        'service_account.json', 
        scopes=SCOPES
    )
    
    service = build('drive', 'v3', credentials=credentials)
    
    # Make sure data/raw exists
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get existing PDFs to avoid duplicates
    existing_pdfs = {f.name for f in output_dir.glob('*.pdf')}
    
    # List PDFs in Google Drive folder
    print(f"📂 Fetching file list from Google Drive...\n")
    
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
        fields="files(id, name, size)"
    ).execute()
    
    pdf_files = results.get('files', [])
    
    if not pdf_files:
        print("⚠️  No PDF files found in Google Drive folder")
        print("\nMake sure:")
        print("1. The folder ID is correct")
        print("2. The folder is shared with the service account email")
        exit(0)
    
    print(f"Found {len(pdf_files)} PDFs in Google Drive\n")
    
    # Download each PDF
    downloaded = 0
    skipped = 0
    
    for file in pdf_files:
        filename = file['name']
        file_id = file['id']
        
        # Skip if already exists
        if filename in existing_pdfs:
            print(f"⏭️  Skipping (already exists): {filename}")
            skipped += 1
            continue
        
        # Download
        print(f"📥 Downloading: {filename}")
        
        request = service.files().get_media(fileId=file_id)
        file_path = output_dir / filename
        
        with open(file_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        
        # Show file size
        size_mb = file_path.stat().st_size / 1_000_000
        print(f"   ✅ Saved ({size_mb:.1f} MB)\n")
        downloaded += 1
    
    # Summary
    print("="*60)
    print(f"✅ Download complete!")
    print(f"   Downloaded: {downloaded} new PDFs")
    print(f"   Skipped: {skipped} existing PDFs")
    print(f"   Total in data/raw/: {len(existing_pdfs) + downloaded}")
    print("="*60)


if __name__ == '__main__':
    download_pdfs()