"""
Sync PDF contracts from Google Drive to local data/raw folder.
"""
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


class GDriveSync:
    """Download PDFs from a shared Google Drive folder."""
    
    def __init__(self, folder_id: str = None):
        """
        Args:
            folder_id: Google Drive folder ID (from URL)
                      e.g., https://drive.google.com/drive/folders/1a2b3c4d5...
                      Use the string after /folders/
        """
        self.folder_id = folder_id or os.getenv('GDRIVE_FOLDER_ID')
        
        if not self.folder_id:
            raise ValueError("Must provide folder_id or set GDRIVE_FOLDER_ID in .env")
        
        # Authenticate
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()  # Opens browser for OAuth
        self.drive = GoogleDrive(gauth)
    
    def download_all_pdfs(self, output_dir: str = 'data/raw') -> list:
        """
        Download all PDFs from the Google Drive folder.
        
        Returns: List of downloaded file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # List files in folder
        file_list = self.drive.ListFile({
            'q': f"'{self.folder_id}' in parents and trashed=false"
        }).GetList()
        
        downloaded = []
        
        for file in file_list:
            if file['title'].endswith('.pdf'):
                print(f"📥 Downloading: {file['title']}")
                
                file_path = output_path / file['title']
                file.GetContentFile(str(file_path))
                
                downloaded.append(file_path)
                print(f"   ✅ Saved to: {file_path}")
        
        print(f"\n✅ Downloaded {len(downloaded)} PDFs")
        return downloaded
    
    def download_specific(self, filename: str, output_dir: str = 'data/raw') -> Path:
        """Download a specific PDF by name."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Search for file
        file_list = self.drive.ListFile({
            'q': f"'{self.folder_id}' in parents and title='{filename}' and trashed=false"
        }).GetList()
        
        if not file_list:
            raise FileNotFoundError(f"File not found: {filename}")
        
        file = file_list[0]
        file_path = output_path / filename
        
        print(f"📥 Downloading: {filename}")
        file.GetContentFile(str(file_path))
        print(f"   ✅ Saved to: {file_path}")
        
        return file_path


def test_sync():
    """Test Google Drive sync."""
    # You'll need to set GDRIVE_FOLDER_ID in .env
    syncer = GDriveSync()
    syncer.download_all_pdfs()


if __name__ == '__main__':
    test_sync()