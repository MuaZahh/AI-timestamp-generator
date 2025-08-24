import os
import requests
import zipfile
import shutil
import sys
from pathlib import Path

def download_ffmpeg():
    """Download and setup FFmpeg for Windows"""
    print("Setting up FFmpeg for Windows...")
    
    # FFmpeg download URL for Windows (static build)
    ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    
    # Create ffmpeg directory in project
    ffmpeg_dir = Path(__file__).parent / "ffmpeg"
    ffmpeg_dir.mkdir(exist_ok=True)
    
    zip_path = ffmpeg_dir / "ffmpeg.zip"
    
    try:
        print("Downloading FFmpeg...")
        response = requests.get(ffmpeg_url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_dir)
        
        # Find the extracted folder (it will have a version number)
        extracted_folders = [d for d in ffmpeg_dir.iterdir() if d.is_dir() and d.name.startswith('ffmpeg')]
        if extracted_folders:
            extracted_folder = extracted_folders[0]
            bin_folder = extracted_folder / "bin"
            
            if bin_folder.exists():
                # Copy executables to main ffmpeg directory
                for exe in bin_folder.glob("*.exe"):
                    shutil.copy2(exe, ffmpeg_dir)
                
                print(f"FFmpeg installed successfully at: {ffmpeg_dir}")
                print(f"FFmpeg executable: {ffmpeg_dir / 'ffmpeg.exe'}")
                
                # Clean up
                os.remove(zip_path)
                shutil.rmtree(extracted_folder)
                
                return str(ffmpeg_dir / 'ffmpeg.exe')
            else:
                print("Error: bin folder not found in extracted archive")
                return None
        else:
            print("Error: No FFmpeg folder found in extracted archive")
            return None
            
    except Exception as e:
        print(f"Error downloading FFmpeg: {e}")
        return None

def test_ffmpeg(ffmpeg_path):
    """Test if FFmpeg is working"""
    try:
        import subprocess
        result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is working correctly!")
            return True
        else:
            print("❌ FFmpeg test failed")
            return False
    except Exception as e:
        print(f"Error testing FFmpeg: {e}")
        return False

if __name__ == "__main__":
    ffmpeg_path = download_ffmpeg()
    if ffmpeg_path and os.path.exists(ffmpeg_path):
        test_ffmpeg(ffmpeg_path)
        
        # Create config file with FFmpeg path
        config_content = f"""# FFmpeg Configuration
FFMPEG_PATH={ffmpeg_path}
"""
        with open(".env.local", "w") as f:
            f.write(config_content)
        
        print(f"\nFFmpeg path saved to .env.local")
        print("You can now run the application!")
    else:
        print("FFmpeg installation failed. Please install manually from https://ffmpeg.org/")