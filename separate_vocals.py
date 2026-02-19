import os, sys
from dotenv import load_dotenv

from demucs.separate import main as separate_main

load_dotenv()

song_path = os.getenv("SONG_PATH")

# Save original sys.argv
original_argv = sys.argv

# Set up arguments as if calling from command line
sys.argv = [
    'demucs',
    '--mp3',                   # Output format (or use --flac, --wav)
    '--two-stems', 'vocals',  # Only separate vocals vs everything else (faster)
    '-n', 'htdemucs',          # Model name
    '--out', './output',       # Output directory
    song_path
]

# Run separation
separate_main()

# Restore original sys.argv
sys.argv = original_argv

