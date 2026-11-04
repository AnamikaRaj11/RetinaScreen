"""Compatibility entry point for the complete RetinaScreen application."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import demo, dark_theme, custom_css

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860,
        share=False, theme=dark_theme, css=custom_css)
