#!/usr/bin/env python3
"""Launcher for the modular Revers-o application"""
from reverso import create_simple_interface

if __name__ == "__main__":
    print("🚀 Starting Simple Revers-o...")
    demo = create_simple_interface()
    print("🌐 Launching interface...")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
