#!/usr/bin/env python
"""
Quick Start Guide for Classification Models Demo
Run this file to set up and launch the application
"""

import subprocess
import sys
import os

def check_and_install_packages():
    """Check if required packages are installed and install if necessary"""
    print("🔍 Checking required packages...")
    
    required_packages = [
        'streamlit',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
    
    try:
        import streamlit
        import numpy
        import pandas
        import sklearn
        import matplotlib
        import seaborn
        print("✅ All packages are already installed!")
        return True
    except ImportError:
        print("📦 Installing required packages...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                '-r', 'requirements.txt'
            ])
            print("✅ Packages installed successfully!")
            return True
        except Exception as e:
            print(f"❌ Error installing packages: {e}")
            return False

def main():
    """Main function to run the application"""
    print("=" * 60)
    print("🤖 Classification Models Demo - Quick Start")
    print("=" * 60)
    
    # Check and install packages
    if not check_and_install_packages():
        print("\n❌ Failed to install packages. Please install manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n🚀 Starting Streamlit application...")
    print("📱 The app will open at: http://localhost:8501")
    print("⚠️  Press Ctrl+C to stop the server\n")
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'classification_models.py',
            '--logger.level=warning'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
