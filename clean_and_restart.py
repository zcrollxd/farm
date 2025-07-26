#!/usr/bin/env python3
"""
Clean up existing commits and restart with correct dates
"""

import subprocess
import os
import shutil

def clean_repository():
    """Clean up the repository and start fresh"""
    print("🧹 Cleaning up repository...")
    
    try:
        # Remove all files except the scripts
        for item in os.listdir('.'):
            if item not in ['hamid_bozabal.py', 'test_commits.py', 'clean_and_restart.py', 'README.md']:
                if os.path.isfile(item):
                    os.remove(item)
                elif os.path.isdir(item):
                    shutil.rmtree(item)
        
        # Remove git repository
        if os.path.exists('.git'):
            shutil.rmtree('.git')
            print("✅ Removed existing .git directory")
        
        print("✅ Repository cleaned up!")
        print("🚀 Ready to run hamid_bozabal.py with correct dates")
        
    except Exception as e:
        print(f"❌ Error cleaning repository: {e}")

if __name__ == "__main__":
    clean_repository()