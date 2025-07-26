#!/usr/bin/env python3
"""
Test script to verify commits are properly dated
"""

import subprocess
import os

def test_commits():
    """Test if commits are properly dated"""
    print("🔍 Testing commit dates...")
    
    try:
        # Check if we're in a git repository
        result = subprocess.run("git status", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Not in a git repository. Run hamid_bozabal.py first.")
            return
        
        # Get recent commits with dates
        result = subprocess.run(
            "git log --oneline --date=short --pretty=format:'%h %ad %s' -5",
            shell=True, capture_output=True, text=True
        )
        
        if result.stdout:
            print("✅ Found commits:")
            print(result.stdout)
            
            # Check if dates are in the past
            import datetime
            today = datetime.datetime.now().date()
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        date_str = parts[1]
                        try:
                            commit_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                            if commit_date <= today:
                                print(f"✅ Commit date {date_str} is valid")
                            else:
                                print(f"⚠️  Commit date {date_str} is in the future")
                        except:
                            print(f"⚠️  Could not parse date: {date_str}")
        else:
            print("❌ No commits found")
            
    except Exception as e:
        print(f"❌ Error testing commits: {e}")

if __name__ == "__main__":
    test_commits()