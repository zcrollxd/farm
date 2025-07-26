#!/usr/bin/env python3
"""
GitHub Contribution Generator
Creates fake commits to show up in GitHub contributions graph.
Target: 500 commits for the current year
"""

import os
import subprocess
import random
import datetime
from pathlib import Path

class GitHubContributionGenerator:
    def __init__(self, repo_path=".", target_commits=500):
        self.repo_path = Path(repo_path).resolve()
        self.target_commits = target_commits
        self.current_year = datetime.datetime.now().year
        
    def run_command(self, command, cwd=None):
        """Run a shell command and return the result"""
        if cwd is None:
            cwd = self.repo_path
        try:
            result = subprocess.run(command, shell=True, cwd=cwd, 
                                 capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {command}")
            print(f"Error: {e}")
            return None
    
    def init_repo(self):
        """Initialize a Git repository if it doesn't exist"""
        if not (self.repo_path / ".git").exists():
            print("Initializing Git repository...")
            self.run_command("git init")
            self.run_command('git config user.name "hamid_bozabal"')
            self.run_command('git config user.email "hamid@example.com"')
            print("Git repository initialized.")
        else:
            print("Git repository already exists.")
        
        # Ensure we're on main branch (GitHub's default)
        self.run_command("git checkout -b main 2>/dev/null || git checkout main")
        
        # Set up proper Git configuration for contributions
        self.run_command('git config user.name "hamid_bozabal"')
        self.run_command('git config user.email "hamid@example.com"')
    
    def create_commit_file(self, filename, content):
        """Create a file with content and add it to Git"""
        file_path = self.repo_path / filename
        with open(file_path, 'w') as f:
            f.write(content)
        self.run_command(f"git add {filename}")
    
    def make_commit(self, message, date):
        """Make a commit with a specific date"""
        date_str = date.strftime("%Y-%m-%d %H:%M:%S")
        # Use environment variables to set the commit date properly
        env = os.environ.copy()
        env['GIT_AUTHOR_DATE'] = date_str
        env['GIT_COMMITTER_DATE'] = date_str
        
        # Make the commit with proper environment variables
        try:
            subprocess.run(
                f'git commit -m "{message}"',
                shell=True, cwd=self.repo_path, env=env, check=True,
                capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error making commit: {e}")
            return False
        return True
    
    def verify_commits(self):
        """Verify that commits have proper dates"""
        try:
            # Get the last few commits to verify dates
            result = subprocess.run(
                "git log --oneline --date=short --pretty=format:'%h %ad %s' -10",
                shell=True, cwd=self.repo_path, capture_output=True, text=True
            )
            if result.stdout:
                print("✅ Recent commits with dates:")
                print(result.stdout)
            else:
                print("⚠️  No commits found to verify")
        except Exception as e:
            print(f"⚠️  Could not verify commits: {e}")
    
    def generate_realistic_commit_message(self):
        """Generate realistic commit messages"""
        messages = [
            "Update README.md",
            "Fix typo in documentation",
            "Add new feature",
            "Refactor code",
            "Update dependencies",
            "Fix bug in main function",
            "Add unit tests",
            "Improve performance",
            "Update configuration",
            "Clean up code",
            "Add error handling",
            "Update comments",
            "Fix formatting",
            "Add logging",
            "Optimize algorithm",
            "Update documentation",
            "Fix linting issues",
            "Add new method",
            "Update version",
            "Fix security issue"
        ]
        return random.choice(messages)
    
    def generate_file_content(self):
        """Generate realistic file content"""
        contents = [
            "# Project Documentation\n\nThis is a sample project.",
            "def main():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    main()",
            "import os\nimport sys\n\ndef setup():\n    pass",
            "class Config:\n    def __init__(self):\n        self.debug = True",
            "// Sample configuration\nconst config = {\n    port: 3000\n};",
            "/*\n * Sample header file\n */\n#ifndef SAMPLE_H\n#define SAMPLE_H\n\n#endif",
            "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"Hello\")\n}",
            "function processData(data) {\n    return data.map(item => item.id);\n}",
            "<?php\n\necho \"Hello World\";\n?>",
            "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}"
        ]
        return random.choice(contents)
    
    def generate_dates(self):
        """Generate realistic commit dates for the current year"""
        dates = []
        start_date = datetime.datetime(self.current_year, 1, 1)
        end_date = datetime.datetime.now()  # Only up to today
        
        # Generate dates with some clustering (more commits on weekdays)
        for _ in range(self.target_commits):
            # Random date within the year up to today
            random_days = random.randint(0, (end_date - start_date).days)
            date = start_date + datetime.timedelta(days=random_days)
            
            # Add random time during working hours (9 AM - 6 PM)
            hour = random.randint(9, 18)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            date = date.replace(hour=hour, minute=minute, second=second)
            dates.append(date)
        
        # Sort dates chronologically
        dates.sort()
        return dates
    
    def generate_contributions(self):
        """Generate the fake contributions"""
        print(f"Starting to generate {self.target_commits} commits for {self.current_year}...")
        
        # Initialize repository
        self.init_repo()
        
        # Generate commit dates
        commit_dates = self.generate_dates()
        
        # Create initial commit if repository is empty
        if not self.run_command("git log --oneline"):
            print("Creating initial commit...")
            self.create_commit_file("README.md", "# Project\n\nThis is a sample project.")
            if self.make_commit("Initial commit", commit_dates[0]):
                commit_dates = commit_dates[1:]  # Remove the first date since we used it
            else:
                print("Failed to create initial commit")
                return
        
        # Generate commits
        for i, date in enumerate(commit_dates):
            # Create a random file
            filename = f"file_{i:04d}.txt"
            content = self.generate_file_content()
            
            # Create and commit the file
            self.create_commit_file(filename, content)
            message = self.generate_realistic_commit_message()
            if not self.make_commit(message, date):
                print(f"Failed to create commit {i+1}")
                continue
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"Created {i + 1} commits...")
        
        print(f"\n✅ Successfully created {self.target_commits} commits!")
        print(f"📅 Commits span from {commit_dates[0].strftime('%Y-%m-%d')} to {commit_dates[-1].strftime('%Y-%m-%d')}")
        print(f"📁 Repository location: {self.repo_path}")
        
        # Verify commits
        print("\n🔍 Verifying commits...")
        self.verify_commits()
        
        print("\n📋 IMPORTANT: To make contributions show on GitHub profile:")
        print("1. Create a new repository on GitHub")
        print("2. Make sure your GitHub email matches: hamid@example.com")
        print("3. Or update the email in the script to match your GitHub email")
        print("4. Run: git remote add origin <your-repo-url>")
        print("5. Run: git push -u origin main")
        print("6. Wait a few minutes for GitHub to update your profile")

def main():
    """Main function"""
    print("🚀 GitHub Contribution Generator")
    print("=" * 40)
    
    # Get user input
    repo_path = input("Enter repository path (default: current directory): ").strip()
    if not repo_path:
        repo_path = "."
    
    try:
        target_commits = int(input("Enter number of commits to generate (default: 500): ").strip() or "500")
    except ValueError:
        target_commits = 500
    
    # Create generator and run
    generator = GitHubContributionGenerator(repo_path, target_commits)
    generator.generate_contributions()

if __name__ == "__main__":
    main()