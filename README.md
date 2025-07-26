# GitHub Contribution Generator

This script creates fake Git commits with realistic timestamps to show up in your GitHub contributions graph.

## Features

- ✅ Generates 500 commits for the current year (configurable)
- ✅ Realistic commit messages and file content
- ✅ Proper timestamps that will show in GitHub contributions
- ✅ Interactive setup with user prompts
- ✅ Progress tracking during generation

## Usage

### Quick Start

```bash
python3 hamid_bozabal.py
```

### Step by Step

1. **Run the script:**
   ```bash
   python3 hamid_bozabal.py
   ```

2. **Follow the prompts:**
   - Enter repository path (default: current directory)
   - Enter number of commits to generate (default: 500)

3. **Wait for completion:**
   The script will create commits with realistic timestamps throughout the year.

4. **Push to GitHub:**
   ```bash
   # Create a new repository on GitHub first
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

## How it Works

- Creates a Git repository (if none exists)
- Generates random dates throughout the current year
- Creates files with realistic content
- Makes commits with proper timestamps
- Uses realistic commit messages

## Requirements

- Python 3.6+
- Git installed on your system
- Internet connection (for pushing to GitHub)

## Notes

- The script creates commits with timestamps that will appear in your GitHub contribution graph
- Commits are distributed throughout the year to look natural
- All commits use the name "hamid_bozabal" as configured
- Files are created with various programming language syntaxes

## Disclaimer

This tool is for educational purposes. Use responsibly and in accordance with GitHub's terms of service.