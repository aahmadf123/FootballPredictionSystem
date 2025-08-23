#!/usr/bin/env python3
"""Installation script for Grid Football Prediction System."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and print output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result


def main():
    """Main installation process."""
    print("Installing Grid Football Prediction System...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("Error: Python 3.11 or higher is required")
        sys.exit(1)
    
    # Install package in development mode
    print("\n1. Installing Python package...")
    run_command("pip install -e .")
    
    # Create .env file from template if it doesn't exist
    print("\n2. Setting up environment file...")
    env_file = Path(".env")
    env_template = Path(".env.template")
    
    if not env_file.exists() and env_template.exists():
        run_command("cp .env.template .env")
        print("Created .env file. Please edit it with your API keys.")
    elif not env_file.exists():
        # Create basic .env file
        with open(".env", "w") as f:
            f.write("CFBD_API_KEY=your_cfbd_api_key_here\n")
            f.write("WEATHER_API_KEY=your_weather_api_key_here\n")
            f.write("NEWS_FEEDS=https://feeds.espn.com/rss/headlines/nfl\n")
        print("Created basic .env file. Please edit it with your API keys.")
    
    # Initialize the system
    print("\n3. Initializing system...")
    run_command("grid init", check=False)
    
    print("\nInstallation completed!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run 'grid update' to fetch initial data")
    print("3. Run 'grid predict' to generate predictions")
    print("4. Run 'grid serve' to start the API server")
    print("\nFor help, run 'grid --help'")


if __name__ == "__main__":
    main()
