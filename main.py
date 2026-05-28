"""
Customer Value Intelligence Hub - Main Entry Point
Run with: streamlit run main.py
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'dashboards',
        'customer_app.py'
    )
    
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("👥 Starting Customer Value Intelligence Hub")
    print("=" * 60)
    print(f"Dashboard: {dashboard_path}")
    print("Starting Streamlit server on http://localhost:8502 ...")
    
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', dashboard_path,
        '--server.port', '8502',
        '--server.address', 'localhost',
        '--browser.gatherUsageStats', 'false'
    ])

if __name__ == "__main__":
    main()