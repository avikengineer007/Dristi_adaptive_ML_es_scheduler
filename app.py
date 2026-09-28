import os
import sys

# Entry point forwarding to dashboard/app.py
dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard")
sys.path.insert(0, dashboard_path)

from dashboard.app import *
