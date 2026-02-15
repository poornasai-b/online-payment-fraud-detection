"""
WSGI entry point for Gunicorn
This file imports the Flask app from the flask directory
"""
import sys
import os

# Add the flask directory to sys.path to avoid naming conflict with flask package
flask_dir = os.path.join(os.path.dirname(__file__), 'flask')
sys.path.insert(0, flask_dir)

# Import the app module from the flask directory
import app as flask_module
app = flask_module.app

if __name__ == "__main__":
    app.run()
