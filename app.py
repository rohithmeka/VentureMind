import os
from flask import Flask, send_from_directory

# This is the object gunicorn needs -- "app:app" means
# "in the file app.py, run the object named app".
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "venturemind_app.html")


if __name__ == "__main__":
    # only used for local testing (python app.py) -- Render uses gunicorn instead
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
