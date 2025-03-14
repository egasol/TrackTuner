import subprocess
import time
from pathlib import Path
from flask import Flask, render_template, request, Response

app = Flask(__name__)

PORT = 5000
HOST = "0.0.0.0"
TEMPLATE = "index.html"


@app.route("/")
def index():
    return render_template(TEMPLATE)


@app.route("/stream")
def stream():
    n_files = request.args.get("n_files", "5")
    trials = request.args.get("trials", "10")

    def generate():
        # print(Path(__file__).parent)
        script_path = Path(__file__).resolve().parent / "../main.py"

        process = subprocess.Popen(
            [
                "python",
                script_path,
                "--sequences",
                n_files,
                "--trials",
                trials,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        for line in iter(process.stdout.readline, ""):
            time.sleep(0.1)
            if line:
                formatted = "data:" + line.replace("\n", "\ndata:")
                yield f"{formatted}\n\n"
        process.stdout.close()
        process.wait()

        yield "data:--- Job finished successfully ---\n\n"
        yield "event:close\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
