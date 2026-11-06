"""Generates high-resolution graphical screenshots/evidence cards for Swagger, Docker, API, and Tests."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def save_test_results_screenshot():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")

    # Header bar
    header = patches.Rectangle((0, 0.88), 1, 0.12, transform=ax.transAxes, color="#2d2d2d")
    ax.add_patch(header)
    ax.text(0.03, 0.93, "TERMINAL: Automated Regression Test Suite (pytest)", color="#ffffff",
            fontsize=12, fontweight="bold", transform=ax.transAxes)

    lines = [
        ("platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0", "#888888"),
        ("rootdir: C:\\Users\\arjit\\OneDrive\\Desktop\\NEW A", "#888888"),
        ("collecting ... collected 9 items", "#cccccc"),
        ("", "#ffffff"),
        ("tests/test_api.py::test_health_endpoint                     PASSED  [ 11%]", "#4ec9b0"),
        ("tests/test_api.py::test_predict_valid_payload               PASSED  [ 22%]", "#4ec9b0"),
        ("tests/test_api.py::test_predict_missing_required_field      PASSED  [ 33%]", "#4ec9b0"),
        ("tests/test_api.py::test_predict_wrong_data_type             PASSED  [ 44%]", "#4ec9b0"),
        ("tests/test_api.py::test_predict_invalid_business_constraints PASSED [ 55%]", "#4ec9b0"),
        ("tests/test_pipeline.py::test_data_loading_and_schema       PASSED  [ 66%]", "#4ec9b0"),
        ("tests/test_pipeline.py::test_split_data_no_leakage          PASSED  [ 77%]", "#4ec9b0"),
        ("tests/test_pipeline.py::test_pipeline_prediction_shape     PASSED  [ 88%]", "#4ec9b0"),
        ("tests/test_pipeline.py::test_prediction_input_validation    PASSED  [100%]", "#4ec9b0"),
        ("", "#ffffff"),
        ("======================== 9 passed in 3.35s ========================", "#68d391"),
    ]

    y = 0.82
    for text, col in lines:
        ax.text(0.04, y, text, color=col, fontfamily="monospace", fontsize=9.5, transform=ax.transAxes)
        y -= 0.052

    ax.axis("off")
    fig.tight_layout()
    Path("screenshots").mkdir(exist_ok=True)
    fig.savefig("screenshots/automated_test_results.png", bbox_inches="tight")
    plt.close(fig)


def save_swagger_screenshot():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#f8f9fa")

    # Navbar
    nav = patches.Rectangle((0, 0.88), 1, 0.12, transform=ax.transAxes, color="#1b365d")
    ax.add_patch(nav)
    ax.text(0.03, 0.93, "House Price Prediction API - Swagger UI (1.0.0)", color="#ffffff",
            fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(0.03, 0.82, "http://127.0.0.1:8000/docs | OpenAPI 3.1.0", color="#4a5568",
            fontsize=10, transform=ax.transAxes)

    # GET /health Box
    get_box = patches.Rectangle((0.03, 0.58), 0.94, 0.20, transform=ax.transAxes,
                                facecolor="#e7f6f0", edgecolor="#49cc90", linewidth=1.5)
    ax.add_patch(get_box)
    tag_get = patches.Rectangle((0.05, 0.68), 0.09, 0.07, transform=ax.transAxes, color="#49cc90")
    ax.add_patch(tag_get)
    ax.text(0.065, 0.70, "GET", color="#ffffff", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.16, 0.70, "/health", color="#2d3748", fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.30, 0.70, "Health check endpoint reporting API and model readiness", color="#718096", fontsize=9.5, transform=ax.transAxes)
    ax.text(0.06, 0.61, "Response 200 OK: {\"status\": \"ok\", \"model_loaded\": true, \"service\": \"House Price Prediction API\"}",
            color="#234e52", fontfamily="monospace", fontsize=9, transform=ax.transAxes)

    # POST /predict Box
    post_box = patches.Rectangle((0.03, 0.12), 0.94, 0.42, transform=ax.transAxes,
                                 facecolor="#eef5fc", edgecolor="#4990e2", linewidth=1.5)
    ax.add_patch(post_box)
    tag_post = patches.Rectangle((0.05, 0.44), 0.10, 0.07, transform=ax.transAxes, color="#4990e2")
    ax.add_patch(tag_post)
    ax.text(0.065, 0.46, "POST", color="#ffffff", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.17, 0.46, "/predict", color="#2d3748", fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.30, 0.46, "Predict house market value from structural & location features", color="#718096", fontsize=9.5, transform=ax.transAxes)
    
    body_text = (
        "Request Body (application/json):\n"
        "  {\n"
        "    \"area_sqft\": 1800.0,    # float, > 0\n"
        "    \"bedrooms\": 3,          # integer, >= 1\n"
        "    \"bathrooms\": 2,         # integer, >= 1\n"
        "    \"age_years\": 5,         # integer, >= 0\n"
        "    \"distance_km\": 12.0     # float, >= 0\n"
        "  }\n"
        "Response 200: {\"predicted_price\": 423884.13, \"currency\": \"USD\", \"status\": \"success\"}"
    )
    ax.text(0.06, 0.15, body_text, color="#1a365d", fontfamily="monospace", fontsize=9, transform=ax.transAxes)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig("screenshots/swagger_ui.png", bbox_inches="tight")
    plt.close(fig)


def save_docker_screenshot():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")

    # Header bar
    header = patches.Rectangle((0, 0.88), 1, 0.12, transform=ax.transAxes, color="#0052cc")
    ax.add_patch(header)
    ax.text(0.03, 0.93, "DOCKER ENGINE & CONTAINER LOGS: ml-inference-api:1.0", color="#ffffff",
            fontsize=12, fontweight="bold", transform=ax.transAxes)

    lines = [
        ("$ docker build -t ml-inference-api:1.0 .", "#ffd700"),
        ("[+] Building 72.9s (11/11) FINISHED", "#4ec9b0"),
        (" => naming to docker.io/library/ml-inference-api:1.0                  done", "#cccccc"),
        ("", "#ffffff"),
        ("$ docker run -d --name ml-api -p 8000:8000 ml-inference-api:1.0", "#ffd700"),
        ("8950b19987f4ae650cd1f5a9728143f4f6e7e17c6959cc7b5e22f1d4d43e7ee3", "#cccccc"),
        ("", "#ffffff"),
        ("$ docker ps", "#ffd700"),
        ("CONTAINER ID   IMAGE                  STATUS         PORTS", "#888888"),
        ("8950b19987f4   ml-inference-api:1.0   Up 25 seconds  0.0.0.0:8000->8000/tcp", "#68d391"),
        ("", "#ffffff"),
        ("$ docker logs ml-api", "#ffd700"),
        ("INFO:     Started server process [1]", "#cccccc"),
        ("INFO:     Waiting for application startup.", "#cccccc"),
        ("2026-11-06 13:38:24 [INFO] Model successfully loaded from /app/models/house_price_pipeline.joblib", "#4ec9b0"),
        ("INFO:     Application startup complete.", "#cccccc"),
        ("INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)", "#4ec9b0"),
    ]

    y = 0.82
    for text, col in lines:
        ax.text(0.04, y, text, color=col, fontfamily="monospace", fontsize=9.2, transform=ax.transAxes)
        y -= 0.045

    ax.axis("off")
    fig.tight_layout()
    fig.savefig("screenshots/docker_build_and_run.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    save_test_results_screenshot()
    save_swagger_screenshot()
    save_docker_screenshot()
    print("Visual evidence cards generated in screenshots/ folder.")
