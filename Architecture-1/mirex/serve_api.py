"""
MusicScope-CL — High-Performance Model Serving API (Zero-Dependency Microservice)

Provides ultra-fast HTTP REST endpoints for real-time model inference:
  - GET  /health         : System health, GPU telemetry, VRAM usage, and checkpoint status.
  - POST /predict        : Real-time single track inference (JSON or file path).
  - POST /predict_batch  : High-throughput batch inference (JSON list of file paths).

Built with Python's high-concurrency ThreadingHTTPServer — zero extra pip dependencies required.

Usage:
    python serve_api.py --port 8000
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"path": "audio.mp3"}'
"""
import argparse
import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

import torch

_CUR_DIR = Path(__file__).resolve().parent
if str(_CUR_DIR) not in sys.path:
    sys.path.insert(0, str(_CUR_DIR))

import config
from new_inference import FastMusicScopeScorer

# Global singleton scorer
SCORER: FastMusicScopeScorer | None = None


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MusicScopeAPIHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            self._send_json(200, {
                "status": "healthy",
                "service": "MusicScope-CL Inference API",
                "device": str(SCORER.device if SCORER else "uninitialized"),
                "gpu_name": gpu_name,
                "vram_allocated_mb": round(vram_mb, 2),
                "model_version": "200k_large_scale",
                "checkpoints": {
                    "simclr": "simclr_200k_best.pt",
                    "supcon": "supcon_200k_best.pt",
                    "fusion": "fusion_200k_best.pt"
                }
            })
        else:
            self._send_json(404, {"error": f"Endpoint '{self.path}' not found. Use /health or /predict"})

    def do_POST(self):
        if self.path == "/predict":
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len == 0:
                self._send_json(400, {"error": "Empty request body. Provide JSON with 'path' field."})
                return

            raw_body = self.rfile.read(content_len)
            try:
                data = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON payload: {e}"})
                return

            track_path = data.get("path")
            if not track_path or not Path(track_path).exists():
                self._send_json(404, {"error": f"Audio file not found at: '{track_path}'"})
                return

            t0 = time.perf_counter()
            try:
                prob = SCORER.score_single(track_path)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                label = "AI-Generated" if prob >= 0.50 else "Human Original"
                self._send_json(200, {
                    "path": track_path,
                    "filename": Path(track_path).name,
                    "probability_ai": round(prob, 4),
                    "prediction": label,
                    "latency_ms": round(elapsed_ms, 2)
                })
            except Exception as e:
                self._send_json(500, {"error": f"Inference failed on track: {e}"})

        elif self.path == "/predict_batch":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len)
            try:
                data = json.loads(raw_body.decode("utf-8"))
                paths = data.get("paths", [])
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON payload: {e}"})
                return

            if not paths:
                self._send_json(400, {"error": "Empty 'paths' list."})
                return

            t0 = time.perf_counter()
            results = SCORER.score_directory(paths, batch_size=data.get("batch_size", 128))
            elapsed_s = time.perf_counter() - t0

            self._send_json(200, {
                "total_tracks": len(results),
                "elapsed_seconds": round(elapsed_s, 2),
                "throughput_tracks_per_sec": round(len(results) / max(elapsed_s, 0.001), 1),
                "predictions": results
            })
        else:
            self._send_json(404, {"error": f"POST endpoint '{self.path}' not found."})

    def log_message(self, format, *args):
        # Clean logging format
        sys.stderr.write(f"[API Server] {self.address_string()} - {format%args}\n")


def main():
    global SCORER
    parser = argparse.ArgumentParser(description="MusicScope-CL Model Serving API")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()

    print("=" * 70)
    print("  MusicScope-CL Model Serving API Initializing...")
    print("=" * 70)

    # Preload global scorer
    SCORER = FastMusicScopeScorer()

    server_address = (args.host, args.port)
    httpd = ThreadedHTTPServer(server_address, MusicScopeAPIHandler)
    print(f"\n[Serving API] Server listening at http://{args.host}:{args.port}")
    print(f"              Endpoints: /health (GET), /predict (POST), /predict_batch (POST)")
    print(f"              Press Ctrl+C to terminate.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Serving API] Shutting down cleanly...")
        httpd.server_close()


if __name__ == "__main__":
    main()
