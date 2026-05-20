#!/usr/bin/env python3
"""Serve large Gaussian splat assets with browser-friendly CORS headers."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class SplatRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header("Access-Control-Expose-Headers", "Accept-Ranges, Content-Length, Content-Range")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/jr/CloudTwin_Splat/public")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"splat root does not exist: {root}")

    handler = partial(SplatRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[mjswan] Serving splats from {root} on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
