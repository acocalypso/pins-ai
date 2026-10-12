#!/usr/bin/env python3
"""Inference-transport fixture ONLY. Not an MCP server; no equipment interfaces."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int)
parser.add_argument('--host')
parser.add_argument('--api-key-file')
args, _ = parser.parse_known_args()
token = Path(args.api_key_file).read_text().strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *unused):
        pass

    def do_GET(self):
        if self.headers.get('Authorization') != 'Bearer ' + token:
            self.send_error(401)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        if self.headers.get('Authorization') != 'Bearer ' + token:
            self.send_error(401)
            return
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body['messages']
        if messages[0]['role'] != 'system' or sum(m['role'] == 'system' for m in messages) != 1:
            self.send_error(500, 'Template requires one leading system message')
            return
        prompt = body['messages'][-1]['content']
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        if prompt == 'oversize':
            self.wfile.write(b'data: ' + b'x' * 65537 + b'\n\n')
            self.wfile.flush()
            return
        if prompt == 'invalid-json':
            self.wfile.write(b'data: {broken}\n\n')
            self.wfile.flush()
            return
        if prompt == 'allocate':
            self.allocation = bytearray(280 * 1024 * 1024)
        pieces = ['First', ' second', ' third']
        for piece in pieces:
            self.wfile.write(('data: ' + json.dumps({'choices': [{'delta': {'content': piece}}]}) + '\n\n').encode())
            self.wfile.flush()
            time.sleep(15 if prompt in ('stall', 'allocate') else .1)
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()


ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
