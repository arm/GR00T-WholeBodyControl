#!/usr/bin/env python3
"""Send one command to the SONIC VLA keyboard subscriber."""

import argparse
import time

import zmq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="k, p, i, [, ], or prompt:<text>")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5580)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    args = parser.parse_args()

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://{args.host}:{args.port}")
    try:
        time.sleep(args.settle_seconds)
        publisher.send_string(args.command)
    finally:
        publisher.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
