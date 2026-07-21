#!/usr/bin/env python3
"""
Simittag app: encode a payload to a marker, or decode an image.

  python app.py encode --id 12345 --out marker.png
  python app.py encode --raw "hi" --out marker.png
  python app.py decode image.png
  python app.py calibrate img1.png img2.png ... --out intrinsics.json
  python app.py decode image.png --intrinsics intrinsics.json

Thin shim around simittag.cli so the repo layout keeps working; the
installed package exposes the same interface as the `simittag` command.
"""
from simittag.cli import main

if __name__ == "__main__":
    main()
