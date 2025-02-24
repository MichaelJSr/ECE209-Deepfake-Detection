import sys
from datetime import datetime


def log(*args, sep=" ", end="\n", file=sys.stdout, flush=False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]", *args, sep=sep, end=end, file=file, flush=flush)
