#!/usr/bin/env python3
"""Drive install_diffrg.sh's wizard on a pseudo-terminal, pressing Enter at
every question (i.e. taking each default), as a user at a terminal would.

  installer_pty.py [--expect TEXT]... [--abort] -- bash install_diffrg.sh [ARGS]

The wizard only asks questions when it can open /dev/tty, so --yes or a pipe
skips exactly the code paths under test; a pty gives it a real terminal.
--expect fails the run unless TEXT appeared in the output (e.g. a question
that must be asked). --abort answers "n" at the final "Proceed?" instead of
installing. Exits with the installer's status, or 1 on a missing --expect.
"""
import argparse
import os
import pty
import re
import select
import sys
import time

# A menu's title line, or the "[default]: " of a free-text question. Both are
# printed last before the wizard blocks on input.
PROMPT = re.compile(r"\((arrows|space toggles)[^)]*\)|\]: $")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
QUIET = 1.0  # seconds without output before a prompt counts as waiting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", action="append", default=[])
    ap.add_argument("--abort", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no installer command given")

    pid, fd = pty.fork()
    if pid == 0:
        os.environ.setdefault("TERM", "xterm")
        os.execvp(cmd[0], cmd)

    transcript, pending, last_output = "", "", time.monotonic()
    while True:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if ready:
            try:
                data = os.read(fd, 65536)
            except OSError:  # EIO: the installer exited (Linux)
                data = b""
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.flush()
            text = ANSI.sub("", data.decode(errors="replace"))
            transcript += text
            pending += text
            last_output = time.monotonic()
        elif pending and time.monotonic() - last_output > QUIET:
            tail = pending.rstrip("\n")
            if PROMPT.search(tail):
                answer = b"n\n" if args.abort and tail.endswith("Proceed? [Y/n]: ") else b"\n"
                print(f"\n[installer_pty] answering {answer!r}", flush=True)
                os.write(fd, answer)
            pending = ""

    _, status = os.waitpid(pid, 0)
    rc = os.waitstatus_to_exitcode(status)
    missing = [e for e in args.expect if e not in transcript]
    for e in missing:
        print(f"\n[installer_pty] FAIL: never saw {e!r}", flush=True)
    print(f"\n[installer_pty] installer exited with {rc}", flush=True)
    sys.exit(1 if missing and rc == 0 else rc)


if __name__ == "__main__":
    main()
