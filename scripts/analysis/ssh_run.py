"""Tiny SSH helper: read password from SSHPASS env var, run a command, print output.

Usage:
    SSHPASS=skh024682 python scripts/analysis/ssh_run.py "ls /data3/khsong"
"""
import os
import sys
import paramiko

HOST = os.environ.get("SSH_HOST", "10.13.74.231")
PORT = int(os.environ.get("SSH_PORT", "66"))
USER = os.environ.get("SSH_USER", "khsong")
PASS = os.environ.get("SSHPASS")

if not PASS:
    print("SSHPASS env var required", file=sys.stderr)
    sys.exit(2)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "echo no-cmd"
stdin, stdout, stderr = client.exec_command(cmd, timeout=300)
out = stdout.read().decode("utf-8", errors="replace")
err = stderr.read().decode("utf-8", errors="replace")
sys.stdout.write(out)
sys.stdout.flush()
if err.strip():
    sys.stderr.write("STDERR: " + err[:1500] + "\n")
client.close()
