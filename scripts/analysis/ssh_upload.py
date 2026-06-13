"""Tiny SFTP uploader: upload a single local file to a remote path.

Usage:
    SSHPASS=xxx python ssh_upload.py <local> <remote>
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

if len(sys.argv) < 3:
    print("usage: ssh_upload.py <local> <remote>", file=sys.stderr)
    sys.exit(2)

local, remote = sys.argv[1], sys.argv[2]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
sftp = client.open_sftp()
sftp.put(local, remote)
print(f"  uploaded: {local} -> {remote}")
sftp.close()
client.close()
