"""
Application runner for my facial recognition system

Author: J-A-Collins
"""

import subprocess

files_to_run = ("nn.py", "upload_user.py", "sys.py")

for file in files_to_run:
    subprocess.run(["python", file])


print("Face recog system complete.")
