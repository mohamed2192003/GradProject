import subprocess, sys, os

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

result = subprocess.run(
    [sys.executable, r'f:\Graduation-Project\model1_survival\pipeline.py'],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace',
    cwd=r'f:\Graduation-Project',
    env=env,
)

with open(r'f:\Graduation-Project\run_log.txt', 'w', encoding='utf-8') as f:
    f.write("EXIT CODE: " + str(result.returncode) + "\n\n")
    f.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)

print("Exit code:", result.returncode)
print("Done - check run_log.txt")