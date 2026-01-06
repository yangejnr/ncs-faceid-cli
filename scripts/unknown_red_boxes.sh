#!/usr/bin/env bash
set -euo pipefail

python - << 'PY'
from pathlib import Path

p = Path("src/faceid/cli.py")
text = p.read_text(encoding="utf-8").splitlines()

out = []
removed_green_rect = 0
patched_puttext = 0
inserted_color_logic = 0

for i, line in enumerate(text):
    # Remove the early green rectangle draw (we will draw after decision is known)
    if "cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)" in line:
        removed_green_rect += 1
        continue

    # Insert color logic + rectangle draw right before the existing putText call
    if "cv2.putText(frame, label" in line and patched_puttext == 0:
        # Insert color logic once per detection loop occurrence. Your file has only one such loop.
        out.append("            # Box color: UNKNOWN -> red, otherwise green")
        out.append("            box_color = (0, 255, 0)")
        out.append('            if decision == "UNKNOWN":')
        out.append("                box_color = (0, 0, 255)")
        out.append("            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)")
        inserted_color_logic += 1

        # Patch putText to use box_color instead of hard-coded green
        newline = line.replace("(0, 255, 0), 2", "box_color, 2")
        out.append(newline)
        patched_puttext += 1
        continue

    # Also patch any other putText using hard-coded green (defensive)
    if "cv2.putText(frame, label" in line and "(0, 255, 0), 2" in line:
        line = line.replace("(0, 255, 0), 2", "box_color, 2")
        patched_puttext += 1

    out.append(line)

new_text = "\n".join(out) + "\n"
p.write_text(new_text, encoding="utf-8")

print("Patch summary:")
print(f"  removed early green rectangles: {removed_green_rect}")
print(f"  inserted color logic blocks:    {inserted_color_logic}")
print(f"  patched putText calls:          {patched_puttext}")

if inserted_color_logic == 0:
    raise SystemExit("ERROR: Did not find expected putText(label) location to insert color logic.")
PY

pip install -e . --force-reinstall >/dev/null

echo "Done."
echo "Run: faceid live --camera 0 --db data/faceid.sqlite3 --history-size 10"
