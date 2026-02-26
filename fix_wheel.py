#!/usr/bin/env python3
"""Script to add metadata to the wheel final (circular diagram)"""

import sys

# Read the file
with open('/workspaces/placegamedraft/streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the section we need to replace
# We're looking for the section that contains ax.text with "Lugar"
old_section = '''    # ----- 6.  TEXTO "LUGAR" ---------------------------------------

    ax.text(
    0, 0, "Lugar",
    ha="center",
    va="center",
    fontsize=7,
    color="#333333",
    fontweight="bold",
    )

    # ----- 7.  MOSTRAR EN STREAMLIT ------------------------------------------------'''

new_section = '''    # ----- 6.  METADATA Y TEXTO "LUGAR" -----------------------------------------
    
    # Timestamp (izquierda)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(
        -5.5, 5.5, timestamp,
        ha="center",
        va="center",
        fontsize=6,
        color="#666666",
        fontweight="normal",
        style="italic"
    )
    
    # Nombre del lugar (centro)
    ax.text(
        0, 0, nombre_lugar or "Lugar",
        ha="center",
        va="center",
        fontsize=8,
        color="#333333",
        fontweight="bold",
    )
    
    # Nombre del evaluador (derecha)
    ax.text(
        5.5, 5.5, nombre_eval or "Evaluador",
        ha="center",
        va="center",
        fontsize=6,
        color="#666666",
        fontweight="normal"
    )

    # ----- 7.  MOSTRAR EN STREAMLIT ------------------------------------------------'''

if old_section in content:
    print("Found the section to replace!")
    content = content.replace(old_section, new_section)
    with open('/workspaces/placegamedraft/streamlit_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("File updated successfully!")
else:
    print("Section not found. Checking for variants...")
    # Check if the text is there but with different quote marks
    if "TEXTO" in content and "LUGAR" in content and 'ax.text(' in content:
        print("Some parts are there but exact match failed")
        # Try to find it with regex
        import re
        # Look for the pattern
        pattern = r'# ----- 6\..*?ax\.text\(.*?\n.*?fontweight="bold",\s*\)'
        matches = list(re.finditer(pattern, content, re.DOTALL))
        if matches:
            print(f"Found {len(matches)} potential matches with regex")
            # For now, let's just extract one to see what's different
            m = matches[-1]  # Get the last one (most likely the one we want)
            print(f"Match: {m.group(0)[:100]}...")
        else:
            print("No regex matches either")
    sys.exit(1)
