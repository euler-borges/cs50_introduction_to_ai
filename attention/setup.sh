# Credits to user Makaze, on the CS50's discord server, for the original setup.sh file.

uv python install 3.10

# Init a project for the entire course
uv init --python 3.10 cs50ai

# Install the requirements
uv add numpy==1.26.4
uv add pillow==10.4.0
uv add tensorflow==2.17.0
uv add tensorflow-io-gcs-filesystem==0.37.0
uv add tf_keras==2.17.0
uv add transformers==4.44.0

# Make sure your projects are inside their own folders inside this project.
# For example:
#
# .
# ├── 0
# │   ├── degrees
# │   ├── degrees.zip
# │   ├── tictactoe
# │   └── tictactoe.zip
# ├── 1
# │   ├── knights
# │   ├── knights.zip
# │   ├── minesweeper
# │   └── minesweeper.zip

# From inside a project, run the project file using uv:
# e.g. from cs50ai/.../attention/ folder:
uv run mask.py