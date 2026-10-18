# geoguessr-utils

A command-line toolkit for extracting structured data from Geoguessr
match videos. It walks through a recording, finds each score screen,
and produces a JSON file with per-player scores, distances, health
points, and usernames.

Built to make tournament analysis and post-match review scriptable.


## What it does

Geoguessr records the end-of-round score screen as a still image
embedded in the video. This tool:

1. Discovers every score screen in a video file and saves the last
   frame of each one to disk (one JPEG per match).
2. Extracts structured data from each saved frame:
   - Match score (0, 1, or 2)
   - Round score (integer)
   - Distance (e.g. 511.0 km)
   - Health points remaining
   - Username (fuzzy-matched against a known roster)

The output is a single JSON file, ready for spreadsheets, databases,
or further analysis.


## How it works

The pipeline has two stages, both exposed as CLI commands:

    match.mp4  --[gg-discover]-->  batch_001_frame_0012345.jpg
                                   batch_002_frame_0045678.jpg
                                   ...

    frames/*.jpg  --[gg-extract]-->  results.json

Discovery (gg-discover) scans the video at roughly 1 frame per second,
checking whether the current frame matches a score screen using color
ratios, text density, and health-bar visibility. When it detects the
start of a score screen, it switches to a 4x-per-second sampling rate
and keeps overwriting the last-valid-frame until the screen ends --
at which point the fully-rendered final frame is saved.

Extraction (gg-extract) reads each saved frame, crops the ROIs for
each player's features, and routes each crop through a feature-
specific OCR and parsing pipeline. Numbers and distances go through
Tesseract with tuned preprocessing; usernames go through fuzzy
matching against the roster.

The architecture is layout-driven: everything screen-specific (ROI
coordinates, usernames, detection heuristic) lives behind a Layout
interface. Adding support for a new Geoguessr UI means writing one
new layout file, not modifying the pipeline.


## Installation

Requires Python 3.12+ and Tesseract OCR.

### Step 1: Install Tesseract

macOS:

    brew install tesseract

Ubuntu / Debian:

    sudo apt install tesseract-ocr

Windows:

Download the installer from the UB-Mannheim tesseract wiki
(https://github.com/UB-Mannheim/tesseract/wiki) and ensure the install
directory is on your PATH.

### Step 2: Install the project

This project uses uv (https://docs.astral.sh/uv/) for dependency
management. If you do not have it:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Then clone the repo and install:

    git clone <your-repo-url> geoguessr-utils
    cd geoguessr-utils
    uv sync

uv sync creates a virtual environment and installs everything from
pyproject.toml and uv.lock. You do not need to activate anything --
uv run handles it.


## Quickstart

Run the two pipeline stages in order:

    uv run gg-discover -v match.mp4 -o frames/
    uv run gg-extract -i frames/ -o results.json

Sample output from results.json:

    [
      {
        "screen_index": "001",
        "frame_index": 12345,
        "source_image": "batch_001_frame_0012345.jpg",
        "time_code": "00:03:25",
        "left_player": {
          "match_score": 1,
          "username": "BLINKY",
          "round_score": 4250,
          "distance": { "amount": 511.0, "unit": "km" },
          "health_points": 85
        },
        "right_player": {
          "match_score": 1,
          "username": "KAPTEN",
          "round_score": 4100,
          "distance": { "amount": 480.5, "unit": "km" },
          "health_points": 92
        }
      }
    ]


## Configuration

gg-extract takes a -c/--config flag pointing to a config.json file:

    {
      "FPS": 59.94006,
      "RESOLUTION": "1080p"
    }

Fields:

    FPS
        Type: float
        Default: 59.94006
        Used to convert frame indices into time codes.

    RESOLUTION
        Type: string
        Default: "1080p"
        Target resolution. Reserved for multi-resolution support;
        currently only "1080p" layouts exist.

Everything else -- ROIs, known usernames, detection thresholds --
lives in the layout, not in the config. See the Layouts section for
details.