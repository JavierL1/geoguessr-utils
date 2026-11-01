# geoguessr-utils

A command-line toolkit for extracting structured data from Geoguessr
match videos. It walks through a recording, finds each score screen,
and produces a JSON file with per-player scores, distances, health
points, and usernames.

Built to make tournament analysis and post-match review scriptable.


## What it does

Geoguessr records the end-of-round score screen as a still image
embedded in the video. This tool provides a suite of commands to:

1. **Discover** every score screen in a video file and save the last
   frame of each one to disk.
2. **Filter** out false positives from the discovered frames.
3. **Extract** structured data from each saved frame:
   - Match score (0, 1, or 2)
   - Round score (integer)
   - Distance (e.g. 511.0 km)
   - Health points remaining
   - Username (fuzzy-matched against a known roster)
4. **Annotate** frames to generate golden test fixtures.

The output is a single JSON file, ready for spreadsheets, databases,
or further analysis.


## How it works

The pipeline is broken into modular CLI commands:

    match.mp4 --[gg-discover]--> batch_001_frame_0012345.jpg
                                  batch_002_frame_0045678.jpg
                                  ...
                                  (moved to _false_positives/)

             --[gg-filter]--->  (clean directory of frames)

             --[gg-extract]-->  results.json

**Discovery** (`gg-discover`) scans the video, checking whether the current
frame matches a score screen. When it detects the start of a score screen, 
it switches to a faster sampling rate and saves the fully-rendered final frame.

**Extraction** (`gg-extract`) reads each saved frame, crops the ROIs for
each player's features, and routes each crop through a feature-
specific OCR and parsing pipeline. Numbers and distances go through
Tesseract with tuned preprocessing; usernames go through fuzzy
matching against the roster.

The architecture is layout-driven: everything screen-specific (ROI
coordinates, usernames, detection heuristic) lives behind a Layout
and ScreenDetector interface. Adding support for a new Geoguessr UI
means writing one new layout file, not modifying the pipeline.


## Installation

Requires Python 3.14+ and Tesseract OCR.

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

`uv sync` creates a virtual environment and installs everything from
`pyproject.toml` and `uv.lock`. You do not need to activate anything --
`uv run` handles it.


## Quickstart

Run the pipeline stages in order. Note that the commands require
you to specify which UI layout the video uses:

    # 1. Discover score screens (using the 2026A layout)
    uv run gg-discover -v match.mp4 -o frames/ --layout 2026a

    # 2. Filter out false positives (optional, but recommended)
    uv run gg-filter -i frames/ --layout 2026a

    # 3. Extract data from the saved frames
    uv run gg-extract -i frames/ -o results.json --layout 2026a

Sample output from results.json:

    [
      {
        "screen_index": "001",
        "frame_index": 12345,
        "source_image": "batch_001_frame_0012345.jpg",
        "time_code": "00:03:25",
        "left_player": {
          "match_score": null,
          "username": "MALFUNCTIONN",
          "round_score": 27,
          "distance": { "amount": 7027.0, "unit": "km" },
          "health_points": 0
        },
        "right_player": {
          "match_score": null,
          "username": "BULGARIAENJOYER",
          "round_score": 3654,
          "distance": { "amount": 423.0, "unit": "km" },
          "health_points": 3948
        }
      }
    ]


## CLI Commands

All commands are registered via `pyproject.toml` and can be run using `uv run`.

**Pipeline Commands:**
- `gg-discover`: Finds score screens in a video and extracts frames.
- `gg-filter`: Classifies a folder of frames and moves false positives to a `_false_positives` subfolder. Optionally renumbers the remaining batches to be contiguous.
- `gg-extract`: Extracts structured player data from a folder of frames into a JSON file.

**Development & Debugging Commands:**
- `gg-annotate-frames`: Interactively prompts you to annotate images in a folder, generating `.json` golden files for your tests.
- `gg-visualize`: Draws the ROIs from a flat `config.json` onto a sample image for debugging.
- `gg-visualize-layout`: Draws the Layout and Detector ROIs defined in the Python layout class onto a sample image.
- `gg-select-roi`: A helper tool to interactively select and print ROI coordinates from an image.

## Configuration

`gg-extract` takes a `-c/--config` flag pointing to a `config.json` file:

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
        Target resolution. Reserved for multi-resolution support.

Everything else -- ROIs, known usernames, detection thresholds --
lives in the layout, not in the config.


## Layouts

Geoguessr periodically updates its broadcast UI. To handle this, the 
project uses a strategy pattern.

Every supported UI version has two components:
1. A `Layout` class (defines where the ROIs are and how to extract text).
2. A `ScreenDetector` class (defines the OpenCV heuristic used to find the score screen).

These are paired together in a central registry (`layout_registry.py`).

Currently supported layouts:

    2026b
        The standard Geoguessr UI.

    2026a
        The World Championships Berlin UI (no match score, different health bars).

To use a specific layout, pass the `-l/--layout` flag to `gg-discover`, 
`gg-filter`, or `gg-extract`.

    uv run gg-discover -v match.mp4 -o frames/ --layout 2026a
