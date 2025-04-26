# VoTranhOmniLearner

**VoTranhOmniLearner** is an advanced, self-evolving simulation system designed to model a dynamic universe of entities, social interactions, and reflective processes. It integrates machine learning, database management, web APIs, and rich console interfaces to create a "digital cosmos" that evolves through complex social dynamics, emotional fields, and introspective mechanisms. The system is built to explore concepts of consciousness, societal evolution, and knowledge archiving in a computational framework.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## Overview
VoTranhOmniLearner simulates a universe with thousands of entities (planets, stars, sentient beings) and imaginary constructs (subworlds, dynasties, social graphs). It uses a combination of probabilistic rules, machine learning (via SentenceTransformer for text vectorization), and a SQLite database to store states, knowledge, and archives. The system is monitored through a FastAPI-based API and a rich console interface, providing real-time insights into its evolution.

The core philosophy of VoTranhOmniLearner is to create a self-sustaining system that reflects, interrogates, and evolves based on internal and external stimuli, mimicking the complexity of a living cosmos.

## Key Features
- **Universe Simulation**: Models a scalable universe with up to 50,000 entities, governed by probabilistic laws (collision, birth, extinction).
- **Imaginary Entities**: Supports a dynamic ecosystem of imaginary entities with roles (servant, co-creator, rival, mythic), evolving through levels and attributes (intelligence, loyalty, creativity).
- **Social Dynamics**: Implements a social graph with allies, rivals, and neutral relationships, driven by message passing (encouragement, conflict, reflection triggers).
- **Emotional Field**: Simulates an emotional landscape (anger, inspiration, calm) that influences entity behavior.
- **Knowledge Archiving**: Uses SentenceTransformer to vectorize and store text-based knowledge, enabling similarity-based querying.
- **Database Management**: Stores states, knowledge, universe entities, reflections, and archives in a SQLite database with batch processing for efficiency.
- **Real-Time Monitoring**: Provides a FastAPI-based API and a rich console interface for live metrics (pulse, growth, entity counts, emotion field).
- **Reflective Mechanisms**: Supports self-reflection and self-interrogation to simulate consciousness-like behavior.
- **File System Integration**: Scans and reacts to text-based files (.txt, .md, .py, .json, .csv, .log) to build an internal language and influence graph.

## Architecture
The system is composed of several interconnected modules:
- **EvolutionVault**: Manages SQLite database operations for storing states, knowledge, universe entities, reflections, and archives.
- **OptimizationFormula**: Computes growth metrics based on diversity, entropy, file count, and node activity.
- **InternalGenesisEngine**: Core engine for managing imaginary entities, subworlds, dynasties, social interactions, and emotional fields.
- **UniverseSimulator**: Simulates a physical universe with entities (planets, stars, sentient beings) governed by probabilistic laws.
- **VoTranhMonitor**: Provides real-time monitoring via a FastAPI API and a rich console interface with live tables and panels.
- **VoTranhOmniLearner**: Orchestrates the entire system, integrating file scanning, machine learning, and threading for continuous operation.

The system uses a heartbeat mechanism to periodically evolve, reflect, and save states, ensuring a dynamic and responsive simulation.

## Installation
### Prerequisites
- Python 3.8+
- SQLite (included with Python)
- CUDA-enabled GPU (optional, for faster ML processing with `torch`)

### Dependencies
Install the required Python packages using `pip`:
```bash
pip install torch numpy pandas fastapi uvicorn sqlite3 sentence-transformers rich
```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/vinhatson/VoTranhOmniLearner.git
   cd VoTranhOmniLearner
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Configure the SentenceTransformer cache:
   - Set the cache folder to avoid repeated downloads:
     ```bash
     export SENTENCE_TRANSFORMERS_HOME=~/sentence_transformers_cache
     ```

5. Run the system:
   ```bash
   python omnilearner.py
   ```

## Usage
### Running the System
Start the VoTranhOmniLearner by executing:
```bash
python omnilearner.py
```
This launches three main components:
- **Heartbeat Thread**: Drives the simulation loop, scanning files, evolving the universe, and triggering reflections.
- **Monitor Thread**: Displays real-time metrics in a rich console interface (pulse, growth, entity counts, emotion field).
- **API Thread**: Runs a FastAPI server on `http://127.0.0.1:8000` for querying system status.

### Accessing the API
- Open a browser or use `curl` to access the status endpoint:
  ```bash
  curl http://127.0.0.1:8000/status
  ```
- The API returns JSON with metrics like:
  ```json
  {
    "timestamp": "2025-04-26 23:06:34",
    "pulse": 75,
    "nodes": 1234,
    "growth": 1.0,
    "universe_entities": 74464,
    "imaginary_entities": 549,
    "subworlds": 0,
    "dynasties": 0,
    "knowledge_entries": 1000,
    "archive_entries": 2000,
    "reflection_count": 50,
    "emotion_field": {"anger": 0.385, "inspiration": 0.344, "calm": 0.271}
  }
  ```

### Monitoring Output
The console displays a live table and panel with:
- Metrics: Pulse, nodes, growth, entity counts, reflections, etc.
- Recent Activity: Knowledge entries, reflections, archives, logs, and legends.
- Emotion Field: Current state of anger, inspiration, and calm.

### Stopping the System
Press `Ctrl+C` to gracefully shut down, generating a final report (`omnilearner_report_YYYYMMDD_HHMMSS.csv`).

## Configuration
Customize the system by modifying constants in `omnilearner.py`:
- `UNIVERSE_SCALE`: Number of universe entities (default: 5000).
- `ENTITY_MESSAGE_LIMIT`: Maximum messages per entity (default: 500).
- `SUBWORLD_SCALE`: Maximum entities per subworld (default: 20).
- `ARCHIVE_SCALE`: Maximum archive entries (default: 10000).
- `SCAN_PATH`: Directory to scan for files (default: `/` on Unix, `C:\\` on Windows).
- `FILE_TYPES`: Supported file extensions (default: `.txt`, `.md`, `.py`, `.json`, `.csv`, `.log`).
- `EXCLUDED_DIRS`: Directories to skip during file scanning (default: `/proc`, `/sys`, etc.).

Example modification:
```python
UNIVERSE_SCALE = 10000  # Increase universe size
SCAN_PATH = "/path/to/custom/directory"  # Custom scan path
```

## Technical Details
### Core Components
- **EvolutionVault**:
  - SQLite database with tables: `states`, `knowledge`, `universe`, `reflections`, `archives`.
  - Supports batch inserts for efficiency (`store_knowledge_batch`, `store_archive_batch`).
  - Stores vectors as `BLOB` for ML-based querying.
- **InternalGenesisEngine**:
  - Manages imaginary entities with attributes (intelligence, loyalty, creativity, emotional_depth).
  - Simulates social dynamics via a `defaultdict`-based social graph and `deque`-based message queues.
  - Supports subworlds, dynasties, and emotional fields.
- **UniverseSimulator**:
  - Simulates physical entities (planets, stars, sentient) with probabilistic laws.
  - Integrates with `InternalGenesisEngine` for hybrid physical-imaginary evolution.
- **VoTranhMonitor**:
  - Uses FastAPI for a `/status` endpoint.
  - Renders live metrics with `rich` (tables, panels, live updates).
  - Generates CSV reports for historical analysis.
- **VoTranhOmniLearner**:
  - Orchestrates the system with a heartbeat loop (`_heartbeat`).
  - Scans files to build an internal language and influence graph.
  - Uses `SentenceTransformer` (`all-MiniLM-L6-v2` or fallback `paraphrase-MiniLM-L3-v2`) for text vectorization.

### Performance Optimizations
- **Batch Processing**: Reduces database I/O with batch inserts.
- **Memory Management**: Uses `deque` with `maxlen` for message queues and archives.
- **Threading**: Separates heartbeat, monitoring, and API tasks to prevent blocking.
- **Error Handling**: Comprehensive `try-except` blocks with logging to `omnilearner.log`.

### Scalability
- Supports up to 50,000 universe entities and 10,000 archive entries.
- Configurable scale parameters (`UNIVERSE_SCALE`, `ARCHIVE_SCALE`) for larger simulations.
- SQLite database may become a bottleneck for very large datasets; consider NoSQL for extreme scaling.

### Logging
- Logs are written to `omnilearner.log` with timestamp, level, and message.
- Key events (entity evolution, message sending, rivalry declarations) are logged for debugging and analysis.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/vinhatson`).
3. Commit changes (`git commit -m 'Add vinhatson'`).
4. Push to the branch (`git push origin feature/vinhatson`).
5. Open a Pull Request.

Please include tests and update documentation for new features.

## License
Licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).  
You are free to use, modify, and distribute under the terms provided.