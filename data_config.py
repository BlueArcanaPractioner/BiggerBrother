"""
BiggerBrother Data Directory Configuration
==========================================

Centralizes all data directory paths to maintain compatibility with existing
modules while allowing behavioral_engine to use the same data structure.
"""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path("C:/BiggerBrother-minimal")
DATA_ROOT = PROJECT_ROOT / "data"
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_output"

# Existing directories (keep using these)
CHUNKS_DIR = DATA_ROOT / "chunks"                    # Existing: data/chunks/
LABELS_DIR = PROJECT_ROOT / "labels"                 # Existing: root/labels/
EMBEDDING_CACHE_FILE = DATA_ROOT / "embedding_cache.pkl"  # Existing: data/embedding_cache.pkl
PROFILES_DIR = ANALYSIS_ROOT / "profiles"            # Existing: analysis_output/profiles/

# New directories for behavioral_engine (will be created in data/)
HARMONIZER_DIR = DATA_ROOT            # New: data/harmonizer/
SCHEDULER_DIR = DATA_ROOT / "scheduler"              # New: data/scheduler/
LOGBOOKS_DIR = DATA_ROOT / "logbooks"               # New: data/logbooks/
FEATURES_DIR = DATA_ROOT / "features"               # New: data/features/
RPG_DIR = DATA_ROOT / "rpg"                         # New: data/rpg/
ROUTINES_DIR = DATA_ROOT / "routines"               # New: data/routines/
TRACKING_DIR = DATA_ROOT / "tracking"               # New: data/tracking/
GRAPH_DIR = DATA_ROOT / "graph"                     # New: data/graph/
PLANNER_DIR = DATA_ROOT / "planner"                 # New: data/planner/
ORCHESTRATOR_DIR = DATA_ROOT / "orchestrator"       # New: data/orchestrator/
CONTEXT_CACHE_DIR = DATA_ROOT / "context_cache"     # New: data/context_cache/

# Harmonizer specific
HARMONIZER_CACHE_FILE = HARMONIZER_DIR / "embedding_cache.pkl"

# LogBook categories
LOGBOOK_CATEGORIES = [
    "medications",
    "meals", 
    "exercise",
    "mood",
    "sleep",
    "productivity",
    "symptoms",
    "sessions"
]

def get_config_dict():
    """
    Returns a dictionary of all configured paths.
    Useful for passing to classes that need multiple directories.
    """
    return {
        'project_root': str(PROJECT_ROOT),
        'data_root': str(DATA_ROOT),
        'chunks_dir': str(CHUNKS_DIR),
        'labels_dir': str(LABELS_DIR),
        'embedding_cache': str(EMBEDDING_CACHE_FILE),
        'profiles_dir': str(PROFILES_DIR),
        'harmonizer_dir': str(HARMONIZER_DIR),
        'harmonizer_cache': str(HARMONIZER_CACHE_FILE),
        'scheduler_dir': str(SCHEDULER_DIR),
        'logbooks_dir': str(LOGBOOKS_DIR),
        'features_dir': str(FEATURES_DIR),
        'rpg_dir': str(RPG_DIR),
        'routines_dir': str(ROUTINES_DIR),
        'tracking_dir': str(TRACKING_DIR),
        'graph_dir': str(GRAPH_DIR),
        'planner_dir': str(PLANNER_DIR),
        'orchestrator_dir': str(ORCHESTRATOR_DIR),
        'context_cache_dir': str(CONTEXT_CACHE_DIR),
    }

def create_missing_directories():
    """
    Creates only the directories that don't exist yet.
    Preserves existing directories and their contents.
    """
    directories_to_create = [
        DATA_ROOT,
        HARMONIZER_DIR,
        SCHEDULER_DIR,
        LOGBOOKS_DIR,
        FEATURES_DIR,
        RPG_DIR,
        ROUTINES_DIR,
        TRACKING_DIR,
        GRAPH_DIR,
        PLANNER_DIR,
        ORCHESTRATOR_DIR,
        CONTEXT_CACHE_DIR,
    ]
    
    # Add logbook subdirectories
    for category in LOGBOOK_CATEGORIES:
        directories_to_create.append(LOGBOOKS_DIR / category)
    
    created = []
    already_existed = []
    
    for directory in directories_to_create:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(str(directory))
        else:
            already_existed.append(str(directory))
    
    # Report what was created
    if created:
        print("✅ Created missing directories:")
        for dir_path in created:
            print(f"   {dir_path}")
    
    if already_existed:
        print("📁 Using existing directories:")
        for dir_path in already_existed[:5]:  # Show first 5
            print(f"   {dir_path}")
        if len(already_existed) > 5:
            print(f"   ... and {len(already_existed) - 5} more")
    
    # Create README in logbooks if needed
    logbooks_readme = LOGBOOKS_DIR / "README.md"
    if not logbooks_readme.exists():
        readme_content = """# LogBooks Directory

Each subdirectory is a separate log category with:
- `{category}.csv` - Structured log entries
- `{category}.jsonl` - Full log entries with metadata
- `daily_YYYYMMDD.json` - Daily summaries
- `README.md` - Category documentation

Categories are created dynamically based on:
1. User requests
2. GPT-3.5-turbo proposals from conversations
3. System defaults

The system uses SmartHarmonizer V3 to intelligently group labels while preserving unique concepts.
"""
        with open(logbooks_readme, "w") as f:
            f.write(readme_content)
        print("📝 Created LogBooks README")
    
    return created, already_existed

def verify_external_dependencies():
    """
    Verify that expected external directories exist.
    These are directories that should already exist from other modules.
    """
    external_dirs = {
        'chunks': CHUNKS_DIR,
        'labels': LABELS_DIR,
        'profiles': PROFILES_DIR,
    }
    
    missing = []
    found = []
    
    for name, path in external_dirs.items():
        if path.exists():
            found.append(name)
        else:
            missing.append((name, str(path)))
    
    if found:
        print("✅ Found external directories:")
        for name in found:
            print(f"   {name}: {external_dirs[name]}")
    
    if missing:
        print("⚠️ Missing expected external directories:")
        for name, path in missing:
            print(f"   {name}: {path}")
            print(f"     → You may need to run the module that creates this directory")
    
    return missing

if __name__ == "__main__":
    """Test the configuration and create missing directories."""
    print("BiggerBrother Data Configuration")
    print("=" * 60)
    
    # Verify external dependencies
    missing_external = verify_external_dependencies()
    
    print()
    
    # Create missing directories
    created, existed = create_missing_directories()
    
    print()
    print("Configuration Summary:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Data root: {DATA_ROOT}")
    print(f"  Using external labels: {LABELS_DIR}")
    print(f"  Using external chunks: {CHUNKS_DIR}")
    print(f"  Using external profiles: {PROFILES_DIR}")
    
    if missing_external:
        print()
        print("⚠️ Some external directories are missing.")
        print("  These should be created by running the relevant modules.")
