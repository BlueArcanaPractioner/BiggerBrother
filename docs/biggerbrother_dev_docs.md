# BiggerBrother Developer Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)
7. [Usage Guide](#usage-guide)
8. [API Reference](#api-reference)
9. [Development Guide](#development-guide)
10. [Troubleshooting](#troubleshooting)

## System Overview

**BiggerBrother** is an advanced behavioral intelligence system that combines conversational AI with intelligent data organization to track, analyze, and provide insights into personal activities and patterns. The system uses a sophisticated two-tier label harmonization approach to organize information intelligently while preserving unique concepts.

### Key Features

- **Dual Scheduler System**: Combines adaptive scheduling (for patterns & scheduling) with context-aware scheduling (for message processing)
- **Two-Tier Label Harmonization**: Intelligent grouping with general (>0.5 similarity) and specific (>0.8 similarity) tiers
- **Dynamic LogBooks**: Automatically creates and manages categorized activity logs
- **Gmail OAuth Integration**: Email-based interaction with multiple conversation modes
- **RPG Gamification**: Motivational system with XP, levels, and achievements
- **Behavioral Routines**: Structured routine tracking and execution
- **Smart Context Loading**: Uses harmonized labels to selectively load relevant context

### Technology Stack

- **Python 3.8+**: Core language
- **OpenAI API**: GPT models for conversation and extraction (supports GPT-4o, GPT-3.5-turbo, GPT-5-nano)
- **Gmail API**: Email integration via OAuth 2.0
- **Data Storage**: JSON, JSONL, CSV, and pickle formats
- **Dependencies**: See `requirements.txt` (not included, but inferred from imports)

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Interface                     │
│        (Gmail OAuth / CLI / Direct API)             │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│            BiggerBrotherEmailSystem                  │
│  (Main orchestrator - enhanced_gmail_runner.py)     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│         CompleteIntegratedSystem                     │
│    (Core system - complete_system.py)               │
├──────────────────────────────────────────────────────┤
│  Components:                                         │
│  • Dual Schedulers (Adaptive + Context-Aware)       │
│  • EnhancedLabelHarmonizer (Two-Tier)              │
│  • DynamicLogBook System                            │
│  • Feature Extractor                                │
│  • RPG System                                       │
│  • Routine Builder                                  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Data Layer                              │
│  • GraphStore (JSONL)                               │
│  • LogBooks (CSV + JSONL)                          │
│  • Labels (JSON)                                    │
│  • Profiles (JSON)                                  │
└──────────────────────────────────────────────────────┘
```

### Directory Structure

```
C:/BiggerBrother/
├── app/                          # Core application modules
│   ├── openai_client.py         # OpenAI API wrapper with offline mode
│   ├── label_generator.py       # Label generation from text
│   ├── label_integration_wrappers.py  # Label system wrappers
│   └── coverage_checker.py      # Label coverage validation
│
├── assistant/                    # Assistant modules
│   ├── behavioral_engine/       # Main behavioral tracking engine
│   │   ├── core/
│   │   │   └── complete_system.py  # Central system orchestrator
│   │   ├── enhanced_gmail_runner.py  # Gmail integration & main entry
│   │   ├── gamification/        # RPG gamification system
│   │   ├── logbooks/            # Dynamic log categorization
│   │   ├── planners/            # Daily planning system
│   │   ├── routines/            # Behavioral routine tracking
│   │   └── schedulers/          # Dual scheduler system
│   │
│   ├── graph/                   # Graph data store
│   │   ├── models.py           # Data models (Node, Edge, LogEntry)
│   │   └── store.py            # JSONL-based graph storage
│   │
│   ├── importers/               # Data import and harmonization
│   │   ├── enhanced_smart_label_importer.py
│   │   └── harmonizer_v2.py    # Semantic harmonization
│   │
│   └── logger/                  # Logging systems
│       ├── unified.py           # Unified logging interface
│       └── temporal_personal_profile.py  # Personal profile tracking
│
├── schemas/                     # JSON schemas for data validation
│   ├── label_record.schema.json
│   ├── behavioral_log.schema.json
│   └── [other schemas...]
│
├── data/                        # Data storage (created at runtime)
│   ├── chunks/                 # Message chunks
│   ├── labels/                 # Generated labels
│   ├── logbooks/               # Activity logs by category
│   ├── features/               # Extracted features
│   ├── rpg/                   # Game state
│   ├── routines/              # Routine definitions
│   ├── scheduler/             # Schedule data
│   ├── graph/                 # Graph store
│   ├── profiles/              # User profiles
│   └── embedding_cache.pkl    # Cached embeddings
│
├── data_config.py              # Centralized path configuration
├── credentials.json            # Gmail OAuth credentials
├── token.pickle               # Gmail OAuth token
└── .env                       # Environment variables
```

## Core Components

### 1. OpenAI Client (`app/openai_client.py`)

Wrapper for OpenAI API with special handling for different models and offline mode.

**Key Features:**
- Model-specific parameter allowlisting (GPT-5-nano only supports response_format)
- Offline mode with deterministic JSON responses
- Automatic fallback between OpenAI SDK versions
- Error handling that never raises exceptions

### 2. Complete Integrated System (`complete_system.py`)

Central orchestrator that ties together all components.

**Key Responsibilities:**
- Initialize and coordinate all subsystems
- Process messages with context awareness
- Manage harmonization and label organization
- Coordinate between dual schedulers
- Handle session management (check-ins, detailed logging, routines)

### 3. Label Harmonization System

**Two-Tier Architecture:**
- **General Groups** (similarity > 0.5): Broad conceptual categories
- **Specific Groups** (similarity > 0.8): Fine-grained matching
- Uses smart filtering to minimize API calls:
  - String matching first (no API)
  - Groups similar strings together
  - Only uses embeddings for ambiguous cases (0.3 < similarity < 0.85)

**Files Generated:**
- `harmonization_general.json`: General tier mappings
- `harmonization_specific.json`: Specific tier mappings
- `harmonization_report.json`: Statistics and insights

### 4. Dynamic LogBook System (`dynamic_logbook_system.py`)

Automatically creates and manages categorized activity logs.

**Default Categories:**
- medications, meals, exercise, mood, sleep, productivity, symptoms

**Storage Format:**
- `{category}.csv`: Structured data for analysis
- `{category}.jsonl`: Full entries with metadata
- `daily_YYYYMMDD.json`: Daily summaries

### 5. Dual Scheduler System

**AdaptiveScheduler:**
- Pattern analysis and behavioral insights
- Daily schedule generation
- Pomodoro session management
- Activity pattern detection

**ContextAwareScheduler:**
- Message processing with context
- Label-based context loading
- Sandbox environment support
- Session metrics tracking

### 6. Gmail Integration (`enhanced_gmail_runner.py`)

Main entry point with email-based interaction.

**Conversation Modes:**
- `CHECK_IN`: Structured check-ins with prompts
- `CHAT`: Open-ended conversation mode
- `DETAILED`: Detailed activity logging with dual models
- `QUICK_LOG`: Quick entry logging
- `ROUTINE`: Guided routine execution

## Data Flow

### Label Generation and Harmonization Pipeline

```
1. User Message
      ↓
2. Generate Raw Labels (GPT-3.5-turbo)
      ↓
3. Harmonize Labels (Two-Tier System)
      ├── Check General Groups (>0.5 similarity)
      └── Check Specific Groups (>0.8 similarity)
      ↓
4. Find Similar Messages (using harmonized labels)
      ↓
5. Select Context Categories (based on labels)
      ↓
6. Load Relevant LogBook Context
      ↓
7. Process with Full Context
      ↓
8. Extract and Log Activities
      ↓
9. Update Harmonization Groups (if needed)
```

### Message Processing Flow

```python
# Simplified flow in process_message_with_context()
1. Generate raw labels from message
2. Harmonize labels using two-tier system
3. Find similar messages using probability-weighted similarity
4. Select logbook categories based on harmonized labels
5. Load context from selected categories
6. Process message based on active session type
7. Save harmonized labels to disk
8. Extract features and update RPG
9. Return response with context analysis
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Google Cloud Console account (for Gmail OAuth)
- Windows/Linux/MacOS

### Step 1: Clone Repository

```bash
git clone [repository-url]
cd BiggerBrother
```

### Step 2: Install Dependencies

```bash
pip install openai google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client python-dotenv plyer schedule
```

### Step 3: Configure Environment

Create `.env` file:
```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
AGENT_OFFLINE=0  # Set to 1 for offline mode
```

### Step 4: Setup Gmail OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Gmail API
4. Create OAuth 2.0 credentials
5. Download as `credentials.json` to project root
6. Run the system once to authenticate (will open browser)

### Step 5: Initialize System

```python
python assistant/behavioral_engine/enhanced_gmail_runner.py
```

## Configuration

### Using Centralized Configuration (`data_config.py`)

The system uses centralized path configuration for consistency:

```python
from data_config import create_missing_directories

# Initialize with config
system = CompleteIntegratedSystem(use_config=True)

# Or use custom paths
system = CompleteIntegratedSystem(
    base_dir="/custom/path",
    use_config=False
)
```

### Model Configuration

Configure models via environment variables or directly:

```python
# Via environment
OPENAI_MODEL=gpt-4o  # For conversations
OPENAI_MODEL=gpt-3.5-turbo  # For extraction
OPENAI_MODEL=gpt-5-nano  # For quick operations

# Or in code
client.chat(messages, model="gpt-4o")
```

### Harmonization Thresholds

Adjust similarity thresholds in `CompleteIntegratedSystem`:

```python
self.harmonizer = EnhancedLabelHarmonizer(
    similarity_threshold=0.80,  # Specific groups
    min_semantic_distance=0.5,  # General groups
    use_real_embeddings=True
)
```

## Usage Guide

### Basic Usage - CLI Mode

```python
from assistant.behavioral_engine.core.complete_system import CompleteIntegratedSystem

# Initialize
system = CompleteIntegratedSystem(use_config=True)

# Process a message
result = system.process_message_with_context("I took my vitamins and went for a run")

# Start a check-in
session = system.start_check_in_with_logs()
print(session['greeting'])

# Get daily summary
summary = system.get_daily_summary_with_logs()
```

### Email Integration Mode

```python
from assistant.behavioral_engine.enhanced_gmail_runner import BiggerBrotherEmailSystem

# Initialize and run
system = BiggerBrotherEmailSystem(use_config=True)
system.run_interactive()  # Interactive CLI with email monitoring
```

### Email Commands

Send emails with "BiggerBrother" in subject:

- **"chat"** - Start open conversation
- **"checkin"** - Start structured check-in
- **"detailed"** - Detailed activity logging
- **"status"** - Get daily summary
- **"help"** - Show commands
- **"done"** - End current conversation

### Creating Custom LogBook Categories

```python
# Programmatically
logbook.create_category(
    name="reading",
    description="Track reading sessions",
    fields={
        "book": "string",
        "pages": "number",
        "duration_minutes": "number",
        "notes": "string"
    },
    required_fields=["book", "pages"]
)

# Or let AI propose
proposed = logbook.propose_category(conversation_text)
```

### Using Harmonization

```python
# Check harmonization status
insights = system.get_harmonization_insights()

# Rebuild two-tier groups
stats = system.rebuild_two_tier_groups()

# Manual harmonization trigger
if system.needs_harmonization(threshold=100):
    system.rebuild_two_tier_groups()
```

## API Reference

### CompleteIntegratedSystem

Main system orchestrator class.

#### Methods

```python
def __init__(self, base_dir=None, use_sandbox=False, sandbox_name=None, use_config=True)
"""Initialize the complete system with all components."""

def process_message_with_context(self, message: str) -> Dict
"""Process a message with full context awareness and harmonization."""

def start_check_in_with_logs(self, check_type=None, include_summary=True) -> Dict
"""Start a structured check-in session with context."""

def start_detailed_activity_logging(self) -> Dict
"""Start detailed logging with dual-model conversation."""

def get_daily_summary_with_logs(self) -> Dict
"""Get comprehensive daily summary including all logs."""

def rebuild_two_tier_groups(self) -> Dict
"""Manually rebuild the two-tier harmonization groups."""

def get_harmonization_insights(self) -> Dict
"""Get insights about label organization and grouping."""
```

### DynamicLogBook

Dynamic log categorization system.

#### Methods

```python
def create_category(name, description, fields=None, required_fields=None, proposed_by="user", examples=None) -> LogCategory
"""Create a new log category with its own directory."""

def log_entry(category_name, data, raw_text=None, extracted_by="manual", confidence=1.0, session_id=None) -> LogEntry
"""Log an entry to a specific category."""

def load_category_context(category_name, days_back=7, max_entries=50) -> List[Dict]
"""Load recent entries from a category for context."""

def search_logs(query, categories=None, date_range=None) -> List[Dict]
"""Search across log books."""
```

### EnhancedLabelHarmonizer

Two-tier label harmonization system.

#### Methods

```python
def process_label_set(labels: Dict, message_context=None) -> Dict
"""Process and harmonize a set of labels using two-tier system."""

def get_label_similarity_scores(query_labels, candidate_labels, category) -> float
"""Calculate similarity between label sets using probabilities."""

def save_harmonization_tiers(self)
"""Save the two-tier groups to disk."""
```

## Development Guide

### Adding New Components

1. **Create module** in appropriate directory
2. **Import** in `complete_system.py`
3. **Initialize** in `CompleteIntegratedSystem.__init__()`
4. **Add paths** to `data_config.py` if needed
5. **Create schema** in `schemas/` if using structured data

### Creating Custom Schedulers

```python
from assistant.behavioral_engine.schedulers.base import BaseScheduler

class CustomScheduler(BaseScheduler):
    def analyze_patterns(self, days=30):
        """Your pattern analysis logic"""
        pass
    
    def generate_schedule(self, date):
        """Your scheduling logic"""
        pass
```

### Extending the RPG System

```python
# Add new skill categories in rpg_system.py
class SkillCategory(Enum):
    PHYSICAL = "physical"
    MENTAL = "mental"
    SOCIAL = "social"
    CREATIVE = "creative"  # New category

# Process activities for XP
rpg.process_activity("creative", 1, {"action": "wrote_poem"})
```

### Custom LogBook Fields

```python
# Define custom field types
fields = {
    "custom_metric": "number",
    "tags": "list",
    "location": "string",
    "photo_url": "string",
    "is_public": "boolean"
}
```

### Working with Embeddings

The system uses smart embedding caching to minimize API calls:

```python
# Embedding cache location
cache_file = "data/embedding_cache.pkl"

# The harmonizer automatically:
# 1. Checks string similarity first (no API)
# 2. Uses cached embeddings when available
# 3. Only calls API for truly ambiguous cases
```

## Troubleshooting

### Common Issues

#### 1. "credentials.json not found"
**Solution:** Download OAuth credentials from Google Cloud Console

#### 2. "OPENAI_API_KEY not set"
**Solution:** Add to `.env` file or set environment variable

#### 3. Harmonization taking too long
**Solution:** 
- Check embedding cache is being used
- Adjust similarity thresholds
- Use `AGENT_OFFLINE=1` for testing

#### 4. Gmail not receiving messages
**Check:**
- Subject contains "BiggerBrother"
- OAuth token is valid (delete `token.pickle` to re-auth)
- Gmail API is enabled in Cloud Console

#### 5. Categories not being created
**Verify:**
- Directory permissions for `data/logbooks/`
- OpenAI API key is valid
- GPT-5-nano model is accessible

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Offline Mode

For development without API access:

```python
# Set in .env
AGENT_OFFLINE=1

# Returns deterministic JSON responses
# Useful for testing data flow
```

### Performance Optimization

1. **Batch Operations**: Process multiple messages before harmonization
2. **Cache Context**: Reuse loaded context across messages in same session
3. **Limit Embeddings**: Adjust thresholds to reduce embedding API calls
4. **Archive Old Data**: Move old logs to archive directories

## Best Practices

### Label Design
- Keep labels concise and descriptive
- Use consistent naming conventions
- Let the harmonizer group similar concepts

### Context Management
- Load only necessary categories
- Use time-based filters for recent data
- Cache frequently accessed context

### Session Handling
- Always close sessions properly
- Save session state periodically
- Handle interruptions gracefully

### Data Privacy
- Store sensitive data encrypted
- Use sandbox mode for testing
- Implement data retention policies

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document all public methods
- Include usage examples in docstrings

### Testing
- Test with `AGENT_OFFLINE=1` first
- Verify harmonization groups are created
- Check all log categories function
- Test email integration separately

### Pull Request Process
1. Create feature branch
2. Update documentation
3. Add tests if applicable
4. Ensure backward compatibility
5. Update `CONTEXT.md` if architecture changes

## License and Credits

This project integrates several technologies and approaches:
- OpenAI GPT models for natural language processing
- Google Gmail API for email integration  
- Two-tier harmonization inspired by semantic similarity research
- Graph-based storage for flexible data relationships

---

*For questions or issues, please refer to the repository issues or contact the maintainers.*