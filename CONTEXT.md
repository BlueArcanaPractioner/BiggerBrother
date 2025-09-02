# BiggerBrother Codebase Context

## Summary
- **Files**: 38 Python files
- **Lines**: 16,907 total lines
- **Classes**: 65 classes
- **Functions**: 488 functions/methods
- **Main Packages**: app, assistant
- **External Dependencies**: __future__, app, argparse, assistant, base64, csv, data_config, difflib, dotenv, email, glob, google, google_auth_oauthlib, googleapiclient, jsonschema, label_generator, logging, numpy, openai, openai_client, pickle, platform, plyer, requests, schedule, smtplib, statistics, threading, traceback

## Architecture Patterns
- Type hints: ✓
- Dataclasses: ✓
- Async/await: ✗
- CLI interface: ✓
- API endpoints: ✗
- AI integration: ✓
- Test coverage: ✗

## Data Flow and Storage

### Primary Data Directories
```
data/
├── chunks/           # Message chunks (JSON)
├── labels/           # Semantic labels (JSON)
├── logbooks/         # Dynamic log categories (CSV + JSONL)
├── features/         # Extracted features (JSON)
├── rpg/             # Game state (JSON)
├── routines/        # Routine definitions (JSON)
├── scheduler/       # Schedule data (JSON)
└── tracking/        # Graph store (JSONL)
```

**Data Formats Used**: csv, json, jsonl, pickle

### Data Flow Patterns

**Key Data Processing Modules:**
- `app.coverage_checker`: _load_chunk_entry(), load_manifest()
- `app.label_generator`: _extract_first_json_object, _extract_first_json_object(), _load_prompt_head()
- `app.label_integration_wrappers`: get_label_statistics()
- `app.openai_client`: _offline_payload()
- `assistant.behavioral_engine.context.similarity_matcher`: _get_embedding(), _load_chunk_by_gid(), _load_embedding_cache()
- `assistant.behavioral_engine.core.complete_system`: _extract_to_logbooks(), analyze_behavioral_patterns(), extracted.get
- `assistant.behavioral_engine.enhanced_gmail_runner`: _extract_body(), _get_conversation_mode(), _get_help_text()
- `assistant.behavioral_engine.features.enhanced_feature_extraction`: _get_logs_in_window(), _load_top_words(), _load_user_vocabulary()
- `assistant.behavioral_engine.gamification.rpg_system`: _load_achievements(), _load_character(), _load_quest_history()
- `assistant.behavioral_engine.logbooks.dynamic_logbook_system`: _extract_activities(), _load_categories(), _save_categories()

## Key Modules and Classes

### app/

**`app\coverage_checker.py`**
_coverage_checker.py_
📥 Reads: `<dynamic>, r`
📁 Uses: `schemas/`
Imports: `dataclasses.dataclass, glob, hashlib, label_generator, logging`
```python
class ChunkRecord:
```

**`app\label_generator.py`**
📥 Reads: `<dynamic>`
📤 Writes: `<json.dumps()>`
📁 Uses: `schemas/`
Imports: `__future__.annotations, dataclasses.dataclass, dataclasses.field, logging, openai_client`
```python
class LabelScore:
class LabelRecord:
    def to_dict(self) -> Dict[Tuple]
```

**`app\label_integration_wrappers.py`**
_Label Integration Wrapper Classes_
📥 Reads: `<dynamic>, <label_file>`
📁 Uses: `schemas/`
Imports: `app.coverage_checker, app.label_generator, app.openai_client.OpenAIClient, pathlib.Path`
```python
class LabelGenerator:
    def __init__(self, openai_client)
    def generate_labels_for_text(self, text: str, metadata: Optional[Dict]=None) -> Dict[Tuple]
    def generate_labels_for_chunk(self, chunk: Dict) -> Dict
    def batch_generate_labels(self, texts: List[str], labels_dir: str='labels', skip_existing: bool=True) -> Dict[Tuple]
class CoverageChecker:
    def __init__(self, openai_client)
    def check_coverage(self, data_root: str='data', labels_dir: str='labels', manifest_glob: Optional[List[str]]=None, autogenerate_missing: bool=False) -> Dict[Tuple]
    def validate_labels(self, labels_dir: str='labels') -> Dict[Tuple]
    def get_label_statistics(self, labels_dir: str='labels') -> Dict[Tuple]
```

**`app\openai_client.py`**
Imports: `__future__.annotations, dotenv.load_dotenv, openai, openai.OpenAI`
```python
class OpenAIClient:
    def __init__(self) -> None
    def chat(self, messages: list[dict], model: BinOp=None, **kwargs) -> str
    def complete(self, prompt: str, model: BinOp=None, **kwargs) -> str
```

### assistant/

**`assistant\behavioral_engine\context\similarity_matcher.py`**
_Context Similarity Matcher for BiggerBrother_
📥 Reads: `<cache_file>, <chunk_file>`
📤 Writes: `<tier>`
📁 Uses: `<header_parts>, data/`
Imports: `__future__.annotations, app.label_integration_wrappers.LabelGenerator, app.openai_client.OpenAIClient, assistant.importers.enhanced_smart_label_importer.EnhancedLabelHarmonizer, collections.defaultdict`
```python
class ContextSimilarityMatcher:
    def __init__(self, labels_dir: str='labels', chunks_dir: str='data/chunks', harmonization_dir: str='data', openai_client: Optional[OpenAIClient]=None, harmonizer: Optional[EnhancedLabelHarmonizer]=None, context_minimum_char_long_term: int=150000, context_minimum_char_recent: int=50000, max_context_messages: int=500000, general_tier_weight: float=0.3, specific_tier_weight: float=0.7, recency_decay_factor: float=0.998, recency_cutoff_days: int=3650)
    def get_message_labels(self, message: str) -> Dict
    def harmonize_and_score_labels(self, labels: Dict) -> Dict
    def calculate_similarity_score(self, current_harmonized: Dict, target_harmonized: Dict, message_timestamp: datetime) -> float
    def find_similar_messages(self, message: str, min_chars_recent: Optional[int]=None, min_chars_long_term: Optional[int]=None) -> Tuple[Tuple]
    def build_context_header(self, message: str, max_chars: int=150000) -> str
```

**`assistant\behavioral_engine\core\complete_system.py`**
_Complete Integrated System with Dynamic LogBooks and Two-Tier Harmonization_
📥 Reads: `<dynamic>, <label_file>`
📤 Writes: `<dynamic>, <readme_content>`
📁 Uses: `/`
Imports: `__future__.annotations, app.label_integration_wrappers.LabelGenerator, app.openai_client.OpenAIClient, assistant.behavioral_engine.context.similarity_matcher.ContextSimilarityMatcher, assistant.behavioral_engine.features.enhanced_feature_extraction.EnhancedFeatureExtractor`
```python
class CompleteIntegratedSystem:
    def __init__(self, base_dir: str=None, use_sandbox: bool=False, sandbox_name: Optional[str]=None, use_config: bool=True)
    def get_similar_messages_for_context(self, message: str, current_labels: Dict[Tuple], limit: int=10) -> List[str]
    def select_context_categories_by_labels(self, harmonized_labels: Dict) -> Tuple[Tuple]
    def process_message_with_context(self, message: str) -> Dict
    def analyze_behavioral_patterns(self, days: int=30)
    def generate_daily_schedule(self, date=None)
    def schedule_pomodoro(self, task: str, start_time=None, duration=25, break_duration=5)
    def needs_harmonization(self, threshold: int=100) -> bool
    def reset_harmonization_counter(self)
    def rebuild_two_tier_groups(self) -> Dict
    def get_harmonization_insights(self) -> Dict
    def start_check_in_with_logs(self, check_type: Optional[CheckInType]=None, include_summary: bool=True) -> Dict[Tuple]
    def start_detailed_activity_logging(self) -> Dict
    def start_routine_with_logging(self, routine_name: str) -> Dict
    def complete_routine_step(self, step_name: str, notes: str='') -> Dict
    def query_with_logs(self, query: str) -> str
    def propose_new_log_category(self, conversation_text: str) -> Optional[Dict]
    def get_daily_summary_with_logs(self) -> Dict
    def create_directory_structure(self)
```

**`assistant\behavioral_engine\enhanced_gmail_runner.py`**
_BiggerBrother Complete System with Gmail OAuth Integration_
📥 Reads: `<dynamic>`
📁 Uses: `<metadata>, <parts>`
Imports: `app.openai_client.OpenAIClient, assistant.behavioral_engine.core.complete_system.CompleteIntegratedSystem, assistant.behavioral_engine.schedulers.adaptive_scheduler.AdaptiveScheduler, assistant.behavioral_engine.schedulers.notification_integration.NotificationConfig, assistant.behavioral_engine.schedulers.notification_integration.NotificationManager`
```python
class ConversationMode(Enum):
class GmailIntegration:
    def __init__(self, credentials_file='credentials.json', token_file='token.pickle')
    def get_unread_messages(self, query='is:unread subject:BiggerBrother')
    def send_email(self, to, subject, body, thread_id=None)
    def mark_as_read(self, msg_id)
class BiggerBrotherEmailSystem:
    def __init__(self, base_dir=None, use_config=True)
    def start(self)
    def run_interactive(self)
    def stop(self)
```

**`assistant\behavioral_engine\features\enhanced_feature_extraction.py`**
_Enhanced Feature Extraction System_
📥 Reads: `<filepath>, <vocab_file>`
📤 Writes: `<dynamic>, <session_features>`
📁 Uses: `
Features saved to data/, <filename>, data/`
Imports: `__future__.annotations, assistant.graph.store.GraphStore, assistant.logger.unified.UnifiedLogger, collections.Counter, collections.defaultdict`
```python
class MessageFeatures:
    def to_dict(self) -> Dict
class ActivityFeatures:
    def to_dict(self) -> Dict
class SessionFeatures:
    def to_dict(self) -> Dict
class EnhancedFeatureExtractor:
    def __init__(self, logger: UnifiedLogger, graph_store: GraphStore, data_dir: str='data/features')
    def extract_message_features(self, message: str, timestamp: datetime, previous_timestamp: Optional[datetime]=None, context_window_hours: int=24) -> MessageFeatures
    def extract_activity_features(self, start_time: datetime, end_time: datetime) -> ActivityFeatures
    def extract_session_features(self, session_id: str, messages: List[Dict], labels: List[List[str]]) -> SessionFeatures
    def save_features(self, features: Any, filename: str) -> None
    def save_vocabulary_stats(self) -> None
    def get_feature_summary(self, session_features: SessionFeatures) -> Dict[Tuple]
```

**`assistant\behavioral_engine\gamification\rpg_system.py`**
_RPG Gamification Layer for Behavioral Tracking_
📥 Reads: `<achievements_file>, <char_file>`
📤 Writes: `<dynamic>`
📁 Uses: `achievements.json, character.json, data/`
Imports: `__future__.annotations, assistant.graph.store.GraphStore, assistant.logger.unified.UnifiedLogger, collections.defaultdict, dataclasses.dataclass`
```python
class SkillCategory(Enum):
class Skill:
    @property def xp_for_next_level(self) -> float
    @property def progress_to_next_level(self) -> float
    def add_xp(self, amount: float) -> bool
class Achievement:
    @property def is_complete(self) -> bool
    def update_progress(self, value: float) -> bool
class DailyQuest:
    def check_completion(self, user_data: Dict) -> bool
class CharacterStats:
    @property def xp_for_next_level(self) -> float
    def add_xp(self, amount: float) -> bool
class RPGSystem:
    def __init__(self, logger: UnifiedLogger, graph_store: GraphStore, data_dir: str='data/rpg')
    def process_activity(self, category: str, value: Any, metadata: Optional[Dict]=None) -> Dict[Tuple]
    def generate_daily_quests(self) -> List[DailyQuest]
    def get_character_summary(self) -> Dict[Tuple]
```

**`assistant\behavioral_engine\logbooks\dynamic_logbook_system.py`**
_Dynamic LogBook System with Category Discovery_
📥 Reads: `<csv_file>, <dynamic>`
📤 Writes: `## Fields

, <category>`
📁 Uses: `<logged>, <safe_name>, README.md`
Imports: `__future__.annotations, app.openai_client.OpenAIClient, assistant.graph.store.GraphStore, assistant.logger.unified.UnifiedLogger, collections.defaultdict`
```python
class LogCategory:
    def to_dict(self) -> Dict
class LogEntry:
    def to_dict(self) -> Dict
class DynamicLogBook:
    def __init__(self, base_dir: str='data/logbooks', openai_client: Optional[OpenAIClient]=None)
    def create_category(self, name: str, description: str, fields: Optional[Dict[Tuple]]=None, required_fields: Optional[List[str]]=None, proposed_by: str='user', examples: Optional[List[str]]=None) -> LogCategory
    def propose_category(self, conversation_text: str) -> Optional[LogCategory]
    def log_entry(self, category_name: str, data: Dict[Tuple], raw_text: Optional[str]=None, extracted_by: str='manual', confidence: float=1.0, session_id: Optional[str]=None) -> LogEntry
    def get_categories_for_context(self) -> List[Dict]
    def load_category_context(self, category_name: str, days_back: int=7, max_entries: int=50) -> List[Dict]
    def search_logs(self, query: str, categories: Optional[List[str]]=None, date_range: Optional[Tuple[Tuple]]=None) -> List[Dict]
class DetailedActivityLogger:
    def __init__(self, logbook: DynamicLogBook, logger: UnifiedLogger, openai_client: Optional[OpenAIClient]=None)
    def start_detailed_logging(self) -> Dict
    def process_activity_description(self, description: str) -> Dict
    def complete_session(self) -> Dict
class LogBookAssistant:
    def __init__(self, logbook: DynamicLogBook, openai_client: Optional[OpenAIClient]=None)
    def decide_context_needed(self, query: str) -> List[str]
    def answer_with_logs(self, query: str) -> str
```

**`assistant\behavioral_engine\planners\integrated_daily_planner.py`**
_Integrated Daily Planner and Logger System_
📥 Reads: `<dynamic>, <filepath>`
📤 Writes: `<dynamic>, <plan>`
📁 Uses: `<filename>, <preferences>, <recent_meals>`
Imports: `__future__.annotations, app.label_integration_wrappers.LabelGenerator, app.openai_client.OpenAIClient, assistant.behavioral_engine.features.enhanced_feature_extraction.EnhancedFeatureExtractor, assistant.behavioral_engine.gamification.rpg_system.RPGSystem`
```python
class DailyPlan:
    def add_check_in(self, time: datetime, check_type: CheckInType, context: Dict=None)
    def add_reminder(self, time: datetime, title: str, description: str='')
    def add_task(self, title: str, priority: int=1, estimated_minutes: int=30)
    def set_menu(self, breakfast: str='', lunch: str='', dinner: str='', snacks: List[str]=None)
    def to_dict(self) -> Dict
class IntegratedDailyPlanner:
    def __init__(self, data_dir: str='data/planner', labels_dir: str='labels', chunks_dir: str='data/chunks', use_sandbox: bool=False, sandbox_name: Optional[str]=None)
    def create_daily_plan(self, date: Optional[datetime]=None, custom_schedule: Optional[Dict]=None) -> DailyPlan
    def start_check_in(self, check_type: Optional[CheckInType]=None) -> Dict
    def process_message(self, message: str) -> Dict
    def complete_check_in(self) -> Dict
    def log_quick_entry(self, entry_text: str) -> Dict
    def generate_menu_suggestions(self) -> Dict
    def get_daily_progress(self) -> Dict
```

**`assistant\behavioral_engine\routines\routine_builder.py`**
_Behavioral Routine Builder with Task Analysis_
📥 Reads: `<dynamic>, <filepath>`
📤 Writes: `<dynamic>, <routine>`
📁 Uses: `<filename>, data/`
Imports: `__future__.annotations, app.openai_client.OpenAIClient, assistant.graph.store.GraphStore, assistant.logger.unified.UnifiedLogger, collections.defaultdict`
```python
class RoutineStep:
    def to_dict(self) -> Dict
    @classmethod def from_dict(cls, data: Dict) -> 'RoutineStep'
class BehavioralRoutine:
    def add_step(self, name: str, description: str, estimated_minutes: float, category: str, **kwargs) -> RoutineStep
    def start_execution(self) -> Dict
    def complete_step(self, step_name: str, actual_minutes: Optional[float]=None) -> bool
    def skip_step(self, step_name: str, reason: str='') -> bool
    def complete_execution(self) -> Dict
    def to_dict(self) -> Dict
class RoutineBuilder:
    def __init__(self, logger: UnifiedLogger, graph_store: GraphStore, openai_client: OpenAIClient, data_dir: str='data/routines')
    def create_routine_from_description(self, description: str) -> BehavioralRoutine
    def start_routine(self, routine_name: str) -> Dict
    def complete_step(self, step_name: str, notes: str='') -> Dict
    def skip_step(self, step_name: str, reason: str) -> Dict
    def complete_routine(self) -> Dict
    def analyze_routine_effectiveness(self, routine_name: str) -> Dict
    def get_all_routines(self) -> Dict[Tuple]
```

**`assistant\behavioral_engine\schedulers\adaptive_scheduler.py`**
_Smart Adaptive Check-in Scheduler_
📥 Reads: `<pattern_file>, <schedule_file>`
📤 Writes: `<date>, <dynamic>`
📁 Uses: `activity_patterns.json, data/, transitions.json`
Imports: `__future__.annotations, assistant.conversational_logger.CheckInType, assistant.conversational_logger.ConversationalLogger, assistant.graph.store.GraphStore, assistant.logger.temporal_personal_profile.TemporalPersonalProfile`
```python
class ActivityPattern:
    def is_active_on_day(self, day: int) -> bool
class CheckInSchedule:
    def to_dict(self) -> Dict
class PomodoroSession:
    @property def end_time(self) -> datetime
class TransitionType(Enum):
class AdaptiveScheduler:
    def __init__(self, logger: UnifiedLogger, graph_store: GraphStore, profile: TemporalPersonalProfile, data_dir: str='data/scheduler')
    def analyze_patterns(self, days: int=30) -> Dict[Tuple]
    def generate_daily_schedule(self, date: datetime) -> List[CheckInSchedule]
    def schedule_pomodoro(self, task: str, start_time: Optional[datetime]=None, duration: int=25, break_duration: int=5) -> PomodoroSession
    def complete_pomodoro(self, session: PomodoroSession, focus_score: float) -> None
    def update_transition(self, transition_type: TransitionType, timestamp: datetime) -> None
    def get_next_check_in(self) -> Optional[CheckInSchedule]
    def complete_check_in(self, schedule: CheckInSchedule) -> None
    def get_daily_summary(self) -> Dict[Tuple]
```

**`assistant\behavioral_engine\schedulers\context_aware_scheduler.py`**
_Context-Aware Conversational Scheduler with Labeling Integration_
📥 Reads: `<char_file>, <chunk_file>`
📤 Writes: `<dynamic>`
📁 Uses: `<char_file>, <label_file>, <modifiers>`
Imports: `__future__.annotations, app.coverage_checker, app.label_generator, app.openai_client.OpenAIClient, assistant.conversational_logger.CheckInType`
```python
class ConversationContext:
    def add_similar_message(self, message: Dict, relevancy: float) -> bool
    def add_temporal_message(self, message: Dict) -> bool
    def get_context_prompt(self) -> str
class SandboxEnvironment:
    def add_character(self, name: str, sheet: Dict) -> None
    def get_context_modifier(self) -> str
class ContextAwareScheduler:
    def __init__(self, logger: UnifiedLogger, graph_store: GraphStore, openai_client: OpenAIClient, data_dir: str='data/context_scheduler', labels_dir: str='labels', chunks_dir: str='data/chunks', sandbox_mode: bool=False, sandbox_name: Optional[str]=None)
    def start_check_in(self, check_in_type: Optional[CheckInType]=None, custom_context: Optional[Dict]=None) -> Dict[Tuple]
    def process_message(self, user_message: str) -> Dict[Tuple]
    def create_sandbox(self, name: str, ruleset: Optional[Dict]=None, character_sheets: Optional[Dict[Tuple]]=None) -> SandboxEnvironment
    def switch_to_sandbox(self, sandbox_name: str) -> None
    def get_session_metrics(self) -> Dict[Tuple]
```

**`assistant\behavioral_engine\schedulers\notification_integration.py`**
_Integration Layer for Smart Scheduler with Notifications_
📥 Reads: `<dynamic>, <pending_file>`
📤 Writes: `<check_in_schedule>, <session>`
📁 Uses: `data/, pending_check_ins.json, pending_pomodoro_reviews.json`
Imports: `app.openai_client.OpenAIClient, assistant.behavioral_engine.schedulers.adaptive_scheduler.AdaptiveScheduler, assistant.behavioral_engine.schedulers.adaptive_scheduler.PomodoroSession, assistant.conversational_logger.CheckInType, assistant.conversational_logger.ConversationalLogger`
```python
class NotificationConfig:
    @classmethod def from_env(cls) -> 'NotificationConfig'
class NotificationManager:
    def __init__(self, config: NotificationConfig)
    def send_notification(self, title: str, message: str, priority: str='normal', context: Optional[Dict]=None) -> bool
class ScheduledCheckInOrchestrator:
    def __init__(self, scheduler, conv_logger: ConversationalLogger, notifier: NotificationManager, data_dir: str='data/orchestrator')
    def start(self) -> None
    def stop(self) -> None
    def start_check_in_from_notification(self, check_in_type: CheckInType) -> Dict
    def process_email_response(self, email_body: str) -> Dict
    def get_dashboard_data(self) -> Dict[Tuple]
```

**`assistant\conversational_logger.py`**
_Conversational AI Logger for natural tracking interactions._
📥 Reads: `<dynamic>`
📁 Uses: `<context_items>`
Imports: `__future__.annotations, enum.Enum, random`
```python
class CheckInType(Enum):
class ConversationalLogger:
    def __init__(self, unified_logger: UnifiedLogger, openai_client: Any, extractor_model: str='gpt-5-mini', conversationalist_model: str='gpt-4o')
    def start_check_in(self, check_in_type: Union[Tuple]=..., custom_prompts: Optional[List[str]]=None) -> Dict[Tuple]
    def process_response(self, user_message: str) -> Dict[Tuple]
    def quick_log(self, message: str) -> Dict[Tuple]
```

**`assistant\graph\models.py`**
Imports: `__future__.annotations, dataclasses.asdict, dataclasses.dataclass, dataclasses.field, uuid`
```python
class ValidationError(ValueError):
class Node:
    def touch(self) -> None
    def to_dict(self) -> Dict[Tuple]
    @classmethod def from_dict(cls: Type['Node'], data: Dict[Tuple]) -> 'Node'
class Edge:
    def to_dict(self) -> Dict[Tuple]
    @classmethod def from_dict(cls: Type['Edge'], data: Dict[Tuple]) -> 'Edge'
class LogEntry:
    def to_dict(self) -> Dict[Tuple]
    @classmethod def from_dict(cls: Type['LogEntry'], data: Dict[Tuple]) -> 'LogEntry'
```

**`assistant\graph\store.py`**
📥 Reads: `<dynamic>, r`
📤 Writes: `<dynamic>`
Imports: `__future__.annotations, pathlib.Path, threading, uuid`
```python
class UnifiedLogger:
    def info(self, *args, **kwargs) -> None
    def debug(self, *args, **kwargs) -> None
    def warning(self, *args, **kwargs) -> None
    def error(self, *args, **kwargs) -> None
class GraphStore:
    def __init__(self, data_dir: BinOp, logger: Optional[Any]=None) -> None
    def get_all_nodes(self) -> List[Dict[Tuple]]
    @property def tag_index(self) -> Dict[Tuple]
    @property def adj_out(self) -> Dict[Tuple]
    @property def adj_in(self) -> Dict[Tuple]
    def load(self) -> None
    def save(self) -> None
    def persist(self) -> None
    def create_node(self, type: str, text: str='', tags: List[str]=None, attrs: dict=None) -> str
    def get_node(self, node_id: str) -> Optional[dict]
    def update_node(self, node_id: str, text: str=None, tags: List[str]=None, attrs: dict=None) -> bool
    def delete_node(self, node_id: str, cascade: bool=True) -> bool
    def create_edge(self, src_id: str, dst_id: str=None, kind: str=None, weight: float=1.0, attrs: dict=None) -> str
    def get_edge(self, edge_id: str) -> Optional[dict]
    def delete_edge(self, edge_id: str) -> bool
    def get_edges(self, src_id: str=None, dst_id: str=None, kind: str=None) -> List[dict]
    def search_nodes(self, tags: List[str]=None, text: str=None, node_type: str=None, created_after: Union[Tuple]=None, created_before: Union[Tuple]=None, limit: int=10) -> List[dict]
    def get_nodes_in_range(self, start_time: Union[Tuple], end_time: Union[Tuple], node_type: str=None, limit: int=100) -> List[dict]
    def get_recent_nodes(self, hours: int=24, node_type: str=None, limit: int=50) -> List[dict]
    def get_neighbors(self, node_id: str, edge_kind: str=None, direction: str='both') -> List[dict]
```

**`assistant\importers\enhanced_smart_label_importer.py`**
_Enhanced Smart Label Importer with Two-Tier Harmonization_
📥 Reads: `<chunk_file>, <dynamic>`
📤 Writes: `<dynamic>`
📁 Uses: `C:/, data/`
Imports: `assistant.graph.store.GraphStore, collections.Counter, collections.defaultdict, difflib.SequenceMatcher, hashlib`
```python
class EnhancedLabelHarmonizer:
    def __init__(self, harmonization_report_path: str='data/harmonization_report.json', target_groups: Dict[Tuple]=None, similarity_threshold: float=0.8, min_semantic_distance: float=0.5, use_real_embeddings: bool=True, embedding_cache_file: str='data/embedding_cache.pkl')
    def enable_batch_mode(self)
    def disable_batch_mode(self)
    def compute_string_similarity(self, label1: str, label2: str) -> float
    def compute_similarity(self, label1: str, label2: str, use_embeddings: bool=True) -> float
    def find_canonical(self, label: str, category: str) -> Tuple[Tuple]
    def harmonize_label_set(self, labels: List[Dict], category: str) -> Tuple[Tuple]
    def get_label_similarity_scores(self, query_labels: List[Dict], candidate_labels: List[Dict], category: str) -> float
    def save_harmonization_tiers(self)
    def get_harmonization_report(self) -> Dict
    def process_label_set(self, labels: Dict[Tuple], message_context: Optional[str]=None) -> Dict[Tuple]
class EnhancedSmartLabelImporter:
    def __init__(self, graph_store: GraphStore, data_dir: str='data', use_harmonization_report: bool=True)
    def load_message_index(self) -> Dict[Tuple]
    def sanitize_tag(self, tag: str) -> str
    def import_labeled_message(self, label_file: Path, message_index: Dict) -> Optional[str]
    def import_all_labels(self, max_files: Optional[int]=None) -> Dict
```

**`assistant\importers\harmonizer_v2.py`**
_SEMANTIC HARMONIZER V2 - Label Topology Manager_
📥 Reads: `<cache_path>, <dynamic>`
Imports: `collections.defaultdict, dataclasses.dataclass, dataclasses.field, numpy, pathlib.Path`
```python
class LabelNode:
class SemanticGroup:
class EnhancedHarmonizer:
    def __init__(self, data_dir: str='data', strict_threshold: float=0.85, loose_threshold: float=0.65, drift_threshold: float=0.15, cache_file: Optional[str]=None)
    def process_label_set(self, labels: Dict[Tuple], message_context: Optional[str]=None) -> Dict[Tuple]
    def get_or_create_label_node(self, label: str, category: str, confidence: float, context: Optional[str]=None) -> LabelNode
    def assign_to_group(self, node: LabelNode, category: str) -> SemanticGroup
    def create_group(self, canonical_label: str, category: str, parent: Optional[str]=None) -> SemanticGroup
    def update_group_centroid(self, group: SemanticGroup)
    def check_drift(self, node: LabelNode, group: SemanticGroup) -> bool
    def detect_label_correlations(self, labels: Dict)
    def get_similar_labels(self, label: str, category: str, k: int=5) -> List[Tuple[Tuple]]
    def update_behavioral_correlation(self, labels: Dict[Tuple], behavior_occurred: bool)
    def get_group_behavioral_stats(self, group_id: str) -> Dict
    def normalize_label(self, label: str) -> str
    def get_embedding(self, text: str) -> np.ndarray
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float
    def save_state(self)
    def load_state(self)
    def get_visual_topology(self, category: str) -> Dict
```

**`assistant\logger\temporal_personal_profile.py`**
_TEMPORAL PERSONAL PROFILE SYSTEM_
📥 Reads: `<save_file>`
📤 Writes: `<dynamic>`
📁 Uses: `<context_parts>, data/`
Imports: `collections.defaultdict, dataclasses.asdict, dataclasses.dataclass, dataclasses.field, enum.Enum`
```python
class ProfileCategory(Enum):
class PrivacyLevel(Enum):
class ProfileFact:
    def is_expired(self, current_time: datetime) -> bool
    def decay_confidence(self, current_time: datetime) -> float
class PersonEntity:
class ProfileSnapshot:
class TemporalPersonalProfile:
    def __init__(self, data_dir: str='data/profiles', max_facts: int=1000, snapshot_interval_days: int=7, anonymize_third_parties: bool=True)
    def get_facts_by_category(self, category: str) -> List[Dict[Tuple]]
    def extract_facts_from_message(self, message: Dict[Tuple], current_time: datetime) -> List[ProfileFact]
    def update_profile(self, new_facts: List[ProfileFact], current_time: datetime) -> Dict[Tuple]
    def get_context_for_inference(self, max_chars: int=150000, categories: Optional[List[ProfileCategory]]=None, privacy_filter: Optional[PrivacyLevel]=None) -> str
    def create_snapshot(self, timestamp: datetime, trigger: str) -> ProfileSnapshot
    def save_profile(self)
    def load_profile(self)
```

**`assistant\logger\unified.py`**
_Unified Logger for the personal assistant system._
📥 Reads: `<path>`
📤 Writes: `<dynamic>`
Imports: `__future__.annotations, assistant.graph.store.GraphStore, csv, jsonschema, jsonschema.FormatChecker`
```python
class ValidationError(ValueError):
class ImportError(Exception):
class UnifiedLogger:
    def __init__(self, data_dir: Optional[str]=None, graph_store: Optional[Any]=None)
    def info(self, message: str, *args, **kwargs) -> None
    def debug(self, message: str, *args, **kwargs) -> None
    def warning(self, message: str, *args, **kwargs) -> None
    def error(self, message: str, *args, **kwargs) -> None
    def log(self, entry: Dict[Tuple]) -> Dict[Tuple]
    def get_node(self, node_id: str) -> Optional[Dict[Tuple]]
    def import_csv(self, csv_path: str) -> Dict[Tuple]
    def get_daily_summary(self, date_str: str, tz: Optional[str]=None) -> Dict[Tuple]
```

### root/

**`context_header_generator.py`**
_Context Header Generator for BiggerBrother_
📥 Reads: `<dynamic>, <filepath>`
📤 Writes: `<header>, <self._extract_path_from_call()>`
📁 Uses: `2. LABEL: chunks/, 3. HARMONIZE: labels/, 4. LOG: Natural language → logbooks/`
Imports: `argparse, ast, collections.defaultdict, pathlib.Path`
```python
class CodebaseAnalyzer:
    def __init__(self, root_path: str='.', exclude_dirs: Set[str]=None)
    def analyze(self) -> Dict
```


## Data Pipeline

```
1. INPUT: Raw conversations → chunks/ (via chunking)
2. LABEL: chunks/ → labels/ (via OpenAI)
3. HARMONIZE: labels/ → canonical labels (via harmonizer)
4. LOG: Natural language → logbooks/ (via GPT-5-nano)
5. EXTRACT: All data → features/ (via FeatureExtractor)
6. PERSIST: All → GraphStore (JSONL)
```

**Input Modules** (data ingestion):
- `context_header_generator`
- `data_config`
- `app.coverage_checker`
- `app.label_generator`
- `app.label_integration_wrappers`

**Processing Modules** (transformation):
- `assistant.conversational_logger`
- `assistant.behavioral_engine.context.similarity_matcher`
- `assistant.behavioral_engine.features.enhanced_feature_extraction`
- `assistant.behavioral_engine.gamification.rpg_system`
- `assistant.behavioral_engine.logbooks.dynamic_logbook_system`

## Key Dependencies

```
app/ → OpenAI API
assistant/behavioral_engine/ → app.openai_client
assistant/behavioral_engine/ → assistant.logger, assistant.graph
assistant/ml/ → sklearn, pandas (when available)
```