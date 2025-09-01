#!/usr/bin/env python3
"""
Context Header Generator for BiggerBrother
Generates a comprehensive overview of the codebase for AI assistants
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

class CodebaseAnalyzer:
    def __init__(self, root_path: str = ".", exclude_dirs: Set[str] = None):
        self.root_path = Path(root_path)
        self.exclude_dirs = exclude_dirs or {
            '__pycache__', '.git', '.venv', 'venv', 
            'node_modules', '.pytest_cache', 'htmlcov',
            'dist', 'build', '*.egg-info'
        }
        self.codebase_info = {
            'summary': {},
            'structure': {},
            'modules': {},
            'dependencies': defaultdict(set),
            'key_classes': {},
            'key_patterns': {}
        }
    
    def analyze(self) -> Dict:
        """Analyze the entire codebase."""
        print("🔍 Analyzing codebase structure...")
        
        # Get all Python files
        py_files = self._get_python_files()
        
        # Analyze each file
        for py_file in py_files:
            self._analyze_file(py_file)
        
        # Generate summary statistics
        self._generate_summary()
        
        # Identify key patterns
        self._identify_patterns()
        
        return self.codebase_info
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files, excluding .bak files."""
        py_files = []
        
        for root, dirs, files in os.walk(self.root_path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                if file.endswith('.py') and not file.endswith('.py.bak'):
                    py_files.append(Path(root) / file)
        
        return py_files
    
    def _analyze_file(self, filepath: Path) -> None:
        """Analyze a single Python file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Get relative path for cleaner output
            rel_path = filepath.relative_to(self.root_path)
            module_path = str(rel_path).replace('/', '.').replace('\\', '.')[:-3]
            
            module_info = {
                'path': str(rel_path),
                'imports': self._extract_imports(tree),
                'classes': self._extract_classes(tree),
                'functions': self._extract_functions(tree),
                'docstring': ast.get_docstring(tree),
                'lines': len(content.splitlines()),
                'has_main': self._has_main_block(tree),
                'data_paths': self._extract_data_paths(tree, content),
                'io_operations': self._extract_io_operations(tree, content)
            }
            
            self.codebase_info['modules'][module_path] = module_info
            
            # Track dependencies
            for imp in module_info['imports']:
                if not imp.startswith('.'):
                    self.codebase_info['dependencies'][imp.split('.')[0]].add(module_path)
        
        except Exception as e:
            print(f"  ⚠️  Error analyzing {filepath}: {e}")
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.level > 0:  # Relative import
                    module = '.' * node.level + module
                for alias in node.names:
                    if alias.name == '*':
                        imports.append(f"{module}.*")
                    else:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        return sorted(list(set(imports)))
    
    def _extract_classes(self, tree: ast.AST) -> Dict[str, Dict]:
        """Extract class definitions with method signatures."""
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'docstring': ast.get_docstring(node),
                    'bases': [self._get_name(base) for base in node.bases],
                    'methods': {},
                    'class_vars': []
                }
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_sig = self._get_function_signature(item)
                        class_info['methods'][item.name] = {
                            'signature': method_sig,
                            'docstring': ast.get_docstring(item),
                            'is_property': self._is_property(item),
                            'is_classmethod': self._is_classmethod(item),
                            'is_staticmethod': self._is_staticmethod(item)
                        }
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        # Class variables with type annotations
                        class_info['class_vars'].append(item.target.id)
                
                classes[node.name] = class_info
                
                # Track key classes
                if self._is_key_class(node.name, class_info):
                    self.codebase_info['key_classes'][node.name] = {
                        'module': '...',  # Will be filled in later
                        'description': class_info['docstring'],
                        'method_count': len(class_info['methods'])
                    }
        
        return classes
    
    def _extract_functions(self, tree: ast.AST) -> Dict[str, Dict]:
        """Extract top-level function definitions."""
        functions = {}
        
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = {
                    'signature': self._get_function_signature(node),
                    'docstring': ast.get_docstring(node)
                }
        
        return functions
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get function signature as string."""
        args = []
        
        # Regular arguments
        for i, arg in enumerate(node.args.args):
            arg_str = arg.arg
            
            # Add type annotation if present
            if arg.annotation:
                arg_str += f": {self._get_name(arg.annotation)}"
            
            # Add default value if present
            default_offset = len(node.args.args) - len(node.args.defaults)
            if i >= default_offset:
                default_idx = i - default_offset
                if default_idx < len(node.args.defaults):
                    default_val = self._get_default_value(node.args.defaults[default_idx])
                    arg_str += f"={default_val}"
            
            args.append(arg_str)
        
        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        # Return type
        return_type = ""
        if node.returns:
            return_type = f" -> {self._get_name(node.returns)}"
        
        return f"({', '.join(args)}){return_type}"
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from various AST node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, str):
            return node
        else:
            return str(type(node).__name__)
    
    def _get_default_value(self, node: ast.AST) -> str:
        """Get default value as string."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return "[]"
        elif isinstance(node, ast.Dict):
            return "{}"
        elif isinstance(node, ast.Call):
            return f"{self._get_name(node.func)}(...)"
        else:
            return "..."
    
    def _is_property(self, node: ast.FunctionDef) -> bool:
        """Check if function is a property."""
        return any(
            isinstance(d, ast.Name) and d.id == 'property'
            for d in node.decorator_list
        )
    
    def _is_classmethod(self, node: ast.FunctionDef) -> bool:
        """Check if function is a classmethod."""
        return any(
            isinstance(d, ast.Name) and d.id == 'classmethod'
            for d in node.decorator_list
        )
    
    def _is_staticmethod(self, node: ast.FunctionDef) -> bool:
        """Check if function is a staticmethod."""
        return any(
            isinstance(d, ast.Name) and d.id == 'staticmethod'
            for d in node.decorator_list
        )
    
    def _has_main_block(self, tree: ast.AST) -> bool:
        """Check if file has if __name__ == '__main__' block."""
        for node in tree.body:
            if (isinstance(node, ast.If) and 
                isinstance(node.test, ast.Compare) and
                isinstance(node.test.left, ast.Name) and
                node.test.left.id == '__name__'):
                return True
        return False
    
    def _extract_data_paths(self, tree: ast.AST, content: str) -> Dict[str, List[str]]:
        """Extract data paths and directory references."""
        data_paths = {
            'reads_from': [],
            'writes_to': [],
            'directories': [],
            'schemas': []
        }
        
        # Look for path patterns in string literals
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                
                # Check for data directories
                if 'data/' in value or 'labels/' in value or 'logbooks/' in value:
                    if '/' in value or '\\' in value:
                        data_paths['directories'].append(value)
                
                # Check for schema references
                if '.schema.json' in value or 'schema' in value.lower():
                    data_paths['schemas'].append(value)
            
            # Check for os.path.join patterns
            elif isinstance(node, ast.Call):
                func_name = self._get_name(node.func)
                
                # File operations
                if 'open' in func_name or 'read' in func_name:
                    data_paths['reads_from'].append(self._extract_path_from_call(node))
                elif 'write' in func_name or 'save' in func_name:
                    data_paths['writes_to'].append(self._extract_path_from_call(node))
                
                # Path joining
                elif 'join' in func_name and len(node.args) > 0:
                    path_parts = []
                    for arg in node.args:
                        if isinstance(arg, ast.Constant):
                            path_parts.append(str(arg.value))
                        elif isinstance(arg, ast.Name):
                            path_parts.append(f"<{arg.id}>")
                    if path_parts:
                        data_paths['directories'].append('/'.join(path_parts))
        
        # Look for common data directory patterns in strings
        import re
        
        # Common BiggerBrother paths
        path_patterns = [
            r'data/chunks/[^"\']*',
            r'data/logbooks/[^"\']*',
            r'data/features/[^"\']*',
            r'data/rpg/[^"\']*',
            r'data/routines/[^"\']*',
            r'labels/[^"\']*',
            r'schemas/[^"\']*'
        ]
        
        for pattern in path_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if match not in data_paths['directories']:
                    data_paths['directories'].append(match)
        
        # Deduplicate and clean
        for key in data_paths:
            data_paths[key] = sorted(list(set(filter(None, data_paths[key]))))
        
        return data_paths
    
    def _extract_io_operations(self, tree: ast.AST, content: str) -> Dict[str, List[str]]:
        """Extract I/O operations and data transformations."""
        io_ops = {
            'file_operations': [],
            'data_formats': [],
            'transformations': []
        }
        
        # Track file operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_name(node.func)
                
                # File I/O
                if 'open(' in content or 'with open' in content:
                    io_ops['file_operations'].append('file_read_write')
                
                # JSON operations
                if 'json.dump' in func_name or 'json.load' in func_name:
                    io_ops['data_formats'].append('json')
                    io_ops['file_operations'].append(f'{func_name}')
                
                # CSV operations
                if 'csv.' in func_name or 'to_csv' in func_name:
                    io_ops['data_formats'].append('csv')
                
                # JSONL operations
                if '.jsonl' in content or 'jsonlines' in content:
                    io_ops['data_formats'].append('jsonl')
                
                # Pickle operations
                if 'pickle.' in func_name:
                    io_ops['data_formats'].append('pickle')
                
                # Data transformations
                if 'transform' in func_name or 'extract' in func_name or 'process' in func_name:
                    io_ops['transformations'].append(func_name)
        
        # Check for class methods that handle data
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                name_lower = node.name.lower()
                
                # Input methods
                if any(word in name_lower for word in ['load', 'read', 'get', 'fetch', 'retrieve']):
                    io_ops['transformations'].append(f"{node.name}()")
                
                # Output methods  
                elif any(word in name_lower for word in ['save', 'write', 'store', 'dump', 'export']):
                    io_ops['transformations'].append(f"{node.name}()")
                
                # Processing methods
                elif any(word in name_lower for word in ['process', 'transform', 'extract', 'analyze']):
                    io_ops['transformations'].append(f"{node.name}()")
        
        # Deduplicate
        for key in io_ops:
            io_ops[key] = sorted(list(set(io_ops[key])))
        
        return io_ops
    
    def _extract_path_from_call(self, node: ast.Call) -> str:
        """Extract path from a function call."""
        if node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant):
                return str(first_arg.value)
            elif isinstance(first_arg, ast.Name):
                return f"<{first_arg.id}>"
            elif isinstance(first_arg, ast.Call):
                return f"<{self._get_name(first_arg.func)}()>"
        return "<dynamic>"
    
    def _is_key_class(self, name: str, info: Dict) -> bool:
        """Determine if a class is a key class."""
        # Heuristics for key classes
        if len(info['methods']) > 5:  # Has significant functionality
            return True
        if 'System' in name or 'Manager' in name or 'Builder' in name:
            return True
        if info['docstring'] and len(info['docstring']) > 100:
            return True
        return False
    
    def _generate_summary(self) -> None:
        """Generate summary statistics."""
        total_files = len(self.codebase_info['modules'])
        total_lines = sum(m['lines'] for m in self.codebase_info['modules'].values())
        total_classes = sum(len(m['classes']) for m in self.codebase_info['modules'].values())
        total_functions = sum(
            len(m['functions']) + 
            sum(len(c['methods']) for c in m['classes'].values())
            for m in self.codebase_info['modules'].values()
        )
        
        self.codebase_info['summary'] = {
            'total_files': total_files,
            'total_lines': total_lines,
            'total_classes': total_classes,
            'total_functions': total_functions,
            'main_packages': self._identify_packages(),
            'external_dependencies': self._identify_external_deps()
        }
    
    def _identify_packages(self) -> List[str]:
        """Identify main packages in the codebase."""
        packages = set()
        for module_path in self.codebase_info['modules'].keys():
            parts = module_path.split('.')
            if len(parts) > 1 and not parts[0].startswith('_'):
                packages.add(parts[0])
        return sorted(list(packages))
    
    def _identify_external_deps(self) -> List[str]:
        """Identify external dependencies."""
        stdlib = {
            'os', 'sys', 'json', 'datetime', 'time', 're', 'math',
            'collections', 'itertools', 'functools', 'typing',
            'pathlib', 'hashlib', 'uuid', 'random', 'copy',
            'ast', 'inspect', 'dataclasses', 'enum', 'abc'
        }
        
        external = set()
        for deps in self.codebase_info['dependencies'].keys():
            if deps and not deps.startswith('.') and deps not in stdlib:
                external.add(deps)
        
        return sorted(list(external))
    
    def _identify_patterns(self) -> None:
        """Identify architectural patterns."""
        patterns = {
            'uses_dataclasses': False,
            'uses_typing': False,
            'uses_async': False,
            'has_tests': False,
            'has_cli': False,
            'has_api': False,
            'uses_ai': False
        }
        
        # Track data flow patterns
        data_flow = {
            'input_modules': [],   # Modules that read external data
            'processing_modules': [],  # Modules that transform data
            'output_modules': [],  # Modules that write data
            'storage_patterns': set()  # Types of storage used
        }
        
        for module_path, info in self.codebase_info['modules'].items():
            # Check for patterns
            if 'dataclasses' in str(info['imports']):
                patterns['uses_dataclasses'] = True
            if 'typing' in str(info['imports']):
                patterns['uses_typing'] = True
            if 'asyncio' in str(info['imports']):
                patterns['uses_async'] = True
            if 'test' in module_path.lower():
                patterns['has_tests'] = True
            if info['has_main']:
                patterns['has_cli'] = True
            if 'fastapi' in str(info['imports']) or 'flask' in str(info['imports']):
                patterns['has_api'] = True
            if 'openai' in str(info['imports']) or 'gpt' in module_path.lower():
                patterns['uses_ai'] = True
            
            # Track data flow
            if 'io_operations' in info:
                io_ops = info['io_operations']
                transforms = io_ops.get('transformations', [])
                
                # Categorize modules by their primary function
                has_load = any('load' in t or 'read' in t or 'get' in t for t in transforms)
                has_save = any('save' in t or 'write' in t or 'store' in t for t in transforms)
                has_process = any('process' in t or 'transform' in t or 'extract' in t for t in transforms)
                
                if has_load and not has_save:
                    data_flow['input_modules'].append(module_path)
                elif has_save and not has_load:
                    data_flow['output_modules'].append(module_path)
                elif has_process or (has_load and has_save):
                    data_flow['processing_modules'].append(module_path)
                
                # Track storage patterns
                for fmt in io_ops.get('data_formats', []):
                    data_flow['storage_patterns'].add(fmt)
        
        self.codebase_info['key_patterns'] = patterns
        self.codebase_info['data_flow'] = data_flow

def generate_context_header(codebase_info: Dict) -> str:
    """Generate a concise context header from codebase analysis."""
    header = []
    
    # Title and summary
    header.append("# BiggerBrother Codebase Context\n")
    header.append("## Summary")
    summary = codebase_info['summary']
    header.append(f"- **Files**: {summary['total_files']} Python files")
    header.append(f"- **Lines**: {summary['total_lines']:,} total lines")
    header.append(f"- **Classes**: {summary['total_classes']} classes")
    header.append(f"- **Functions**: {summary['total_functions']} functions/methods")
    header.append(f"- **Main Packages**: {', '.join(summary['main_packages'])}")
    header.append(f"- **External Dependencies**: {', '.join(summary['external_dependencies'])}")
    
    # Architectural patterns
    header.append("\n## Architecture Patterns")
    patterns = codebase_info['key_patterns']
    header.append(f"- Type hints: {'✓' if patterns['uses_typing'] else '✗'}")
    header.append(f"- Dataclasses: {'✓' if patterns['uses_dataclasses'] else '✗'}")
    header.append(f"- Async/await: {'✓' if patterns['uses_async'] else '✗'}")
    header.append(f"- CLI interface: {'✓' if patterns['has_cli'] else '✗'}")
    header.append(f"- API endpoints: {'✓' if patterns['has_api'] else '✗'}")
    header.append(f"- AI integration: {'✓' if patterns['uses_ai'] else '✗'}")
    header.append(f"- Test coverage: {'✓' if patterns['has_tests'] else '✗'}")
    
    # Data Flow Paths
    header.append("\n## Data Flow and Storage\n")
    
    # Collect all data paths
    all_paths = defaultdict(set)
    data_formats = set()
    
    for module_info in codebase_info['modules'].values():
        if 'data_paths' in module_info:
            for path in module_info['data_paths'].get('directories', []):
                if 'data/' in path:
                    all_paths['data'].add(path)
                elif 'labels/' in path:
                    all_paths['labels'].add(path)
                elif 'schemas/' in path:
                    all_paths['schemas'].add(path)
        
        if 'io_operations' in module_info:
            data_formats.update(module_info['io_operations'].get('data_formats', []))
    
    header.append("### Primary Data Directories")
    header.append("```")
    header.append("data/")
    header.append("├── chunks/           # Message chunks (JSON)")
    header.append("├── labels/           # Semantic labels (JSON)")
    header.append("├── logbooks/         # Dynamic log categories (CSV + JSONL)")
    header.append("├── features/         # Extracted features (JSON)")
    header.append("├── rpg/             # Game state (JSON)")
    header.append("├── routines/        # Routine definitions (JSON)")
    header.append("├── scheduler/       # Schedule data (JSON)")
    header.append("└── tracking/        # Graph store (JSONL)")
    header.append("```")
    
    if data_formats:
        header.append(f"\n**Data Formats Used**: {', '.join(sorted(data_formats))}")
    
    # Key Data Flow Patterns
    header.append("\n### Data Flow Patterns\n")
    
    # Identify modules with significant I/O
    io_modules = []
    for module_path, info in codebase_info['modules'].items():
        if 'io_operations' in info and info['io_operations'].get('transformations'):
            io_modules.append((module_path, info))
    
    if io_modules:
        header.append("**Key Data Processing Modules:**")
        for module_path, info in sorted(io_modules)[:10]:  # Top 10
            transforms = info['io_operations']['transformations'][:3]  # First 3
            if transforms:
                header.append(f"- `{module_path}`: {', '.join(transforms)}")
    
    # Key modules with classes
    header.append("\n## Key Modules and Classes\n")
    
    # Group modules by package
    packages = defaultdict(list)
    for module_path, info in codebase_info['modules'].items():
        if info['classes']:  # Only include modules with classes
            parts = module_path.split('.')
            package = parts[0] if len(parts) > 1 else 'root'
            packages[package].append((module_path, info))
    
    for package in sorted(packages.keys()):
        header.append(f"### {package}/\n")
        
        for module_path, info in sorted(packages[package]):
            # Module header
            rel_path = info['path']
            header.append(f"**`{rel_path}`**")
            
            if info['docstring']:
                # First line of docstring
                first_line = info['docstring'].split('\n')[0]
                header.append(f"_{first_line}_")
            
            # Data paths if present
            if 'data_paths' in info and any(info['data_paths'].values()):
                paths = info['data_paths']
                if paths.get('reads_from'):
                    header.append(f"📥 Reads: `{', '.join(paths['reads_from'][:2])}`")
                if paths.get('writes_to'):
                    header.append(f"📤 Writes: `{', '.join(paths['writes_to'][:2])}`")
                if paths.get('directories'):
                    unique_dirs = set(d.split('/')[0] + '/' if '/' in d else d for d in paths['directories'])
                    header.append(f"📁 Uses: `{', '.join(sorted(unique_dirs)[:3])}`")
            
            # Imports summary (only key external imports)
            external_imports = [
                imp for imp in info['imports']
                if not imp.startswith('.') and not imp.split('.')[0] in {
                    'os', 'sys', 'json', 'datetime', 'typing', 're'
                }
            ]
            if external_imports:
                header.append(f"Imports: `{', '.join(external_imports[:5])}`")
            
            # Classes
            if info['classes']:
                header.append("```python")
                for class_name, class_info in info['classes'].items():
                    # Class definition
                    bases = f"({', '.join(class_info['bases'])})" if class_info['bases'] else ""
                    header.append(f"class {class_name}{bases}:")
                    
                    # Key methods (init, public methods)
                    for method_name, method_info in class_info['methods'].items():
                        if (method_name == '__init__' or 
                            not method_name.startswith('_') or 
                            method_info['is_property']):
                            
                            prefix = ""
                            if method_info['is_property']:
                                prefix = "@property "
                            elif method_info['is_classmethod']:
                                prefix = "@classmethod "
                            elif method_info['is_staticmethod']:
                                prefix = "@staticmethod "
                            
                            header.append(f"    {prefix}def {method_name}{method_info['signature']}")
                header.append("```")
            
            # Key functions (if any)
            if info['functions'] and not info['classes']:
                header.append("```python")
                for func_name, func_info in info['functions'].items():
                    if not func_name.startswith('_'):
                        header.append(f"def {func_name}{func_info['signature']}")
                header.append("```")
            
            header.append("")  # Blank line between modules
    
    # Data Pipeline
    header.append("\n## Data Pipeline\n")
    header.append("```")
    header.append("1. INPUT: Raw conversations → chunks/ (via chunking)")
    header.append("2. LABEL: chunks/ → labels/ (via OpenAI)")
    header.append("3. HARMONIZE: labels/ → canonical labels (via harmonizer)")
    header.append("4. LOG: Natural language → logbooks/ (via GPT-5-nano)")
    header.append("5. EXTRACT: All data → features/ (via FeatureExtractor)")
    header.append("6. PERSIST: All → GraphStore (JSONL)")
    header.append("```")
    
    # Module categorization by data flow
    if 'data_flow' in codebase_info:
        flow = codebase_info['data_flow']
        
        if flow.get('input_modules'):
            header.append("\n**Input Modules** (data ingestion):")
            for mod in flow['input_modules'][:5]:
                header.append(f"- `{mod}`")
        
        if flow.get('processing_modules'):
            header.append("\n**Processing Modules** (transformation):")
            for mod in flow['processing_modules'][:5]:
                header.append(f"- `{mod}`")
        
        if flow.get('output_modules'):
            header.append("\n**Output Modules** (persistence):")
            for mod in flow['output_modules'][:5]:
                header.append(f"- `{mod}`")
    
    # Dependency graph
    header.append("\n## Key Dependencies\n")
    header.append("```")
    header.append("app/ → OpenAI API")
    header.append("assistant/behavioral_engine/ → app.openai_client")
    header.append("assistant/behavioral_engine/ → assistant.logger, assistant.graph")
    header.append("assistant/ml/ → sklearn, pandas (when available)")
    header.append("```")
    
    return '\n'.join(header)

def main():
    """Generate context header for BiggerBrother."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate context header for codebase')
    parser.add_argument('--root', default='.', help='Root directory to analyze')
    parser.add_argument('--output', default='CONTEXT.md', help='Output file')
    parser.add_argument('--json', action='store_true', help='Also save JSON analysis')
    args = parser.parse_args()
    
    # Analyze codebase
    analyzer = CodebaseAnalyzer(args.root)
    codebase_info = analyzer.analyze()
    
    # Generate context header
    header = generate_context_header(codebase_info)
    
    # Save context header
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(header)
    
    print(f"\n✅ Context header saved to {args.output}")
    print(f"   {len(header.splitlines())} lines generated")
    
    # Optionally save full JSON analysis
    if args.json:
        json_file = args.output.replace('.md', '_analysis.json')
        
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return sorted(list(obj))
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(item) for item in obj]
            return obj
        
        serializable_info = convert_sets(codebase_info)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_info, f, indent=2)
        
        print(f"   Full analysis saved to {json_file}")

if __name__ == '__main__':
    main()
