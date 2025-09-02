"""
Label Integration Wrapper Classes
==================================

These wrapper classes provide a clean interface to the labeling and coverage
functionality while handling the mismatch between the expected class-based
API and the actual function-based implementation.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json

# Import the actual modules
from app import label_generator
from app import coverage_checker


class LabelGenerator:
    """
    Wrapper class for label_generator module functions.
    Provides a class-based interface to the label generation functionality.
    """
    
    def __init__(self, openai_client):
        """Initialize the label generator with an OpenAI client."""
        self.openai_client = openai_client
        self.data_root = Path("data")
        
    def generate_labels_for_text(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, List[Dict]]:
        """
        Generate labels for a single text string.
        
        Args:
            text: The text to label
            metadata: Optional metadata (gid, role, conversation_id, etc.)
            
        Returns:
            Dictionary with categorized labels (topic, tone, intent)
        """
        # Handle None metadata
        if metadata is None:
            metadata = {}
        
        # Create a chunk structure
        chunk = {
            "gid": metadata.get("gid", "temp_" + str(hash(text))[:8]),
            "content_text": text,
            "role": metadata.get("role", "user"),
            "conversation_id": metadata.get("conversation_id", "")
        }
        
        # Get the prompt head
        prompt_head = label_generator._load_prompt_head(self.data_root)
        
        # Build the prompt
        prompt = label_generator._build_prompt(prompt_head, chunk)
        
        try:
            # Call OpenAI to generate labels
            if hasattr(self.openai_client, "chat"):
                raw = self.openai_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-nano",
                    response_format={"type": "json_object"},
                )
            else:
                raw = self.openai_client.complete(prompt=prompt, model="gpt-5-nano")
            
            # Parse the response
            payload = label_generator._coerce_json(raw)
            
            # Return categorized labels as expected by the system
            return {
                "topic": label_generator._norm_list(payload.get("topic", [])),
                "tone": label_generator._norm_list(payload.get("tone", [])),
                "intent": label_generator._norm_list(payload.get("intent", [])),
                "confidence": payload.get("confidence", 0.5)
            }
            
        except Exception as e:
            print(f"Error generating labels: {e}")
            return {"topic": [], "tone": [], "intent": [], "confidence": 0.0}
    
    def generate_labels_for_chunk(self, chunk: Dict) -> Dict:
        """
        Generate labels for a chunk dictionary.
        
        Args:
            chunk: Dictionary with 'gid', 'content_text', etc.
            
        Returns:
            Dictionary with categorized labels (topic, tone, intent)
        """
        prompt_head = label_generator._load_prompt_head(self.data_root)
        prompt = label_generator._build_prompt(prompt_head, chunk)
        
        try:
            if hasattr(self.openai_client, "chat"):
                raw = self.openai_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-nano",
                    response_format={"type": "json_object"},
                )
            else:
                raw = self.openai_client.complete(prompt=prompt, model="gpt-5-nano")
            
            payload = label_generator._coerce_json(raw)
            
            return {
                "topic": label_generator._norm_list(payload.get("topic", [])),
                "tone": label_generator._norm_list(payload.get("tone", [])),
                "intent": label_generator._norm_list(payload.get("intent", [])),
                "confidence": payload.get("confidence", 0.0)
            }
            
        except Exception as e:
            print(f"Error generating labels for chunk: {e}")
            return {"topic": [], "tone": [], "intent": [], "confidence": 0.0}
    
    def batch_generate_labels(
        self, 
        texts: List[str], 
        labels_dir: str = "labels",
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Generate labels for multiple texts in batch.
        
        Args:
            texts: List of text strings to label
            labels_dir: Directory to save label files
            skip_existing: Whether to skip texts that already have labels
            
        Returns:
            Summary dictionary with counts and file paths
        """
        Path(labels_dir).mkdir(parents=True, exist_ok=True)
        
        results = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "labels": []
        }
        
        for text in texts:
            gid = f"text_{hash(text)}".replace("-", "")[:16]
            label_file = Path(labels_dir) / f"{gid}.json"
            
            if skip_existing and label_file.exists():
                results["skipped"] += 1
                continue
            
            labels = self.generate_labels_for_text(text, {"gid": gid})
            
            if labels:
                # Save to file
                try:
                    label_data = {
                        "gid": gid,
                        "text": text[:500],  # Store first 500 chars for reference
                        "labels": labels,
                        "timestamp": str(Path.ctime(Path.cwd()))
                    }
                    
                    with open(label_file, "w") as f:
                        json.dump(label_data, f, indent=2)
                    
                    results["processed"] += 1
                    results["labels"].append(label_data)
                except Exception as e:
                    print(f"Error saving labels for {gid}: {e}")
                    results["errors"] += 1
            else:
                results["errors"] += 1
        
        return results


class CoverageChecker:
    """
    Wrapper class for coverage_checker module functions.
    Provides coverage analysis and validation for labeled data.
    """
    
    def __init__(self, openai_client):
        """Initialize the coverage checker with an OpenAI client."""
        self.openai_client = openai_client
        
    def check_coverage(
        self,
        data_root: str = "data",
        labels_dir: str = "labels",
        manifest_glob: Optional[List[str]] = None,
        autogenerate_missing: bool = False
    ) -> Dict[str, Any]:
        """
        Check label coverage for chunks in the data directory.
        
        Args:
            data_root: Root directory containing chunk data
            labels_dir: Directory containing label files
            manifest_glob: Patterns to find manifest files
            autogenerate_missing: Whether to generate missing labels
            
        Returns:
            Coverage statistics and details
        """
        if manifest_glob is None:
            manifest_glob = ["chunk_manifest.json", "*.manifest.json"]
        
        inputs = {
            "data_root": data_root,
            "manifest_glob": manifest_glob
        }
        
        outputs = {
            "labels_dir": labels_dir,
            "records_schema": "schemas/label_record.schema.json"
        }
        
        # Call the coverage checker main function
        result = coverage_checker.main(
            inputs,
            outputs,
            openai_client=self.openai_client,
            autogenerate_missing=autogenerate_missing
        )
        
        return result
    
    def validate_labels(self, labels_dir: str = "labels") -> Dict[str, Any]:
        """
        Validate all label files in a directory.
        
        Args:
            labels_dir: Directory containing label files
            
        Returns:
            Validation results with counts and issues
        """
        labels_path = Path(labels_dir)
        
        results = {
            "total_files": 0,
            "valid": 0,
            "invalid": 0,
            "issues": []
        }
        
        if not labels_path.exists():
            results["issues"].append(f"Labels directory {labels_dir} does not exist")
            return results
        
        for label_file in labels_path.glob("*.json"):
            results["total_files"] += 1
            
            try:
                with open(label_file, "r") as f:
                    data = json.load(f)
                
                # Check required fields
                issues = []
                
                if "gid" not in data:
                    issues.append("Missing 'gid' field")
                
                # Check label structure
                if "labels" in data:
                    if not isinstance(data["labels"], list):
                        issues.append("'labels' field is not a list")
                    else:
                        for label in data["labels"]:
                            if not isinstance(label, dict):
                                issues.append("Label entry is not a dictionary")
                                break
                            if "label" not in label:
                                issues.append("Label entry missing 'label' field")
                            if "confidence" not in label and "p" not in label:
                                issues.append("Label entry missing confidence score")
                
                # Check categorized structure (alternative format)
                elif any(cat in data for cat in ["topic", "tone", "intent"]):
                    for category in ["topic", "tone", "intent"]:
                        if category in data and not isinstance(data[category], list):
                            issues.append(f"'{category}' field is not a list")
                else:
                    issues.append("No recognizable label structure found")
                
                if issues:
                    results["invalid"] += 1
                    results["issues"].append({
                        "file": str(label_file),
                        "problems": issues
                    })
                else:
                    results["valid"] += 1
                    
            except json.JSONDecodeError as e:
                results["invalid"] += 1
                results["issues"].append({
                    "file": str(label_file),
                    "problems": [f"Invalid JSON: {e}"]
                })
            except Exception as e:
                results["invalid"] += 1
                results["issues"].append({
                    "file": str(label_file),
                    "problems": [f"Error reading file: {e}"]
                })
        
        results["coverage_percentage"] = (
            (results["valid"] / results["total_files"] * 100) 
            if results["total_files"] > 0 else 0
        )
        
        return results
    
    def get_label_statistics(self, labels_dir: str = "labels") -> Dict[str, Any]:
        """
        Get statistics about the labels in a directory.
        
        Args:
            labels_dir: Directory containing label files
            
        Returns:
            Statistics about label distribution and frequency
        """
        labels_path = Path(labels_dir)
        
        stats = {
            "total_files": 0,
            "total_labels": 0,
            "categories": {
                "topic": {},
                "tone": {},
                "intent": {}
            },
            "confidence_distribution": {
                "0.0-0.2": 0,
                "0.2-0.4": 0,
                "0.4-0.6": 0,
                "0.6-0.8": 0,
                "0.8-1.0": 0
            }
        }
        
        if not labels_path.exists():
            return stats
        
        for label_file in labels_path.glob("*.json"):
            try:
                with open(label_file, "r") as f:
                    data = json.load(f)
                
                stats["total_files"] += 1
                
                # Process flat label list
                if "labels" in data and isinstance(data["labels"], list):
                    for label in data["labels"]:
                        if isinstance(label, dict):
                            category = label.get("category", "unknown")
                            label_text = label.get("label", "")
                            confidence = label.get("confidence", label.get("p", 0))
                            
                            if category in stats["categories"]:
                                if label_text not in stats["categories"][category]:
                                    stats["categories"][category][label_text] = 0
                                stats["categories"][category][label_text] += 1
                            
                            # Update confidence distribution
                            if confidence <= 0.2:
                                stats["confidence_distribution"]["0.0-0.2"] += 1
                            elif confidence <= 0.4:
                                stats["confidence_distribution"]["0.2-0.4"] += 1
                            elif confidence <= 0.6:
                                stats["confidence_distribution"]["0.4-0.6"] += 1
                            elif confidence <= 0.8:
                                stats["confidence_distribution"]["0.6-0.8"] += 1
                            else:
                                stats["confidence_distribution"]["0.8-1.0"] += 1
                            
                            stats["total_labels"] += 1
                
                # Process categorized structure
                else:
                    for category in ["topic", "tone", "intent"]:
                        if category in data and isinstance(data[category], list):
                            for item in data[category]:
                                if isinstance(item, dict):
                                    label_text = item.get("label", "")
                                    confidence = item.get("p", item.get("probability", 0))
                                    
                                    if label_text not in stats["categories"][category]:
                                        stats["categories"][category][label_text] = 0
                                    stats["categories"][category][label_text] += 1
                                    
                                    # Update confidence distribution
                                    if confidence <= 0.2:
                                        stats["confidence_distribution"]["0.0-0.2"] += 1
                                    elif confidence <= 0.4:
                                        stats["confidence_distribution"]["0.2-0.4"] += 1
                                    elif confidence <= 0.6:
                                        stats["confidence_distribution"]["0.4-0.6"] += 1
                                    elif confidence <= 0.8:
                                        stats["confidence_distribution"]["0.6-0.8"] += 1
                                    else:
                                        stats["confidence_distribution"]["0.8-1.0"] += 1
                                    
                                    stats["total_labels"] += 1
                        
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
        
        # Sort labels by frequency
        for category in stats["categories"]:
            sorted_labels = sorted(
                stats["categories"][category].items(),
                key=lambda x: x[1],
                reverse=True
            )
            stats["categories"][category] = dict(sorted_labels[:20])  # Top 20
        
        return stats


# Example usage
if __name__ == "__main__":
    from app.openai_client import OpenAIClient
    
    # Initialize the OpenAI client
    client = OpenAIClient()
    
    # Create wrapper instances
    label_gen = LabelGenerator(client)
    coverage = CoverageChecker(client)
    
    # Example: Generate labels for a text
    text = "I had a great workout this morning, ran 5 miles and feeling energized!"
    labels = label_gen.generate_labels_for_text(text)
    print("Generated labels:", labels)
    
    # Example: Check coverage
    coverage_result = coverage.check_coverage(
        data_root="data",
        labels_dir="labels",
        autogenerate_missing=False
    )
    print("Coverage:", coverage_result["coverage"])
    
    # Example: Get label statistics
    stats = coverage.get_label_statistics("labels")
    print("Label statistics:", stats)
