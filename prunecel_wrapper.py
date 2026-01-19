"""
PruneCEL Wrapper for Python Integration
Provides a Python interface to the Java-based PruneCEL learner.
"""
import os
import json
import subprocess
import tempfile
import time
import shutil
from pathlib import Path
from typing import Set, Dict, Any
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.parser import DLSyntaxParser
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard


class PruneCELWrapper:
    """
    Python wrapper for the Java-based PruneCEL concept learner.
    
    Requirements:
    - PruneCEL JAR file compiled: target/prune-cel-0.0.1-SNAPSHOT.jar
    - Triple store running with SPARQL endpoint (e.g., Fuseki, Tentris)
    - Java Runtime Environment (JRE) installed
    
    Usage:
        prunecel = PruneCELWrapper(
            jar_path="path/to/prune-cel-0.0.1-SNAPSHOT.jar",
            sparql_url="http://localhost:3030/family/sparql",
            knowledge_base=kb,
            max_runtime=60  # seconds
        )
        pred = prunecel.fit(learning_problem).best_hypothesis()
    """
    
    def __init__(self, 
                 jar_path: str,
                 sparql_url: str,
                 knowledge_base: KnowledgeBase,
                 max_runtime: int = 60,
                 accuracy_function: int = 0,  # 0=F1, 1=BalancedAcc, 2=Accuracy
                 recursive: bool = True,       # -R extension
                 skip_none: bool = True,       # -S extension
                 ontology: str = "ALC",
                 punish_long: bool = True,
                 avoid_picky: bool = True):
        """
        Initialize PruneCEL wrapper.
        
        Args:
            jar_path: Path to compiled PruneCEL JAR file
            sparql_url: URL of SPARQL endpoint with loaded knowledge base
            knowledge_base: Ontolearn KnowledgeBase instance
            max_runtime: Maximum runtime in seconds
            accuracy_function: 0=F1, 1=BalancedAccuracy, 2=Accuracy
            recursive: Enable recursive extension (-R)
            skip_none: Enable skip-none extension (-S)
            ontology: Ontology language (default: ALC)
            punish_long: Punish long expressions
            avoid_picky: Avoid picky solutions decorator
        """
        self.jar_path = Path(jar_path)
        if not self.jar_path.exists():
            raise FileNotFoundError(f"PruneCEL JAR not found at: {jar_path}")
        
        self.sparql_url = sparql_url
        self.kb = knowledge_base
        self.max_runtime_seconds = max_runtime
        self.max_runtime_ms = max_runtime * 1000
        self.accuracy_function = accuracy_function
        self.recursive = recursive
        self.skip_none = skip_none
        self.ontology = ontology
        self.punish_long = punish_long
        self.avoid_picky = avoid_picky
        
        # For tracking
        self._number_of_tested_concepts = 0
        self._last_prediction = None
        self._last_runtime = 0
        self._last_f1_score = 0.0
        self._last_train_f1 = 0.0
        self._last_pos_count = 0
        self._last_neg_count = 0
        
        # Create temp directories for I/O
        self.temp_dir = Path(tempfile.mkdtemp(prefix="prunecel_"))
        
        # Verify Java is available
        self._check_java()
    
    def _check_java(self):
        """Verify Java is installed and accessible."""
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Java is not properly installed")
        except FileNotFoundError:
            raise RuntimeError("Java not found. Please install Java Runtime Environment (JRE)")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Java version check timed out")
    
    def fit(self, learning_problem: PosNegLPStandard):
        """
        Fit PruneCEL on the given learning problem.
        
        Args:
            learning_problem: Learning problem with positive and negative examples
            
        Returns:
            self (to allow method chaining like learner.fit(lp).best_hypothesis())
        """
        # Create temporary input JSON file
        lp_file = self.temp_dir / f"lp_{time.time()}.json"
        output_file = self.temp_dir / f"output_{time.time()}.csv"
        
        # Convert learning problem to PruneCEL JSON format
        lp_json = self._create_learning_problem_json(learning_problem)
        
        with open(lp_file, 'w') as f:
            json.dump(lp_json, f, indent=2)
        
        # Build PruneCEL command
        cmd = [
            "java",
            "-cp", str(self.jar_path),
            "org.example.cel.PruneCEL_CLI",
            "--sparqlUrl", self.sparql_url,
            "--ontology", self.ontology,
            "--accuracyfunction", str(self.accuracy_function),
            "--punishLongExpression", str(self.punish_long).lower(),
            "--avoidPickySolutionsDecorator", str(self.avoid_picky).lower(),
            "--iteration", "0",
            "--time", str(self.max_runtime_ms),
            "--recursive", str(self.recursive).lower(),
            "--skipNone", str(self.skip_none).lower(),
            "--inputFile", str(lp_file),
            "--outputFile", str(output_file),
            "--cluster", "false",
            "--folds", "1"
        ]
        
        # Run PruneCEL
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.max_runtime_seconds + 30  # Add buffer
            )
            self._last_runtime = time.time() - start_time
            
            if result.returncode != 0:
                print(f"PruneCEL Error (stderr): {result.stderr}")
                raise RuntimeError(f"PruneCEL execution failed with code {result.returncode}")
            
            # Parse number of tested concepts from stdout/stderr
            # PruneCEL prints: "Stopping search. Saw X expressions."
            self._parse_concepts_tested(result.stdout, result.stderr)
            
            # Parse output
            self._parse_output(output_file)
            
        except subprocess.TimeoutExpired:
            self._last_runtime = self.max_runtime_seconds
            print(f"PruneCEL timed out after {self.max_runtime_seconds} seconds")
            # Use a fallback simple concept
            self._last_prediction = "⊤"  # OWL Thing
            self._number_of_tested_concepts = 0
        
        finally:
            # Cleanup temporary files (keep for debugging)
            # if lp_file.exists():
            #     lp_file.unlink()
            # if output_file.exists():
            #     output_file.unlink()
            pass
        
        return self
    
    def _create_learning_problem_json(self, lp: PosNegLPStandard) -> Dict[str, Any]:
        """
        Convert Ontolearn learning problem to PruneCEL JSON format.
        
        PruneCEL expects format:
        {
            "problems": {
                "ProblemName": {
                    "positive_examples": ["http://example.org/ind1", ...],
                    "negative_examples": ["http://example.org/ind2", ...]
                }
            }
        }
        """
        pos_examples = [ind.str if hasattr(ind, 'str') else str(ind.iri) for ind in lp.pos]
        neg_examples = [ind.str if hasattr(ind, 'str') else str(ind.iri) for ind in lp.neg]
        
        return {
            "problems": {
                "temp_problem": {
                    "positive_examples": pos_examples,
                    "negative_examples": neg_examples
                }
            }
        }
    
    def _parse_concepts_tested(self, stdout: str, stderr: str):
        """
        Parse the number of tested concepts from PruneCEL output.
        
        PruneCEL prints: "Stopping search. Saw X expressions."
        """
        import re
        
        # Check both stdout and stderr
        combined_output = stdout + "\n" + stderr
        
        # Pattern: "Stopping search. Saw X expressions."
        match = re.search(r'Stopping search\.\s+Saw\s+(\d+)\s+expressions?\.?', combined_output, re.IGNORECASE)
        if match:
            self._number_of_tested_concepts = int(match.group(1))
        else:
            # Fallback: try to find any "Saw X expressions" pattern
            match = re.search(r'Saw\s+(\d+)\s+expressions?', combined_output, re.IGNORECASE)
            if match:
                self._number_of_tested_concepts = int(match.group(1))
            else:
                # If we can't find it, keep whatever was set (default 0)
                pass
    
    def _parse_output(self, output_file: Path):
        """
        Parse PruneCEL CSV output to extract the learned concept.
        
        PruneCEL outputs a CSV with columns including the learned expression.
        """
        if not output_file.exists():
            print(f"Warning: PruneCEL output file not found: {output_file}")
            exit(0)
        
        try:
            # Read CSV output
            import csv
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    print("Warning: PruneCEL output is empty")
                    exit(0)
                
                # Get first row (should be the best hypothesis)
                row = rows[0]
                
                # DEBUG: Print what we're actually getting
                print(f"\n=== PruneCEL CSV Output Debug ===")
                print(f"Available columns: {list(row.keys())}")
                print(f"First 3 columns with values:")
                for i, (k, v) in enumerate(list(row.items())[:3]):
                    print(f"  {k}: {v[:200] if len(str(v)) > 200 else v}")
                print(f"=================================\n")
                
                # Extract F1 score and other metrics from PruneCEL output
                try:
                    self._last_f1_score = float(row.get('F1-score', 0.0))
                except (ValueError, TypeError):
                    self._last_f1_score = 0.0
                
                try:
                    self._last_pos_count = int(row.get('PosCount', 0))
                except (ValueError, TypeError):
                    self._last_pos_count = 0
                
                try:
                    self._last_neg_count = int(row.get('NegCount', 0))
                except (ValueError, TypeError):
                    self._last_neg_count = 0
                
                # Compute train F1 from pos/neg counts
                if 'Number-of-Pos' in row and 'Number-of-Neg' in row:
                    try:
                        total_pos = int(row['Number-of-Pos'])
                        total_neg = int(row['Number-of-Neg'])
                        if total_pos > 0 or total_neg > 0:
                            # Compute F1: correctly classified pos / total examples
                            precision = self._last_pos_count / (self._last_pos_count + self._last_neg_count) if (self._last_pos_count + self._last_neg_count) > 0 else 0
                            recall = self._last_pos_count / total_pos if total_pos > 0 else 0
                            self._last_train_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        else:
                            self._last_train_f1 = self._last_f1_score
                    except (ValueError, TypeError, KeyError):
                        self._last_train_f1 = self._last_f1_score
                else:
                    self._last_train_f1 = self._last_f1_score
                
                # Extract the concept expression (column name may vary)
                # Common column names: 'Prediction', 'Expression', 'Hypothesis'
                concept_str = None
                for col in ['Expressions']:
                    if col in row:
                        concept_str = row[col]
                        break
                
                if not concept_str:
                    print(f"Warning: Could not find concept expression in output. Available columns: {list(row.keys())}")
                    concept_str = "⊤"
                
                # Parse PruneCEL format: extract first expression from list
                # Format: [ScoredClassExpression [classExpression=..., cScore=...], ...]
                if concept_str.startswith('[ScoredClassExpression'):
                    # Extract first classExpression (no parentheses around the value)
                    import re
                    match = re.search(r'classExpression=([^,]+),', concept_str)
                    if match:
                        concept_str = match.group(1).strip()
                    else:
                        print(f"Warning: Could not parse ScoredClassExpression format {concept_str}")
                        exit(0)
                
                # Clean up the expression - PruneCEL uses full IRIs
                # We need to keep them as-is for now
                self._last_prediction = concept_str.strip()
                
                # Note: _number_of_tested_concepts should be set by _parse_concepts_tested()
                # from stdout/stderr, not by counting expressions in output
                
        except Exception as e:
            print(f"Error parsing PruneCEL output: {e}")
            self._last_prediction = "⊤"
            self._number_of_tested_concepts = 0
            self._last_f1_score = 0.0
            self._last_train_f1 = 0.0
            self._last_pos_count = 0
            self._last_neg_count = 0
    
    def _preprocess_prunecel_expression(self, expression: str) -> str:
        """
        Preprocess PruneCEL expression to make it parseable by DLSyntaxParser.
        PruneCEL outputs full IRIs, but we need short names (just the fragment after #).
        Also adds spaces around operators for proper parsing.
        
        Example: 
        Input:  http://example.org#Female⊓∃http://example.org#married.∃http://example.org#hasSibling.⊤
        Output: Female ⊓ ∃ married.(∃ hasSibling.⊤)
        """
        import re
        # Find all full IRIs and replace them with just the fragment (short name)
        # Match: http(s)://...#fragment where fragment is the short name
        # The pattern captures everything from http to # and then captures the fragment
        iri_pattern = r'https?://[^#]+#([a-zA-Z0-9_]+)'
        
        # Replace with just the fragment (short name) - group 1
        processed = re.sub(iri_pattern, r'\1', expression)
        
        # Add spaces around operators for the parser
        operators = ['⊓', '⊔', '∃', '∀', '≤', '≥', '=']
        for op in operators:
            processed = processed.replace(op, f' {op} ')
        
        # Clean up multiple spaces
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # Fix successive existential/universal quantifiers with proper parentheses
        # Pattern: ". ∃" or ". ∀" should become ".(∃" and we need to close parens
        # We need to find where to close each opened parenthesis
        
        # Count and mark positions where we need opening parens
        def add_nested_parens(text):
            # Find all positions where we have ". quantifier"
            pattern = r'\.(\s+)(∃|∀)'
            matches = list(re.finditer(pattern, text))
            
            if not matches:
                return text
            
            # Work backwards to avoid index shifting
            for match in reversed(matches):
                start = match.start()
                quantifier = match.group(2)
                
                # Find the extent of this quantifier's scope
                # It ends at: ⊓, ⊔, ), or end of string
                rest = text[match.end():]
                
                # Find where this quantifier scope ends
                paren_depth = 0
                end_pos = len(rest)
                
                for i, char in enumerate(rest):
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        if paren_depth == 0:
                            end_pos = i
                            break
                        paren_depth -= 1
                    elif char in ['⊓', '⊔'] and paren_depth == 0:
                        # Need to skip the space before the operator
                        end_pos = i
                        while end_pos > 0 and rest[end_pos - 1] == ' ':
                            end_pos -= 1
                        break
                
                # Insert closing paren at end_pos and opening paren after dot
                actual_end = match.end() + end_pos
                text = text[:start + 1] + '(' + quantifier + ' ' + text[match.end():actual_end] + ')' + text[actual_end:]
            
            return text
        
        processed = add_nested_parens(processed)
        
        return processed
    
    def best_hypothesis(self):
        """
        Return the best hypothesis (learned concept expression).
        
        Returns:
            OWLClassExpression that can be evaluated with kb.individuals()
        """
        if self._last_prediction is None:
            raise RuntimeError("No prediction available. Call fit() first.")
        
        # Try to parse the DL syntax expression into OWL class expression
        # try:
        # Preprocess to wrap IRIs in angle brackets
        processed_expr = self._preprocess_prunecel_expression(self._last_prediction)
        # print(f"Processed PruneCEL expression for parsing: {processed_expr}")
        namespace =  list(self.kb.ontology.classes_in_signature())[0].iri.get_namespace()
        parser = DLSyntaxParser(namespace=namespace)

        # print(f"Attempting to parse PruneCEL expression: {self._last_prediction}")
        # print(f"Using processed expression: {processed_expr}")
        # exit(0)
        owl_expr = parser.parse(processed_expr)
        # print(f"Parsed OWL Expression: {owl_expr}" )
        # exit(0)
        return owl_expr
        # except Exception as e:
        #     print(f"Warning: Could not parse PruneCEL expression '{self._last_prediction}': {e}")
        #     print(f"Falling back to OWL Thing (⊤)")
        #     # Fallback to Thing if parsing fails
        #     from owlapy.class_expression import OWLThing
        #     return OWLThing
    
    @property
    def number_of_tested_concepts(self) -> int:
        """Get the number of concepts tested by PruneCEL."""
        return self._number_of_tested_concepts
    
    @property
    def last_runtime(self) -> float:
        """Get the runtime of the last PruneCEL execution in seconds."""
        return self._last_runtime
    
    def clean(self):
        """Clean up temporary files."""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass  # Ignore cleanup errors during shutdown
    
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.clean()
        except Exception:
            pass  # Ignore errors during interpreter shutdown


def check_prunecel_available(jar_path: str = None) -> bool:
    """
    Check if PruneCEL is available and properly set up.
    
    Args:
        jar_path: Optional path to PruneCEL JAR. If None, looks in common locations.
        
    Returns:
        True if PruneCEL can be used, False otherwise
    """
    # Check Java
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            print("Java is not available")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("Java is not available")
        return False
    
    # Check JAR file
    if jar_path:
        jar_paths = [Path(jar_path)]
    else:
        # Common locations
        jar_paths = [
            Path("target/prune-cel-0.0.1-SNAPSHOT.jar"),
            Path("PruneCEL/target/prune-cel-0.0.1-SNAPSHOT.jar"),
            Path("../PruneCEL/target/prune-cel-0.0.1-SNAPSHOT.jar"),
        ]
    
    for path in jar_paths:
        if path.exists():
            print(f"Found PruneCEL JAR at: {path}")
            return True
    
    print(f"PruneCEL JAR not found. Checked locations: {[str(p) for p in jar_paths]}")
    print("Please compile PruneCEL using: cd PruneCEL && mvn clean package")
    return False
