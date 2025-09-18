import ipywidgets as widgets
from IPython.display import display, clear_output
from itertools import product
from typing import Dict, Any, Callable, List, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pickle
import joblib
import json
import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from tabpfn import TabPFNClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

@dataclass
class ComponentConfig:
    """Configuration for a pipeline component"""
    name: str
    factory_func: Callable
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    dependencies: List[str] = field(default_factory=list)  # Components this depends on
    conflicts: List[str] = field(default_factory=list)     # Components this conflicts with


class ComponentFactory:
    """Factory for creating pipeline components"""
    
    @staticmethod
    def create_scaler(scaler_type: str, **params) -> Any:
        """Create scaler components"""
        if scaler_type == "standard":
            return StandardScaler()
        elif scaler_type.startswith("minmax"):
            feature_range = params.get("feature_range", (0, 1))
            return MinMaxScaler(feature_range=feature_range)
        elif scaler_type == "robust":
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    @staticmethod
    def create_pca(n_components: Union[int, float], **params) -> PCA:
        """Create PCA component"""
        return PCA(n_components=n_components, **params)
    
    @staticmethod
    def create_model(model_type: str, **params) -> Any:
        """Create model components"""
        random_state = params.get("random_state", 42)
        
        if model_type == "logistic":
            return LogisticRegression(max_iter=500, random_state=random_state, **params)
        elif model_type == "svm":
            kernel = params.get("kernel", "rbf")
            return SVC(kernel=kernel, random_state=random_state)
        elif model_type == "random_forest":
            return RandomForestClassifier(random_state=random_state, **params)
        elif model_type == "xgboost":
            return XGBClassifier(
                use_label_encoder=False, 
                eval_metric="logloss", 
                random_state=random_state,
                **params
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(random_state=random_state, **params)
        elif model_type == "decision_tree":
            return DecisionTreeClassifier(random_state=random_state, **params)
        elif model_type == "TabPFN":
            return TabPFNClassifier(**params)
        elif model_type == "TabNet":
            return TabNetClassifier(verbose=0, seed=random_state, device_name="cuda", **params)

        raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_imputer(strategy: str = "median", **params) -> SimpleImputer:
        """Create imputer component"""
        return SimpleImputer(strategy=strategy, **params)
    
    @staticmethod
    def create_encoder(**params) -> OneHotEncoder:
        """Create encoder component"""
        return OneHotEncoder(handle_unknown="ignore", **params)

class PipelineBuilder:
    """Builds scikit-learn pipelines from component configurations"""
    
    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
    
    def build_pipeline(self, components: Dict[str, ComponentConfig]) -> Pipeline:
        """Build a complete pipeline from component configurations"""
        
        # Determine preprocessing strategy based on model requirements
        model_config = components.get("model")
        needs_imputation = self._needs_imputation(model_config)
        
        # Build numeric transformer
        numeric_steps = []
        if needs_imputation:
            numeric_steps.append(("imputer", ComponentFactory.create_imputer("median")))
        
        if "scaler" in components:
            scaler = components["scaler"].factory_func(**components["scaler"].params)
            numeric_steps.append(("scaler", scaler))
        
        # Build categorical transformer
        categorical_steps = []
        if needs_imputation:
            categorical_steps.append(("imputer", ComponentFactory.create_imputer("most_frequent")))
        categorical_steps.append(("onehot", ComponentFactory.create_encoder()))
        
        # Create column transformer
        preprocessor = ColumnTransformer(transformers=[
            ("num", Pipeline(numeric_steps), self.numeric_features),
            ("cat", Pipeline(categorical_steps), self.categorical_features)
        ])
        
        # Build main pipeline
        pipeline_steps = [("preprocessor", preprocessor)]
        
        # Add PCA if specified
        if "pca" in components:
            # Check for PCA conflicts (e.g., XGBoost self-handling missing values)
            if self._has_pca_conflict(components):
                print(f"[WARN] Skipping PCA due to conflict with {model_config.name}")
            else:
                pca = components["pca"].factory_func(**components["pca"].params)
                pipeline_steps.append(("pca", pca))
        
        # Add model
        if "model" in components:
            model = components["model"].factory_func(**components["model"].params)
            pipeline_steps.append(("model", model))
        
        return Pipeline(pipeline_steps)
    
    def _needs_imputation(self, model_config: ComponentConfig) -> bool:
        """Determine if the model needs explicit imputation"""
        if model_config is None:
            return True
        
        # XGBoost can handle missing values natively if configured to do so
        if model_config.name == "xgboost":
            return not model_config.params.get("self_handle_missing", False)
        
        return True
    
    def _has_pca_conflict(self, components: Dict[str, ComponentConfig]) -> bool:
        """Check if PCA conflicts with other components"""
        model_config = components.get("model")
        if model_config and model_config.name == "xgboost":
            return model_config.params.get("self_handle_missing", False)
        return False


class RobustConfigurator:
    """Main configurator class that uses component-based approach"""
    
    def __init__(self, X_train, X_test, y_train, y_test, 
                 numeric_features: List[str], categorical_features: List[str],
                 experiment_name: str = "ml_experiment"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.experiment_name = experiment_name
        
        self.component_registry = {}
        self.experiments = []
        self.results = {}
        self.trained_models = {}  # Store trained pipelines
        self.pipeline_builder = PipelineBuilder(numeric_features, categorical_features)
        
        # Create directories for saving
        self.base_dir = f"/content/{experiment_name}"
        self.models_dir = os.path.join(self.base_dir, "models")
        self.results_dir = os.path.join(self.base_dir, "results")
        self.config_dir = os.path.join(self.base_dir, "configs")
        
        # Register default components
        self._register_default_components()
    
    def _register_default_components(self):
        """Register default ML components"""
        
        # Scalers
        self.register_component("standard_scaler", ComponentConfig(
            name="standard_scaler",
            factory_func=lambda: ComponentFactory.create_scaler("standard"),
            description="Standard Scaler (mean=0, std=1)"
        ))
        
        self.register_component("minmax_01", ComponentConfig(
            name="minmax_01",
            factory_func=lambda: ComponentFactory.create_scaler("minmax", feature_range=(0, 1)),
            description="MinMax Scaler (0-1)"
        ))
        
        self.register_component("minmax_neg11", ComponentConfig(
            name="minmax_neg11", 
            factory_func=lambda: ComponentFactory.create_scaler("minmax", feature_range=(-1, 1)),
            description="MinMax Scaler (-1 to 1)"
        ))
        
        self.register_component("minmax_02", ComponentConfig(
            name="minmax_02",
            factory_func=lambda: ComponentFactory.create_scaler("minmax", feature_range=(0, 2)),
            description="MinMax Scaler (0-2)"
        ))
        
        self.register_component("robust_scaler", ComponentConfig(
            name="robust_scaler",
            factory_func=lambda: ComponentFactory.create_scaler("robust"),
            description="Robust Scaler (median and IQR)"
        ))
        
        # PCA components
        for variance in [0.85, 0.90, 0.95, 0.98]:
            self.register_component(f"pca_{variance}", ComponentConfig(
                name=f"pca_{variance}",
                factory_func=lambda v=variance: ComponentFactory.create_pca(v),
                description=f"PCA retaining {variance*100}% variance"
            ))
        
        # Models
        self.register_component("logistic", ComponentConfig(
            name="logistic",
            factory_func=lambda: ComponentFactory.create_model("logistic"),
            description="Logistic Regression"
        ))
        
        self.register_component("random_forest", ComponentConfig(
            name="random_forest",
            factory_func=lambda: ComponentFactory.create_model("random_forest"),
            description="Random Forest Classifier"
        ))
        
        self.register_component("gradient_boosting", ComponentConfig(
            name="gradient_boosting",
            factory_func=lambda: ComponentFactory.create_model("gradient_boosting"),
            description="Gradient Boosting Classifier"
        ))
        
        self.register_component("decision_tree", ComponentConfig(
            name="decision_tree", 
            factory_func=lambda: ComponentFactory.create_model("decision_tree"),
            description="Decision Tree Classifier"
        ))
        
        # SVM variants
        for kernel in ["linear", "rbf", "poly"]:
            self.register_component(f"svm_{kernel}", ComponentConfig(
                name="svm",
                factory_func=lambda k=kernel: ComponentFactory.create_model("svm", kernel=k),
                description=f"SVM with {kernel} kernel"
            ))
        
        # XGBoost variants
        self.register_component("xgboost_impute", ComponentConfig(
            name="xgboost",
            factory_func=lambda: ComponentFactory.create_model("xgboost", self_handle_missing=False),
            description="XGBoost with imputation",
            conflicts=["pca_*"]  # PCA conflicts when self-handling missing values
        ))
        
        self.register_component("xgboost_native", ComponentConfig(
            name="xgboost", 
            factory_func=lambda: ComponentFactory.create_model("xgboost", self_handle_missing=True),
            description="XGBoost handling missing values natively",
            conflicts=["pca_*"]  # PCA conflicts when self-handling missing values
        ))

        # Deep model
        self.register_component("TabPFN", ComponentConfig(
            name="TabPFN", 
            factory_func=lambda: ComponentFactory.create_model("TabPFN"),
            description="TabPFN Deep Learning"
        ))

        self.register_component("TabNet", ComponentConfig(
            name="TabNet", 
            factory_func=lambda: ComponentFactory.create_model("TabNet"),
            description="TabNet Deep Learning"
        ))
    
    def register_component(self, key: str, config: ComponentConfig):
        """Register a new component configuration"""
        self.component_registry[key] = config
    
    def get_components_by_type(self, component_type: str) -> Dict[str, ComponentConfig]:
        """Get all components of a specific type"""
        if component_type == "scaler":
            return {k: v for k, v in self.component_registry.items() 
                   if "scaler" in k}
        elif component_type == "pca":
            return {k: v for k, v in self.component_registry.items() 
                   if k.startswith("pca_")}
        elif component_type == "model":
            return {k: v for k, v in self.component_registry.items() 
                   if k in ["logistic", "random_forest"] or k.startswith(("svm_", "xgboost"))}
        else:
            return {}
    
    def generate_experiments(self, selected_components: Dict[str, List[str]] = None):
        """Generate all possible experiment combinations"""
        if selected_components is None:
            # Use all components by default
            selected_components = {
                "scaler": list(self.get_components_by_type("scaler").keys()),
                "pca": ["none"] + list(self.get_components_by_type("pca").keys()),
                "model": list(self.get_components_by_type("model").keys())
            }
        
        self.experiments.clear()
        
        # Generate all combinations
        scaler_options = selected_components.get("scaler", [])
        pca_options = selected_components.get("pca", [])  
        model_options = selected_components.get("model", [])
        
        for scaler_key, pca_key, model_key in product(scaler_options, pca_options, model_options):
            # Check for conflicts
            components = {}
            
            # Add scaler
            if scaler_key in self.component_registry:
                components["scaler"] = self.component_registry[scaler_key]
            
            # Add PCA (if not "none")
            if pca_key != "none" and pca_key in self.component_registry:
                components["pca"] = self.component_registry[pca_key]
            
            # Add model
            if model_key in self.component_registry:
                components["model"] = self.component_registry[model_key]
            
            # Check conflicts
            if self._has_conflicts(components):
                continue
            
            # Generate experiment name
            exp_name = self._generate_experiment_name(scaler_key, pca_key, model_key)
            
            self.experiments.append({
                "name": exp_name,
                "components": components
            })
        
        print(f"[INFO] Generated {len(self.experiments)} experiments")
        for exp in self.experiments:
            print(f" - {exp['name']}")
    
    def _has_conflicts(self, components: Dict[str, ComponentConfig]) -> bool:
        """Check if components have conflicts"""
        # Example: XGBoost self-handling missing values conflicts with PCA
        model_config = components.get("model")
        if model_config and "pca" in components:
            if model_config.name == "xgboost" and model_config.params.get("self_handle_missing", False):
                return True
        return False
    
    def _generate_experiment_name(self, scaler_key: str, pca_key: str, model_key: str) -> str:
        """Generate descriptive experiment name"""
        parts = [scaler_key]
        
        if pca_key != "none":
            parts.append(pca_key)
        else:
            parts.append("no_pca")
        
        parts.append(model_key)
        
        return "_".join(parts)
    
    def run_experiments(self, random_state: int = 42, save_models: bool = True):
        """Run all generated experiments"""
        self.results.clear()
        self.trained_models.clear()
        
        # Create directories if saving
        if save_models:
            self._create_directories()
        
        print(f"[INFO] Running {len(self.experiments)} experiments...")
        
        for exp in tqdm(self.experiments, desc="[INFO] Training"):
            # print(f"\n[INFO] Training {exp['name']}")
            try:
                # Build pipeline
                pipeline = self.pipeline_builder.build_pipeline(exp["components"])
                
                # Train and evaluate
                pipeline.fit(self.X_train, self.y_train)
                y_pred = pipeline.predict(self.X_test)
                
                # Calculate metrics
                metrics = {
                    "accuracy": accuracy_score(self.y_test, y_pred),
                    "weighted_precision": precision_score(self.y_test, y_pred, average="weighted"),
                    "weighted_recall": recall_score(self.y_test, y_pred, average="weighted"),
                    "weighted_f1": f1_score(self.y_test, y_pred, average="weighted"),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.results[exp["name"]] = metrics
                self.trained_models[exp["name"]] = pipeline
                
                # Save model if requested
                if save_models:
                    self._save_model(exp["name"], pipeline, metrics)
                
            except Exception as e:
                print(f"   -> ERROR: {str(e)}")
                self.results[exp["name"]] = {"error": str(e), "timestamp": datetime.now().isoformat()}
        
        # Save results summary
        if save_models:
            self._save_results()
            self._save_experiment_config()
            print(f"\n[INFO] All results saved to: {self.base_dir}")
        
        print("\n[INFO] DONE")
        return self.results
    
    def _create_directories(self):
        """Create necessary directories for saving"""
        for directory in [self.models_dir, self.results_dir, self.config_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _save_model(self, model_name: str, pipeline: Pipeline, metrics: dict):
        """Save individual model and its metadata"""
        try:
            # Save the model using joblib (more efficient for sklearn models)
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            joblib.dump(pipeline, model_path)
            
            # Save model metadata
            metadata = {
                "model_name": model_name,
                "metrics": metrics,
                "model_path": model_path,
                "components": {k: str(v) for k, v in self.experiments[0]["components"].items()},  # Convert to serializable
                "features": {
                    "numeric": self.numeric_features,
                    "categorical": self.categorical_features
                }
            }
            
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"   [WARN] Could not save model {model_name}: {str(e)}")
    
    def _save_results(self):
        """Save results summary"""
        # Save as JSON
        results_path = os.path.join(self.results_dir, "results_summary.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV for easy analysis
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        csv_path = os.path.join(self.results_dir, "results_summary.csv")
        results_df.to_csv(csv_path)
        
        print(f"[INFO] Results saved to {results_path} and {csv_path}")
    
    def _save_experiment_config(self):
        """Save the experiment configuration"""
        config = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.experiments),
            "features": {
                "numeric": self.numeric_features,
                "categorical": self.categorical_features
            },
            "experiments": [
                {
                    "name": exp["name"],
                    "components": {k: str(v) for k, v in exp["components"].items()}
                } for exp in self.experiments
            ]
        }
        
        config_path = os.path.join(self.config_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[INFO] Configuration saved to {config_path}")
    
    def load_model(self, model_name: str, base_dir: str = None):
        """Load a saved model"""
        if base_dir is None:
            base_dir = self.base_dir
        
        model_path = os.path.join(base_dir, "models", f"{model_name}.joblib")
        metadata_path = os.path.join(base_dir, "models", f"{model_name}_metadata.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        pipeline = joblib.load(model_path)
        
        # Load metadata if available
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return pipeline, metadata
    
    def load_results(self, base_dir: str = None):
        """Load saved results"""
        if base_dir is None:
            base_dir = self.base_dir
        
        results_path = os.path.join(base_dir, "results", "results_summary.json")
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def zip_experiment_results(self):
        """Create a ZIP file of all experiment results for download"""
        import zipfile
        
        zip_path = f"{self.base_dir}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(self.base_dir))
                    zipf.write(file_path, arcname)
        
        print(f"[INFO] Experiment results zipped to: {zip_path}")
        return zip_path
    
    def get_best_results(self, metric: str = "accuracy", top_k: int = 5):
        """Get top k results sorted by metric"""
        valid_results = {name: results for name, results in self.results.items() 
                        if "error" not in results and metric in results}
        
        sorted_results = sorted(valid_results.items(), 
                              key=lambda x: x[1][metric], 
                              reverse=True)
        
        return sorted_results[:top_k]