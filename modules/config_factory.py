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
            return TabPFNClassifier(device="cuda", **params)
        elif model_type == "TabNet":
            return TabNetClassifier(verbose=0, seed=random_state, device="cuda", **params)

        raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_imputer(strategy: str = "median", **params) -> SimpleImputer:
        """Create imputer component"""
        return SimpleImputer(strategy=strategy, **params)
    
    @staticmethod
    def create_encoder(**params) -> OneHotEncoder:
        """Create encoder component"""
        return OneHotEncoder(handle_unknown="ignore", **params)
