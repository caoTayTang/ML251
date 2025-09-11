import ipywidgets as widgets
from IPython.display import display, clear_output
from itertools import product

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb


class Configurator:
    def __init__(self, config_dict, X_train, X_test, y_train, y_test):
        self.config_dict = config_dict
        self.widget_groups = {}
        self.experiments = []
        self.results = {}
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    # -------------------------
    # Helpers
    # -------------------------
    def _make_float_input(self, value, width="80px"):
        return widgets.FloatText(value=value, layout=widgets.Layout(width=width))

    # -------------------------
    # Render UI
    # -------------------------
    def render(self):
        blocks = []
        for dim_name, spec in self.config_dict.items():
            if dim_name == "features":  # skip features, not interactive
                continue

            option_widgets = {}
            caption = widgets.Label(value=spec.get("caption", dim_name))
            children = [caption]

            for opt_name, opt_spec in spec["options"].items():
                opt_type = opt_spec.get("type", "checkbox")
                default = opt_spec.get("default", False)
                params = opt_spec.get("params", {})

                # main widget
                if opt_type == "checkbox":
                    parent_widget = widgets.Checkbox(value=default, description=opt_name)
                elif opt_type == "radio":
                    parent_widget = widgets.RadioButtons(options=params.get("choices", []),
                                                         value=default,
                                                         description=opt_name)
                else:
                    parent_widget = widgets.Label(value=f"[WARN] Unknown type {opt_type}")

                # suboptions
                sub_widgets = {}
                if "suboptions" in opt_spec:
                    sub_children = []
                    for sub_name, sub_spec in opt_spec["suboptions"].items():
                        stype = sub_spec.get("type", "float")
                        sdefault = sub_spec.get("default")
                        sparams = sub_spec.get("params", {})

                        if stype == "float":
                            sw = self._make_float_input(sdefault, width=sparams.get("width", "80px"))
                            row = widgets.HBox([widgets.Label(sub_name, layout=widgets.Layout(width="110px")), sw])
                        elif stype == "range":
                            a = self._make_float_input(sdefault[0], width="70px")
                            b = self._make_float_input(sdefault[1], width="70px")
                            row = widgets.HBox([
                                widgets.Label(sub_name, layout=widgets.Layout(width="110px")),
                                widgets.Label("min", layout=widgets.Layout(width="30px")), a,
                                widgets.Label("max", layout=widgets.Layout(width="30px")), b
                            ])
                            sw = {"min": a, "max": b}
                        elif stype == "radio":
                            sw = widgets.RadioButtons(options=sparams.get("choices", []),
                                                      value=sdefault,
                                                      layout=widgets.Layout(width="200px"))
                            row = widgets.HBox([widgets.Label(sub_name, layout=widgets.Layout(width="110px")), sw])
                        elif stype == "checkbox":
                            sw = widgets.Checkbox(value=sdefault, description=sub_name)
                            row = sw
                        else:
                            sw = widgets.Label(value=f"[WARN] Unknown sub-type {stype}")
                            row = widgets.HBox([widgets.Label(sub_name), sw])

                        # disable if parent unchecked
                        if isinstance(parent_widget, widgets.Checkbox):
                            if isinstance(sw, dict):
                                for child in sw.values():
                                    child.disabled = not parent_widget.value
                            else:
                                sw.disabled = not parent_widget.value

                        sub_widgets[sub_name] = sw
                        sub_children.append(row)

                    sub_box = widgets.VBox(sub_children, layout={'margin': '0 0 0 20px'})
                    combined = widgets.VBox([parent_widget, sub_box])

                    if isinstance(parent_widget, widgets.Checkbox):
                        def make_toggle(subs):
                            def _toggle(change):
                                enabled = change["new"]
                                for s in subs.values():
                                    if isinstance(s, dict):
                                        for child in s.values():
                                            child.disabled = not enabled
                                    else:
                                        s.disabled = not enabled
                            return _toggle
                        parent_widget.observe(make_toggle(sub_widgets), names="value")

                    box = combined
                else:
                    box = parent_widget

                children.append(box)
                option_widgets[opt_name] = {"widget": parent_widget, "type": opt_type, "subwidgets": sub_widgets}

            container = widgets.VBox(children, layout={'border': '1px solid #ddd', 'padding': '8px'})
            self.widget_groups[dim_name] = option_widgets
            blocks.append(container)

        # self.output_area = widgets.Output()

        # self.generate_btn.on_click(self._on_generate)
        # self.run_btn.on_click(self._on_run)

        ui = widgets.VBox([
            widgets.HBox(blocks),
            # self.output_area
        ])
        display(ui)

    # -------------------------
    # Generate experiments
    # -------------------------
    def generate(self):
        self.experiments.clear()
        selected = {}

        for dim, opts in self.widget_groups.items():
            selected[dim] = []
            for opt_name, meta in opts.items():
                w = meta["widget"]
                swidgets = meta["subwidgets"]

                if isinstance(w, widgets.Checkbox):
                    if not w.value:
                        continue

                    if not swidgets:  # no suboptions
                        selected[dim].append({"name": opt_name})
                    else:
                        for sname, sw in swidgets.items():
                            if isinstance(sw, widgets.Checkbox):
                                if sw.value:
                                    selected[dim].append({"name": opt_name, sname: True})
                            elif isinstance(sw, dict):  # range
                                selected[dim].append({
                                    "name": opt_name,
                                    f"{sname}_min": sw["min"].value,
                                    f"{sname}_max": sw["max"].value
                                })
                            else:
                                selected[dim].append({"name": opt_name, sname: sw.value})

                elif isinstance(w, widgets.RadioButtons):
                    selected[dim].append({"name": opt_name, "value": w.value})

        # Cartesian product
        dims = list(selected.keys())
        for combo in product(*selected.values()):
            exp = {}
            for d, opt in zip(dims, combo):
                exp[d] = opt

            # --- Build descriptive experiment name ---
            parts = []

            # Scaler
            scaler_cfg = exp.get("scaler", {})
            if scaler_cfg.get("name") == "minmax":
                parts.append(f"minmax[{scaler_cfg.get('feature_range_min',0)}-{scaler_cfg.get('feature_range_max',1)}]")
            elif scaler_cfg.get("name"):
                parts.append(scaler_cfg["name"])

            # PCA
            pca_cfg = exp.get("pca", {})
            pca_val = pca_cfg.get("name", "None")
            parts.append(f"pca={pca_val}")

            # Model
            model_cfg = exp.get("model", {})
            model_name = model_cfg.get("name", "Unknown")
            if model_name == "SVM":
                kernel = model_cfg.get("kernel", "rbf")
                parts.append(f"SVM(kernel={kernel})")
            elif model_name == "XGBoost":
                if model_cfg.get("Self-handle missing", False):
                    parts.append("XGBoost[self-handle]")
                elif model_cfg.get("Impute missing", False):
                    parts.append("XGBoost[impute]")
                else:
                    parts.append("XGBoost")
            else:
                parts.append(model_name)

            exp["experiment_name"] = "_".join(parts)
            self.experiments.append(exp)

        print(f"[INFO] Generated {len(self.experiments)} experiments")
        for e in self.experiments:
            print(" -", e["experiment_name"])


    # -------------------------
    # Run experiments
    # -------------------------
    def run(self):
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        numeric_features = self.config_dict["features"]["numeric"]
        categorical_features = self.config_dict["features"]["categorical"]

        self.results = {}

        
        print(f"[INFO] Running {len(self.experiments)} experiments...")

        for exp in self.experiments:
            model_cfg = exp.get("model", {})
            scaler_cfg = exp.get("scaler", {})
            pca_cfg = exp.get("pca", {})

            model_name = model_cfg.get("name")
            print(f"\n[INFO] Training {exp['experiment_name']}")

            # Scaler
            if scaler_cfg.get("name") == "minmax":
                scaler = MinMaxScaler(
                    feature_range=(scaler_cfg.get("feature_range_min", 0.0),
                                    scaler_cfg.get("feature_range_max", 1.0))
                )
            else:
                scaler = StandardScaler()

            if model_name == "XGBoost":
                if model_cfg.get("Self-handle missing", False):
                    numeric_transformer = Pipeline(steps=[("scaler", scaler)])
                    categorical_transformer = Pipeline(steps=[
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ])
                elif model_cfg.get("Impute missing", False):
                    numeric_transformer = Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", scaler)
                    ])
                    categorical_transformer = Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ])
            else:
                # Non-XGB models → always impute
                numeric_transformer = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", scaler)
                ])
                categorical_transformer = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ])

            # ✅ Always build ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ])

            # PCA
            skip_pca = False
            if pca_cfg.get("name") and pca_cfg["name"] not in ["None"]:
                if model_name == "XGBoost" and model_cfg.get("Self-handle missing", False):
                    print("   [WARN] Skipping PCA because XGBoost self-handles NaN and PCA cannot.")
                    skip_pca = True
                else:
                    n_components = float(pca_cfg["name"])
                    preprocessor = Pipeline([
                        ("coltrans", preprocessor),
                        ("pca", PCA(n_components=n_components))
                    ])
            if skip_pca:
                old_name = exp["experiment_name"]
                pca_val = pca_cfg.get("name")
                exp["experiment_name"] = old_name.replace(pca_val, "None")
                        
            # Model selection
            if model_name == "XGBoost":
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=SEED)
            elif model_name == "Random Forest":
                model = RandomForestClassifier(random_state=SEED)
            elif model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=500, random_state=SEED)
            elif model_name == "SVM":
                kernel = model_cfg.get("kernel", "rbf")
                model = SVC(kernel=kernel, random_state=SEED)
            else:
                print(f"[WARN] Unknown model {model_name}, skipping.")
                continue

            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "weighted precision": precision_score(y_test, y_pred, average="weighted"),
                "weighted recall": recall_score(y_test, y_pred, average="weighted"),
                "weighted f1": f1_score(y_test, y_pred, average="weighted"),
            }
            self.results[exp["experiment_name"]] = metrics
            print("   ->", metrics)

        print("\n[INFO] DONE")
        return self.results