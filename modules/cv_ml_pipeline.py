import csv
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import xgboost as xgb
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm.autonotebook import tqdm


class ImageDataset(Dataset):
    """
    Dataset for pipeline
    """

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.categories = []

        for label in os.listdir(dataset_path):
            self.categories.append(label)
            label_path = os.path.join(dataset_path, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    self.image_paths.append(os.path.join(label_path, image_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.categories.index(label)
        return image, label


@dataclass
class ComponentConfig:
    """
    Config corresponding to the Component in pipeline
    """

    name: str
    params: Dict[str, Any] = field(default_factory=Dict)


class ComponentFactory:
    """
    Class where contains function generating components
    """

    @staticmethod
    def create_scaler(name, params):
        if name == "standard":
            return StandardScaler()
        elif name == "minmax":
            feature_range = params.get("feature_range", (0, 1))
            return MinMaxScaler(feature_range=feature_range)
        elif name == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {name}")

    @staticmethod
    def create_extractor(name, params):
        if name == "HOG":

            def hog_extractor(loader):
                resize_size = params.get("resize_size", 224)
                winSize = (resize_size, resize_size)
                blockSize = (16, 16)
                blockStride = (8, 8)
                cellSize = (8, 8)
                nbins = 9
                hog = cv2.HOGDescriptor(
                    winSize, blockSize, blockStride, cellSize, nbins
                )
                features = []
                labels = []
                progress_bar = tqdm(loader, colour="cyan", leave=True)
                for idx, (images, lbls) in enumerate(progress_bar):
                    labels.extend(lbls.numpy())
                    images = images.permute(0, 2, 3, 1).numpy()
                    for img in images:
                        img_gray = cv2.cvtColor(
                            (img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
                        )
                        feature = hog.compute(img_gray)
                        features.append(feature.flatten())
                    progress_bar.set_description(
                        f"> Extracting HOG {idx/len(progress_bar):.4f}"
                    )
                return np.array(features), np.array(labels)

            return hog_extractor

        elif name == "SIFT":

            def sift_extractor(loader, kmeans=None):
                nfeatures = params.get("nfeatures", 100)
                sift = cv2.SIFT_create(nfeatures=nfeatures)
                features = []
                labels = []
                kmeans_clusters = params.get("kmeans_clusters", 100)
                if kmeans is None:
                    descriptors_all = []
                    progress_bar = tqdm(loader, colour="cyan", leave=True)
                    for idx, (images, lbls) in enumerate(progress_bar):
                        images = images.permute(0, 2, 3, 1).numpy()
                        for img in images:
                            img_gray = cv2.cvtColor(
                                (img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
                            )
                            _, desc = sift.detectAndCompute(img_gray, None)
                            if desc is not None:
                                descriptors_all.append(desc)
                        progress_bar.set_description(
                            f"> Collecting SIFT Descriptors {idx/len(progress_bar):.4f}"
                        )
                    descriptors_all = np.vstack(descriptors_all)
                    kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
                    kmeans.fit(descriptors_all)
                progress_bar = tqdm(loader, colour="cyan", leave=True)
                for idx, (images, lbls) in enumerate(progress_bar):
                    labels.extend(lbls.cpu().numpy())
                    images = images.permute(0, 2, 3, 1).numpy()
                    for img in images:
                        img_gray = cv2.cvtColor(
                            (img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
                        )
                        _, desc = sift.detectAndCompute(img_gray, None)
                        if desc is not None:
                            labels_kmeans = kmeans.predict(desc)
                            hist = np.histogram(
                                labels_kmeans,
                                bins=kmeans_clusters,
                                range=(0, kmeans_clusters),
                            )[0]
                            hist = hist / (np.sum(hist) + 1e-6)
                        else:
                            hist = np.zeros(kmeans_clusters)
                        features.append(hist)
                    progress_bar.set_description(
                        f"> BoVW histogram: {idx/len(progress_bar):.4f}"
                    )

                return np.array(features), np.array(labels), kmeans

            return sift_extractor

        elif name == "resnet":

            def resnet_extractor(loader):
                model = models.resnet50(pretrained=True)
                model = nn.Sequential(*list(model.children())[:-1])
                model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                features = []
                labels = []
                progress_bar = tqdm(loader, colour="cyan", leave=True)
                with torch.no_grad():
                    for idx, (images, lbls) in enumerate(progress_bar):
                        labels.extend(lbls.numpy())
                        images = images.to(device)
                        feat = (
                            model(images).cpu().numpy().squeeze(-1).squeeze(-1)
                        )  # Shape: (batch, 2048)
                        features.append(feat)
                        progress_bar.set_description(
                            f"> Extracting ResNet {idx/len(progress_bar):.4f}"
                        )
                return np.vstack(features), np.array(labels)

            return resnet_extractor

        elif name == "efficientnet":

            def efficientnet_extractor(loader):
                model = models.efficientnet_b0(pretrained=True)
                model = nn.Sequential(*list(model.children())[:-1])
                model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                features = []
                labels = []
                progress_bar = tqdm(loader, colour="cyan", leave=True)
                with torch.no_grad():
                    for idx, (images, lbls) in enumerate(progress_bar):
                        labels.extend(lbls.numpy())
                        images = images.to(device)
                        feat = model(images).cpu().numpy().squeeze(-1).squeeze(-1)
                        features.append(feat)
                        progress_bar.set_description(
                            f"> Extracting EfficientNet {idx/len(progress_bar):.4f}"
                        )
                return np.vstack(features), np.array(labels)

            return efficientnet_extractor

        elif name == "vgg":

            def vgg_extractor(loader):
                model = models.vgg16(pretrained=True)
                model.classifier = nn.Sequential(
                    *list(model.classifier.children())[:-1]
                )
                model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                features = []
                labels = []
                progress_bar = tqdm(loader, colour="cyan", leave=True)
                with torch.no_grad():
                    for idx, (images, lbls) in enumerate(progress_bar):
                        labels.extend(lbls.numpy())
                        images = images.to(device)
                        feat = model(images).cpu().numpy()
                        features.append(feat)
                        progress_bar.set_description(
                            f"> Extracting VGG {idx/len(progress_bar):.4f}"
                        )
                return np.vstack(features), np.array(labels)

            return vgg_extractor

        elif name == "ViT":

            def vit_extractor(loader):
                model = models.vit_b_16(pretrained=True)
                model.heads = nn.Identity()
                model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                features = []
                labels = []
                progress_bar = tqdm(loader, colour="cyan", leave=True)
                with torch.no_grad():
                    for idx, (images, lbls) in enumerate(progress_bar):
                        labels.extend(lbls.numpy())
                        images = images.to(device)
                        feat = model(images).cpu().numpy()
                        features.append(feat)
                        progress_bar.set_description(
                            f"> Extracting ViT {idx/len(progress_bar):.4f}"
                        )
                return np.vstack(features), np.array(labels)

            return vit_extractor

        else:
            raise ValueError(f"Unknown extractor: {name}")

    @staticmethod
    def create_PCA(name, params):
        n_components = params.get("n_components", 200)
        return PCA(n_components=n_components)

    @staticmethod
    def create_model(name, params):
        if name == "logistic":
            return LogisticRegression(**params)
        elif name == "svm":
            return SVC(**params)
        elif name == "random":
            return RandomForestClassifier(**params)
        elif name == "xgboost":
            return xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {name}")


class TranditionalPipelineRunner:
    """
    Tranditional Pipeline for Image Task

    Main method
    ----------
    run_experiment():
        Run pipeline
    """

    def __init__(self, configs: List[ComponentConfig]):
        self.pipeline = configs
        self.extractor = ComponentFactory.create_extractor(
            configs[0].name, configs[0].params
        )
        self.scaler = ComponentFactory.create_scaler(configs[1].name, configs[1].params)
        self.pca = ComponentFactory.create_PCA(configs[2].name, configs[2].params)
        self.model = ComponentFactory.create_model(configs[3].name, configs[3].params)

    def run_experiment(
        self,
        train_loader,
        test_loader,
        use_feature_file=True,  # Extract Again. Let's set False :>
        feature_dir="features/image_features",
    ):
        feature_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", feature_dir)
        )
        print(
            f" >>>>> Start the Experiment with the following Pipeline <<<<<\n {self.pipeline}",
            flush=True,
        )

        feature_name = self.pipeline[0].name.lower()
        train_feature_file = os.path.join(
            feature_dir, f"features_train_{feature_name}.npy"
        )
        test_feature_file = os.path.join(
            feature_dir, f"features_test_{feature_name}.npy"
        )
        label_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../labels")
        )

        train_label_file = os.path.join(label_dir, f"labels_train_{feature_name}.npy")
        test_label_file = os.path.join(label_dir, f"labels_test_{feature_name}.npy")
        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        # ---------------------------------------------------------
        # - Step 1: extract feature/Load feature extracted        -
        # ---------------------------------------------------------
        if (
            use_feature_file
            and os.path.exists(train_feature_file)
            and os.path.exists(test_feature_file)
        ):
            print(" >>>>>> Step 1: Loading features extracted", flush=True)
            x_train = np.load(train_feature_file)
            x_test = np.load(test_feature_file)
            y_train = np.load(train_label_file)
            y_test = np.load(test_label_file)
        else:
            print(" >>>>>> Step 1: Extracting features ...", flush=True)
            if feature_name == "sift":
                x_train, y_train, kmeans = self.extractor(train_loader)
                x_test, y_test, _ = self.extractor(test_loader, kmeans=kmeans)
            else:
                x_train, y_train = self.extractor(train_loader)
                x_test, y_test = self.extractor(test_loader)

            np.save(train_feature_file, x_train)
            np.save(test_feature_file, x_test)
            np.save(train_label_file, y_train)
            np.save(test_label_file, y_test)
            print(f"Features saved to {feature_dir} and labels saved to {label_dir}")
        # ---------------------------------------------------------
        # - Step 2: Scaling Features                              -
        # ---------------------------------------------------------
        print(" >>>>>> Step 2: Scaling features ...", flush=True)
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        # ---------------------------------------------------------
        # - Step 3: PCA: Reduce Dimension                         -
        # ---------------------------------------------------------
        print(" >>>>>> Step 3: Reducing dimension ...", flush=True)
        self.pca.fit(x_train_scaled)
        x_train_reduced = self.pca.transform(x_train_scaled)
        x_test_reduced = self.pca.transform(x_test_scaled)
        print(f"Variance Retained: {sum(self.pca.explained_variance_ratio_):.4f}")
        # ---------------------------------------------------------
        # - Step 4: Train model                                   -
        # ---------------------------------------------------------
        print(" >>>>>> Step 4: Training model ...", flush=True)
        start_train = time.time()
        self.model.fit(x_train_reduced, y_train)
        train_time = time.time() - start_train
        # ---------------------------------------------------------
        # - Step 5: Eval model                                    -
        # ---------------------------------------------------------
        print(" >>>>>> Step 5: Evaluating model ...", flush=True)
        start_inference = time.time()
        y_pred = self.model.predict(x_test_reduced)
        inference_time = time.time() - start_inference

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")

        print(
            f"Accuracy: {acc:.4f}, F1-macro: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )
        print(f"Train time: {train_time:.4f}s - Inference time: {inference_time:.4f}s")
        print(" >>>>> Pipeline successfully! <<<<<")
        return train_time, inference_time, acc, f1, precision, recall


if __name__ == "__main__":
    RESIZE = (224, 224)  # Resize Image Input
    transform = Compose(
        [
            Resize(RESIZE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Path
    image_data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "image_data")
    )
    train_path = os.path.join(image_data_path, "seg_train", "seg_train")
    test_path = os.path.join(image_data_path, "seg_test", "seg_test")
    # Dataset and DataLoader
    train_dataset = ImageDataset(train_path, transform=transform)
    test_dataset = ImageDataset(test_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    configs_list = [
        [
            ComponentConfig("HOG", {"resize_size": 224}),
            ComponentConfig("standard", {}),
            ComponentConfig("PCA", {"n_components": 100}),
            ComponentConfig("logistic", {"max_iter": 1000}),
        ],
        [
            ComponentConfig("SIFT", {"kmeans_clusters": 100}),
            ComponentConfig("standard", {}),
            ComponentConfig("PCA", {"n_components": 50}),
            ComponentConfig("svm", {"kernel": "rbf"}),
        ],
        [
            ComponentConfig("resnet", {"resize_size": 224}),
            ComponentConfig("minmax", {"feature_range": (0, 1)}),
            ComponentConfig("PCA", {"n_components": 100}),
            ComponentConfig("random", {"n_estimators": 100}),
        ],
        [
            ComponentConfig("efficientnet", {"resize_size": 224}),
            ComponentConfig("robust", {}),
            ComponentConfig("PCA", {"n_components": 100}),
            ComponentConfig("xgboost", {"n_estimators": 100}),
        ],
        [
            ComponentConfig("vgg", {"resize_size": 224}),
            ComponentConfig("standard", {}),
            ComponentConfig("PCA", {"n_components": 100}),
            ComponentConfig("logistic", {}),
        ],
        [
            ComponentConfig("ViT", {"resize_size": 224}),
            ComponentConfig("standard", {}),
            ComponentConfig("PCA", {"n_components": 100}),
            ComponentConfig("svm", {"kernel": "rbf"}),
        ],
    ]

    results = {}
    for idx, configs in enumerate(configs_list):
        print(f"\n=== Running Pipeline {idx + 1} ===")
        runner = TranditionalPipelineRunner(configs)
        train_time, inference_time, acc, f1, precision, recall = runner.run_experiment(
            train_loader, test_loader, True
        )
        results[f"pipeline_{idx + 1}"] = {
            "pipeline": configs,
            "train_time": train_time,
            "inference_time": inference_time,
            "acc": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        # To export result csv file, let uncomment this
        # csv_file = f"experiment_results_pipeline_{idx + 1}.csv"
        # with open(csv_file, mode="w", newline="") as file:
        #     writer = csv.DictWriter(
        #         file, fieldnames=results[f"Config_{idx + 1}"].keys()
        #     )
        #     writer.writeheader()
        #     writer.writerow(results[f"Config_{idx + 1}"])

    print("\n=== Results ===")
    for key, value in results.items():
        print(
            f">>>>>{key}: Accuracy = {value['acc']:.4f}, F1-macro = {value['f1']:.4f}, Train Time = {value['train_time']:.4f}s, Inference Time = {value['inference_time']:.4f}s"
        )
        print(" -> ".join([str(e) for e in value["pipeline"]]))
