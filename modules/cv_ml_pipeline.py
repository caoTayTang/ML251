import csv
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm.autonotebook import tqdm
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy("file_system")


class ImageDataset(Dataset):
    """
    Dataset for pipeline
    """

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.categories = [
            "buildings",
            "forest",
            "glacier",
            "mountain",
            "sea",
            "street",
        ]

        for label in self.categories:
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

    def __repr__(self):
        return "Pipeline(" + "->".join(self.pipeline[i].name for i in range(4)) + ")"

    def run_experiment(
        self,
        train_loader,
        test_loader,
        use_feature_file=True,  # Extract Again. Let's set False :>
        feature_path="./features/image_features",
        label_path="./labels",
    ):
        # feature_dir = os.path.abspath(
        #     os.path.join(os.path.dirname(__file__), "..", feature_dir)
        # )
        print(
            f" >>>>> Start the Experiment with the following Pipeline <<<<<\n {self.__repr__()}",
            flush=True,
        )

        feature_name = self.pipeline[0].name.lower()
        train_feature_file = os.path.join(
            feature_path, f"features_train_{feature_name}.npy"
        )
        test_feature_file = os.path.join(
            feature_path, f"features_test_{feature_name}.npy"
        )
        train_label_file = os.path.join(label_path, f"labels_train_{feature_name}.npy")
        test_label_file = os.path.join(label_path, f"labels_test_{feature_name}.npy")
        os.makedirs(feature_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

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
            print(f"Features saved to {feature_path} and labels saved to {label_dir}")
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
        print("*****Result*****")
        print(
            f"Accuracy: {acc:.4f}, F1-macro: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )
        print(f"Train time: {train_time:.4f}s - Inference time: {inference_time:.4f}s")
        print(" >>>>> Pipeline successfully! <<<<<")
        return train_time, inference_time, acc, f1, precision, recall


class DLPipelineRunner:
    def __init__(self, root, ckpt_path, config: dict):
        self.config = config
        self.TRAIN_PATH = os.path.join(
            root, "data", "image_data", "seg_train/seg_train"
        )
        self.TEST_PATH = os.path.join(root, "data", "image_data", "seg_test/seg_test")
        self.ckpt = ckpt_path

    def run_experiment(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        RESIZE_SIZE = self.config.get("resize_size", 224)
        BATCH_SIZE = self.config.get("batch_size", 32)

        transforms = Compose(
            [
                Resize((RESIZE_SIZE, RESIZE_SIZE)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # Dataset
        val_dataset = ImageDataset(self.TEST_PATH, transform=transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
        class_nums = len(val_dataset.categories)
        # Model
        MODEL_NAME = self.config.get("model_name", "")

        if MODEL_NAME == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, class_nums)

        elif MODEL_NAME == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, class_nums)
        elif MODEL_NAME == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            )
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, class_nums)
        elif MODEL_NAME == "vit_b_16":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, class_nums)
        else:
            raise ValueError(f"Unknown Model in this pipeline: {MODEL_NAME}")
        # self.load_checkpoint(model, optimizer=None, name_model=MODEL_NAME, is_best=True)
        self.load_model_state(model, MODEL_NAME)
        model.to(device)
        # Metric
        precision_metric = Precision(
            task="multiclass", average="weighted", num_classes=class_nums
        ).to(device)
        recall_metric = Recall(
            task="multiclass", average="weighted", num_classes=class_nums
        ).to(device)
        f1_score_metric = F1Score(
            task="multiclass", average="weighted", num_classes=class_nums
        ).to(device)
        accuracy_metric = Accuracy(task="multiclass", num_classes=class_nums).to(device)
        # Evaluation
        model.eval()
        progress_bar = tqdm(val_dataloader, colour="cyan", leave=True)
        start_time = time.time()
        with torch.inference_mode():
            for image, label in progress_bar:
                image = image.to(device)
                label = torch.tensor(label, dtype=torch.int64).to(device)
                output = model(image)
                pred = torch.argmax(output, dim=-1)
                accuracy_metric.update(pred, label)
                precision_metric.update(pred, label)
                recall_metric.update(pred, label)
                f1_score_metric.update(pred, label)
                progress_bar.set_description("Evaluating val dataset ...")
        inference_time = time.time() - start_time
        accuracy = accuracy_metric.compute().cpu().item()
        precision = precision_metric.compute().cpu().item()
        recall = recall_metric.compute().cpu().item()
        f1_score = f1_score_metric.compute().cpu().item()
        print("*****Result*****")
        print(
            f"Accuracy {accuracy:.4f} - Precision {precision:.4f} - Recall {recall:.4f} - F1_score {f1_score:.4f}"
        )
        print(f"Inference time: {inference_time:.4f}s")
        return inference_time, accuracy, f1_score, precision, recall

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = self.config.get("batch_size", 32)
        resize_size = self.config.get("resize_size", 224)
        lr = self.config.get("learning_rate", 1e-3)
        num_epochs = self.config.get("num_epochs", 100)
        transforms = Compose(
            [
                Resize((resize_size, resize_size)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        train_dataset = ImageDataset(self.TRAIN_PATH, transform=transforms)
        val_dataset = ImageDataset(self.TEST_PATH, transform=transforms)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
        class_nums = len(train_dataset.categories)
        MODEL_NAME = self.config.get("model_name", "")
        if MODEL_NAME == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, class_nums)

        elif MODEL_NAME == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, class_nums)
        elif MODEL_NAME == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            )
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, class_nums)
        elif MODEL_NAME == "vit_b_16":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, class_nums)
        else:
            raise ValueError(f"Unknown Model in this pipeline: {MODEL_NAME}")
        criterion = nn.CrossEntropyLoss()
        # Metric
        precision_metric = Precision(
            task="multiclass", average="weighted", num_classes=class_nums
        ).to(device)
        recall_metric = Recall(
            task="multiclass", average="weighted", num_classes=class_nums
        ).to(device)
        f1_score_metric = F1Score(
            task="multiclass", average="weighted", num_classes=class_nums
        ).to(device)
        accuracy_metric = Accuracy(task="multiclass", num_classes=class_nums).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if self.config.get("is_load_model", False):
            start_epoch, best_metric = self.load_checkpoint(
                model, optimizer, MODEL_NAME
            )
        else:
            start_epoch, best_metric = 0, 0
        model.to(device)
        writer = SummaryWriter(f"run/{MODEL_NAME}")
        params_table = "| Parameter | Value |\n|---|---|\n"
        for key, value in self.config.items():
            params_table += f"| {key} | {value} |\n"
        writer.add_text("Hyperparameters", params_table)
        global_step = start_epoch * len(train_dataloader)
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = []
            progress_bar = tqdm(train_dataloader, colour="cyan", leave=True)
            for i, (image, label) in enumerate(progress_bar):
                image = image.to(device)
                output = model(image)
                label = torch.tensor(label, dtype=torch.int64).to(device)
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                mean_loss = np.mean(train_loss)
                writer.add_scalar("Train/loss", mean_loss, global_step=global_step)
                progress_bar.set_description(
                    f"(Train) Epoch {epoch}/{num_epochs} - Mean Loss {mean_loss:.4f}"
                )
                global_step += 1
            model.eval()
            progress_bar = tqdm(val_dataloader, colour="cyan", leave=True)
            with torch.inference_mode():
                for image, label in progress_bar:
                    image = image.to(device)
                    label = torch.tensor(label, dtype=torch.int64).to(device)
                    output = model(image)
                    pred = torch.argmax(output, dim=-1)
                    accuracy_metric.update(pred, label)
                    precision_metric.update(pred, label)
                    recall_metric.update(pred, label)
                    f1_score_metric.update(pred, label)

                    # num_accuracy += torch.eq(label, pred).sum()
                    progress_bar.set_description(
                        f"(Evaluation) - Accuracy {accuracy_metric.compute().item():.4f} - Precision {precision_metric.compute().item(): .4f} - Recall {recall_metric.compute().item():.4f} - F1 Score {f1_score_metric.compute().item():.4f}"
                    )

            accuracy = accuracy_metric.compute()
            precision = precision_metric.compute()
            recall = recall_metric.compute()
            f1_score = f1_score_metric.compute()
            writer.add_scalar("Val/Accuracy", accuracy, global_step=epoch)
            writer.add_scalar("Val/Precision", precision, global_step=epoch)
            writer.add_scalar("Val/Recall", recall, global_step=epoch)
            writer.add_scalar("Val/F1 Score", f1_score, global_step=epoch)
            accuracy_metric.reset()
            precision_metric.reset()
            recall_metric.reset()
            f1_score_metric.reset()
            if best_metric < accuracy.item():
                best_metric = accuracy.item()
                self.save_checkpoint(
                    model, optimizer, epoch, best_metric, MODEL_NAME, best=True
                )

            self.save_checkpoint(model, optimizer, epoch, best_metric, MODEL_NAME)

    def save_checkpoint(
        self, model, optimizer, epoch, best_metric, name_model, best=False
    ):
        print("=> Save checkpoint ...")
        if best:
            extension = "best"
        else:
            extension = "last"
        file_name = f"{extension}_{name_model}.pt.tar"
        ckpt_path = os.path.join(self.ckpt, file_name)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        }
        torch.save(checkpoint, ckpt_path)

    def load_checkpoint(self, model, optimizer, name_model, is_best=False):
        print("=> Load checkpoint ...")
        if is_best:
            file_name = f"best_{name_model}.pt.tar"
        else:
            file_name = f"last_{name_model}.pt.tar"
        ckpt_path = os.path.join(self.ckpt, file_name)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint["epoch"], checkpoint["best_metric"]

    def load_model_state(self, model, name_model):
        file_name = "best_" + name_model + ".pt"
        print("=> Load model state ...")
        model_state_path = os.path.join(self.ckpt, file_name)
        model.load_state_dict(torch.load(model_state_path))


if __name__ == "__main__":
    # root = os.getcwd()
    # feature_path = os.path.join(root, "features", "image_features")
    # label_path = os.path.join(root, "labels")
    # RESIZE = (224, 224)  # Resize Image Input
    # transform = Compose(
    #     [
    #         Resize(RESIZE),
    #         ToTensor(),
    #         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    # # Path
    # image_data_path = os.path.abspath(
    #     os.path.join(os.path.dirname(__file__), "..", "data", "image_data")
    # )
    # train_path = os.path.join(image_data_path, "seg_train", "seg_train")
    # test_path = os.path.join(image_data_path, "seg_test", "seg_test")
    # # Dataset and DataLoader
    # train_dataset = ImageDataset(train_path, transform=transform)
    # test_dataset = ImageDataset(test_path, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # configs_list = [
    #     [
    #         ComponentConfig("HOG", {"resize_size": 224}),
    #         ComponentConfig("standard", {}),
    #         ComponentConfig("PCA", {"n_components": 100}),
    #         ComponentConfig("logistic", {"max_iter": 1000}),
    #     ],
    #     [
    #         ComponentConfig("SIFT", {"kmeans_clusters": 100}),
    #         ComponentConfig("standard", {}),
    #         ComponentConfig("PCA", {"n_components": 50}),
    #         ComponentConfig("svm", {"kernel": "rbf"}),
    #     ],
    #     [
    #         ComponentConfig("resnet", {"resize_size": 224}),
    #         ComponentConfig("minmax", {"feature_range": (0, 1)}),
    #         ComponentConfig("PCA", {"n_components": 100}),
    #         ComponentConfig("random", {"n_estimators": 100}),
    #     ],
    #     [
    #         ComponentConfig("efficientnet", {"resize_size": 224}),
    #         ComponentConfig("robust", {}),
    #         ComponentConfig("PCA", {"n_components": 100}),
    #         ComponentConfig("xgboost", {"n_estimators": 100}),
    #     ],
    #     [
    #         ComponentConfig("vgg", {"resize_size": 224}),
    #         ComponentConfig("standard", {}),
    #         ComponentConfig("PCA", {"n_components": 100}),
    #         ComponentConfig("logistic", {}),
    #     ],
    #     [
    #         ComponentConfig("ViT", {"resize_size": 224}),
    #         ComponentConfig("standard", {}),
    #         ComponentConfig("PCA", {"n_components": 100}),
    #         ComponentConfig("svm", {"kernel": "rbf"}),
    #     ],
    # ]

    # results = {}
    # for idx, configs in enumerate(configs_list):
    #     print("\n==========================")
    #     print(f"=== Running Pipeline {idx + 1} ===")
    #     print("==========================")
    #     runner = TranditionalPipelineRunner(configs)
    #     train_time, inference_time, acc, f1, precision, recall = runner.run_experiment(
    #         train_loader,
    #         test_loader,
    #         True,
    #         feature_path=feature_path,
    #         label_path=label_path,
    #     )
    #     results[f"pipeline_{idx + 1}"] = {
    #         "pipeline": configs,
    #         "train_time": train_time,
    #         "inference_time": inference_time,
    #         "acc": acc,
    #         "f1": f1,
    #         "precision": precision,
    #         "recall": recall,
    #     }
    #     # To export result csv file, let uncomment this
    #     # csv_file = f"experiment_results_pipeline_{idx + 1}.csv"
    #     # with open(csv_file, mode="w", newline="") as file:
    #     #     writer = csv.DictWriter(
    #     #         file, fieldnames=results[f"Config_{idx + 1}"].keys()
    #     #     )
    #     #     writer.writeheader()
    #     #     writer.writerow(results[f"Config_{idx + 1}"])

    # print("\n=== Results ===")
    # for key, value in results.items():
    #     print(
    #         f">>>>>{key}: Accuracy = {value['acc']:.4f}, F1-macro = {value['f1']:.4f}, Train Time = {value['train_time']:.4f}s, Inference Time = {value['inference_time']:.4f}s"
    #     )
    #     print(" -> ".join([str(e) for e in value["pipeline"]]))
    # checkpoint_path = "."
    # dl_configs = {
    #     "config 1": {"resize_size": 224, "batch_size": 32, "model_name": "resnet18"},
    #     "config 2": {
    #         "resize_size": 224,
    #         "batch_size": 32,
    #         "model_name": "efficientnet_b0",
    #     },
    #     "config 3": {
    #         "resize_size": 224,
    #         "batch_size": 32,
    #         "model_name": "mobilenet_v3_large",
    #     },
    #     "config 4": {"resize_size": 224, "batch_size": 32, "model_name": "vit_b_16"},
    # }
    # dl_results = []
    # print("\n----- Bắt đầu Mục 5.2: Thực thi pipeline Deep Learning -----")
    # for idx, (name, config) in enumerate(dl_configs.items()):
    #     print("============================================================")
    #     print(f"Running Deep Learning Pipeline {idx + 1}")
    #     print(config)
    #     print("============================================================")
    #     runner = DLPipelineRunner(root=".", ckpt_path=checkpoint_path, config=config)
    #     inference_time, acc, f1, precision, recall = runner.run_experiment()
    #     dl_results.append(
    #         {
    #             "pipeline": str(config),
    #             "inference_time": inference_time,
    #             "acc": acc,
    #             "f1": f1,
    #             "precision": precision,
    #             "recall": recall,
    #         }
    #     )
    # import pandas as pd

    # dl_df = pd.DataFrame(dl_results)
    # print("\n----- TẤT CẢ CÁC THỬ NGHIỆM HỌC SÂU ĐÃ HOÀN TẤT! -----")
    # print("Bảng kết quả tổng hợp:")
    # print(dl_df)

    # Train
    # config = {
    #     "batch_size": 32,
    #     "resize_size": 224,
    #     "learning_rate": 1e-3,
    #     "num_epochs": 100,
    #     "model_name": "vit_b_16",
    #     "is_load_model": False,
    # }
    # runner = DLPipelineRunner(".", config)
    # runner.train()

    dl_configs = {
        "config 1": {"resize_size": 224, "batch_size": 32, "model_name": "resnet18"},
        "config 2": {
            "resize_size": 224,
            "batch_size": 32,
            "model_name": "efficientnet_b0",
        },
        "config 3": {
            "resize_size": 224,
            "batch_size": 32,
            "model_name": "mobilenet_v3_large",
        },
        "config 4": {"resize_size": 224, "batch_size": 32, "model_name": "vit_b_16"},
    }
    checkpoint_path = "."
    dl_results = []
    print("\n----- Bắt đầu Mục 5.2: Thực thi pipeline Deep Learning -----")
    for idx, (name, config) in enumerate(dl_configs.items()):
        print("============================================================")
        print(f"Running Deep Learning Pipeline {idx + 1}")
        print(config)
        print("============================================================")
        runner = DLPipelineRunner(root=".", ckpt_path=checkpoint_path, config=config)
        inference_time, acc, f1, precision, recall = runner.run_experiment()
        dl_results.append(
            {
                "pipeline": str(config),
                "inference_time": inference_time,
                "acc": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )
    dl_df = pd.DataFrame(dl_results)
    print("\n----- TẤT CẢ CÁC THỬ NGHIỆM HỌC SÂU ĐÃ HOÀN TẤT! -----")
    print("Bảng kết quả tổng hợp:")
    # display(dl_df)
