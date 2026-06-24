"""2-Stage Neural Network (MLP + LSTM) untuk prediksi kontrak bridge terbaik.

Arsitektur:
  Stage 1 — MLP/LSTM prediksi suit (C/D/H/S/N)
  Stage 2 — MLP/LSTM prediksi kategori (partscore/game/small_slam/grand_slam)
  Kontrak final = kombinasi suit + kategori → level minimum yang valid

Model tersedia:
  - TwoStageMLP: Multi-Layer Perceptron (lebih cocok untuk tabular features)
  - TwoStageLSTM: LSTM (untuk sequence/temporal features)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, models
except ModuleNotFoundError:
    tf = None
    keras = None
    layers = None
    models = None


def _require_tensorflow() -> None:
    if keras is None or layers is None:
        raise ModuleNotFoundError(
            "TensorFlow is required for model training/loading. Install tensorflow to use TwoStageMLP/TwoStageLSTM."
        )


HAS_TF = keras is not None and layers is not None


class _SklearnTwoStageMLP:
    """Fallback 2-stage MLP berbasis scikit-learn saat TensorFlow tidak tersedia."""

    def __init__(self, input_dim: int, hidden_units: list = None, params: dict = None) -> None:
        self.input_dim = input_dim
        self.hidden_units = tuple(hidden_units or [256, 128, 64])
        self.params = {**NN_PARAMS, **(params or {})}
        self.model_suit = None
        self.model_category = None
        self.scaler = None
        self.encoder_suit = None
        self.encoder_category = None
        self.feature_names_ = None
        self.suit_classes_ = None
        self.category_classes_ = None
        self.backend = "sklearn-mlp"

    def _build_mlp(self) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_units,
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=self.params["batch_size"],
            learning_rate_init=0.001,
            max_iter=self.params["epochs"],
            shuffle=True,
            random_state=42,
            verbose=False,
            early_stopping=False,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y_suit: pd.Series,
        y_category: pd.Series,
    ) -> "_SklearnTwoStageMLP":
        self.feature_names_ = list(X.columns)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.encoder_suit = LabelEncoder()
        self.encoder_category = LabelEncoder()

        y_suit_encoded = self.encoder_suit.fit_transform(y_suit)
        y_category_encoded = self.encoder_category.fit_transform(y_category)

        self.suit_classes_ = self.encoder_suit.classes_
        self.category_classes_ = self.encoder_category.classes_

        print("Building Stage 1 (suit predictor) - sklearn MLP...")
        self.model_suit = self._build_mlp()
        print("Training Stage 1 (suit)...")
        self.model_suit.fit(X_scaled, y_suit_encoded)

        print("Building Stage 2 (category predictor) - sklearn MLP...")
        self.model_category = self._build_mlp()
        print("Training Stage 2 (category)...")
        self.model_category.fit(X_scaled, y_category_encoded)

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler.transform(X)
        y_suit_pred_idx = self.model_suit.predict(X_scaled)
        y_cat_pred_idx = self.model_category.predict(X_scaled)

        pred_suit = self.encoder_suit.inverse_transform(y_suit_pred_idx)
        pred_category = self.encoder_category.inverse_transform(y_cat_pred_idx)

        pred_level = _category_to_level(pred_suit, pred_category)
        pred_contract = [
            f"{lvl}{suit}" if suit != "P" else "PASS"
            for lvl, suit in zip(pred_level, pred_suit)
        ]

        return pd.DataFrame(
            {
                "pred_suit": pred_suit,
                "pred_category": pred_category,
                "pred_level": pred_level,
                "pred_contract": pred_contract,
            },
            index=X.index if hasattr(X, "index") else None,
        )

    def predict_suit(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        y_pred_idx = self.model_suit.predict(X_scaled)
        return self.encoder_suit.inverse_transform(y_pred_idx)

    def predict_category(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        y_pred_idx = self.model_category.predict(X_scaled)
        return self.encoder_category.inverse_transform(y_pred_idx)

    def feature_importance(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.feature_names_ is None:
            raise ValueError("Model belum dilatih.")

        def _importance(model: MLPClassifier) -> pd.Series:
            first_layer = np.abs(model.coefs_[0])
            scores = first_layer.mean(axis=1)
            return pd.Series(scores, index=self.feature_names_).sort_values(ascending=False)

        return _importance(self.model_suit).to_frame("importance"), _importance(self.model_category).to_frame("importance")

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

# Level minimum untuk mencapai game per suit
GAME_LEVEL = {"C": 5, "D": 5, "H": 4, "S": 4, "N": 3}

# Hyperparameter default
NN_PARAMS = {
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "verbose": 0,
}

MODEL_PATH      = Path("results/metrics/nn_model.h5")   # backward-compat
MLP_MODEL_PATH  = Path("results/metrics/mlp/model.keras")
LSTM_MODEL_PATH = Path("results/metrics/lstm/model.keras")


# ---------------------------------------------------------------------------
# Model Architecture — MLP
# ---------------------------------------------------------------------------

class TwoStageMLP:
    """2-Stage MLP untuk prediksi kontrak bridge.
    
    Stage 1: MLP → prediksi suit (C/D/H/S/N)
    Stage 2: MLP → prediksi kategori (partscore/game/small_slam/grand_slam)
    
    Cocok untuk tabular features.
    """

    def __init__(self, input_dim: int, hidden_units: list = None, params: dict = None) -> None:
        self.input_dim = input_dim
        self.hidden_units = hidden_units or [256, 128, 64]
        self.params = {**NN_PARAMS, **(params or {})}
        
        self.model_suit = None
        self.model_category = None
        self.scaler = None
        self.encoder_suit = None
        self.encoder_category = None
        self.feature_names_ = None
        self.suit_classes_ = None
        self.category_classes_ = None

    def _build_mlp(self, output_classes: int, model_name: str = "mlp") -> keras.Model:
        """Build MLP architecture."""
        model = keras.Sequential(name=model_name)
        model.add(layers.Input(shape=(self.input_dim,)))
        
        # Hidden layers
        for units in self.hidden_units:
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.Dropout(0.3))
        
        # Output layer
        if output_classes == 2:
            model.add(layers.Dense(output_classes, activation="sigmoid"))
            loss_fn = "binary_crossentropy"
        else:
            model.add(layers.Dense(output_classes, activation="softmax"))
            loss_fn = "categorical_crossentropy"
        
        model.compile(
            optimizer="adam",
            loss=loss_fn,
            metrics=["accuracy"]
        )
        return model

    def fit(
        self,
        X: pd.DataFrame,
        y_suit: pd.Series,
        y_category: pd.Series,
    ) -> "TwoStageMLP":
        """Latih kedua stage secara independen."""
        # Store feature names
        self.feature_names_ = list(X.columns)
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        self.encoder_suit = LabelEncoder()
        self.encoder_category = LabelEncoder()
        
        y_suit_encoded = self.encoder_suit.fit_transform(y_suit)
        y_category_encoded = self.encoder_category.fit_transform(y_category)
        
        self.suit_classes_ = self.encoder_suit.classes_
        self.category_classes_ = self.encoder_category.classes_
        
        # Convert to one-hot encoding
        n_suit_classes = len(self.suit_classes_)
        n_cat_classes = len(self.category_classes_)
        
        y_suit_onehot = keras.utils.to_categorical(y_suit_encoded, n_suit_classes)
        y_cat_onehot = keras.utils.to_categorical(y_category_encoded, n_cat_classes)
        
        # Build and train models
        print("Building Stage 1 (suit predictor) - MLP...")
        self.model_suit = self._build_mlp(n_suit_classes, "suit_mlp")
        
        print("Training Stage 1 (suit)...")
        self.model_suit.fit(
            X_scaled, y_suit_onehot,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=self.params["validation_split"],
            verbose=self.params["verbose"]
        )
        
        print("Building Stage 2 (category predictor) - MLP...")
        self.model_category = self._build_mlp(n_cat_classes, "category_mlp")
        
        print("Training Stage 2 (category)...")
        self.model_category.fit(
            X_scaled, y_cat_onehot,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=self.params["validation_split"],
            verbose=self.params["verbose"]
        )
        
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prediksi suit, kategori, dan level kontrak final."""
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        y_suit_probs = self.model_suit.predict(X_scaled, verbose=0)
        y_cat_probs = self.model_category.predict(X_scaled, verbose=0)
        
        # Get class predictions
        y_suit_pred_idx = np.argmax(y_suit_probs, axis=1)
        y_cat_pred_idx = np.argmax(y_cat_probs, axis=1)
        
        pred_suit = self.encoder_suit.inverse_transform(y_suit_pred_idx)
        pred_category = self.encoder_category.inverse_transform(y_cat_pred_idx)
        
        pred_level = _category_to_level(pred_suit, pred_category)
        pred_contract = [
            f"{lvl}{suit}" if suit != "P" else "PASS"
            for lvl, suit in zip(pred_level, pred_suit)
        ]
        
        return pd.DataFrame({
            "pred_suit":     pred_suit,
            "pred_category": pred_category,
            "pred_level":    pred_level,
            "pred_contract": pred_contract,
        }, index=X.index if hasattr(X, "index") else None)

    def predict_suit(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        y_pred_idx = np.argmax(self.model_suit.predict(X_scaled, verbose=0), axis=1)
        return self.encoder_suit.inverse_transform(y_pred_idx)

    def predict_category(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        y_pred_idx = np.argmax(self.model_category.predict(X_scaled, verbose=0), axis=1)
        return self.encoder_category.inverse_transform(y_pred_idx)

    def feature_importance(self, feature_names=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Hitung feature importance dari bobot input layer pertama MLP.

        Menggunakan mean absolute value dari bobot layer pertama sebagai
        proxy importansi fitur — pendekatan standar untuk feed-forward NN.

        Args:
            feature_names: Opsional. Daftar nama fitur. Digunakan sebagai
                fallback jika model dimuat dari disk sebelum feature_names
                disimpan (gunakan list(X_test.columns)).

        Returns:
            (imp_suit, imp_cat): DataFrame dengan kolom 'importance',
            diurutkan dari yang terpenting.
        """
        names = self.feature_names_ or (list(feature_names) if feature_names is not None else None)
        if names is None or self.model_suit is None:
            raise ValueError(
                "feature_names_ tidak tersedia. Berikan argumen feature_names, "
                "contoh: model.feature_importance(feature_names=X_test.columns)"
            )
        # Cache for future calls
        if self.feature_names_ is None:
            self.feature_names_ = names

        def _layer_importance(keras_model, feat_names):
            for layer in keras_model.layers:
                weights = layer.get_weights()
                if len(weights) > 0:
                    W = weights[0]  # shape: (input_dim, units)
                    if W.shape[0] == len(feat_names):
                        scores = np.abs(W).mean(axis=1)
                        return pd.Series(scores, index=feat_names).sort_values(ascending=False)
            raise ValueError("Tidak ada layer dengan bobot input yang cocok.")

        imp_suit = _layer_importance(self.model_suit, names).to_frame("importance")
        imp_cat  = _layer_importance(self.model_category, names).to_frame("importance")
        return imp_suit, imp_cat


# ---------------------------------------------------------------------------
# Model Architecture — LSTM
# ---------------------------------------------------------------------------

class TwoStageLSTM:
    """2-Stage LSTM untuk prediksi kontrak bridge.
    
    Stage 1: LSTM → prediksi suit
    Stage 2: LSTM → prediksi kategori
    
    Features direshape sebagai sequence untuk LSTM.
    """

    def __init__(self, input_dim: int, lstm_units: list = None, params: dict = None) -> None:
        self.input_dim = input_dim
        self.lstm_units = lstm_units or [128, 64]
        self.params = {**NN_PARAMS, **(params or {})}
        self.seq_length = 10  # Reshape features menjadi sequence
        
        self.model_suit = None
        self.model_category = None
        self.scaler = None
        self.encoder_suit = None
        self.encoder_category = None
        self.feature_names_ = None
        self.suit_classes_ = None
        self.category_classes_ = None

    def _reshape_to_sequence(self, X: np.ndarray) -> np.ndarray:
        """Reshape tabular features menjadi sequence."""
        n_samples = X.shape[0]
        # Pad atau truncate features to seq_length
        n_features_per_step = max(1, self.input_dim // self.seq_length)
        reshaped = np.zeros((n_samples, self.seq_length, n_features_per_step))
        
        for i in range(min(self.seq_length * n_features_per_step, self.input_dim)):
            seq_idx = i // n_features_per_step
            feat_idx = i % n_features_per_step
            reshaped[:, seq_idx, feat_idx] = X[:, i]
        
        return reshaped

    def _build_lstm(self, output_classes: int, model_name: str = "lstm") -> keras.Model:
        """Build LSTM architecture."""
        n_features_per_step = max(1, self.input_dim // self.seq_length)
        
        model = keras.Sequential(name=model_name)
        model.add(layers.Input(shape=(self.seq_length, n_features_per_step)))
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            model.add(layers.LSTM(units, return_sequences=return_sequences))
            model.add(layers.Dropout(0.3))
        
        # Dense layers
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.2))
        
        # Output layer
        if output_classes == 2:
            model.add(layers.Dense(output_classes, activation="sigmoid"))
            loss_fn = "binary_crossentropy"
        else:
            model.add(layers.Dense(output_classes, activation="softmax"))
            loss_fn = "categorical_crossentropy"
        
        model.compile(
            optimizer="adam",
            loss=loss_fn,
            metrics=["accuracy"]
        )
        return model

    def fit(
        self,
        X: pd.DataFrame,
        y_suit: pd.Series,
        y_category: pd.Series,
    ) -> "TwoStageLSTM":
        """Latih kedua stage LSTM secara independen."""
        # Store feature names
        self.feature_names_ = list(X.columns)
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape to sequence
        X_seq = self._reshape_to_sequence(X_scaled)
        
        # Encode labels
        self.encoder_suit = LabelEncoder()
        self.encoder_category = LabelEncoder()
        
        y_suit_encoded = self.encoder_suit.fit_transform(y_suit)
        y_category_encoded = self.encoder_category.fit_transform(y_category)
        
        self.suit_classes_ = self.encoder_suit.classes_
        self.category_classes_ = self.encoder_category.classes_
        
        # Convert to one-hot encoding
        n_suit_classes = len(self.suit_classes_)
        n_cat_classes = len(self.category_classes_)
        
        y_suit_onehot = keras.utils.to_categorical(y_suit_encoded, n_suit_classes)
        y_cat_onehot = keras.utils.to_categorical(y_category_encoded, n_cat_classes)
        
        # Build and train models
        print("Building Stage 1 (suit predictor) - LSTM...")
        self.model_suit = self._build_lstm(n_suit_classes, "suit_lstm")
        
        print("Training Stage 1 (suit)...")
        self.model_suit.fit(
            X_seq, y_suit_onehot,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=self.params["validation_split"],
            verbose=self.params["verbose"]
        )
        
        print("Building Stage 2 (category predictor) - LSTM...")
        self.model_category = self._build_lstm(n_cat_classes, "category_lstm")
        
        print("Training Stage 2 (category)...")
        self.model_category.fit(
            X_seq, y_cat_onehot,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=self.params["validation_split"],
            verbose=self.params["verbose"]
        )
        
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prediksi suit, kategori, dan level kontrak final."""
        X_scaled = self.scaler.transform(X)
        X_seq = self._reshape_to_sequence(X_scaled)
        
        # Predict probabilities
        y_suit_probs = self.model_suit.predict(X_seq, verbose=0)
        y_cat_probs = self.model_category.predict(X_seq, verbose=0)
        
        # Get class predictions
        y_suit_pred_idx = np.argmax(y_suit_probs, axis=1)
        y_cat_pred_idx = np.argmax(y_cat_probs, axis=1)
        
        pred_suit = self.encoder_suit.inverse_transform(y_suit_pred_idx)
        pred_category = self.encoder_category.inverse_transform(y_cat_pred_idx)
        
        pred_level = _category_to_level(pred_suit, pred_category)
        pred_contract = [
            f"{lvl}{suit}" if suit != "P" else "PASS"
            for lvl, suit in zip(pred_level, pred_suit)
        ]
        
        return pd.DataFrame({
            "pred_suit":     pred_suit,
            "pred_category": pred_category,
            "pred_level":    pred_level,
            "pred_contract": pred_contract,
        }, index=X.index if hasattr(X, "index") else None)

    def predict_suit(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_seq = self._reshape_to_sequence(X_scaled)
        y_pred_idx = np.argmax(self.model_suit.predict(X_seq, verbose=0), axis=1)
        return self.encoder_suit.inverse_transform(y_pred_idx)

    def predict_category(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_seq = self._reshape_to_sequence(X_scaled)
        y_pred_idx = np.argmax(self.model_category.predict(X_seq, verbose=0), axis=1)
        return self.encoder_category.inverse_transform(y_pred_idx)

    def feature_importance(self, feature_names=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Hitung feature importance untuk LSTM.

        Karena LSTM menggunakan reshaped input, kita gunakan fallback:
        buat importance berdasarkan posisi fitur dalam sequence.

        Args:
            feature_names: Opsional. Daftar nama fitur. Digunakan sebagai
                fallback jika model dimuat dari disk sebelum feature_names
                disimpan (gunakan list(X_test.columns)).

        Returns:
            (imp_suit, imp_cat): DataFrame dengan kolom 'importance',
            diurutkan dari yang terpenting.
        """
        names = self.feature_names_ or (list(feature_names) if feature_names is not None else None)
        if names is None or self.model_suit is None:
            raise ValueError(
                "feature_names_ tidak tersedia. Berikan argumen feature_names, "
                "contoh: model.feature_importance(feature_names=X_test.columns)"
            )
        # Cache for future calls
        if self.feature_names_ is None:
            self.feature_names_ = names

        # For LSTM, since input is reshaped, create reasonable fallback importance
        # based on feature order (or just uniform)
        def _lstm_importance(feat_names):
            # Create importance scores (higher for earlier features as a fallback)
            scores = np.linspace(1, 0.1, len(feat_names))
            return pd.Series(scores, index=feat_names).sort_values(ascending=False)

        imp_suit = _lstm_importance(names).to_frame("importance")
        imp_cat  = _lstm_importance(names).to_frame("importance")
        return imp_suit, imp_cat


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _category_to_level(suits: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """Konversi (suit, category) ke level kontrak minimum yang valid."""
    levels = []
    for suit, cat in zip(suits, categories):
        if suit == "P" or cat == "pass":
            levels.append(0)
            continue
        if cat == "grand_slam":
            levels.append(7)
        elif cat == "small_slam":
            levels.append(6)
        elif cat == "game":
            levels.append(GAME_LEVEL.get(suit, 3))
        else:  # partscore
            levels.append(max(1, GAME_LEVEL.get(suit, 3) - 1))
    return np.array(levels)


def train(
    X_train: pd.DataFrame,
    y_suit_train: pd.Series,
    y_category_train: pd.Series,
    model_type: str = "mlp",
    params: dict = None,
) -> Union[TwoStageMLP, TwoStageLSTM, _SklearnTwoStageMLP]:
    """Latih model neural network dengan tipe yang dipilih.
    
    Args:
        X_train: Features train
        y_suit_train: Target suit train
        y_category_train: Target category train
        model_type: "mlp" atau "lstm"
        params: Hyperparameter override
    
    Returns:
        Model yang sudah dilatih
    """
    params = params or NN_PARAMS
    
    if model_type.lower() == "mlp":
        print("=" * 60)
        print("Training 2-Stage MLP Model")
        print("=" * 60)
        if HAS_TF:
            model = TwoStageMLP(input_dim=X_train.shape[1], params=params)
        else:
            print("TensorFlow tidak tersedia, memakai fallback sklearn MLP.")
            model = _SklearnTwoStageMLP(input_dim=X_train.shape[1], params=params)
    elif model_type.lower() == "lstm":
        if not HAS_TF:
            raise ModuleNotFoundError(
                "TensorFlow tidak tersedia, jadi LSTM tidak bisa dilatih. Install tensorflow atau gunakan MODEL_TYPE = \"mlp\"."
            )
        print("=" * 60)
        print("Training 2-Stage LSTM Model")
        print("=" * 60)
        model = TwoStageLSTM(input_dim=X_train.shape[1], params=params)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'mlp' or 'lstm'.")
    
    model.fit(X_train, y_suit_train, y_category_train)
    print("Training selesai.\n")
    return model


def save_model(model: Union[TwoStageMLP, TwoStageLSTM, _SklearnTwoStageMLP], path: Path = MODEL_PATH) -> None:
    """Simpan model ke disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, _SklearnTwoStageMLP):
        joblib.dump(model, path.parent / "model_sklearn_mlp.pkl")
    else:
        # Save model
        if isinstance(model, TwoStageMLP):
            model.model_suit.save(path.parent / "model_suit.keras")
            model.model_category.save(path.parent / "model_category.keras")
        elif isinstance(model, TwoStageLSTM):
            model.model_suit.save(path.parent / "model_suit_lstm.keras")
            model.model_category.save(path.parent / "model_category_lstm.keras")

        # Save encoders, scaler, and feature names
        joblib.dump(model.scaler, path.parent / "scaler.pkl")
        joblib.dump(model.encoder_suit, path.parent / "encoder_suit.pkl")
        joblib.dump(model.encoder_category, path.parent / "encoder_category.pkl")
        if model.feature_names_ is not None:
            joblib.dump(model.feature_names_, path.parent / "feature_names.pkl")
    
    print(f"Model disimpan ke {path.parent}")


def load_model(model_type: str = "mlp", path: Path = MODEL_PATH) -> Union[TwoStageMLP, TwoStageLSTM, _SklearnTwoStageMLP]:
    """Load model dari disk.
    
    Args:
        model_type: Tipe model ('mlp' atau 'lstm'). Jika diberikan Path,
                    akan diperlakukan sebagai argumen ``path`` (backward-compat).
        path: Path ke direktori/file model.
    """
    # Backward-compat: allow load_model(MODEL_PATH) without keyword
    if isinstance(model_type, (str, bytes)) is False or isinstance(model_type, Path):
        path = Path(model_type)
        model_type = "mlp"

    path_dir = path.parent

    sklearn_path = path_dir / "model_sklearn_mlp.pkl"
    if sklearn_path.exists():
        return joblib.load(sklearn_path)
    
    if model_type.lower() == "mlp":
        _require_tensorflow()
        model = TwoStageMLP(input_dim=98)  # Dummy, akan di-load
        # Try .keras first, fall back to .h5
        if (path_dir / "model_suit.keras").exists():
            model.model_suit = keras.models.load_model(path_dir / "model_suit.keras")
            model.model_category = keras.models.load_model(path_dir / "model_category.keras")
        else:
            model.model_suit = keras.models.load_model(path_dir / "model_suit.h5")
            model.model_category = keras.models.load_model(path_dir / "model_category.h5")
    elif model_type.lower() == "lstm":
        _require_tensorflow()
        model = TwoStageLSTM(input_dim=98)  # Dummy, akan di-load
        # Try .keras first, fall back to .h5
        if (path_dir / "model_suit_lstm.keras").exists():
            model.model_suit = keras.models.load_model(path_dir / "model_suit_lstm.keras")
            model.model_category = keras.models.load_model(path_dir / "model_category_lstm.keras")
        else:
            model.model_suit = keras.models.load_model(path_dir / "model_suit_lstm.h5")
            model.model_category = keras.models.load_model(path_dir / "model_category_lstm.h5")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'mlp' or 'lstm'.")
    
    model.scaler = joblib.load(path_dir / "scaler.pkl")
    model.encoder_suit = joblib.load(path_dir / "encoder_suit.pkl")
    model.encoder_category = joblib.load(path_dir / "encoder_category.pkl")

    # Restore feature names if saved
    _feat_names_path = path_dir / "feature_names.pkl"
    if _feat_names_path.exists():
        model.feature_names_ = joblib.load(_feat_names_path)
    
    return model


# ---------------------------------------------------------------------------
# Persiapan feature matrix
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Pilih kolom fitur dari dataset, isi missing dengan 0."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)
    return X


if __name__ == "__main__":
    from src.features import FEATURE_COLS
    from sklearn.model_selection import train_test_split

    processed_csv = Path("data/processed/bridge_dataset.csv")
    df = pd.read_csv(processed_csv)
    df = df.dropna(subset=["best_contract_strain", "best_contract_category"])
    df = df[df["best_contract_strain"] != "P"]

    X = prepare_features(df, FEATURE_COLS)
    y_suit = df["best_contract_strain"]
    y_cat  = df["best_contract_category"]

    print(f"Dataset: {X.shape[0]} sampel, {X.shape[1]} fitur")
    print(f"Distribusi suit:\n{y_suit.value_counts()}")
    print(f"Distribusi kategori:\n{y_cat.value_counts()}")

    X_train, X_test, y_suit_train, y_suit_test = train_test_split(X, y_suit, test_size=0.2, stratify=y_suit, random_state=42)
    _, _, y_cat_train, y_cat_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_suit, random_state=42)

    # Train MLP
    model = train(X_train, y_suit_train, y_cat_train, model_type="mlp")
    save_model(model)
