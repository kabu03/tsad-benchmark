import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore #
from tensorflow.keras.layers import Dense, Input # type: ignore #
from tensorflow.keras.optimizers import Adam # type: ignore #
import tensorflow.keras.backend as K # type: ignore #

from ..base import Detector

class AutoencoderDetector(Detector):
    """
    Anomaly detector using a deep Autoencoder.
    The autoencoder is trained on normal data to learn its reconstruction.
    Anomaly scores are based on the reconstruction error of new data.
    """

    def __init__(self, window_size: int = 50,
                 hidden_dims: list = [32, 16, 32],
                 activation: str = 'relu',
                 output_activation: str = 'sigmoid',
                 epochs: int = 30,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 verbose: int = 0):
        """
        Args:
            window_size (int): Size of the sliding window for creating sequences.
            hidden_dims (list): List of integers specifying the number of units in hidden dense layers.
                                The first half is encoder, middle is latent (if odd len), second half is decoder.
                                E.g., [64, 32, 64] for a 3-layer AE (encoder, latent, decoder).
            activation (str): Activation function for hidden layers.
            output_activation (str): Activation function for the output layer.
                                     'sigmoid' is common if data is scaled to [0,1].
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the Adam optimizer.
            verbose (int): Verbosity mode for model training (0, 1, or 2).
        """
        super().__init__()
        self.window_size = window_size
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_activation = output_activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.model = None
        self.scaler = MinMaxScaler()

    def _create_sequences(self, data: np.ndarray):
        """
        Creates sequences from a 1D time series array.
        Args:
            data (np.ndarray): 1D NumPy array of time series data.
        Returns:
            np.ndarray: 2D NumPy array of sequences, shape (num_sequences, window_size).
                        Returns empty array if data is too short.
        """
        if len(data) < self.window_size:
            return np.empty((0, self.window_size))
        
        sequences = []
        for i in range(len(data) - self.window_size + 1):
            sequences.append(data[i:(i + self.window_size)])
        return np.array(sequences)

    def _build_model(self):
        """Builds the Keras Sequential Autoencoder model."""
        model_layers = [Input(shape=(self.window_size,))]
        
        # Encoder layers
        for dim in self.hidden_dims:
            model_layers.append(Dense(dim, activation=self.activation))
        
        # Decoder layers (match encoder)
        model_layers.append(Dense(self.window_size, activation=self.output_activation))
        
        model = Sequential(model_layers)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        if self.verbose > 0:
            model.summary()
        return model

    def fit(self, x_train: np.ndarray):
        """
        Fit the Autoencoder model to the training data.
        Args:
            x_train (np.ndarray): The training data (1D).
        """
        if x_train.ndim != 1:
            raise ValueError("Input x_train must be a 1D NumPy array.")
        if len(x_train) == 0:
            raise ValueError("Input x_train cannot be empty.")

        # Reshape for scaler, scale, then flatten
        x_train_reshaped = x_train.reshape(-1, 1)
        scaled_x_train = self.scaler.fit_transform(x_train_reshaped).flatten()

        train_sequences = self._create_sequences(scaled_x_train)

        if train_sequences.shape[0] == 0:
            raise ValueError(f"Not enough training data to create sequences with window_size {self.window_size}. "
                             f"Need at least {self.window_size} data points, got {len(x_train)}.")

        # Clear previous session if any model exists (good practice for Keras)
        if self.model is not None:
            K.clear_session()
            
        self.model = self._build_model()
        self.model.fit(train_sequences, train_sequences,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       shuffle=True,
                       verbose=self.verbose)

    def score(self, x_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (reconstruction errors) for the test data.
        Args:
            x_test (np.ndarray): The test data (1D).
        Returns:
            np.ndarray: Reconstruction error (MSE) for each window in x_test,
                        padded to match the length of x_test.
        """
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() first.")
        if x_test.ndim != 1:
            raise ValueError("Input x_test must be a 1D NumPy array.")

        if len(x_test) == 0:
            return np.array([])

        # Reshape for scaler, scale, then flatten
        x_test_reshaped = x_test.reshape(-1, 1)
        scaled_x_test = self.scaler.transform(x_test_reshaped).flatten()
        
        test_sequences = self._create_sequences(scaled_x_test)
        
        output_scores = np.zeros(len(x_test), dtype=float)

        if test_sequences.shape[0] > 0:
            reconstructions = self.model.predict(test_sequences, verbose=self.verbose)
            mse_per_window = np.mean(np.square(test_sequences - reconstructions), axis=1)
            
            # Assign score of window i (starting at x_test[i]) to point x_test[i]
            # and pad the rest.
            output_scores[:len(mse_per_window)] = mse_per_window
            if len(mse_per_window) > 0 and len(mse_per_window) < len(x_test):
                padding_value = np.mean(mse_per_window) 
                output_scores[len(mse_per_window):] = padding_value
        elif len(x_test) > 0 : # Not enough data for any sequence, but x_test is not empty
            pass
            
        return output_scores

    def __del__(self):
        # Attempt to clear Keras session when object is deleted
        # This is to help manage resources if many models are created/destroyed.
        if hasattr(K, 'clear_session'):
            K.clear_session()
