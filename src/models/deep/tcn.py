import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Activation, Add, Dense, Lambda, Dropout, SpatialDropout1D #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from sklearn.preprocessing import StandardScaler

from src.models.base import Detector

# Helper function for TCN Residual Block
def _tcn_residual_block(input_tensor, filters, kernel_size, dilation_rate, dropout_rate):
    """
    Defines a TCN residual block.
    """
    prev_x = input_tensor
    
    # First Convolution
    x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal')(prev_x)
    x = Activation('relu')(x)
    if dropout_rate > 0:
        x = SpatialDropout1D(dropout_rate)(x) # Use SpatialDropout1D for Conv1D

    # Second Convolution
    x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
    x = Activation('relu')(x)
    if dropout_rate > 0:
        x = SpatialDropout1D(dropout_rate)(x)

    # Residual connection
    if prev_x.shape[-1] != filters:  # Match filter dimensions for residual connection
        shortcut = Conv1D(filters, 1, padding='same')(prev_x)
    else:
        shortcut = prev_x
    
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

class TCNDetector(Detector):
    """
    Temporal Convolutional Network (TCN) for anomaly detection.
    It predicts the next time step, and the prediction error is used as the anomaly score.
    """
    def __init__(self, window_size=64, num_filters=32, kernel_size=3,
                 dilations=(1, 2, 4, 8), dropout_rate=0.1, 
                 learning_rate=0.001, epochs=30, batch_size=32, verbose=0):
        super().__init__()
        self.window_size = window_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        # It's good practice to add data scaling (e.g., StandardScaler)
        self.scaler = StandardScaler() 
        # For simplicity, scaling is omitted here but recommended for real use.

    def _create_sequences(self, data: np.ndarray):
        """
        Creates sequences (windows) and corresponding targets for TCN.
        Input: data (1D array)
        Output: X (n_samples, window_size, 1), y (n_samples,)
        """
        X, y = [], []
        if len(data) <= self.window_size:
            # Not enough data to form even one sequence
            return np.array(X).reshape(-1, self.window_size, 1), np.array(y)
            
        for i in range(len(data) - self.window_size):
            X.append(data[i:(i + self.window_size)])
            y.append(data[i + self.window_size])
        return np.array(X).reshape(-1, self.window_size, 1), np.array(y)

    def _build_model(self):
        """
        Builds the TCN Keras model.
        """
        input_layer = Input(shape=(self.window_size, 1))
        x = input_layer

        for dilation_rate in self.dilations:
            x = _tcn_residual_block(x, self.num_filters, self.kernel_size, 
                                    dilation_rate, self.dropout_rate)
        
        # Use the output of the last time step from the TCN stack for prediction
        # x has shape (batch_size, window_size, num_filters)
        # We want to predict the single next value
        x = Lambda(lambda l: l[:, -1, :])(x) # Takes the last time step features
        output_layer = Dense(1)(x) # Predicts the single next value

        model = Model(inputs=input_layer, outputs=output_layer)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        if self.verbose > 0:
            model.summary()
        return model

    def fit(self, x_train: np.ndarray):
        """
        Fit the TCN detector to the training data.
        x_train is expected to be a 1D NumPy array.
        """
        if x_train.ndim != 1:
            raise ValueError("x_train must be a 1D NumPy array.")
        
        # Data scaling
        x_train_processed = self.scaler.fit_transform(x_train.reshape(-1, 1)).flatten()

        X_train_seq, y_train_seq = self._create_sequences(x_train_processed)

        if X_train_seq.shape[0] == 0:
            print(f"Warning: Not enough training data to create sequences with window_size {self.window_size}. Model not trained.")
            self.model = None # Ensure model is None if not trainable
            return

        if self.model is None:
            self.model = self._build_model()
        
        self.model.fit(X_train_seq, y_train_seq, 
                       epochs=self.epochs, 
                       batch_size=self.batch_size, 
                       verbose=self.verbose)

    def score(self, x_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for the test data.
        Scores are the absolute prediction errors.
        x_test is expected to be a 1D NumPy array.
        """
        if self.model is None:
            print("Warning: Model not trained. Returning zero scores.")
            return np.zeros_like(x_test, dtype=float)
        
        if x_test.ndim != 1:
            raise ValueError("x_test must be a 1D NumPy array.")

        # Data scaling 
        x_test_processed = self.scaler.transform(x_test.reshape(-1, 1)).flatten()

        X_test_seq, y_test_actual_seq = self._create_sequences(x_test_processed)
        
        anomaly_scores = np.zeros_like(x_test_processed, dtype=float)

        if X_test_seq.shape[0] > 0:
            predictions_seq = self.model.predict(X_test_seq, verbose=self.verbose).flatten()
            errors = np.abs(y_test_actual_seq - predictions_seq)
            
            # Scores correspond to x_test[window_size:]
            # The error for y_test_actual_seq[i] (which is x_test_processed[i+window_size])
            # is the score for x_test_processed[i+window_size]
            anomaly_scores[self.window_size : self.window_size + len(errors)] = errors

            # Pad the beginning part (where no prediction was possible)
            # A common strategy is to use the mean of calculated scores or replicate the first valid score
            if len(errors) > 0:
                padding_value = np.mean(errors) # Or errors[0]
                anomaly_scores[:self.window_size] = padding_value
            else: # Not enough test data to make any predictions
                anomaly_scores[:] = 0 # Or some other default
        else:
            # Not enough test data to create any sequences, fill with zeros or a default
            print(f"Warning: Not enough test data to create sequences with window_size {self.window_size}. Returning zero scores for all points.")
            anomaly_scores[:] = 0

        return anomaly_scores
