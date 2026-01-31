import numpy as np
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler for data normalization
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import tensorflow.keras.backend as K # type: ignore

from ..base import Detector

class LSTMDetector(Detector):
    """
    Anomaly detector using a Long Short-Term Memory (LSTM) network.
    The LSTM is trained to predict the next value in a time series.
    Anomalies are detected based on high prediction errors (Mean Squared Error).
    """

    def __init__(self, 
                 window_size: int = 50,       # Number of past time steps to use for prediction
                 lstm_units: int = 50,        # Number of units in the LSTM layer
                 epochs: int = 30,            # Number of training epochs
                 batch_size: int = 32,        # Batch size for training
                 learning_rate: float = 0.001, # Learning rate for the Adam optimizer
                 verbose: int = 0):           # Verbosity mode for model training (0, 1, or 2)
        """
        Initializes the LSTMDetector.

        Args:
            window_size (int): The number of past time steps to consider for predicting the next step.
            lstm_units (int): The number of LSTM units in the hidden layer.
            epochs (int): The number of epochs to train the model.
            batch_size (int): The number of samples per gradient update.
            learning_rate (float): The learning rate for the optimizer.
            verbose (int): Verbosity mode for Keras model training (0 = silent, 1 = progress bar, 2 = one line per epoch).
        """
        super().__init__() # Call the constructor of the base Detector class
        self.window_size = window_size # Store the window size
        self.lstm_units = lstm_units # Store the number of LSTM units
        self.epochs = epochs # Store the number of epochs
        self.batch_size = batch_size # Store the batch size
        self.learning_rate = learning_rate # Store the learning rate
        self.verbose = verbose # Store the verbosity mode

        self.model = None # Initialize the Keras model as None
        self.scaler = MinMaxScaler(feature_range=(0, 1)) # Initialize MinMaxScaler to scale data to [0, 1]

    def _create_sequences(self, data: np.ndarray):
        """
        Creates input sequences (X) and corresponding target values (y) for the LSTM.
        Each sequence in X contains 'window_size' past data points, and the corresponding y is the next data point.

        Args:
            data (np.ndarray): 1D NumPy array of time series data (already scaled).

        Returns:
            tuple: (X, y)
                X (np.ndarray): 3D array of input sequences, shape (num_sequences, window_size, 1).
                y (np.ndarray): 1D array of target values, shape (num_sequences,).
        """
        X, y = [], [] # Initialize empty lists for sequences and targets
        # Iterate through the data to create sequences
        for i in range(len(data) - self.window_size):
            # Extract a sequence of 'window_size' points
            sequence = data[i:(i + self.window_size)] 
            # The target is the point immediately following the sequence
            target = data[i + self.window_size] 
            X.append(sequence) # Append the sequence to X
            y.append(target) # Append the target to y
        
        # Convert lists to NumPy arrays
        # Reshape X to (num_samples, window_size, num_features=1) as expected by LSTM
        return np.array(X).reshape(-1, self.window_size, 1), np.array(y)

    def _build_model(self):
        """
        Builds the Keras LSTM model.
        The model consists of an LSTM layer followed by a Dense output layer.
        """
        model = Sequential() # Initialize a Sequential model
        # Add an LSTM layer with 'lstm_units' and input shape (window_size, num_features=1)
        # `return_sequences=False` because the next layer is a Dense layer expecting a 2D input (after flattening by LSTM)
        model.add(LSTM(self.lstm_units, input_shape=(self.window_size, 1)))
        # Add a Dense output layer with 1 unit (to predict the single next value)
        model.add(Dense(1)) 
        # Compile the model with Adam optimizer and Mean Squared Error loss
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        if self.verbose > 0: # If verbose mode is enabled
            model.summary() # Print the model summary
        return model # Return the compiled model

    def fit(self, x_train: np.ndarray):
        """
        Fit the LSTM model to the training data.
        The training data is assumed to be normal (anomaly-free).

        Args:
            x_train (np.ndarray): The training data (1D NumPy array).
        """
        if x_train.ndim != 1: # Check if input is 1D
            raise ValueError("Input x_train must be a 1D NumPy array.")
        if len(x_train) <= self.window_size: # Check if there's enough data to form at least one sequence
            raise ValueError(f"Training data length ({len(x_train)}) must be greater than window_size ({self.window_size}).")

        # Reshape x_train to 2D for the scaler, fit and transform, then flatten back to 1D
        scaled_x_train = self.scaler.fit_transform(x_train.reshape(-1, 1)).flatten()
        
        # Create sequences from the scaled training data
        X_train_seq, y_train_seq = self._create_sequences(scaled_x_train)

        if X_train_seq.shape[0] == 0: # If no sequences could be created (should be caught by earlier check)
            raise ValueError("Not enough training data to create sequences after applying window_size.")

        # Clear previous Keras session if a model already exists (good practice for re-fitting)
        if self.model is not None:
            K.clear_session()
            
        self.model = self._build_model() # Build the LSTM model
        # Train the model
        self.model.fit(X_train_seq, y_train_seq,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       shuffle=True, # Shuffle training data before each epoch
                       verbose=self.verbose) # Control training verbosity

    def score(self, x_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for the test data.
        Scores are the Mean Squared Error (MSE) of the LSTM's predictions.

        Args:
            x_test (np.ndarray): The test data (1D NumPy array).

        Returns:
            np.ndarray: Anomaly scores (MSE) for each data point in x_test.
                        The scores are padded at the beginning to match the length of x_test.
        """
        if self.model is None: # Check if the model has been fitted
            raise RuntimeError("The model has not been fitted yet. Call fit() first.")
        if x_test.ndim != 1: # Check if input is 1D
            raise ValueError("Input x_test must be a 1D NumPy array.")

        if len(x_test) == 0: # Handle empty test data
            return np.array([])

        # Reshape x_test for scaler, transform using the FITTED scaler, then flatten
        scaled_x_test = self.scaler.transform(x_test.reshape(-1, 1)).flatten()
        
        # Initialize an array for output scores with zeros, same length as x_test
        output_scores = np.zeros(len(x_test), dtype=float)

        # Create sequences from test data
        X_test_seq, y_test_target = self._create_sequences(scaled_x_test)
        
        if X_test_seq.shape[0] > 0:
            # Predict the next values using the LSTM model
            y_pred = self.model.predict(X_test_seq, verbose=0).flatten()
            
            # Calculate Mean Squared Error between prediction and actual target for each step
            # Note: y_test_target is the actual value at step t, y_pred is the predicted value for step t
            mse_errors = (y_test_target - y_pred) ** 2
            
            # The first 'window_size' points do not have predictions, so we leave them as 0 or pad them.
            # A common strategy is to pad with the mean error of the first processed batch or just 0s.
            # Here we fill the scores array starting from index 'window_size'.
            output_scores[self.window_size:] = mse_errors
            
            # Optionally backfill the first 'window_size' scores with the first valid score to avoid zeros
            if len(output_scores) > self.window_size:
                 output_scores[:self.window_size] = np.mean(mse_errors)

        return output_scores

    def __del__(self):
        # Attempt to clear Keras session when object is deleted
        if hasattr(K, 'clear_session'):
            K.clear_session()
