"""
PyTorch-based time series prediction module.
Implements LSTM-based deep learning models for financial time series forecasting,
including data preparation, model training, and prediction generation.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import logging
from src.tools.logger import logger

log = logger(log_file='DAPS_Log.log', log_level=logging.DEBUG)

class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for time series data.
    
    Handles the preparation and access of sequential financial data
    for model training and evaluation.
    
    Args:
        X (array-like): Input features
        y (array-like): Target values
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTM(nn.Module): #gru rnn
    """
    LSTM-based neural network model for time series prediction.
    
    Implements a multi-layer LSTM architecture with a final fully connected layer
    for time series forecasting.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units in LSTM layers
        num_layers (int): Number of LSTM layers
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesPredictor:
    """
    High-level class for time series prediction using LSTM models.
    
    Handles the entire prediction pipeline including data preprocessing,
    model training, and prediction generation.
    
    Args:
        n_components (float): PCA variance ratio threshold (0-1)
        sequence_length (int): Length of input sequences
        hidden_size (int): Size of LSTM hidden layers
        num_layers (int): Number of LSTM layers
        batch_size (int): Training batch size
        num_epochs (int): Number of training epochs
        learning_rate (float): Model learning rate
    """
    def __init__(self, n_components=0.95, sequence_length=10, hidden_size=64, num_layers=2,
                 batch_size=32, num_epochs=100, learning_rate=0.001):
        self.n_components = n_components
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_pca_components = None
        self.last_sequence_data = None
        self.training_data = None
        self.epoch_losses = []

    def _prepare_sequences(self, X, y):
        """
        Prepares input sequences for the LSTM model.
        
        Creates overlapping sequences of specified length from input data
        for sequential prediction.
        
        Args:
            X (array-like): Input features
            y (array-like): Target values
            
        Returns:
            tuple: Processed sequences and corresponding targets
        """
        sequences = []
        targets = []

        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:(i + self.sequence_length)])
            targets.append(y[i + self.sequence_length])

        return np.array(sequences), np.array(targets).reshape(-1, 1)

    def predict_multiple_days(self, sample_data, num_days, target_column):
        """
        Generates predictions for multiple future time steps.
        
        Uses an iterative approach to predict multiple days ahead,
        using each prediction as input for the next prediction.
        
        Args:
            sample_data (pd.DataFrame): Initial data for prediction
            num_days (int): Number of days to predict
            target_column (str): Name of the target variable
            
        Returns:
            list: Predicted values for specified number of days
        """
        self.model.eval()
        predictions = []
        current_sequence = sample_data.copy()

        with torch.no_grad():
            for _ in range(num_days):
                # Prepare features (excluding target column)
                features = current_sequence.drop(columns=[target_column])

                # Scale features
                X_scaled = self.feature_scaler.transform(features)

                # Apply PCA
                X_pca = self.pca.transform(X_scaled)

                # Prepare sequence
                X_seq = X_pca[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                X_seq = torch.FloatTensor(X_seq).to(self.device)

                # Make prediction
                prediction = self.model(X_seq)
                prediction = prediction.view(-1, 1)

                # Scale back to original scale
                prediction_value = self.target_scaler.inverse_transform(
                    prediction.cpu().numpy()
                )[0][0]

                predictions.append(prediction_value)

                # Update sequence for next prediction
                new_row = current_sequence.iloc[-1:].copy()
                new_row.index = [new_row.index[0] + pd.Timedelta(days=1)]
                new_row[target_column] = prediction_value

                # Remove oldest day and add prediction
                current_sequence = pd.concat([current_sequence[1:], new_row])

        return predictions

    def train(self, df, target_column, cutoff_date):
        """
        Trains the LSTM model on provided data.
        
        Handles data preprocessing, model initialization, and training process,
        including PCA transformation and sequence preparation.
        
        Args:
            df (pd.DataFrame): Training data
            target_column (str): Name of target variable
            cutoff_date (str): Date to split training data
        """
        # Store initial training data
        self.training_data = df[df.index < cutoff_date].copy()
        
        # Separate features and target
        X = self.training_data.drop(columns=[target_column])
        y = self.training_data[target_column]

        # Scale features and target
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1))

        # Apply PCA and store components
        X_pca = self.pca.fit_transform(X_scaled)
        self.last_pca_components = self.pca

        # Prepare sequences and store sample
        X_seq, y_seq = self._prepare_sequences(X_pca, y_scaled)
        self.last_sequence_data = X_seq

        # Create dataset and dataloader
        dataset = TimeSeriesDataset(X_seq, y_seq)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        self.model = LSTM(
            input_size=X_pca.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        self.epoch_losses = []
        for epoch in range(self.num_epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            epoch_loss = total_loss / len(dataloader)
            self.epoch_losses.append(epoch_loss)
            if (epoch + 1) % 10 == 0:
                log.log(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/len(dataloader):.4f}", logging.INFO)

    def predict(self, sample_data):
        """
        Generates a single prediction from input data.
        
        Processes input data through the same preprocessing pipeline
        used during training and generates a prediction.
        
        Args:
            sample_data (pd.DataFrame): Input data for prediction
            
        Returns:
            float: Predicted value
        """
        self.model.eval()
        with torch.no_grad():
            # Scale features
            X_scaled = self.feature_scaler.transform(sample_data)
            # Apply PCA
            X_pca = self.pca.transform(X_scaled)       
            # Prepare sequence
            X_seq = X_pca[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            X_seq = torch.FloatTensor(X_seq).to(self.device)
            # Make prediction
            prediction = self.model(X_seq)
            # Ensure prediction has the right shape
            prediction = prediction.view(-1, 1)
            # Scale back to original scale
            prediction = self.target_scaler.inverse_transform(
                prediction.cpu().numpy()
            )
            return prediction[0][0]
    def get_processing_data(self):
        """
        Retrieves data processing and training information.
        
        Returns a dictionary containing PCA components, sequence data,
        training data, and training loss history.
        
        Returns:
            dict: Processing and training information
        """
        return {
            'pca_components': self.last_pca_components,
            'sequence_data': self.last_sequence_data,
            'training_data': self.training_data,
            'epoch_losses': self.epoch_losses
        }