import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, ParameterGrid
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from scipy import stats

class Regressor():

    def __init__(self, x, nb_epoch=1000):
        """
        initialize regressor
        """
        # preprocessor variables
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.category_binarizer = LabelBinarizer()
        self.nb_epoch = nb_epoch
        self.category_mapping = None
        self.fill_values = {}
        self.numeric_cols = None
        self.categorical_cols = None
        self.encoded_features = None

        # get processed dimensions after preprocessing
        X_processed, _ = self._preprocessor(x, training=True)
        input_dim = X_processed.shape[1]
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.nb_epoch = nb_epoch

    def _preprocessor(self, x, y = None, training = False):
        """
        Preprocess data
        """
        try:
            x_processed = x.copy()
            
            if training:
                # 1. identify numeric and categorical columns
                self.numeric_cols = x_processed.select_dtypes(include=['float64', 'int64']).columns
                self.categorical_cols = x_processed.select_dtypes(exclude=['float64', 'int64']).columns
                
                # 2. fill missing values
                self.fill_values = {}
                for col in x_processed.columns:
                    if col in self.numeric_cols:
                        fill_value = x_processed[col].median()
                    else:
                        fill_value = x_processed[col].mode()[0]
                    self.fill_values[col] = fill_value

            # Apply missing values
            for col, value in self.fill_values.items():
                x_processed.loc[:, col] = x_processed[col].fillna(value)

            # 3. categorical features with one-hot encoding
            if training:
                self.encoder = OneHotEncoder(handle_unknown='ignore')
                x_encoded = self.encoder.fit_transform(x_processed[self.categorical_cols]).toarray()
            else:
                x_encoded = self.encoder.transform(x_processed[self.categorical_cols]).toarray()

            # 4. standardize numeric features
            if training:
                self.scaler = StandardScaler()
                x_standard = self.scaler.fit_transform(x_processed[self.numeric_cols])
            else:
                x_standard = self.scaler.transform(x_processed[self.numeric_cols])

            # 5. combine numeric and categorical features
            x_processed = np.hstack([x_standard, x_encoded])
            
            # 6. process target variable if provided
            if y is not None:
                y_processed = y.values 
                if training:
                    self.y_scaler = StandardScaler()
                    y_processed = self.y_scaler.fit_transform(y_processed.reshape(-1, 1))
                else:
                    y_processed = self.y_scaler.transform(y_processed.reshape(-1, 1))
                return torch.tensor(x_processed, dtype=torch.float32), torch.tensor(y_processed, dtype=torch.float32)
            
            return torch.tensor(x_processed, dtype=torch.float32), None

        except Exception as e:
            print(f"Fail to preprocesss: {str(e)}")
            raise


    def fit(self, X, y):
        """
        fit the model
        """
        X_processed, y_processed = self._preprocessor(X, y, training=True)
        
        for epoch in range(self.nb_epoch):
            # train
            self.network.train()
            self.optimizer.zero_grad()
            y_pred = self.network(X_processed)
            loss = F.mse_loss(y_pred, y_processed)
            
            # backward
            loss.backward()
            self.optimizer.step()
            
            # print test
            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}, Loss: {loss.item():.4f}")



        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, X):
        """
        Regressor prediction function
        """
        X_processed, _ = self._preprocessor(X, training=False)
        
        # set to eval mode
        self.network.eval()
        with torch.no_grad():
            y_pred = self.network(X_processed)
        
        # use y_scaler to transform back to original range
        if hasattr(self, 'y_scaler'):
            y_pred = self.y_scaler.inverse_transform(y_pred)
        
        return y_pred

    def score(self, X, y, plot=False):
        """
        Score function for regressor
        """
        # preprocess data
        X_processed, y_processed = self._preprocessor(X, y, training=False)
        
        # set to eval mode
        self.network.eval()
        with torch.no_grad():
            y_pred = self.network(X_processed)
        
        # convert predictions back to original scale
        y_pred_orig = self.y_scaler.inverse_transform(y_pred.numpy())
        y_true = y.values.reshape(-1, 1)
        
        # calculate metrics
        rmse = np.sqrt(np.mean((y_pred_orig - y_true) ** 2))
        mae = np.mean(np.abs(y_pred_orig - y_true))
        r2 = 1 - np.sum((y_true - y_pred_orig) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # plot
            y_true = y_true.ravel()
            y_pred_orig = y_pred_orig.ravel()
            
            # 1. 
            ax1.scatter(y_true, y_pred_orig, alpha=0.5)
            ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            ax1.set_xlabel('Actual House Values')
            ax1.set_ylabel('Predicted House Values')
            ax1.set_title('Predicted vs Actual Values')
            
            # 2.
            residuals = y_true - y_pred_orig
            ax2.scatter(y_pred_orig, residuals, alpha=0.5)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals vs Predicted Values')

            plt.tight_layout()
            plt.savefig('evaluation_plots.png')
            plt.close()
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mse': rmse ** 2
        }


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def perform_hyperparameter_search(x_train, y_train):
    """
    perform hyperparameter search
    """
    # define parameter grid
    param_grid = {
        'nb_epoch': [500, 1000],
        'learning_rate': [0.001, 0.01],
        'batch_size': [32, 64]
    }
    
    best_score = float('inf')
    best_params = None
    
    # split validation set
    X_train, X_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    for params in ParameterGrid(param_grid):
        print(f"\nTesting parameters: {params}")
        
        # create and train model
        model = Regressor(X_train, nb_epoch=params['nb_epoch'])
        model.optimizer = torch.optim.Adam(
            model.network.parameters(), 
            lr=params['learning_rate']
        )
        
        try:
            model.fit(X_train, y_train)
            val_score = model.score(X_val, y_val)

            scores = model.score(X_val, y_val)
            
            print(f"Validation scores:")
            print(f"RMSE: {scores['rmse']:.2f}")
            print(f"MAE: {scores['mae']:.2f}")
            print(f"R²: {scores['r2']:.4f}")
            
            # Use RMSE as the main evaluation 
            val_score = scores['rmse']
            
            if val_score < best_score:
                best_score = val_score
                best_params = params
                print(f"New best score(RMSE): {best_score:.4f}")
                
        except Exception as e:
            print(f"Error with params {params}: {str(e)}")
            continue
    
    print("\nBest Parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best RMSE: {best_score:.4f}")
    
    return best_params



def example_main():
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv") 
    
    # 1. First split into train+val and test sets
    X = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 2. Perform hyperparameter search using train_val data
    print("\nHyperparameter search...")
    best_params = perform_hyperparameter_search(X_train_val, y_train_val)

    # 3. Train final model with best parameters
    print("\nTraining final model with best parameters...")
    regressor = Regressor(X_train_val, nb_epoch=best_params['nb_epoch'])
    regressor.optimizer = torch.optim.Adam(
        regressor.network.parameters(), 
        lr=best_params['learning_rate']
    )
    
    # Important: Fit the model with training data
    regressor.fit(X_train_val, y_train_val)
    
    # 4. Evaluate on test set and generate visualization plots
    print("\nEvaluating on test set...")
    test_scores = regressor.score(X_test, y_test, plot=True)  # Only generate plots here
    print("\nTest Set Performance:")
    print(f"RMSE: {test_scores['rmse']:.2f}")
    print(f"MAE: {test_scores['mae']:.2f}")
    print(f"R²: {test_scores['r2']:.4f}")
    print(f"MSE: {test_scores['mse']:.2f}")
    print("\nEvaluation plots have been saved as 'evaluation_plots.png'")
    
    # 5. Save the model
    save_regressor(regressor)


if __name__ == "__main__":
    example_main()

