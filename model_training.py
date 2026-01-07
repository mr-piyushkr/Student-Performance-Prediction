import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf')
        }
        
        self.results = {}
    
    def train_and_evaluate(self):
        for name, model in self.models.items():
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Predict
            y_pred = model.predict(self.X_test)
            
            # Evaluate
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            self.results[name] = {
                'MSE': mse,
                'MAE': mae,
                'R2 Score': r2
            }
        
        return self.results
    
    def plot_results(self):
        # Create result DataFrame
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Plot bar chart of R2 Scores
        plt.figure(figsize=(10, 6))
        results_df['R2 Score'].plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('R2 Score')
        plt.tight_layout()
        plt.show()
    
    def feature_importance(self, feature_names):
        # Get feature importance for Random Forest
        rf_model = self.models['Random Forest']
        importances = rf_model.feature_importances_
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title('Feature Importance in CGPA Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df