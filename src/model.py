from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(preprocessed_data):
    X, y = preprocessed_data
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))

