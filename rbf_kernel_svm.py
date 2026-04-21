from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset load karein
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Model initialization (RBF Kernel)
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("RBF Kernel Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

