
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


digits = datasets.load_digits()


X = digits.data   
y = digits.target 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = svm.SVC(kernel='rbf')   


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Total Accuracy: {accuracy * 100:.2f}%")


print("Predicted Digit:", model.predict([X_test[0]]))
print("Actual Digit:", y_test[0])
