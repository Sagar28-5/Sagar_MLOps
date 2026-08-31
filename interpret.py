from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

print("Feature Importance:")
print(model.feature_importances_)

plt.bar(iris.feature_names, model.feature_importances_)
plt.title("Feature Importance")
plt.xticks(rotation=20)
plt.show()