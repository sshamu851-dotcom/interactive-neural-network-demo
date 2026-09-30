
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = np.array([0,1,1,0])
X = StandardScaler().fit_transform(X)

model = MLPClassifier(
    hidden_layer_sizes=(4,4),
    activation="tanh",
    solver="sgd",
    learning_rate_init=0.1,
    max_iter=1,
    warm_start=True,
    shuffle=False,
    random_state=42,
    batch_size=4,
    tol=0,
    n_iter_no_change=1000,
)
for _ in range(200):
    model.partial_fit(X, y, classes=np.array([0,1]))
print("Predictions:", model.predict(X).tolist())
print("Accuracy:", model.score(X, y))
