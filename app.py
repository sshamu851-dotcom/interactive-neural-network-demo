
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Interactive Neural Network Learning Demo",
    page_icon="🧠",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        color: #667085;
        font-size: 1.02rem;
        margin-bottom: 1.2rem;
    }
    .info-card {
        padding: 1rem 1.15rem;
        border-radius: 14px;
        border: 1px solid #e4e7ec;
        background: #f8fafc;
        margin-bottom: 0.7rem;
    }
    .metric-label {
        color: #667085;
        font-size: 0.82rem;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Dataset: XOR
# -----------------------------
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y = np.array([0, 1, 1, 0])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dense grid for the decision boundary.
grid = np.linspace(-0.25, 1.25, 220)
xx, yy = np.meshgrid(grid, grid)
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid_points)

# -----------------------------
# Session state
# -----------------------------
if "result" not in st.session_state:
    st.session_state.result = None

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Neural Network Controls")

hidden_layers = st.sidebar.slider(
    "Hidden layers",
    min_value=1,
    max_value=3,
    value=2,
    help="The number of hidden layers in the neural network."
)

neurons = st.sidebar.slider(
    "Neurons per hidden layer",
    min_value=2,
    max_value=16,
    value=4,
    help="Every hidden layer will contain this many neurons."
)

activation = st.sidebar.selectbox(
    "Activation function",
    ["relu", "tanh", "logistic"],
    index=1,
    help="Activation functions introduce non-linearity so the network can learn patterns such as XOR."
)

learning_rate = st.sidebar.slider(
    "Learning rate",
    min_value=0.001,
    max_value=1.0,
    value=0.10,
    step=0.001,
    format="%.3f",
    help="Controls how large each parameter update is."
)

epochs = st.sidebar.slider(
    "Training epochs",
    min_value=10,
    max_value=500,
    value=150,
    step=10,
    help="Number of complete passes through the four XOR examples."
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=999,
    value=42,
    step=1,
    help="Use the same seed to reproduce the same run."
)

train_button = st.sidebar.button(
    "🧠 Train Neural Network",
    type="primary",
    use_container_width=True
)

if st.sidebar.button("↻ Reset", use_container_width=True):
    st.session_state.result = None
    st.rerun()

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🧠 Interactive Neural Network Learning Demo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Train a real neural network on the XOR problem and observe how architecture and training settings change what it learns.</div>',
    unsafe_allow_html=True
)

st.info(
    "XOR is deliberately small but non-linearly separable. A neural network with hidden neurons can learn the pattern, "
    "making it useful for visualizing why hidden layers and non-linear activation functions matter."
)

# -----------------------------
# Dataset preview
# -----------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("1. Training data")
    df = pd.DataFrame({
        "Input X₁": X[:, 0].astype(int),
        "Input X₂": X[:, 1].astype(int),
        "Target": y
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

with right:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for label, marker in [(0, "o"), (1, "s")]:
        pts = X[y == label]
        ax.scatter(pts[:, 0], pts[:, 1], s=150, marker=marker, label=f"Class {label}")
    ax.set_xlabel("Input X₁")
    ax.set_ylabel("Input X₂")
    ax.set_title("XOR training pattern")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# -----------------------------
# Train
# -----------------------------
if train_button:
    architecture = tuple([neurons] * hidden_layers)

    model = MLPClassifier(
        hidden_layer_sizes=architecture,
        activation=activation,
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=learning_rate,
        max_iter=1,
        warm_start=True,
        shuffle=False,
        random_state=int(random_seed),
        batch_size=4,
        tol=0,
        n_iter_no_change=epochs + 1,
    )

    # partial_fit is used one epoch at a time so we can record genuine learning progress.
    losses = []
    for epoch in range(epochs):
        if epoch == 0:
            model.partial_fit(X_scaled, y, classes=np.array([0, 1]))
        else:
            model.partial_fit(X_scaled, y)
        losses.append(float(model.loss_))

    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = float(np.mean(predictions == y) * 100)

    # Decision boundary over the 2D input space.
    grid_prob = model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

    st.session_state.result = {
        "model": model,
        "losses": losses,
        "probabilities": probabilities,
        "predictions": predictions,
        "accuracy": accuracy,
        "grid_prob": grid_prob,
        "architecture": architecture,
    }

# -----------------------------
# Results
# -----------------------------
result = st.session_state.result

if result is None:
    st.subheader("2. Train the network")
    st.markdown("""
    **How to use the demo**

    1. Choose the number of hidden layers and neurons.
    2. Adjust the activation function, learning rate, and epochs.
    3. Click **Train Neural Network**.
    4. Compare the loss curve and decision boundary.
    5. Try different settings and observe the effect on learning.
    """)

    st.markdown('<div class="info-card"><b>Key idea:</b> the network is not given an XOR rule. It learns the relationship from the four training examples by updating its weights during training.</div>', unsafe_allow_html=True)
else:
    st.subheader("2. What the network learned")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Architecture", f"2 → {' → '.join(map(str, result['architecture']))} → 1")
    with m2:
        st.metric("Final loss", f"{result['losses'][-1]:.4f}")
    with m3:
        st.metric("Training accuracy", f"{result['accuracy']:.0f}%")
    with m4:
        st.metric("Epochs", str(len(result["losses"])))

    chart1, chart2 = st.columns(2)

    with chart1:
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        ax.plot(range(1, len(result["losses"]) + 1), result["losses"], linewidth=2)
        ax.set_title("Learning curve: loss vs. epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with chart2:
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        contour = ax.contourf(xx, yy, result["grid_prob"], levels=30, alpha=0.75)
        ax.contour(xx, yy, result["grid_prob"], levels=[0.5], linewidths=2)
        ax.scatter(
            X[y == 0, 0], X[y == 0, 1],
            s=150, marker="o", edgecolors="black", label="Class 0"
        )
        ax.scatter(
            X[y == 1, 0], X[y == 1, 1],
            s=150, marker="s", edgecolors="black", label="Class 1"
        )
        ax.set_title("Learned decision boundary")
        ax.set_xlabel("Input X₁")
        ax.set_ylabel("Input X₂")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("3. Predictions on the training examples")
    pred_df = pd.DataFrame({
        "X₁": X[:, 0].astype(int),
        "X₂": X[:, 1].astype(int),
        "Target": y,
        "Predicted class": result["predictions"],
        "P(class 1)": np.round(result["probabilities"], 4),
        "Correct?": np.where(result["predictions"] == y, "✓", "✗"),
    })
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="info-card"><b>Interpretation:</b> the loss curve shows whether training is improving. '
        'The decision-boundary plot shows the actual function learned by the neural network. '
        'Changing the architecture, activation, learning rate, or epochs and retraining lets you observe these effects directly.</div>',
        unsafe_allow_html=True
    )

st.divider()
st.caption("College ML Demonstration • Interactive Neural Network Learning • Built with Streamlit + scikit-learn")
