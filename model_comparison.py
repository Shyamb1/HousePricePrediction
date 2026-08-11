import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Model": [
        "Linear Regression",
        "Tuned Random Forest"
    ],
    "R2 Score": [
        0.8671,
        0.8893
    ]
}

df = pd.DataFrame(data)

plt.figure(figsize=(8, 5))

plt.bar(
    df["Model"],
    df["R2 Score"]
)

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("R² Score")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300
)

plt.show()