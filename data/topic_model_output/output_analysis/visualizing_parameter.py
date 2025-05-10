import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you already have a DataFrame `df` with these columns:
# ['learning_rate', 'dropout', 'TD', 'Inverted_RBO', 'TC_umass', 'TC_cv', 'TC_cnpmi']

# Step 1: Compute diversity, coherence, and total score
df["Diversity"] = df[["TD", "Inverted_RBO"]].mean(axis=1)
df["Coherence"] = df[["TC_umass", "TC_cv", "TC_cnpmi"]].mean(axis=1)
df["Total_Score"] = df["Diversity"] + df["Coherence"]

# Step 2: Create pivot table
pivot_lr_dropout = df.pivot_table(
    index="learning_rate",
    columns="dropout",
    values="Total_Score",
    aggfunc="mean"
)

# Step 3: Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_lr_dropout, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Diversity + Coherence'})
plt.title("Heatmap of Learning Rate and Dropout (Total Score)", fontsize=14)
plt.xlabel("Dropout")
plt.ylabel("Learning Rate")
plt.tight_layout()
plt.savefig("/{dataset}/{topic_model}/learning_rate_vs_dropout_heatmap.pdf", format="pdf")
plt.show()
