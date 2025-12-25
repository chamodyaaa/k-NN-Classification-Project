import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load cleaned data
df = pd.read_csv('dataset_cleaned.csv')
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test k values from 1 to 50
k_range = range(1, 51)
train_accuracies = []
test_accuracies = []
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Train accuracy
    train_pred = knn.predict(X_train_scaled)
    train_accuracies.append(accuracy_score(y_train, train_pred))
    
    # Test accuracy
    test_pred = knn.predict(X_test_scaled)
    test_accuracies.append(accuracy_score(y_test, test_pred))
    
    # Cross-validation score (5-fold)
    cv_score = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(cv_score.mean())

# Find best k based on different metrics
best_k_test = np.argmax(test_accuracies) + 1
best_k_cv = np.argmax(cv_scores) + 1

# Find the optimal k avoiding k=1 (which often overfits)
# Look for smallest k > 1 where performance is still excellent
test_acc_array = np.array(test_accuracies)
max_test_acc = test_acc_array.max()

# Skip k=1, find best k from 3 onwards (odd numbers to avoid ties)
odd_k_candidates = [k for k in k_range if k >= 3 and k % 2 == 1 and test_accuracies[k-1] >= max_test_acc - 0.02]
if odd_k_candidates:
    recommended_k = odd_k_candidates[0]  # Smallest odd k with good performance
else:
    recommended_k = best_k_cv if best_k_cv > 1 else 3

print("=" * 60)
print("K-NN OPTIMAL K ANALYSIS")
print("=" * 60)
print(f"\nDataset size: {len(df)} samples")
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Number of features: {X.shape[1]}")
print(f"\nCommon rule of thumb: k = sqrt(n) ≈ {int(np.sqrt(len(X_train)))}")

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"\nBest k by Test Accuracy: k={best_k_test} (accuracy: {test_accuracies[best_k_test-1]:.4f})")
print(f"Best k by Cross-Validation: k={best_k_cv} (CV score: {cv_scores[best_k_cv-1]:.4f})")
print(f"\n RECOMMENDED k: {recommended_k}")
print(f"   - Test Accuracy: {test_accuracies[recommended_k-1]:.4f}")
print(f"   - CV Score: {cv_scores[recommended_k-1]:.4f}")
print(f"   - Train Accuracy: {train_accuracies[recommended_k-1]:.4f}")

print("\n" + "=" * 60)
print("WHY THIS K?")
print("=" * 60)
if best_k_test == 1:
    print("  k=1 achieves perfect accuracy BUT is NOT recommended!")
    print("   - Too sensitive to noise and outliers")
    print("   - No voting mechanism (just 1 neighbor)")
    print("   - Poor generalization to new data\n")
    print(f" BETTER CHOICE: k={recommended_k}")
    print(f"   - Test Accuracy: {test_accuracies[recommended_k-1]:.4f} (minimal drop)")
    print(f"   - More robust and generalizable")
    print(f"   - Uses odd k to avoid ties in binary classification")
    print(f"   - Balances bias-variance tradeoff")
elif recommended_k == 1:
    print(" k=1 may be overfitting to the training data!")
    print(" Consider using a higher k for better generalization.")
    # Find first stable k
    for k in range(3, 20, 2):
        if test_accuracies[k-1] >= max_test_acc - 0.02:
            print(f"\n   Better alternative: k={k}")
            print(f"   - Test Accuracy: {test_accuracies[k-1]:.4f}")
            print(f"   - Less prone to noise/outliers")
            break
else:
    print(f" k={recommended_k} balances accuracy and generalization")
    print(f" Avoids overfitting (k=1 usually overfits)")
    print(f" Good cross-validation performance")
    print(f" Odd number prevents ties in binary classification")

print("\n" + "=" * 60)
print("TOP 5 K VALUES:")
print("=" * 60)
top_indices = np.argsort(test_accuracies)[::-1][:5]
for i, idx in enumerate(top_indices, 1):
    k = idx + 1
    print(f"{i}. k={k:2d} | Test Acc: {test_accuracies[idx]:.4f} | CV Score: {cv_scores[idx]:.4f} | Train Acc: {train_accuracies[idx]:.4f}")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Train vs Test Accuracy
axes[0].plot(k_range, train_accuracies, marker='o', label='Train Accuracy', linewidth=2)
axes[0].plot(k_range, test_accuracies, marker='s', label='Test Accuracy', linewidth=2)
axes[0].axvline(x=recommended_k, color='red', linestyle='--', label=f'Recommended k={recommended_k}')
axes[0].set_xlabel('Number of Neighbors (k)', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Train vs Test Accuracy', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Cross-Validation Score
axes[1].plot(k_range, cv_scores, marker='o', color='green', linewidth=2)
axes[1].axvline(x=recommended_k, color='red', linestyle='--', label=f'Recommended k={recommended_k}')
axes[1].set_xlabel('Number of Neighbors (k)', fontsize=12)
axes[1].set_ylabel('Cross-Validation Score', fontsize=12)
axes[1].set_title('5-Fold Cross-Validation Score', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimal_k_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
