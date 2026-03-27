# EAS 587 - Phase 2 Report
**Team Members:** [Fill in Names]

## 1. Algorithm 1: k-Nearest Neighbors (k-NN)

**1. Why This Algorithm? (Justification)**
- **Phase 1 Use Case**: Predicting loan default (charge-off) probability to assist lending decisions.
- **Appropriateness**: k-NN is a non-parametric method that makes predictions based on the closest historical loans in the feature space. It's excellent for capturing localized patterns where similar borrower profiles result in similar outcomes.
- **Expectation**: We expected k-NN to provide a reasonable baseline by grouping similar borrowers together, although we anticipated struggles with the high dimensionality (110 features) of our dataset due to the curse of dimensionality.

**2. How We Used It (Implementation Summary)**
- **Preprocessing**: Since k-NN is distance-based, our robust pipeline scaling (StandardScaler) was essential. High-cardinality categoricals were one-hot encoded.
- **Hyperparameters**: `n_neighbors = 20`. We chose a higher k to reduce noise and variance, considering the class imbalance.
- **Tuning/Runtime**: Computing distances for all 833,173 training rows against validation rows was too computationally expensive ($O(n \times d)$). Thus, we sampled a random subset of 20,000 rows for training to make runtime feasible.

**3. What Happened (Results)**
- **Metrics**: 
  - Valid AUC-ROC: 0.6650
  - Valid AUC-PR: 0.3171
  - KS Statistic: 0.2380
- **Interpretation**: The model performed adequately for a random subset but fell short of a robust predictive model. The ROC and PR scores indicate it is better than random guessing but struggles to confidently separate fully paid vs. charged-off loans.
- *(Include `KNN_confusion_matrix.png` and `KNN_roc_curve.png` here)*

**4. What We Learned (Insight or Failure)**
- **Insight**: k-NN is effectively a "Dead End" for this dataset at scale. The required downsampling limits the model's ability to learn from the full breadth of the data. Furthermore, distance-based metrics lose meaning in our 110-dimensional feature space.

---

## 2. Algorithm 2: Support Vector Machine (LinearSVC)

**1. Why This Algorithm? (Justification)**
- **Phase 1 Use Case**: Identifying high-risk loans.
- **Appropriateness**: Linear SVMs try to find the optimal hyperplane to separate classes. Given the large dataset size, a Linear SVM is much more scalable than kernel-based SVMs. 
- **Expectation**: We expected it to perform competitively with linear models by maximizing the margin between default and non-default classes.

**2. How We Used It (Implementation Summary)**
- **Preprocessing**: Required strictly scaled data.
- **Hyperparameters**: `max_iter=2000`, `class_weight="balanced"`. We wrapped the LinearSVC in a `CalibratedClassifierCV(cv=3)` specifically to extract probability estimates, which are necessary for AUC-ROC calculation.
- **Runtime**: Completed training in ~112 seconds on the full 833k row dataset.

**3. What Happened (Results)**
- **Metrics**: 
  - Valid AUC-ROC: 0.7118
  - Valid AUC-PR: 0.3758
  - KS Statistic: 0.3057
- **Interpretation**: A significant jump in performance compared to k-NN. The SVM successfully utilized the full dataset and established a strong linear decision boundary.
- *(Include `SVM_confusion_matrix.png` and `SVM_roc_curve.png` here)*

**4. What We Learned (Insight or Failure)**
- **Insight**: The Linear SVM is a highly effective, fast-training algorithm for large, sparse datasets. However, because we only used a linear kernel, it could not capture non-linear relationships, slightly capping its potential.

---

## 3. Algorithm 3: Logistic Regression (Baseline)

**1. Why This Algorithm? (Justification)**
- **Phase 1 Use Case**: Explainable credit scoring system.
- **Appropriateness**: Logistic Regression provides explicit coefficients for each feature, establishing exactly how much every variable (like interest rate or DTI) contributes to the probability of default.
- **Expectation**: We expected it to act as our primary interpretability benchmark against the more complex tree models.

**2. How We Used It (Implementation Summary)**
- **Preprocessing**: Handled heavily skewed variables with log transformations (log1p) prior to scaling to satisfy linear assumptions.
- **Hyperparameters**: Used standard L2 penalty with `max_iter=1000` and `class_weight="balanced"`.
- **Runtime**: Extremely fast. Fit the entire dataset in ~18 seconds.

**3. What Happened (Results)**
- **Metrics**: 
  - Valid AUC-ROC: 0.7121
  - Valid AUC-PR: 0.3757
  - KS Statistic: 0.3061
- **Interpretation**: Logistic regression achieved nearly identical performance to the SVM but in a fraction of the time. This indicates that the linearly separable signal in the dataset peaks around an AUC-ROC of 0.71.
- *(Include `LogisticRegression_confusion_matrix.png` and `LogisticRegression_roc_curve.png` here)*

**4. What We Learned (Insight or Failure)**
- **Insight**: The features have strong linear relationships with the target. A complex model must beat 0.71 AUC-ROC to justify its lack of interpretability and longer training times.

---

## 4. Algorithm 4: Decision Tree

**1. Why This Algorithm? (Justification)**
- **Phase 1 Use Case**: Automated lending rule creation.
- **Appropriateness**: Decision trees naturally handle non-linearities and interactions between features without requiring heavy scaling or transformations. 
- **Expectation**: We expected an un-tuned tree to heavily overfit, but a tuned tree to provide decent non-linear splits.

**2. How We Used It (Implementation Summary)**
- **Tuning**: We utilized Optuna for Bayesian Optimization over 50 trials (taking ~22 minutes). 
- **Hyperparameters**: Best params found were `max_depth=9`, `min_samples_split=80`, `min_samples_leaf=50`, `criterion='entropy'`.

**3. What Happened (Results)**
- **Metrics**: 
  - Valid AUC-ROC: 0.7010
  - Valid AUC-PR: 0.3572
  - KS Statistic: 0.2896
- **Interpretation**: The untuned tree (depth=72, 121k leaves) drastically overfit with a validation AUC-ROC of just 0.5490. Tuning rescued the model, but a single decision tree still slightly underperformed our linear baselines.
- *(Include `DecisionTree_confusion_matrix.png` and `DecisionTree_roc_curve.png` here)*

**4. What We Learned (Insight or Failure)**
- **Insight**: This was a great learning experience in overfitting. A single decision tree lacks the robustness to accurately model financial risk across 800,000 individual loans without ensembling.

---

## 5. Algorithm 5: XGBoost (Outside Algorithm)

**1. Why This Algorithm? (Justification)**
- **Phase 1 Use Case**: Maximizing predictive accuracy for institutional risk management.
- **Appropriateness**: XGBoost is state-of-the-art for tabular data, utilizing gradient-boosted decision trees to progressively correct the errors of preceding trees.
- **Expectation**: We expected XGBoost to be our best-performing model by capturing complex non-linear combinations of credit features.

**2. How We Used It (Implementation Summary)**
- **Tuning**: Optimized using Optuna for 50 trials. We set `scale_pos_weight = 4.19` to handle the heavy class imbalance dynamically during training.
- **Hyperparameters**: `n_estimators=468`, `max_depth=7`, `learning_rate=0.074`. 

**3. What Happened (Results)**
- **Metrics**: 
  - Valid AUC-ROC: 0.7240
  - Valid AUC-PR: 0.3928
  - KS Statistic: 0.3244
- **Interpretation**: XGBoost produced our best overall result. The gradient boosting significantly outperformed the single decision tree and the linear baseline. Test set evaluation yielded a Recall of 66.48%.
- *(Include `XGBoost_confusion_matrix.png`, `XGBoost_roc_curve.png`, and `XGBoost_feature_importance.png` here)*

**4. What We Learned (Insight or Failure)**
- **Insight**: Boosting is highly effective on our dataset. The feature importances revealed that interest rate (`int_rate`) and term length (`term`) were overwhelmingly the most predictive factors of loan default.

---

## 6. Algorithm 6: HistGradientBoosting (Outside Algorithm)

**1. Why This Algorithm? (Justification)**
- **Phase 1 Use Case**: High-performance modeling on large data.
- **Appropriateness**: HistGradientBoosting (HGB) is Scikit-Learn’s answer to LightGBM. It bins features into integer-valued buckets, drastically speeding up the splitting process for datasets >10,000 samples.
- **Expectation**: We expected similar performance to XGBoost but with potentially faster training times.

**2. How We Used It (Implementation Summary)**
- **Tuning**: Optuna optimization (50 trials). Tuning took significantly longer than XGBoost (~66 minutes total).
- **Hyperparameters**: `max_iter=495`, `max_depth=10`, `learning_rate=0.031`, `max_leaf_nodes=103`.

**3. What Happened (Results)**
- **Metrics**: 
  - Valid AUC-ROC: 0.7233
  - Valid AUC-PR: 0.3933
  - KS Statistic: 0.3222
- **Interpretation**: HGB matched XGBoost almost identically in predictive power (0.7233 vs 0.7240), cementing the ~0.724 range as the maximum extractable signal from our subset of LendingClub features.
- *(Include `HistGradientBoosting_confusion_matrix.png` and `HistGradientBoosting_roc_curve.png` here)*

**4. What We Learned (Insight or Failure)**
- **Insight**: While theoretically faster, the Scikit-Learn implementation of HGB proved to be significantly slower to tune and train than highly optimized external libraries like XGBoost (3985s vs 1320s for tuning).

---

## Algorithm Comparison
We compared our Baseline Logistic Regression, Single Decision Tree, and XGBoost models. 
1. **Performance**: XGBoost was the clear winner (AUC-ROC: 0.7240) because it could model non-linear interactions across variables, unlike Logistic Regression (0.7121). A single Decision Tree (0.7010) proved too unstable.
2. **Trade-offs**: Logistic regression was incredibly fast (18s) and interpretable, making it ideal for regulatory compliance. XGBoost took nearly two orders of magnitude longer to tune and train and operates as a "black box," but it compensates with superior recall for catching defaults, which saves institutions money.
3. **Conclusion**: If pure predictive power is the goal, XGBoost wins. However, Logistic Regression captured 98% of the signal in 1% of the time, making it an excellent MVP.

## Dead Ends
1. **k-Nearest Neighbors Runtime Constraint**: We attempted to run k-NN on the full dataset, but quickly realized building distance matrices for 833,173 rows of 110 dimensions would take days to process. We learned that distance-based, lazy-evaluated models are fundamentally incompatible with high-dimensional big data.
2. **Untuned Decision Tree Overfitting**: Our base Decision Trees severely overfit the training data. The tree grew to a depth of 72 with 121,000 leaves, memorizing the training set but scoring an abysmal 0.549 AUC-ROC on the validation set. We learned that strict regularization (`max_depth`, `min_samples_leaf`) is strictly mandatory for tree-based models on complex data.

---

## Code Verification
*Verification Statement:* We conducted a "dry run" of our entire pipeline on a clean terminal instance. Using our customized `generate_visualizations.py` and modular pipeline, the code executes completely without manual intervention. Random seeds (`42`) were fixed across all tuning, splitting, and training files to ensure reproducibility.
