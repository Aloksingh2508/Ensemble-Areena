"""All teaching text lives here so pages stay short and easy to edit."""

_STYLE = 'rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontcolor="#111111"]; edge [color="#888888"];'

DOT_SINGLE_VS_ENSEMBLE = f"""digraph G {{ {_STYLE}
  subgraph cluster_a {{ label="Single model"; fontcolor="#888888"; color="#888888";
    D1 [label="Data", fillcolor="#E8E8E8"]; M [label="One model", fillcolor="#F6C6C6"]; P1 [label="Prediction", fillcolor="#CDE8CD"]; D1 -> M -> P1; }}
  subgraph cluster_b {{ label="Ensemble"; fontcolor="#888888"; color="#888888";
    D2 [label="Data", fillcolor="#E8E8E8"];
    A [label="Model A", fillcolor="#CFE0F5"]; B [label="Model B", fillcolor="#CFE0F5"]; C [label="Model C", fillcolor="#CFE0F5"];
    V [label="Combine\\n(vote / average /\\nmeta-model)", fillcolor="#FBE3B5"]; P2 [label="Final prediction", fillcolor="#CDE8CD"];
    D2 -> A; D2 -> B; D2 -> C; A -> V; B -> V; C -> V; V -> P2; }}
}}"""

DOT_VOTING = f"""digraph G {{ {_STYLE}
  X [label="New sample", fillcolor="#E8E8E8"];
  A [label="Logistic Reg.\\n→ Class 1", fillcolor="#CFE0F5"]; B [label="Decision Tree\\n→ Class 0", fillcolor="#CFE0F5"]; C [label="KNN\\n→ Class 1", fillcolor="#CFE0F5"];
  V [label="Majority vote\\n(2 vs 1)", fillcolor="#FBE3B5"]; F [label="Class 1", fillcolor="#CDE8CD"];
  X -> A; X -> B; X -> C; A -> V; B -> V; C -> V; V -> F; }}"""

DOT_BAGGING = f"""digraph G {{ {_STYLE}
  D [label="Training set", fillcolor="#E8E8E8"];
  S1 [label="Bootstrap\\nsample 1", fillcolor="#FBE3B5"]; S2 [label="Bootstrap\\nsample 2", fillcolor="#FBE3B5"]; S3 [label="Bootstrap\\nsample 3", fillcolor="#FBE3B5"];
  T1 [label="Tree 1", fillcolor="#CFE0F5"]; T2 [label="Tree 2", fillcolor="#CFE0F5"]; T3 [label="Tree 3", fillcolor="#CFE0F5"];
  V [label="Vote / average", fillcolor="#FBE3B5"]; F [label="Prediction", fillcolor="#CDE8CD"];
  D -> S1 -> T1 -> V; D -> S2 -> T2 -> V; D -> S3 -> T3 -> V; V -> F; }}"""

DOT_BOOSTING = f"""digraph G {{ {_STYLE}
  D [label="Training set\\n(equal weights)", fillcolor="#E8E8E8"];
  M1 [label="Weak model 1", fillcolor="#CFE0F5"]; W1 [label="Increase weight of\\nmistakes", fillcolor="#F6C6C6"];
  M2 [label="Weak model 2", fillcolor="#CFE0F5"]; W2 [label="Increase weight of\\nmistakes", fillcolor="#F6C6C6"];
  M3 [label="Weak model 3", fillcolor="#CFE0F5"]; F [label="Weighted sum\\nof all models", fillcolor="#CDE8CD"];
  D -> M1 -> W1 -> M2 -> W2 -> M3 -> F; }}"""

DOT_STACKING = f"""digraph G {{ {_STYLE}
  D [label="Training set", fillcolor="#E8E8E8"];
  A [label="Decision Tree", fillcolor="#CFE0F5"]; B [label="Logistic Reg.", fillcolor="#CFE0F5"]; C [label="Random Forest", fillcolor="#CFE0F5"];
  O [label="Out-of-fold\\npredictions\\n(new features)", fillcolor="#FBE3B5"]; M [label="Meta-model\\n(Logistic Regression)", fillcolor="#F6C6C6"]; F [label="Final prediction", fillcolor="#CDE8CD"];
  D -> A; D -> B; D -> C; A -> O; B -> O; C -> O; O -> M -> F; }}"""

INTRO_MD = """
### What is ensemble learning?
Ensemble learning trains **several models** and **combines their answers** into one prediction.
Think of asking a panel of doctors instead of one: individual doctors make different mistakes, and when
they are combined, many of those mistakes cancel out.

### Why do we use it?
- **Lower variance** - averaging many unstable models (e.g. deep trees) gives a steadier result (*bagging*).
- **Lower bias** - many simple models added one after another can learn complex patterns (*boosting*).
- **Best of several worlds** - different model families make different mistakes (*voting, stacking*).

The key ingredient is **diversity**: if every model makes the *same* mistake, combining them changes nothing.
"""

SINGLE_VS_ENSEMBLE = [
    ("Training", "One model", "Many models + a combination rule"),
    ("Stability", "Can change a lot if the data changes slightly", "Usually more stable"),
    ("Accuracy", "Good baseline", "Often better, **not guaranteed**"),
    ("Interpretability", "Easier to explain (e.g. one tree)", "Harder to explain"),
    ("Speed / memory", "Fast and small", "Slower and larger (N models)"),
]

PROS = ["Often higher and more stable accuracy", "Reduces overfitting (bagging) or underfitting (boosting)",
        "Works well on tabular data", "Many flexible, ready-made implementations"]
CONS = ["Slower to train and predict", "Harder to interpret", "More hyper-parameters to tune",
        "May give little or no gain if the base model is already strong or the data is tiny/noisy"]
APPLICATIONS = [
    ("Finance", "Credit scoring and fraud detection (gradient boosting is very common)."),
    ("Healthcare", "Disease risk prediction from lab results - e.g. the Breast Cancer dataset in this app."),
    ("E-commerce", "Recommendation and click-through-rate prediction."),
    ("Competitions", "Most winning tabular-data solutions on Kaggle use boosting or stacking."),
    ("Cyber-security", "Malware and intrusion detection with random forests."),
]

VOTING_MD = """
**Voting** trains several *different* models on the **same** data and lets them vote.

| | Hard voting | Soft voting |
|---|---|---|
| Each model gives | a class label | class probabilities |
| Final answer | the label with most votes | the class with the highest **average probability** |
| Needs `predict_proba`? | No | Yes |
| Good when | models are roughly equally good | models are well-calibrated; confident models should count more |

*Example:* three models predict P(class 1) = 0.9, 0.4, 0.4. Hard voting sees labels 1, 0, 0 → **class 0**.
Soft voting averages to 0.57 → **class 1**. The two rules can disagree!
"""

BAGGING_MD = """
**Bagging = Bootstrap AGGregatING.**
1. Draw many **bootstrap samples** (random rows *with replacement*, same size as the training set).
2. Train one model on each sample - this can be done **in parallel**.
3. Combine by majority vote (classification) or average (regression).

Because each tree sees slightly different data, each makes different mistakes. Averaging them **reduces variance**
(overfitting to noise) without increasing bias much. Each bootstrap sample contains about **63.2%** of the unique rows;
the other ~36.8% are **out-of-bag (OOB)** and can be used as a free validation set.

**Random Forest** = bagging of decision trees **plus** a random subset of features considered at every split.
That extra randomness makes the trees even more different from each other.
"""

BOOSTING_MD = """
**Boosting** trains models **one after another**; each new model focuses on what the previous ones got wrong.

- **AdaBoost** - raises the *weights* of misclassified samples, so the next weak learner (often a depth-1 "stump")
  concentrates on them. Models with lower error get a larger say in the final vote.
- **Gradient Boosting** - each new tree is fitted to the *errors (gradients of the loss)* of the current ensemble,
  then added with a small step called the **learning rate**.
- **HistGradientBoosting** - gradient boosting that first bins features into histograms; much faster on large data.
- **XGBoost** (concept) - an engineered gradient-boosting library with regularisation, second-order gradient
  information, built-in handling of missing values and parallel tree building. It is used in this app only if
  `pip install xgboost` has been run.

Boosting mainly **reduces bias**, but with too many iterations or a high learning rate it can overfit - watch the
gap between the train and test curves.
"""

STACKING_MD = """
**Stacking** learns *how* to combine models instead of using a fixed rule.
1. Train several **base models** (here: Decision Tree, Logistic Regression, Random Forest).
2. For every training row, collect each base model's prediction made **while that row was held out** -
   these are **out-of-fold (OOF) predictions** from cross-validation.
3. Train a **meta-model** (here: Logistic Regression) on those OOF predictions to produce the final answer.

Why out-of-fold? If a base model predicted rows it had already *seen*, its predictions would look unrealistically
good and the meta-model would learn to over-trust it (**leakage**).
"""

CONCEPTS = [
    ("Bias vs. variance", """
**Bias** = error from a model being too simple (it misses real patterns). **Variance** = error from a model being too
sensitive to the particular training sample (it chases noise). Total error ≈ bias² + variance + irreducible noise.
- Deep decision tree: low bias, **high variance**. → *Bagging* lowers variance.
- Decision stump: **high bias**, low variance. → *Boosting* lowers bias.
"""),
    ("Overfitting and underfitting", """
- **Underfitting**: poor on both training and test data (model too simple).
- **Overfitting**: great on training data, noticeably worse on test data (model memorised noise).
- Diagnose by comparing train vs. test scores - the *Boosting curves* in the playground show this gap directly.
"""),
    ("Bootstrap sampling", """
Draw *n* rows from *n* rows **with replacement**. Chance a given row is never picked = (1 − 1/n)ⁿ → 1/e ≈ **36.8%**,
so a bootstrap sample holds ≈ **63.2%** unique rows (some repeated). Try it in the Bagging tab.
"""),
    ("Majority voting", """
Each model casts one vote for a class; the class with the most votes wins. With *independent* voters that are each
better than chance, accuracy grows as you add voters (see the slider demo in *Introduction*). Real models are correlated,
so the gain is smaller.
"""),
    ("Hard voting vs. soft voting", """
Hard = count labels. Soft = average probabilities (so a confident model counts more than an unsure one). Soft voting
usually performs equal or better when probabilities are reasonable, but needs `predict_proba`.
"""),
    ("Bagging vs. boosting", """
| | Bagging | Boosting |
|---|---|---|
| Training | independent, parallel | sequential, each model depends on the last |
| Base model | strong / high-variance (deep trees) | weak / high-bias (stumps, shallow trees) |
| Main effect | reduces **variance** | reduces **bias** |
| Overfitting with more models | rarely | possible |
| Noise sensitivity | robust | can chase noisy labels |
"""),
    ("Random Forest vs. Decision Tree", """
A single tree is easy to read but unstable. A forest averages many trees trained on bootstrap samples with random
feature subsets → better accuracy and stability, at the cost of interpretability and speed.
"""),
    ("AdaBoost vs. Gradient Boosting", """
| | AdaBoost | Gradient Boosting |
|---|---|---|
| How it focuses on errors | re-weights **samples** | fits new trees to the **gradient (residual errors)** of a loss |
| Weak learner | usually stumps | shallow trees (depth 3-6) |
| Loss | exponential-loss-like | any differentiable loss |
| Flexibility | lower | higher (basis of XGBoost, LightGBM...) |
"""),
    ("Stacking vs. voting", """
Voting uses a **fixed** rule (count or average). Stacking **learns** the rule with a meta-model, so it can trust
good models more and ignore weak ones - but it needs more data and cross-validation, and is easier to overfit/leak.
"""),
    ("Out-of-bag (OOB) evaluation", """
Each tree in a bagged ensemble never saw ≈ 36.8% of the rows. Predict every row using only the trees that did *not*
train on it and you get a validation score **without a separate validation set**. Use the demo below to compare it
with a real test score.
"""),
    ("Feature importance and its limitations", """
Tree ensembles report how much each feature reduced impurity. Limitations:
- Biased toward features with many distinct values.
- **Correlated features share** (and hide) importance.
- Computed on **training** data, so it can rank features that only help overfit.
- Says *how much* a feature is used, not *in which direction* or *why*.

**Permutation importance** (shuffle one feature on the test set and measure the accuracy drop) is more reliable and
works for any model - the playground can show both.
"""),
]

# (category, question, answer)
INTERVIEW_QA = [
    ("Basics", "What is ensemble learning?",
     "Combining several models so the group predicts better than a typical single model. Example: 100 trees voting in a Random Forest."),
    ("Basics", "Why do ensembles often work better?",
     "Different models make different errors; combining them cancels some of those errors (lower variance) or fixes systematic ones (lower bias). This needs *diversity* - identical models add nothing."),
    ("Basics", "Explain bias and variance with an example.",
     "A depth-1 tree is too simple for curved data (high bias). A fully grown tree memorises noise and changes a lot with new data (high variance). Ensembles target one or the other."),
    ("Basics", "What is overfitting and how do you detect it?",
     "The model fits training noise. Detect it by a large gap between training and validation/test score, e.g. 100% train vs 85% test."),
    ("Bagging", "What is bootstrap sampling?",
     "Sampling n rows from n rows with replacement. About 63.2% of rows appear at least once; the rest are out-of-bag."),
    ("Bagging", "How does Random Forest differ from plain bagging of trees?",
     "A forest also picks a random subset of features at each split. This decorrelates the trees, so averaging them reduces variance more."),
    ("Bagging", "What is the OOB score?",
     "Accuracy measured on each row using only trees that did not see that row in training - an almost free validation estimate (`oob_score=True`)."),
    ("Bagging", "Does adding more trees to a Random Forest cause overfitting?",
     "Generally no - accuracy plateaus. It only costs more time and memory. (Individual deep trees can still overfit.)"),
    ("Voting", "Hard vs. soft voting?",
     "Hard counts predicted labels; soft averages predicted probabilities. Soft can let one confident model outweigh two unsure ones, and needs `predict_proba`."),
    ("Boosting", "How does AdaBoost work?",
     "Train a weak learner, increase the weights of the samples it misclassified, train the next learner on the re-weighted data, and finally combine all learners using weights based on their accuracy."),
    ("Boosting", "AdaBoost vs. Gradient Boosting?",
     "AdaBoost re-weights samples; Gradient Boosting fits each new tree to the gradient (residuals) of a loss function, which allows many loss functions and is more flexible."),
    ("Boosting", "What does the learning rate do in boosting?",
     "It scales each model's contribution. Small values need more iterations but usually generalise better - there is a trade-off between learning rate and number of estimators."),
    ("Boosting", "Why can boosting overfit but bagging usually doesn't?",
     "Boosting keeps chasing remaining errors, including noisy labels. Bagging averages independent models, so extra models mainly smooth predictions."),
    ("Boosting", "How is XGBoost different from sklearn's Gradient Boosting?",
     "XGBoost adds regularisation on tree complexity, uses second-order gradient information, handles missing values natively, and is heavily optimised (parallelism, cache use)."),
    ("Stacking", "Stacking vs. voting?",
     "Voting uses a fixed rule. Stacking trains a meta-model on base-model predictions to learn the best combination."),
    ("Stacking", "Why does stacking use out-of-fold predictions?",
     "Base models must predict rows they did not train on; otherwise their predictions are over-optimistic and the meta-model leaks information and overfits."),
    ("Practice", "Can an ensemble be worse than a single model?",
     "Yes. If base models are highly correlated, the data is tiny, or a single model is already near the noise ceiling, the gain can be zero - or the ensemble may overfit. Always compare on the same split/CV."),
    ("Practice", "What are the limits of feature importance?",
     "Biased toward high-cardinality features, split between correlated features, computed from training data, and non-causal. Prefer permutation importance on held-out data for a sanity check."),
    ("Practice", "How do you avoid data leakage when scaling features?",
     "Fit the scaler on training data only. Put it in a `Pipeline` so cross-validation re-fits it inside every fold. This app does exactly that."),
    ("Practice", "Which ensembles can be trained in parallel?",
     "Bagging and Random Forest (independent models). Classic boosting is sequential, though the work inside each tree can be parallelised."),
]
