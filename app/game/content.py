"""Static game content: missions, achievements, quiz bank and theory topics."""
from src import content as C

XP_GOAL = 1000
QUIZ_LENGTH = 5
TITLES = {1: "ML Rookie", 2: "ML Apprentice", 3: "ML Learner", 4: "ML Strategist", 5: "ML Expert", 6: "ML Ninja"}

MISSIONS = [
    dict(id=1, title="Prediction Rookie", sub="Accuracy & Basics", reward=100, icon="🎮"),
    dict(id=2, title="Random Forest Squad", sub="Bagging & Variance", reward=150, icon="🌲"),
    dict(id=3, title="Boosting Challenge", sub="Focus on Mistakes", reward=200, icon="🚀"),
    dict(id=4, title="Voting Arena", sub="Hard & Soft Voting", reward=200, icon="🗳️"),
    dict(id=5, title="Stacking Mastership", sub="Meta Learner", reward=250, icon="🧠"),
    dict(id=6, title="Final Championship", sub="Pick the Winner", reward=300, icon="🏆"),
]

ACHIEVEMENTS = [
    dict(id="first_step", title="First Step", desc="Complete Level 1", xp=100, icon="👣"),
    dict(id="forest_explorer", title="Forest Explorer", desc="Complete Level 2 (Random Forest Squad)", xp=200, icon="🌲"),
    dict(id="boost_hero", title="Boost Hero", desc="Complete Level 3 (Boosting Challenge)", xp=200, icon="🚀"),
    dict(id="voting_veteran", title="Voting Veteran", desc="Complete Level 4 (Voting Arena)", xp=200, icon="🗳️"),
    dict(id="stack_master", title="Stack Master", desc="Complete Level 5 (Stacking Mastership)", xp=250, icon="🧠"),
    dict(id="champion", title="Champion", desc="Win the Final Championship", xp=300, icon="🏆"),
    dict(id="quiz_master", title="Quiz Master", desc="Score 4/5 or better in a quiz", xp=150, icon="🎓"),
    dict(id="perfect_score", title="Perfect Score", desc="Score 5/5 in a quiz", xp=200, icon="💯"),
    dict(id="streak_starter", title="Streak Starter", desc="Get 3 correct answers in a row", xp=100, icon="🔥"),
    dict(id="hot_streak", title="On Fire", desc="Get 5 correct answers in a row", xp=150, icon="⚡"),
    dict(id="theory_reader", title="Bookworm", desc="Read 6 topics in the Theory Library", xp=100, icon="📚"),
    dict(id="experimenter", title="Experimenter", desc="Train models 5 times", xp=100, icon="🧪"),
]

# Demo players for the (local) leaderboard
DEMO_PLAYERS = [("Priya", 1250, 6), ("Rahul", 980, 5), ("Neha", 360, 3), ("Aman", 280, 2)]

# id, question, options, correct index, explanation
QUIZ = [
    (1, "Three classifiers predict Class A, Class A, and Class B. What is the Hard Voting prediction?",
     ["Class A", "Class B", "Both A and B", "Depends on probabilities"], 0,
     "In Hard Voting, the class with the majority of votes wins. Here, Class A gets 2 votes, so the final prediction is Class A."),
    (2, "What does Soft Voting average?", ["Class labels", "Predicted probabilities", "Input features", "Training times"], 1,
     "Soft voting averages each model's predicted class probabilities and picks the class with the highest average."),
    (3, "Bagging mainly reduces…", ["Bias", "Variance", "The number of features", "The size of the data"], 1,
     "Averaging many models trained on different bootstrap samples smooths out their individual noise (variance)."),
    (4, "How are the models trained in boosting?", ["In parallel on bootstrap samples", "Independently on different features",
     "Sequentially, each focusing on the previous errors", "Only on the test set"], 2,
     "Boosting adds models one after another; each new model concentrates on what earlier models got wrong."),
    (5, "A bootstrap sample is drawn…", ["Without replacement", "From the test set only", "Sorted by label", "With replacement"], 3,
     "Rows are drawn with replacement, so some rows repeat and ~36.8% are left out (out-of-bag)."),
    (6, "Out-of-bag (OOB) samples are…", ["Rows a particular tree did not train on", "Rows with missing values", "Outliers removed from the data",
     "The test set"], 0, "Each tree misses about 36.8% of rows. Predicting them with the trees that never saw them gives a free validation score."),
    (7, "What extra randomness does Random Forest add to bagging?", ["Random labels", "A random subset of features at each split",
     "Random learning rates", "Random test sets"], 1, "Feature subsampling makes the trees less alike, so averaging them removes more variance."),
    (8, "Training accuracy 100% but test accuracy 80% suggests…", ["Underfitting", "Perfect generalisation", "Data leakage in the test set",
     "Overfitting"], 3, "A big gap between training and test performance means the model memorised noise."),
    (9, "A depth-1 tree (stump) on curved data has…", ["High variance", "High bias", "Zero error", "No bias and no variance"], 1,
     "A model that is too simple misses the pattern (high bias). Boosting stacks many stumps to fix this."),
    (10, "AdaBoost increases the weight of…", ["Correctly classified samples", "Misclassified samples", "Random samples", "Test samples"], 1,
     "Misclassified samples get larger weights so the next weak learner focuses on them."),
    (11, "Gradient Boosting fits each new tree to…", ["The original labels", "Random noise", "The errors (gradient of the loss) of the current ensemble",
     "The test set"], 2, "Each new tree corrects what the current ensemble still gets wrong, scaled by the learning rate."),
    (12, "A smaller learning rate in boosting usually needs…", ["Fewer estimators", "More estimators", "No estimators", "A bigger dataset"], 1,
     "Smaller steps are safer but you need more of them - the learning-rate / n_estimators trade-off."),
    (13, "Why does stacking use out-of-fold predictions?", ["To make training faster", "To avoid leakage - base models predict rows they have not seen",
     "To increase the number of features", "To remove outliers"], 1,
     "If base models predicted rows they trained on, their outputs would look too good and the meta-model would over-trust them."),
    (14, "What does the meta-model in stacking learn?", ["Which features to drop", "How to combine the base models' predictions",
     "The best random seed", "How to scale the data"], 1, "Unlike voting's fixed rule, the meta-model learns how much to trust each base model."),
    (15, "Is an ensemble guaranteed to beat a single model?", ["Yes, always", "Yes, if it has 100+ models", "No - it often helps but not always", "Only for neural networks"], 2,
     "On small, easy or very noisy data the gain can be tiny or negative. Always compare on the same split."),
    (16, "Where should a feature scaler be fitted?", ["On the full dataset", "On the test set", "On the training data only", "After predicting"], 2,
     "Fitting on test data leaks information. Put the scaler inside a Pipeline so it only sees training data."),
    (17, "Recall answers which question?", ["Of everything predicted positive, how much was right?", "Of all real positives, how many did we find?",
     "How fast is the model?", "How many features were used?"], 1, "Recall = true positives / (true positives + false negatives)."),
    (18, "Ensembles work best when their members are…", ["Identical", "Diverse (make different mistakes)", "Very small", "Trained on the test set"], 1,
     "If all models make the same mistake, combining them changes nothing. Diversity is the key ingredient."),
]

_RF_MD = """
A **Random Forest** is bagging of decision trees **plus** a random subset of features at every split.
Each tree votes and the majority wins. More trees → more stable predictions (accuracy plateaus rather than overfitting).
"""
_ADA_MD = """
**AdaBoost** trains weak learners (often depth-1 stumps) one after another. After each round it **raises the weights of the samples
that were misclassified**, so the next learner concentrates on them. Learners with lower error get a bigger say in the final weighted vote.
"""
_GB_MD = """
**Gradient Boosting** builds trees sequentially; each new tree is fitted to the **errors (gradient of the loss)** of the current ensemble and
added with a small step - the **learning rate**. It is flexible (many loss functions) and is the basis of XGBoost and LightGBM.
`HistGradientBoosting` is a fast version that bins features into histograms.
"""
_CON = dict(C.CONCEPTS)

# topic -> (markdown, optional graphviz diagram)
THEORY = {
    "Ensemble Learning": (C.INTRO_MD + "\n**Real-world examples:** fraud detection, medical diagnosis, recommendation systems.", C.DOT_SINGLE_VS_ENSEMBLE),
    "Bias and Variance": (_CON["Bias vs. variance"], None),
    "Overfitting & Underfitting": (_CON["Overfitting and underfitting"], None),
    "Bagging": (C.BAGGING_MD, C.DOT_BAGGING),
    "Random Forest": (_RF_MD + "\n" + _CON["Random Forest vs. Decision Tree"], None),
    "Boosting": (C.BOOSTING_MD + "\n" + _CON["Bagging vs. boosting"], C.DOT_BOOSTING),
    "AdaBoost": (_ADA_MD, None),
    "Gradient Boosting": (_GB_MD + "\n" + _CON["AdaBoost vs. Gradient Boosting"], None),
    "Hard and Soft Voting": (C.VOTING_MD + "\n" + _CON["Majority voting"], C.DOT_VOTING),
    "Stacking": (C.STACKING_MD + "\n" + _CON["Stacking vs. voting"], C.DOT_STACKING),
    "Bootstrap Sampling": (_CON["Bootstrap sampling"], None),
    "Out-of-Bag Evaluation": (_CON["Out-of-bag (OOB) evaluation"], None),
    "Feature Importance": (_CON["Feature importance and its limitations"], None),
    "Interview Q&A": ("", None),
}
