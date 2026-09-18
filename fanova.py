import pandas as pd
import optuna
from optuna.importance import FanovaImportanceEvaluator

METRIC = "goal_rate"


def run_fanova(df, param_names, title):
    study = optuna.create_study(direction = "maximize")

    for _, row in df.iterrows():
        trial = optuna.trial.create_trial(
            params = {p: row[p] for p in param_names},
            distributions = {
                p: optuna.distributions.CategoricalDistribution(df[p].unique().tolist())
                for p in param_names
            },
            value=row[METRIC],
        )
        study.add_trial(trial)

    evaluator = FanovaImportanceEvaluator(seed = 0)
    importances = optuna.importance.get_param_importances(study, evaluator = evaluator)

    print(f"\n{title}")
    for param, importance in importances.items():
        print(f"{param}: {importance:.4f}")


mppo_df = pd.read_csv("results/gridsearch_eval/mppo_results.csv")
mppo_params = ["learning_rate", "n_steps", "ent_coef", "clip_range"]
run_fanova(mppo_df, mppo_params, "MPPO - fANOVA-Importance")

ppo_df = pd.read_csv("results/gridsearch_eval/ppo_results.csv")
ppo_params = ["learning_rate", "n_steps", "ent_coef", "clip_range"]
ppo_df = ppo_df[ppo_df["learning_rate"] < 0.1]
run_fanova(ppo_df, ppo_params, "PPO (learning_rate < 0.1) - fANOVA-Importance")
