# Import SkLearn Classes and Functions
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_val_score
# Import Ml Model and Optimizer
import optuna
from xgboost import XGBRegressor
# Import Custom Data
import sys
# Add Directory Python To Path
sys.path.append(r"K:\Projects\vscProject\Kaggle\HousePrices")
# Then Import Custom Made Preprocessed Data Sets for Housing Prices Problem.
from Script.data import data

# The Maim Function
def Main(time_out: float=None) -> optuna.Study:
    # Import Data
    X_train, y_train, X_test, y_samp = data()
    # Print Train- Test Data Shapes
    print(f"{X_train.shape = }")
    print(f"{X_test.shape  = }")
    # Form Custom Loss Function
    RMSE = make_scorer(mean_squared_error, greater_is_better=False, **{'squared': False})
    # Form Cross Validation Loop
    rKF = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2021)
    # Form Objective Function
    def Objective(trial):
        param_dict = {
            'n_estimators'     : trial.suggest_int('n_estimators', 10, 1000),
            'learning_rate'    : trial.suggest_float('learning_rate ', 1e-3, 10, log=True),
            'gamma'            : trial.suggest_float('gamma', 1e-5, 1, log=True),
            'subsample'        : trial.suggest_float('subsample', 0.2, 1),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.2, 1),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 0, 100),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 0, 100),
            'use_label_encoder': False,
            'random_state'     : 2021,
            'verbosity'        : 1,
        }
        regrModel = XGBRegressor( **param_dict )
        SCORE = cross_val_score(regrModel, X_train, y_train, scoring=RMSE, cv=rKF, n_jobs=None, verbose=0).mean()
        return SCORE
    # Form Optuna Study:
    Study = optuna.create_study(direction='maximize')
    # Perform Otimization search
    time_out = time_out*36000 if time_out else time_out
    Study.optimize(Objective, n_trials=5_000, timeout=time_out)
    return Study

if __name__ == '__main__':
    time_out_in_hours = float(input("The Maximun Number of Hours this Script is allowed to Run = "))
    output = Main(time_out_in_hours)
    print(output.best_params)
