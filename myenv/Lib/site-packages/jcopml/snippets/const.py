from IPython.display import HTML
        

SNIPPETS = {
    "name": "J.COp ML Snippets",
    "menu": {
        "Import Common Packages": """
            import numpy as np
            import pandas as pd
            from ydata_profiling import ProfileReport
            
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            
            from jcopml.pipeline import num_pipe, cat_pipe
            from jcopml.utils import save_model, load_model, get_inferred_type_from_report
        """,
        "Import csv data": """
            df = pd.read_csv("____________", index_col="___________", parse_dates=["____________"])
            df.head()
        """,
        "Dataset Splitting": {
            "Shuffle Split": """
                X = df.drop(columns="___________")
                y = "_____________"
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train.shape, X_test.shape, y_train.shape, y_test.shape
            """,
            "Stratified Shuffle Split": """
                X = df.drop(columns="___________")
                y = "_____________"
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                X_train.shape, X_test.shape, y_train.shape, y_test.shape
            """
        },
        "Preprocessor": {
            "Common": """
                preprocessor = ColumnTransformer([
                    ('numeric', num_pipe(), ["______________"]),
                    ('categoric', cat_pipe(encoder='onehot'), ["_____________"]),
                ])            
            """,
            "Advance": """
                # Note: You could not use gsp, rsp, and bsp recommendation in advance mode
                # You should specify your own parameter grid / interval when tuning
                preprocessor = ColumnTransformer([
                    ('numeric1', num_pipe(impute='mean', poly=2, scaling='standard', transform='yeo-johnson'), ["______________"]),
                    ('numeric2', num_pipe(impute='median', poly=2, scaling='robust'), ["______________"]),
                    ('categoric1', cat_pipe(encoder='ordinal'), ["_____________"]),
                    ('categoric2', cat_pipe(encoder='onehot'), ["_____________"])
                ])            
            """
        },
        "Supervised Learning Pipeline": {
            "KNeighborsRegressor": """
                from sklearn.neighbors import KNeighborsRegressor
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('algo', KNeighborsRegressor())
                ])
            """,
            "SVR": """
                from sklearn.svm import SVR
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('algo', SVR(max_iter=500))
                ])
                """,
            "RandomForestRegressor": """
                from sklearn.ensemble import RandomForestRegressor
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('algo', RandomForestRegressor(n_jobs=-1, random_state=42))
                ])
            """,
            "XGBRegressor": """
                from xgboost import XGBRegressor
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('algo', XGBRegressor(n_jobs=-1, random_state=42))
                ])
            """,
            "LinearRegression": """
                from sklearn.linear_model import LinearRegression
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('algo', LinearRegression())
                ])
            """,
            "ElasticNet": """
                from sklearn.linear_model import ElasticNet
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('algo', ElasticNet())
                ])
            """,
            "KNeighborsClassifier": """
                from sklearn.neighbors import KNeighborsClassifier
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('algo', KNeighborsClassifier())
                ])
            """,
            "SVC": """
                from sklearn.svm import SVC
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('algo', SVC(max_iter=500))
                ])
            """,
            "RandomForestClassifier": """
                from sklearn.ensemble import RandomForestClassifier
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('algo', RandomForestClassifier(n_jobs=-1, random_state=42))
                ])
            """,
            "XGBClassifier": """
                from xgboost import XGBClassifier
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('algo', XGBClassifier(n_jobs=-1, random_state=42))
                ])
            """,
            "LogisticRegression": """
                from sklearn.linear_model import LogisticRegression
                pipeline = Pipeline([
                    ('prep', preprocessor),
                    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42))
                ])
            """
        },
        "Hyperparameter Tuning": {
            "Grid Search": """
                from sklearn.model_selection import GridSearchCV
                from jcopml.tuning import grid_search_params as gsp
                
                model = GridSearchCV(pipeline, gsp."_______________", cv="___", scoring='___', n_jobs=-1, verbose=1)
                model.fit(X_train, y_train)
                
                print(model.best_params_)
                print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
            """,
            "Randomized Search": """
                from sklearn.model_selection import RandomizedSearchCV
                from jcopml.tuning import random_search_params as rsp
                
                model = RandomizedSearchCV(pipeline, rsp."_______________", cv="___", scoring='___', n_iter="___", n_jobs=-1, verbose=1, random_state=42)
                model.fit(X_train, y_train)
                
                print(model.best_params_)
                print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
            """,
            "Bayesian Search": """
                from jcopml.tuning.skopt import BayesSearchCV
                from jcopml.tuning import bayes_search_params as bsp
                
                model = BayesSearchCV(pipeline, bsp."_______________", cv="___", scoring="__", n_iter="___", n_jobs=-1, verbose=1, random_state=42)
                model.fit(X_train, y_train)
                
                print(model.best_params_)
                print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
            """
        },
        "Save Model": {
            "Save the whole search object": """
                save_model(model, "__________.pkl")
            """,
            "Save best estimator only": """
                save_model(model.best_estimator_, "__________.pkl")
            """
        }        
    }
}

COPIED_HTML = HTML("""
    <style>
        /* Style for the box */
        .jcopml-snippet-box {
            background-color: #008000;
            color: white;
            padding: 5px 10px;
            line-height: 30px;
            border-radius: 5px;
        }

        /* Fading animation */
        .jcopml-snippet-fade-out {
            opacity: 1;
            animation: fadeOut 3s ease forwards;
        }
        @keyframes fadeOut {
            from { opacity: 1 }
            to { opacity: 0 }
        }    
    </style>
    <span class="jcopml-snippet-box jcopml-snippet-fade-out">Copied &#10003;</span>
""")
