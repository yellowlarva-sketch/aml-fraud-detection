# === Imports ===
import os
import math
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from optbinning import BinningProcess
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp

# === Settings ===
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 999

# === BinningModelFunction Class ===
class BinningModelFunction:
    def __init__(self, X_train, y_train, features, special_codes, search_space, model_dir):
        self.X_train = X_train
        self.y_train = y_train
        self.features = features
        self.special_codes = special_codes
        self.search_space = search_space
        self.model_dir = model_dir

    # === Internal Utilities ===
    def _apply_binning(self, params):
        bp = BinningProcess(
            variable_names=self.features,
            selection_criteria={"iv": {"min": params["min_iv"]}},
            max_n_bins=params["max_n_bins"],
            min_n_bins=params["min_n_bins"],
            special_codes=self.special_codes,
            binning_fit_params={
                var: {"monotonic_trend": params["monotonic_trend"]}
                for var in self.features
            }
        )
        bp.fit(self.X_train, self.y_train)
        X_binned = bp.transform(self.X_train, metric=params["binned_metric"])
        return bp, X_binned

    def _apply_encoding(self, X_binned, params):
        if params['binned_metric'] == 'bins':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoder.fit(X_binned)
            X_encoded = encoder.transform(X_binned)
            encoded_feature_names = encoder.get_feature_names_out(X_binned.columns)

            X_binned = pd.DataFrame(X_encoded, columns=[
                col.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
                  .replace(' ', '_').replace(',', '_').replace("'", "").replace('(', '_').replace(')', '_')
                for col in encoded_feature_names
            ], index=X_binned.index)

            return X_binned, encoder
        return X_binned, None

    def _apply_feature_selection(self, X_binned, params):
        if params['num_features'] == 'all':
            return X_binned.copy(), list(X_binned.columns)
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=params['num_features'])
            X_selected = selector.fit_transform(X_binned, self.y_train)
            selected_mask = selector.get_support()
            selected_features = np.array(X_binned.columns)[selected_mask].tolist()
            return X_selected, selected_features

    def _load_and_transform_model_data(self, df, m_method, binned_metric):
        # Load model components
        bp = pickle.load(open(f"{self.model_dir}/{m_method}/{m_method}_binning.pkl", 'rb'))
        pipeline = pickle.load(open(f"{self.model_dir}/{m_method}/{m_method}.pkl", 'rb'))
        final_features = pd.read_csv(f"{self.model_dir}/{m_method}/{m_method}_features.csv")['0'].tolist()

        # Transform input data
        df_binned = bp.transform(df[self.features], metric=binned_metric)
        if binned_metric == 'bins':
            encoder = pickle.load(open(f"{self.model_dir}/{m_method}/{m_method}_onehotencoder.pkl", 'rb'))
            df_encoded = encoder.transform(df_binned)
            df_binned = pd.DataFrame(df_encoded, columns=final_features, index=df_binned.index)

            df_binned.columns = [
                col.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
                  .replace(' ', '_').replace(',', '_').replace("'", "")
                for col in df_binned.columns
            ]

        return df_binned, pipeline, final_features

    def _get_model(self, params):
        m_method = params['m_method']
        
        if m_method == 'lr':
            return LogisticRegression(
                C=params['C'],
                penalty=params['penalty'],
                l1_ratio=params['l1_ratio'],
                solver='saga',
                random_state=42
            )
        elif m_method == 'rf':
            return RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=42
            )
        elif m_method == 'lgbm':
            return LGBMClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                verbose=-1,
                random_state=42
            )
        elif m_method == 'gbm':
            return GradientBoostingClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=42
            )
        elif m_method == 'xgb':
            return XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=42
            )

    # === Core Methods ===
    def binning_model_selection(self, params):
        bp, X_binned = self._apply_binning(params)
        X_binned, _ = self._apply_encoding(X_binned, params)
        X_selected, _ = self._apply_feature_selection(X_binned, params)

        model = self._get_model(params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_selected, self.y_train, cv=cv, scoring='roc_auc')
        loss = -scores.mean()

        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    def run_model(self, m_method, num_iterations):
        hyperopt_file = f"/{m_method}.hyperopt"
        model_path = os.path.join(self.model_dir, m_method)
        os.makedirs(model_path, exist_ok=True)

        search_space_updated = self.search_space.copy()
        search_space_updated['m_method'] = hp.choice('m_method', [m_method])

        trials = Trials()
        fmin(
            fn=self.binning_model_selection,
            space=search_space_updated,
            algo=tpe.suggest,
            max_evals=num_iterations,
            trials=trials
        )

        with open(model_path + hyperopt_file, "wb") as f:
            pickle.dump(trials, f)

    def final_model_fitting(self, params):
        m_method = params['m_method']
        bp, X_binned = self._apply_binning(params)
        X_binned, encoder = self._apply_encoding(X_binned, params)
        X_selected, selected_features = self._apply_feature_selection(X_binned, params)

        if encoder:
            with open(f"{self.model_dir}/{m_method}/{m_method}_onehotencoder.pkl", 'wb') as f:
                pickle.dump(encoder, f)

        with open(f"{self.model_dir}/{m_method}/{m_method}_binning.pkl", 'wb') as f:
            pickle.dump(bp, f)

        pd.Series(selected_features).to_csv(
            f"{self.model_dir}/{m_method}/{m_method}_features.csv", index=False
        )

        model = self._get_model(params)
        model.fit(X_selected, self.y_train)

        with open(f"{self.model_dir}/{m_method}/{m_method}.pkl", 'wb') as f:
            pickle.dump(model, f)

        return bp, model, selected_features

    def load_best_params(self, m_method):
        hyperopt_file = f"{m_method}/{m_method}.hyperopt"
        trials = pickle.load(open(os.path.join(self.model_dir, hyperopt_file), 'rb'))
        sorted_results = sorted(trials.results, key=lambda x: x['loss'])

        results = {dic['loss']: dic['params'] for dic in sorted_results}
        best_loss = list(results.keys())[0]
        best_params = results[best_loss]

        print(f'Best loss: {best_loss:.4f}')
        print(f'Best parameters: {best_params}')
        return best_params

    def variable_importance(self, best_params, final_features, pipeline):
        if best_params['m_method'] == 'lr':
            feature_importance = pd.DataFrame({
                'features': final_features,
                'Feature Importance': list(pipeline.coef_[0])
            }).sort_values('Feature Importance', ascending=True).set_index('features')
        else:
            feature_importance = pd.DataFrame(
                index=final_features,
                data=pipeline.feature_importances_,
                columns=['Feature Importance']
            ).sort_values('Feature Importance', ascending=False)

        plt.figure(figsize=(12, 10))
        colors = ['green' if val > 0 else 'red' for val in feature_importance['Feature Importance']]
        plt.barh(np.arange(len(feature_importance)), feature_importance['Feature Importance'], color=colors)
        plt.yticks(np.arange(len(feature_importance)), feature_importance.index)
        plt.show()

        feature_importance.to_csv(
            f"{self.model_dir}/{best_params['m_method']}/{best_params['m_method']}_feature_importance.csv"
        )

        return feature_importance

    def final_fitting_overall(self, model_list, df, binned_metric, target, id_col, non_predictive_cols, roc_curve_graph):
        for m_method in model_list:
            best_params = self.load_best_params(m_method)
            bp, pipeline, final_features = self.final_model_fitting(best_params)

            df_binned, pipeline, final_features = self._load_and_transform_model_data(df, m_method, binned_metric)
            df[m_method] = pipeline.predict_proba(df_binned[final_features])[:, 1]

            roc_curve_graph(df, target=target, feature=m_method, by='X_fold')
            plt.savefig(f"{self.model_dir}/{m_method}/roc_curve_all.png")

            pd.concat([df[[id_col] + non_predictive_cols], df_binned[final_features]], axis=1).to_csv(
                f"{self.model_dir}/{m_method}/{m_method}_{binned_metric}.csv", index=False
            )

            summary = bp.summary()
            summary['selected_postmodel'] = [int(i in final_features) for i in summary['name']]
            summary.to_csv(f"{self.model_dir}/{m_method}/{m_method}_iv.csv", index=False)

            self.variable_importance(best_params, final_features, pipeline)

        return df[[id_col] + non_predictive_cols + model_list]

    def volume_simulation(self, model_list, df_pop, threshold_list, transaction_limit_list, binned_metric):
        results = []

        for m_method in model_list:
            df_binned, pipeline, final_features = self._load_and_transform_model_data(df_pop, m_method, binned_metric)
            df_pop[m_method] = pipeline.predict_proba(df_binned[final_features])[:, 1]

            for thr in threshold_list:
                for limit in transaction_limit_list:
                    criteria = (
                        (df_pop[m_method] >= thr) &
                        ((df_pop['sum_all_cr_amt_1w'] + df_pop['sum_all_dr_amt_1w']) >= limit)
                    )

                    results.append({
                        'Model_name': m_method,
                        'Threshold': thr,
                        'Transaction_limit': limit,
                        'Volume': len(df_pop[criteria])
                    })

        return pd.DataFrame(results)

    def feature_consolidation(self, model_list, binned_metric):
        if binned_metric != 'bins':
            features_df = pd.DataFrame({'features': self.features})
        else:
            features_df = pd.DataFrame()

        for m_method in model_list:
            tmp = pd.read_csv(
                f"{self.model_dir}/{m_method}/{m_method}_feature_importance.csv",
                names=["features", "Imp"], header=0
            )
            tmp = tmp.set_index("features").reset_index()[["features", "Imp"]]
            tmp["Imp"] = tmp["Imp"].abs()

            # Normalize importance values between 0 and 1
            tmp["Imp"] = tmp["Imp"].apply(
                lambda x: 0 if math.isnan(x) else (x - tmp["Imp"].min()) / (tmp["Imp"].max() - tmp["Imp"].min())
            )

            tmp = tmp.rename(columns={"Imp": f"Imp_{m_method}"})

            if binned_metric != 'bins':
                features_df = features_df.merge(tmp, on="features", how="outer")
            else:
                features_df = pd.concat([features_df, tmp.set_index('features')], axis=1)

        return features_df

    def plot_save(self, binning_table, m_method, i):
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Bar plot for event and non-event counts
        ax1.bar(
            binning_table['Bin'][:-1].astype(str),
            binning_table['Non-event'][:-1],
            color='skyblue',
            label='Non-event'
        )
        ax1.bar(
            binning_table['Bin'][:-1].astype(str),
            binning_table['Event'][:-1],
            bottom=binning_table['Non-event'][:-1],
            color='salmon',
            label='Event'
        )
        ax1.set_ylabel('Count')
        ax1.set_xlabel('Bin')
        ax1.tick_params(axis='y')

        # Line plot for event rate
        ax2 = ax1.twinx()
        ax2.plot(
            binning_table['Bin'][:-1].astype(str),
            binning_table['Event rate'][:-1],
            marker='o',
            color='black',
            label='Event Rate'
        )
        ax2.set_ylabel('Event Rate')
        ax2.tick_params(axis='y')

        # Final touches
        plt.title(f'Binning Table: {i}')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()

        # Ensure directory exists
        save_path = os.path.join(self.model_dir, m_method, 'eda')
        os.makedirs(save_path, exist_ok=True)

        # Save plot
        plt.savefig(os.path.join(save_path, f'{i}.png'))
        plt.close()
