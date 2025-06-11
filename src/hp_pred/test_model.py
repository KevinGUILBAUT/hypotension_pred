from pathlib import Path
import pickle
from itertools import chain, repeat

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV

import hp_pred.experiments as expe
from hp_pred.supermodel import SuperModelXGBoost

BASELINE_FEATURE = "last_map_value"

def _revert_dict(d):
    return dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))


def _grouped_shap(shap_vals, features, groups):
    groupmap = _revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T
    return shap_grouped


def expected_calibration_error(confidences, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get predictions from confidences (positional in this case)
    predicted_label = confidences > 0.5

    confidences = np.abs(confidences - 0.5) + 0.5  # get the confidence of the prediction

    # get a boolean list of correct/false predictions
    accuracies = predicted_label == true_labels

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower &amp; upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece


class TestModel():
    def __init__(
            self,
            model_configs: list,
            output_name: str,
            n_bootstraps: int = 200,
    ):

        self.model_configs = model_configs
        self.output_name = output_name
        self.n_bootstraps = n_bootstraps
        self.rng_seed = 42  # control reproducibility
        
        # Setup result directories
        self.result_folder = Path("data/models/results")
        if not self.result_folder.exists():
            self.result_folder.mkdir(parents=True, exist_ok=True)
            
        # Create figures directory if it doesn't exist
        figures_dir = Path("data/models/figures")
        if not figures_dir.exists():
            figures_dir.mkdir(parents=True, exist_ok=True)
            
        # Process each model configuration
        self.models = []
        self.test_datasets = []
        self.train_datasets = []
        self.features_names_list = []
        self.model_names = []
        self.model_result_files = []
        self.baseline_result_files = []
        
        for i, config in enumerate(model_configs):
            # Extract configuration
            test_data = config['test_data']
            train_data = config['train_data']
            model_filename = config['model_filename']
            features_names = config['features_names']
            model_name = config.get('name', f"model_{i}")
            model_type = config.get('model_type', 'xgboost')  # Default to xgboost

            # Load the model based on its type
            model_path = Path("data/models") / model_filename
            
            try:
                if model_type == 'xgboost':
                    # Load as XGBoost model
                    model = xgb.XGBClassifier()
                    model.load_model(model_path)

                elif model_type == 'super_model':
                    model = SuperModelXGBoost()
                    model.load(model_path)
                    
                else:
                    # Load as pickle model (for other model types)
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
            except Exception as e:
                raise ValueError(f"Error loading model {model_filename}: {e}")
            
            # Clean data
            #test_data.dropna(subset=features_names, inplace=True)
            #test_data.dropna(subset=[BASELINE_FEATURE], inplace=True)
            
            #train_data.dropna(subset=features_names, inplace=True)
            #train_data.dropna(subset=[BASELINE_FEATURE], inplace=True)
            
            # Store everything
            self.models.append(model)
            self.test_datasets.append(test_data)
            self.train_datasets.append(train_data)
            self.features_names_list.append(features_names)
            self.model_names.append(model_name)
            
            # Set up result files
            self.model_result_files.append(self.result_folder / f"{model_name}_{self.output_name}.pkl")
            self.baseline_result_files.append(self.result_folder / f"baseline_{model_name}_{self.output_name}.pkl")
            
            print(f"Model {model_name} loaded")
            print(f"Number of points in test data: {len(test_data)}")
            print(f"Prevalence of hypotension: {test_data['label'].mean():.2%}")
            print("---")

    def test_baseline(self, force_computation=False):
        self.y_pred_baseline = []
        self.dict_results_baseline = []
        self.baseline_recalls = []
        
        for i, (test_data, train_data, baseline_result_file) in enumerate(
                zip(self.test_datasets, self.train_datasets, self.baseline_result_files)):
            
            #Load baseline results
            if not force_computation and baseline_result_file.exists():
                with baseline_result_file.open("rb") as f:
                    baseline_results = pickle.load(f)
                    self.dict_results_baseline.append(baseline_results)
                
                model_baseline = CalibratedClassifierCV()
                x_train = train_data[BASELINE_FEATURE].to_numpy().reshape(-1, 1)
                y_train = train_data["label"].to_numpy()
                model_baseline.fit(x_train, y_train)
                x_test = test_data[BASELINE_FEATURE].to_numpy()
                y_pred = model_baseline.predict_proba(x_test.reshape(-1, 1))[:, 1]
                self.y_pred_baseline.append(y_pred)
                
                # Extract baseline recall
                self.baseline_recalls.append(np.median(baseline_results["recall_threshold"]))
                print(f"Baseline {self.model_names[i]} results loaded from file")
                continue
                
            print(f"Computing baseline for {self.model_names[i]}...")
            
            # Train and predict with baseline model
            model_baseline = CalibratedClassifierCV()
            x_train = train_data[BASELINE_FEATURE].to_numpy().reshape(-1, 1)
            y_train = train_data["label"].to_numpy()
            model_baseline.fit(x_train, y_train)
            
            x_test = test_data[BASELINE_FEATURE].to_numpy()
            y_test = test_data["label"].to_numpy()
            y_label_id = test_data["label_id"].to_numpy()
            
            y_pred = model_baseline.predict_proba(x_test.reshape(-1, 1))[:, 1]
            self.y_pred_baseline.append(y_pred)
            
            # Run bootstrap testing
            dict_results, _, _ = expe.bootstrap_test(
                y_test,
                y_pred,
                y_label_id,
                n_bootstraps=self.n_bootstraps,
                rng_seed=self.rng_seed,
                strategy="max_precision",
                target=0.24
            )
            
            # Convert threshold to MAP value for the baseline
            for j in range(len(dict_results["threshold_opt"])):
                dict_results["threshold_opt"][j] = test_data[BASELINE_FEATURE].iloc[np.argmin(np.abs(
                    dict_results["threshold_opt"][j] - y_pred))]

            for j in range(len(dict_results["threshold"])):
                dict_results["threshold"][j] = test_data[BASELINE_FEATURE].iloc[np.argmin(np.abs(
                    dict_results["threshold"][j] - y_pred))]
            
            # Save results
            self.dict_results_baseline.append(dict_results)
            self.baseline_recalls.append(np.median(dict_results["recall_threshold"]))
            
            with baseline_result_file.open("wb") as f:
                pickle.dump(dict_results, f)

    def test_models(self, force_computation=False):
        self.y_pred_models = []
        self.dict_results_models = []
        
        for i, (model, test_data, model_result_file, features_names) in enumerate(
                zip(self.models, self.test_datasets, self.model_result_files, self.features_names_list)):
            
            #Load models results
            if not force_computation and model_result_file.exists():
                with model_result_file.open("rb") as f:
                    model_results = pickle.load(f)
                    self.dict_results_models.append(model_results)
                
                y_pred = model.predict_proba(test_data[features_names])[:, 1]
                self.y_pred_models.append(y_pred)
                
                print(f"Model {self.model_names[i]} results loaded from file")
                continue
                
            print(f"Testing model {self.model_names[i]}...")
            
            y_pred = model.predict_proba(test_data[features_names])[:, 1]
            self.y_pred_models.append(y_pred)
            
            # Get test labels
            y_test = test_data["label"].to_numpy()
            y_label_ids = test_data["label_id"].to_numpy()
            
            # Run bootstrap testing
            dict_result, _, _ = expe.bootstrap_test(
                y_test,
                y_pred,
                y_label_ids,
                n_bootstraps=self.n_bootstraps,
                rng_seed=self.rng_seed,
                strategy="targeted_recall",
                target=self.baseline_recalls[i]  # Use corresponding baseline recall
            )
            
            self.dict_results_models.append(dict_result)
            
            with model_result_file.open("wb") as f:
                pickle.dump(dict_result, f)

    def print_results(self):
        if not hasattr(self, "dict_results_baseline") or not hasattr(self, "dict_results_models"):
            raise ValueError("Results not loaded.")
            
        print('\n')
        print(f"Results for {self.output_name}")
        
        for i, model_name in enumerate(self.model_names):
            print(f"\n===== Results for {model_name} =====")
            print('Baseline:')
            expe.print_statistics(self.dict_results_baseline[i])
            
            print('\nModel:')
            expe.print_statistics(self.dict_results_models[i])
            print("----------------------------")

    def plot_precision_recall(self, model_indices=None):
        if not hasattr(self, "dict_results_baseline") or not hasattr(self, "dict_results_models"):
            raise ValueError("Results not loaded. Run test_baseline() and test_models() first")
            
        if model_indices is None:
            model_indices = list(range(len(self.models)))
            
        plt.figure(figsize=(12, 8))
        
        # Plot each selected model
        for idx in model_indices:
            model_name = self.model_names[idx]
            dict_results_model = self.dict_results_models[idx]
            dict_results_baseline = self.dict_results_baseline[idx]
            
            # Plot model PR curve
            recall = np.linspace(0, 1, 1000)
            precision_mean = dict_results_model['precision'].mean(0)
            precision_std = dict_results_model['precision'].std(0)
            
            plt.fill_between(
                recall, precision_mean - 2 * precision_std, precision_mean + 2 * precision_std, alpha=0.2
            )
            plt.plot(recall, precision_mean, 
                    label=f"{model_name} (AUPRC = {expe.print_one_stat(pd.Series(dict_results_model['auprcs']), False)})")
            
            # Plot baseline PR curve
            plt.fill_between(
                dict_results_baseline['fprs'],
                dict_results_baseline['precision'].mean(0) - 2 * dict_results_baseline['precision'].std(0),
                dict_results_baseline['precision'].mean(0) + 2 * dict_results_baseline['precision'].std(0),
                alpha=0.2, color='gray'
            )
            plt.plot(
                dict_results_baseline['fprs'],
                dict_results_baseline['precision'].mean(0),
                '--', color='gray',
                label=f"baseline {model_name} (AUPRC = {expe.print_one_stat(pd.Series(dict_results_baseline['auprcs']), False)})",
            )
        
        plt.plot([0, 1], [self.dict_results_baseline[0]['precision'].mean(0)[-1]]*2, "k--")
        
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.05, 1.05)
        plt.title("Precision-Recall Curves")
        plt.legend()
        plt.grid()
        plt.savefig("./data/models/figures/PRC_curve_comparison.png")
        plt.show()

    def plot_calibration_curve(self, model_indices=None, n_bins=11):

        if model_indices is None:
            model_indices = list(range(len(self.models)))
        
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')
        
        # Plot each selected model
        for idx in model_indices:
            model_name = self.model_names[idx]
            y_test = self.test_datasets[idx]["label"].to_numpy()
            
            # Model calibration
            y_pred = self.y_pred_models[idx]
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_pred, n_bins=n_bins, strategy='uniform'
            )
            ece = expected_calibration_error(y_pred, y_test, M=n_bins)
            plt.plot(mean_predicted_value, fraction_of_positives,
                    marker='o', label=f'{model_name} (ECE={ece:.3f})')
            
            # Baseline calibration
            y_pred_baseline = self.y_pred_baseline[idx]
            fraction_of_positives_baseline, mean_predicted_value_baseline = calibration_curve(
                y_test, y_pred_baseline, n_bins=n_bins, strategy='uniform'
            )
            ece_baseline = expected_calibration_error(y_pred_baseline, y_test, M=n_bins)
            plt.plot(mean_predicted_value_baseline, fraction_of_positives_baseline,
                    marker='s', linestyle='--', 
                    label=f'Baseline {model_name} (ECE={ece_baseline:.3f})')
        
        plt.xlabel('Predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration curve comparison')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig("./data/models/figures/Calibration_comparison.png")
        plt.show()
        
        n_models = len(model_indices)
        fig, ax = plt.subplots(2, n_models, figsize=(4*n_models, 8), layout='constrained', sharey=True)
        
        for i, idx in enumerate(model_indices):
            model_name = self.model_names[idx]
            
            # Baseline histogram
            ax[0, i].hist(self.y_pred_baseline[idx], bins=20, alpha=0.7, color='gray')
            ax[0, i].set_title(f'Baseline {model_name}')
            ax[0, i].set_xlabel('Predicted probability')
            if i == 0:
                ax[0, i].set_ylabel('Frequency')
                
            # Model histogram
            ax[1, i].hist(self.y_pred_models[idx], bins=20, alpha=0.7)
            ax[1, i].set_title(f'Model {model_name}')
            ax[1, i].set_xlabel('Predicted probability')
            if i == 0:
                ax[1, i].set_ylabel('Frequency')
                
        fig.suptitle('Histogram of predicted probabilities')
        plt.savefig("./data/models/figures/histogram_comparison.png")
        plt.show()

    def compute_shap_values(self, model_idx=0):
        print(f"Computing SHAP values for {self.model_names[model_idx]}...")
        # Use SHAP to explain the model
        shap.initjs()
        explainer = shap.TreeExplainer(self.models[model_idx])
        self.shap_values = explainer.shap_values(
            self.test_datasets[model_idx][self.features_names_list[model_idx]]
        )
        self.shap_model_idx = model_idx  # Store which model was analyzed

    def plot_shap_values(self, nb_max_feature=10):
        model_idx = self.shap_model_idx
        features_names = self.features_names_list[model_idx]
        test_data = self.test_datasets[model_idx][features_names]
        
        plt.figure(figsize=(12, 5))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        for i in range(nb_max_feature):
            plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)
        shap.summary_plot(self.shap_values, test_data, feature_names=features_names,
                        show=False, plot_type="bar", max_display=nb_max_feature)
        plt.xlabel('mean($|$SHAP value$|$)')
        
        names = plt.gca().get_yticklabels()
        names = [name.get_text().replace("constant", "intercept") for name in names]
        names = [name.replace("mbp", "MAP") for name in names]
        names = [name.replace("sbp", "SAP") for name in names]
        names = [name.replace("dbp", "DAP") for name in names]
        names = [name.replace("hr", "HR") for name in names]
        names = [name.replace("rf_ct", "RF_CT") for name in names]
        plt.gca().set_yticklabels(names)
        
        # Beeswarm plot
        plt.subplot(1, 2, 2)
        shap.summary_plot(self.shap_values, test_data, feature_names=features_names,
                        show=False, max_display=nb_max_feature)
        # Remove the y tick labels
        plt.gca().set_yticklabels([])
        plt.xlabel('SHAP value')
        plt.tight_layout()
        
        # Add horizontal lines
        for i in range(nb_max_feature):
            plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)

        plt.savefig(f"./data/models/figures/SHAP_values_{self.model_names[model_idx]}.png")
        plt.show()

    def group_shap_values(self):
        if not hasattr(self, "shap_values"):
            raise ValueError("SHAP values not computed. Run compute_shap_values() first")
            
        model_idx = self.shap_model_idx
        features_names = self.features_names_list[model_idx]
        test_data = self.test_datasets[model_idx][features_names]
        
        # Define feature groups
        groups = {
            'MAP': [name for name in features_names if 'mbp' in name],
            'DAP': [name for name in features_names if 'dbp' in name],
            'SAP': [name for name in features_names if 'sbp' in name],
            'MAC': [name for name in features_names if 'mac' in name],
            'HR': [name for name in features_names if 'hr' in name],
            'RR': [name for name in features_names if 'rr' in name],
            'SPO2': [name for name in features_names if 'spo2' in name],
            'ETCO2': [name for name in features_names if 'etco2' in name],
            'PROPO': [name for name in features_names if 'pp_ct' in name],
            'REMI': [name for name in features_names if 'rf_ct' in name],
            'TEMP': [name for name in features_names if 'body_temp' in name],
            'AGE': ['age'] if 'age' in features_names else [],
            'BMI': ['bmi'] if 'bmi' in features_names else [],
            'ASA': ['asa'] if 'asa' in features_names else [],
            'PREOP_CR': ['preop_cr'] if 'preop_cr' in features_names else [],
            'PREOP_HTN': ['preop_htn'] if 'preop_htn' in features_names else [],
        }
        
        #groups = {k: v for k, v in groups.items() if v}
        
        self.shap_grouped = _grouped_shap(self.shap_values, features_names, groups)
        self.test_data_group = _grouped_shap(test_data, features_names, groups)

    def plot_shap_grouped(self, nb_max_feature=10):
        model_idx = self.shap_model_idx
        font_size = 16

        plt.figure(figsize=(12, 5))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        for i in range(nb_max_feature):
            plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)
        shap.summary_plot(self.shap_grouped.values, self.test_data_group.values, 
                          feature_names=self.shap_grouped.columns,
                          show=False, plot_type="bar", max_display=nb_max_feature)
        plt.xlabel('mean($|$SHAP value$|$)', fontsize=font_size)
        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=font_size)

        plt.subplot(1, 2, 2)
        shap.summary_plot(self.shap_grouped.values, self.test_data_group.values,
                          max_display=nb_max_feature, show=False)
        plt.xlabel('SHAP value', fontsize=font_size)
        
        # Remove y tick labels
        ax = plt.gca()
        ax.set_yticklabels([])
        for i in range(min(nb_max_feature, len(self.shap_grouped.columns))):
            plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)
            
        plt.tight_layout()
        plt.savefig(f"./data/models/figures/SHAP_grouped_{self.model_names[model_idx]}.png")
        plt.show()

    def run(self, force_baseline_computation=False, force_model_computation=False):
        """Run the full evaluation pipeline"""
        # Test baseline models
        self.test_baseline(force_computation=force_baseline_computation)
        
        # Test ML models
        self.test_models(force_computation=force_model_computation)
        
        # Print results
        self.print_results()
        
        # Plot curves
        self.plot_precision_recall()
        self.plot_calibration_curve()