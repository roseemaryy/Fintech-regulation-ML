import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


class FintechRegulationML:
    def __init__(self):
        self.data = {}
        self.models = {}
        self.scalers = {}
        self.results = {}

    def create_sample_data(self):
        np.random.seed(42)

        countries = [
            "Brazil", "India", "Nigeria", "Kenya", "Indonesia", "Mexico",
            "Philippines", "South Africa", "Thailand", "Vietnam"
        ]
        fintech_types = [
            "Digital Payments", "Digital Lending", "Cryptocurrency",
            "InsurTech", "WealthTech", "RegTech", "Neobanking"
        ]

        # Regulatory data
        reg_data = []
        for c in countries:
            reg_data.append({
                "country": c,
                "licenses_required": np.random.randint(1, 5),
                "capital_requirement_usd": np.random.lognormal(15, 1),
                "sandbox_available": np.random.choice([0, 1]),
                "approval_time_days": np.random.randint(60, 200),
                "compliance_burden_score": np.random.uniform(0.2, 0.9),
                "regulatory_clarity_score": np.random.uniform(0.3, 0.9),
            })
        self.data["regulatory"] = pd.DataFrame(reg_data)

        # Market data
        market_data = []
        for c in countries:
            market_data.append({
                "country": c,
                "gdp_per_capita": np.random.lognormal(8, 1),
                "mobile_penetration": np.random.uniform(0.5, 0.95),
                "internet_penetration": np.random.uniform(0.3, 0.8),
                "financial_inclusion_index": np.random.uniform(0.3, 0.8),
            })
        self.data["market"] = pd.DataFrame(market_data)

        # Fintech data
        fintech_data = []
        for i in range(200):
            fintech_data.append({
                "company_id": i + 1,
                "country": np.random.choice(countries),
                "fintech_type": np.random.choice(fintech_types),
                "funding_amount_usd": np.random.lognormal(14, 2),
                "employees": np.random.randint(10, 500),
                "uses_ai_ml": np.random.choice([0, 1]),
                "success_outcome": np.random.choice([0, 1], p=[0.4, 0.6]),
            })
        self.data["fintech"] = pd.DataFrame(fintech_data)

        # Time-series
        ts_data = []
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="M")
        for c in countries[:3]:
            for i, d in enumerate(dates):
                ts_data.append({
                    "country": c,
                    "date": d,
                    "fintech_adoption_rate": np.clip(0.3 + 0.01 * i + np.random.normal(0, 0.05), 0, 1),
                    "regulatory_change_event": 1 if i == 24 else 0,
                })
        self.data["timeseries"] = pd.DataFrame(ts_data)

        print("✅ Sample data created!")

    def train_regulatory_classifier(self):
        merged = pd.merge(self.data["regulatory"], self.data["market"], on="country")
        merged["regulatory_class"] = np.where(
            merged["regulatory_clarity_score"] > 0.6, "Innovation-Friendly", "Restrictive"
        )

        X = merged.drop(["country", "regulatory_class"], axis=1)
        y = merged["regulatory_class"]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        self.models["reg_classifier"] = model
        self.results["reg_classifier"] = {
            "accuracy": model.score(X_test_scaled, y_test),
            "report": classification_report(y_test, y_pred, target_names=le.classes_),
            "feature_importances": dict(zip(X.columns, model.feature_importances_))
        }

        print("✅ Regulatory Classifier trained!")

    def train_fintech_success_predictor(self):
        fintech = pd.merge(self.data["fintech"], self.data["regulatory"], on="country")
        fintech["company_age"] = 2024 - np.random.randint(2015, 2023, len(fintech))

        X = fintech[["funding_amount_usd", "employees", "uses_ai_ml", "licenses_required"]]
        y = fintech["success_outcome"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        self.models["fintech_success"] = model
        self.results["fintech_success"] = {
            "accuracy": model.score(X_test_scaled, y_test),
            "report": classification_report(y_test, y_pred),
        }

        print("✅ Fintech Success Predictor trained!")

    def perform_market_clustering(self):
        merged = pd.merge(self.data["regulatory"], self.data["market"], on="country")
        X = merged.drop("country", axis=1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42)
        merged["cluster"] = kmeans.fit_predict(X_scaled)

        self.results["clusters"] = merged[["country", "cluster"]]
        print("✅ Market Clustering done!")

        # --- Visualization ---
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=merged["gdp_per_capita"],
            y=merged["regulatory_clarity_score"],
            hue=merged["cluster"],
            palette="Set2",
            s=100
        )
        for i, txt in enumerate(merged["country"]):
            plt.annotate(txt, (merged["gdp_per_capita"].iloc[i], merged["regulatory_clarity_score"].iloc[i]))
        plt.title("🌍 Market Clusters by GDP & Regulatory Clarity")
        plt.xlabel("GDP per Capita")
        plt.ylabel("Regulatory Clarity Score")
        plt.show()

    def analyze_time_series_trends(self):
        ts = self.data["timeseries"]
        correlations = {}
        for c in ts["country"].unique():
            subset = ts[ts["country"] == c]
            correlations[c] = subset["fintech_adoption_rate"].corr(subset["regulatory_change_event"])
        self.results["timeseries"] = correlations
        print("✅ Time-Series Analysis done!")

        # --- Visualization ---
        plt.figure(figsize=(10, 6))
        for c in ts["country"].unique():
            subset = ts[ts["country"] == c]
            sns.lineplot(data=subset, x="date", y="fintech_adoption_rate", label=c)
        plt.axvline(ts["date"].iloc[24], color="red", linestyle="--", label="Regulatory Change")
        plt.title("📈 Fintech Adoption Over Time")
        plt.xlabel("Date")
        plt.ylabel("Adoption Rate")
        plt.legend()
        plt.show()

    def generate_report(self):
        print("\n📊 FINTECH REGULATION INSIGHTS REPORT 📊")

        if "reg_classifier" in self.results:
            print(f"Accuracy: {self.results['reg_classifier']['accuracy']:.2f}")
            print("Classification Report:")
            print(self.results["reg_classifier"]["report"])

            # --- Feature Importance Plot ---
            importances = self.results["reg_classifier"]["feature_importances"]
            plt.figure(figsize=(8, 6))
            sns.barplot(x=list(importances.values()), y=list(importances.keys()))
            plt.title("🔑 Top Regulatory Features")
            plt.show()

        if "fintech_success" in self.results:
            print(f"\nFintech Success Model Accuracy: {self.results['fintech_success']['accuracy']:.2f}")
            print(self.results["fintech_success"]["report"])

        if "clusters" in self.results:
            print("\nClusters:")
            print(self.results["clusters"])

        if "timeseries" in self.results:
            print("\nTime-Series Correlations:")
            for c, val in self.results["timeseries"].items():
                print(f"{c}: {val:.3f}")

    def run_complete_analysis(self):
        print("🔎 Running Complete Analysis Pipeline...")
        self.create_sample_data()
        self.train_regulatory_classifier()
        self.train_fintech_success_predictor()
        self.perform_market_clustering()
        self.analyze_time_series_trends()
        self.generate_report()
        print("🎉 Analysis Completed!")


if __name__ == "__main__":
    print("🚀 Script started...")
    project = FintechRegulationML()
    project.run_complete_analysis()
    print("✅ MAIN block executed!")
