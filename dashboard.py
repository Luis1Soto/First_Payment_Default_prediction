import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

st.set_page_config(page_title="FPD Model Dashboard", layout="wide")
st.title("📊 First Payment Default - Model Evaluation Dashboard")

results_path = "../results/model_evaluation_results.csv"
models_dir = "../models/tuned_models"

if not os.path.exists(results_path):
    st.error("No evaluation file found. Run model training first.")
else:
    results_df = pd.read_csv(results_path)
    st.header("🏆 Model Evaluation Results")
    st.dataframe(results_df)

    if 'Model' in results_df.columns:
        summary_model = results_df.groupby('Model').agg(
            mean_auc=('ROC AUC', 'mean'),
            std_auc=('ROC AUC', 'std'),
            mean_f1=('F1', 'mean'),
            std_f1=('F1', 'std')
        ).sort_values(by='mean_auc', ascending=False).reset_index()

        st.header("📈 Model Comparison")
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.barplot(data=summary_model, x='mean_auc', y='Model', ax=ax[0], palette='viridis')
        ax[0].set_title("Average ROC AUC per Model")
        ax[0].set_xlabel("ROC AUC")

        sns.barplot(data=summary_model, x='mean_f1', y='Model', ax=ax[1], palette='plasma')
        ax[1].set_title("Average F1 Score per Model")
        ax[1].set_xlabel("F1 Score")
        st.pyplot(fig)

    if 'Processed_File' in results_df.columns:
        summary_file = results_df.groupby('Processed_File').agg(
            mean_auc=('ROC AUC', 'mean'),
            std_auc=('ROC AUC', 'std'),
            count=('Model', 'count')
        ).sort_values(by='mean_auc', ascending=False).reset_index()

        st.header("📂 Dataset Performance Ranking")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=summary_file, x='mean_auc', y='Processed_File', ax=ax, palette='coolwarm')
        ax.set_title("Average ROC AUC by Dataset")
        st.pyplot(fig)

    st.header("🔍 Best Performing Models")
    if 'ROC AUC' in results_df.columns:
        best_row = results_df.loc[results_df['ROC AUC'].idxmax()]
        st.success(f"Best Model: {best_row['Model']} on {best_row['Processed_File']}")
        st.write(f"**ROC AUC**: {best_row['ROC AUC']:.4f}")
        st.write(f"**F1 Score**: {best_row['F1']:.4f}")

    if 'ROC AUC' in results_df.columns and 'std_auc' in results_df.columns:
        unstable_models = results_df.groupby(['Processed_File', 'Model']).agg(
            auc_std=('ROC AUC', 'std'),
            f1_std=('F1', 'std')
        ).reset_index()
        unstable_models = unstable_models[unstable_models['auc_std'] > 0.02]
        if not unstable_models.empty:
            st.warning("⚠️ Unstable Models Detected")
            st.dataframe(unstable_models.sort_values(by='auc_std', ascending=False))

    st.header("🔮 Predict New Applicants")
    uploaded_file = st.file_uploader("Upload new applicants CSV file", type="csv")
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.write("📄 Loaded new data:", new_data.shape)

            model_path = os.path.join(models_dir, "LightGBM_tuned.joblib")
            imputer_path = os.path.join(models_dir, "imputer_Data_iterative_imputed.pkl.joblib")
            scaler_path = os.path.join(models_dir, "scaler_Data_iterative_imputed.pkl.joblib")

            if os.path.exists(model_path) and os.path.exists(imputer_path) and os.path.exists(scaler_path):
                model = load(model_path)
                imputer = load(imputer_path)
                scaler = load(scaler_path)

                X_new = new_data.drop(columns=['ID'], errors='ignore')
                X_imp = imputer.transform(X_new)
                X_scaled = scaler.transform(X_imp)

                probs = model.predict_proba(X_scaled)[:, 1]
                predictions = (probs > 0.5).astype(int)

                new_data['default_probability'] = probs
                new_data['predicted_default'] = predictions

                st.success("✅ Predictions completed")
                st.dataframe(new_data[['default_probability', 'predicted_default']].head())
                st.download_button(
                    label="Download Predictions",
                    data=new_data.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ Missing required model files for inference.")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    
    st.sidebar.markdown("## 📝 Notes")
    st.sidebar.info("Ensure you have run model training and saved results before running this dashboard.")
    st.sidebar.markdown("- [Model evaluation file](../results/model_evaluation_results.csv)")
    st.sidebar.markdown("- [Trained models](../models/tuned_models/)")
    st.sidebar.markdown("- Upload new applicant data to predict defaults.")
