import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f9fafb;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #6366f1;
    }
    .fraud-high   { color: #dc2626; font-weight: 700; }
    .fraud-medium { color: #f59e0b; font-weight: 700; }
    .fraud-low    { color: #16a34a; font-weight: 700; }
    .stProgress > div > div { background-color: #6366f1; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "fraud_model.pkl"
SCALER_PATH = "scaler.pkl"
COLS_PATH   = "feature_cols.pkl"

FREE_EMAIL_PROVIDERS = ['gmail.com', 'yahoo.com', 'hotmail.com']


# ── Preprocessing (mirrors the notebook exactly) ──────────────────────────────
def preprocess_for_prediction(test_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Preprocess a test CSV using the same steps as the notebook."""
    df = test_df.copy()

    # ── Type safety ──
    if 'TransactionDT' in df.columns:
        df['TransactionDT'] = pd.to_numeric(df['TransactionDT'], errors='coerce').fillna(0).astype(int)
    if 'TransactionAmt' in df.columns:
        df['TransactionAmt'] = pd.to_numeric(df['TransactionAmt'], errors='coerce').fillna(0.0)

    # ── Transaction amount features ──
    if 'TransactionAmt' in df.columns:
        df['TransactionAmt_log']     = np.log1p(df['TransactionAmt'])
        df['TransactionAmt_decimal'] = df['TransactionAmt'] % 1
        if 'card1' in df.columns:
            grp_mean = df.groupby('card1')['TransactionAmt'].transform('mean')
            grp_std  = df.groupby('card1')['TransactionAmt'].transform('std').replace(0, np.nan)
            df['Amt_to_mean'] = df['TransactionAmt'] / (grp_mean + 1e-9)
            df['Amt_to_std']  = df['TransactionAmt'] / (grp_std  + 1e-9)

    # ── Time features ──
    if 'TransactionDT' in df.columns:
        df['Transaction_hour']  = (df['TransactionDT'] // 3600) % 24
        df['Transaction_day']   = (df['TransactionDT'] // 86400) % 7
        df['Transaction_month'] = df['TransactionDT'] // (86400 * 30)
        df['is_night']          = (df['Transaction_hour'] <= 6).astype(int)

    # ── Email features ──
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            df[f'{col}_is_free'] = df[col].isin(FREE_EMAIL_PROVIDERS).astype(int)

    # ── Device features ──
    if 'DeviceInfo' in df.columns:
        df['DeviceInfo'] = df['DeviceInfo'].fillna('Unknown').astype(str)
        df['DeviceInfo_simple'] = df['DeviceInfo'].str.split('/').str[0].str.lower()
    if 'DeviceType' in df.columns:
        df['DeviceType'] = df['DeviceType'].fillna('Unknown').astype(str)

    # ── Card & address frequency ──
    for c in ['card1','card2','card3','card4','card5','addr1','addr2']:
        if c in df.columns:
            df[c + '_freq'] = df.groupby(c)[c].transform('count')

    # ── UID combos ──
    if 'card1' in df.columns:
        df['uid'] = df['card1'].astype(str)
    if {'card1','card2'}.issubset(df.columns):
        df['uid2'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
    if {'card1','card2','addr1'}.issubset(df.columns):
        df['uid3'] = df['card1'].astype(str) + '_' + df['card2'].astype(str) + '_' + df['addr1'].astype(str)
    if {'card1','card2','addr1','P_emaildomain'}.issubset(df.columns):
        df['uid4'] = (df['card1'].astype(str) + '_' + df['card2'].astype(str) + '_' +
                      df['addr1'].astype(str) + '_' + df['P_emaildomain'].astype(str))

    for c in ['uid','uid2','uid3','uid4']:
        if c in df.columns:
            df[c + '_freq'] = df.groupby(c)[c].transform('count')

    # ── User-level aggregations ──
    if 'card1' in df.columns and 'TransactionID' in df.columns:
        df['user_trans_count'] = df.groupby('card1')['TransactionID'].transform('count')
        df['user_mean_amt']    = df.groupby('card1')['TransactionAmt'].transform('mean')
        df['user_std_amt']     = df.groupby('card1')['TransactionAmt'].transform('std').fillna(0)

    # ── Time delta per uid3 ──
    if 'TransactionDT' in df.columns and 'uid3' in df.columns:
        df = df.reset_index().rename(columns={'index': 'orig_index'})
        df = df.sort_values(['uid3', 'TransactionDT']).reset_index(drop=True)
        df['time_diff_uid3'] = df.groupby('uid3')['TransactionDT'].diff().fillna(-1)
        df = df.sort_values('orig_index').drop(columns=['orig_index']).reset_index(drop=True)

    # ── Fill missing ──
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('Unknown').astype(str)

    # ── Label-encode objects ──
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['TransactionID']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # ── Align to training feature columns ──
    missing = set(feature_cols) - set(df.columns)
    for col in missing:
        df[col] = 0
    df = df[feature_cols]

    return df


# ── Model training (lite version for the app) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model_from_uploaded(train_transaction: pd.DataFrame,
                               train_identity: pd.DataFrame | None,
                               sample_n: int = 50_000):
    """Train LightGBM on the uploaded IEEE-CIS data and cache the result."""

    # Merge
    if train_identity is not None:
        df = train_transaction.merge(train_identity, on='TransactionID', how='left')
    else:
        df = train_transaction.copy()

    # Sample for speed
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)

    target = df['isFraud'].copy()
    df = df.drop(columns=['isFraud'], errors='ignore')

    # Preprocess (pass a dummy test identical to train – we only need features)
    dummy_test = df.head(1).copy()
    combined   = pd.concat([df.assign(is_train=1), dummy_test.assign(is_train=0)], ignore_index=True)

    # Reuse same preprocessing logic inline (simplified since no split needed)
    combined = combined.copy()
    if 'TransactionDT' in combined.columns:
        combined['TransactionDT'] = pd.to_numeric(combined['TransactionDT'], errors='coerce').fillna(0).astype(int)
    if 'TransactionAmt' in combined.columns:
        combined['TransactionAmt'] = pd.to_numeric(combined['TransactionAmt'], errors='coerce').fillna(0.0)
        combined['TransactionAmt_log']     = np.log1p(combined['TransactionAmt'])
        combined['TransactionAmt_decimal'] = combined['TransactionAmt'] % 1
        if 'card1' in combined.columns:
            gm = combined.groupby('card1')['TransactionAmt'].transform('mean')
            gs = combined.groupby('card1')['TransactionAmt'].transform('std').replace(0, np.nan)
            combined['Amt_to_mean'] = combined['TransactionAmt'] / (gm + 1e-9)
            combined['Amt_to_std']  = combined['TransactionAmt'] / (gs + 1e-9)
    if 'TransactionDT' in combined.columns:
        combined['Transaction_hour']  = (combined['TransactionDT'] // 3600) % 24
        combined['Transaction_day']   = (combined['TransactionDT'] // 86400) % 7
        combined['Transaction_month'] = combined['TransactionDT'] // (86400 * 30)
        combined['is_night']          = (combined['Transaction_hour'] <= 6).astype(int)
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in combined.columns:
            combined[col] = combined[col].fillna('Unknown').astype(str)
            combined[f'{col}_is_free'] = combined[col].isin(FREE_EMAIL_PROVIDERS).astype(int)
    if 'DeviceInfo' in combined.columns:
        combined['DeviceInfo'] = combined['DeviceInfo'].fillna('Unknown').astype(str)
        combined['DeviceInfo_simple'] = combined['DeviceInfo'].str.split('/').str[0].str.lower()
    for c in ['card1','card2','card3','card4','card5','addr1','addr2']:
        if c in combined.columns:
            combined[c + '_freq'] = combined.groupby(c)[c].transform('count')
    if 'card1' in combined.columns:
        combined['uid'] = combined['card1'].astype(str)
    if {'card1','card2'}.issubset(combined.columns):
        combined['uid2'] = combined['card1'].astype(str) + '_' + combined['card2'].astype(str)
    if {'card1','card2','addr1'}.issubset(combined.columns):
        combined['uid3'] = (combined['card1'].astype(str) + '_' +
                            combined['card2'].astype(str) + '_' + combined['addr1'].astype(str))
    if {'card1','card2','addr1','P_emaildomain'}.issubset(combined.columns):
        combined['uid4'] = (combined['card1'].astype(str) + '_' + combined['card2'].astype(str) + '_' +
                            combined['addr1'].astype(str) + '_' + combined['P_emaildomain'].astype(str))
    for c in ['uid','uid2','uid3','uid4']:
        if c in combined.columns:
            combined[c + '_freq'] = combined.groupby(c)[c].transform('count')
    if 'card1' in combined.columns and 'TransactionID' in combined.columns:
        combined['user_trans_count'] = combined.groupby('card1')['TransactionID'].transform('count')
        combined['user_mean_amt']    = combined.groupby('card1')['TransactionAmt'].transform('mean')
        combined['user_std_amt']     = combined.groupby('card1')['TransactionAmt'].transform('std').fillna(0)
    if 'TransactionDT' in combined.columns and 'uid3' in combined.columns:
        combined = combined.reset_index().rename(columns={'index': 'orig_index'})
        combined = combined.sort_values(['uid3', 'TransactionDT']).reset_index(drop=True)
        combined['time_diff_uid3'] = combined.groupby('uid3')['TransactionDT'].diff().fillna(-1)
        combined = combined.sort_values('orig_index').drop(columns=['orig_index']).reset_index(drop=True)
    missing_pct = combined.isnull().mean()
    drop_cols = missing_pct[missing_pct > 0.90].index.tolist()
    if drop_cols:
        combined = combined.drop(columns=drop_cols)
    for col in combined.columns:
        if combined[col].dtype in ['float64','int64']:
            combined[col] = combined[col].fillna(combined[col].median())
        else:
            combined[col] = combined[col].fillna('Unknown').astype(str)
    for col in combined.select_dtypes(include=['object']).columns:
        if col not in ['TransactionID']:
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))

    feature_cols = [c for c in combined.columns if c not in ['TransactionID','is_train']]
    X = combined[combined['is_train'] == 1][feature_cols].reset_index(drop=True)
    y = target.reset_index(drop=True).iloc[:len(X)]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    params = {
        'objective':        'binary',
        'metric':           'auc',
        'boosting_type':    'gbdt',
        'num_leaves':       31,
        'learning_rate':    0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq':     5,
        'verbose':         -1,
        'random_state':     42,
    }
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_valid = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        params, lgb_train,
        valid_sets=[lgb_valid],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(period=-1)],
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    auc       = roc_auc_score(y_val, val_preds)

    return model, feature_cols, auc, X_val, y_val, val_preds


# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_label(score: float) -> str:
    if score >= 0.7:   return "🔴 High Risk"
    if score >= 0.35:  return "🟡 Medium Risk"
    return "✅ Low Risk"

def risk_class(score: float) -> str:
    if score >= 0.7:   return "fraud-high"
    if score >= 0.35:  return "fraud-medium"
    return "fraud-low"


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔍 Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">IEEE-CIS Transaction Fraud Detection · LightGBM · Batch CSV Predictions</p>',
            unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    st.markdown("**Step 1 – Train the model**")
    st.markdown("Upload your IEEE-CIS `train_transaction.csv` (and optionally `train_identity.csv`) to train.")
    st.markdown("**Step 2 – Run predictions**")
    st.markdown("Upload your `test_transaction.csv` (and optionally `test_identity.csv`) to get predictions.")

    st.divider()
    sample_size = st.slider("Training sample size (rows)", 10_000, 100_000, 50_000, 5_000,
                            help="Larger = more accurate but slower training")
    threshold   = st.slider("Fraud probability threshold", 0.1, 0.9, 0.5, 0.05,
                            help="Transactions above this are flagged as FRAUD")
    st.divider()
    st.caption("Built from Harsh98245/FraudDetection · Deployed on Streamlit Cloud")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Train Model", "🔎 Batch Predict", "📊 Model Insights"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 – TRAIN
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Training Data")
    col1, col2 = st.columns(2)
    with col1:
        train_tx_file  = st.file_uploader("train_transaction.csv ✱ required",  type="csv", key="train_tx")
    with col2:
        train_id_file  = st.file_uploader("train_identity.csv (optional)",      type="csv", key="train_id")

    if train_tx_file:
        if st.button("🚀 Train LightGBM Model", type="primary"):
            with st.spinner("Loading data…"):
                train_tx = pd.read_csv(train_tx_file)
                train_id = pd.read_csv(train_id_file) if train_id_file else None

            with st.spinner(f"Training on up to {sample_size:,} rows – this takes ~1–2 min…"):
                model, feature_cols, auc, X_val, y_val, val_preds = train_model_from_uploaded(
                    train_tx, train_id, sample_n=sample_size
                )

            st.session_state['model']        = model
            st.session_state['feature_cols'] = feature_cols

            st.success(f"✅ Model trained! Validation AUC = **{auc:.4f}**")

            # Confusion matrix on validation set
            y_pred_bin = (val_preds >= threshold).astype(int)
            cm = confusion_matrix(y_val, y_pred_bin)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # ROC curve
            fpr, tpr, _ = roc_curve(y_val, val_preds)
            axes[0].plot(fpr, tpr, color='#6366f1', lw=2, label=f'AUC = {auc:.4f}')
            axes[0].plot([0,1],[0,1],'k--', lw=1)
            axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title('ROC Curve'); axes[0].legend()

            # Confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                        xticklabels=['Not Fraud','Fraud'], yticklabels=['Not Fraud','Fraud'])
            axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
            axes[1].set_title('Confusion Matrix (validation set)')

            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("👆 Upload `train_transaction.csv` to begin.")


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 – BATCH PREDICT
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload Test Data for Batch Predictions")

    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first (go to the **Train Model** tab).")
    else:
        col1, col2 = st.columns(2)
        with col1:
            test_tx_file = st.file_uploader("test_transaction.csv ✱ required",  type="csv", key="test_tx")
        with col2:
            test_id_file = st.file_uploader("test_identity.csv (optional)",      type="csv", key="test_id")

        if test_tx_file:
            if st.button("🔍 Run Predictions", type="primary"):
                with st.spinner("Preprocessing and predicting…"):
                    test_tx = pd.read_csv(test_tx_file)
                    test_id = pd.read_csv(test_id_file) if test_id_file else None

                    if test_id is not None:
                        test_merged = test_tx.merge(test_id, on='TransactionID', how='left')
                    else:
                        test_merged = test_tx.copy()

                    ids  = test_merged.get('TransactionID', pd.Series(range(len(test_merged)), name='TransactionID'))
                    X    = preprocess_for_prediction(test_merged, st.session_state['feature_cols'])
                    preds = st.session_state['model'].predict(
                        X, num_iteration=st.session_state['model'].best_iteration
                    )

                results = pd.DataFrame({
                    'TransactionID':   ids.values,
                    'fraud_probability': np.round(preds, 4),
                    'predicted_fraud':   (preds >= threshold).astype(int),
                    'risk_level':        [risk_label(p) for p in preds],
                })

                st.session_state['results'] = results

            if 'results' in st.session_state:
                results = st.session_state['results']

                # Summary metrics
                n_total  = len(results)
                n_fraud  = results['predicted_fraud'].sum()
                n_safe   = n_total - n_fraud
                fraud_rt = n_fraud / n_total * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Transactions", f"{n_total:,}")
                c2.metric("🔴 Flagged Fraud",   f"{n_fraud:,}")
                c3.metric("✅ Clean",            f"{n_safe:,}")
                c4.metric("Fraud Rate",          f"{fraud_rt:.2f}%")

                st.divider()

                # Prob distribution
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.hist(results['fraud_probability'], bins=50, color='#6366f1', edgecolor='white', alpha=0.85)
                ax.axvline(threshold, color='#dc2626', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
                ax.set_xlabel('Fraud Probability'); ax.set_ylabel('Count')
                ax.set_title('Distribution of Fraud Probability Scores')
                ax.legend()
                st.pyplot(fig)

                st.divider()

                # Table
                st.subheader("📋 Results Table")
                st.dataframe(
                    results.sort_values('fraud_probability', ascending=False).reset_index(drop=True),
                    use_container_width=True,
                    height=400,
                )

                # Download
                csv_buf = io.BytesIO()
                results.to_csv(csv_buf, index=False)
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_buf.getvalue(),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 – MODEL INSIGHTS
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Insights")

    if 'model' not in st.session_state:
        st.info("Train a model first to see feature importances.")
    else:
        model        = st.session_state['model']
        feature_cols = st.session_state['feature_cols']

        importance = pd.DataFrame({
            'feature':    feature_cols,
            'importance': model.feature_importance(importance_type='gain'),
        }).sort_values('importance', ascending=False).head(25)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=importance, y='feature', x='importance', palette='Blues_r', ax=ax)
        ax.set_title('Top 25 Feature Importances (Gain)')
        ax.set_xlabel('Importance (Gain)')
        ax.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("📖 Model Details"):
            st.markdown(f"""
| Property | Value |
|---|---|
| Algorithm | LightGBM (GBDT) |
| Objective | Binary classification |
| Best iteration | {model.best_iteration} |
| Num features | {len(feature_cols)} |
| Learning rate | 0.05 |
| Num leaves | 31 |
| Fraud threshold | {threshold} |
            """)
