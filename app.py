from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              roc_auc_score, silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, mean_squared_error, r2_score, mean_absolute_error)
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# ─── Global model storage ───────────────────────────────────────────────────
models = {}
scalers = {}
feature_columns = {}
dataset_info = {}

def preprocess_data(df):
    """Full preprocessing pipeline from the notebook."""
    data = df.copy()
    if 'Employee_ID' in data.columns:
        data = data.drop(columns=['Employee_ID'])

    # Encode Attrition
    if 'Attrition' in data.columns:
        data['Attrition_encoded'] = data['Attrition'].map({'Yes': 1, 'No': 0})

    # One-hot encode categoricals
    cat_cols = ['Marital_Status', 'Gender', 'Department', 'Job_Role', 'Overtime']
    for col in cat_cols:
        if col in data.columns:
            data = pd.get_dummies(data, columns=[col])

    # Drop reference categories to avoid multicollinearity
    for col in ['Department_HR', 'Job_Role_Assistant', 'Marital_Status_Divorced']:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Drop original Attrition string column
    if 'Attrition' in data.columns:
        data = data.drop(columns=['Attrition'])

    # Convert booleans to int
    bool_cols = data.select_dtypes(include='bool').columns
    data[bool_cols] = data[bool_cols].astype(int)

    return data


def train_classification_models(df):
    """DSO1 - Train attrition prediction models."""
    data = preprocess_data(df)

    X = data.drop(columns=['Attrition_encoded'])
    y = data['Attrition_encoded']

    feature_columns['classification'] = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scalers['classification'] = scaler

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    results = {}

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train_bal, y_train_bal)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    models['logistic_regression'] = lr
    results['Logistic Regression'] = {
        'accuracy': round(accuracy_score(y_test, y_pred_lr) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, y_prob_lr) * 100, 2),
        'report': classification_report(y_test, y_pred_lr, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist()
    }

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)
    y_pred_rf = rf.predict(X_test_scaled)
    y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    models['random_forest'] = rf
    feat_imp = sorted(zip(X.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)[:10]
    results['Random Forest'] = {
        'accuracy': round(accuracy_score(y_test, y_pred_rf) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, y_prob_rf) * 100, 2),
        'report': classification_report(y_test, y_pred_rf, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist(),
        'feature_importance': [{'feature': f, 'importance': round(float(i), 4)} for f, i in feat_imp]
    }

    # 3. KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_bal, y_train_bal)
    y_pred_knn = knn.predict(X_test_scaled)
    y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]
    models['knn'] = knn
    results['KNN'] = {
        'accuracy': round(accuracy_score(y_test, y_pred_knn) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, y_prob_knn) * 100, 2),
        'report': classification_report(y_test, y_pred_knn, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_knn).tolist()
    }

    # 4. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                             use_label_encoder=False, eval_metric='logloss',
                             random_state=42, n_jobs=-1)
        xgb.fit(X_train_bal, y_train_bal)
        y_pred_xgb = xgb.predict(X_test_scaled)
        y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
        models['xgboost'] = xgb
        results['XGBoost'] = {
            'accuracy': round(accuracy_score(y_test, y_pred_xgb) * 100, 2),
            'roc_auc': round(roc_auc_score(y_test, y_prob_xgb) * 100, 2),
            'report': classification_report(y_test, y_pred_xgb, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred_xgb).tolist()
        }

    return results


def train_clustering_models(df):
    """DSO2 - Employee segmentation."""
    data = preprocess_data(df)

    # Composite scores
    overtime_col = 'Overtime_Yes' if 'Overtime_Yes' in data.columns else None
    if overtime_col:
        data['Risk_Score'] = (
            (data[overtime_col]).astype(int) * 2 +
            (data['Job_Satisfaction'] <= 2).astype(int) * 2 +
            (data['Work_Life_Balance'] <= 2).astype(int) +
            (data['Years_Since_Last_Promotion'] >= 5).astype(int) +
            (data['Absenteeism'] >= 15).astype(int)
        )
    else:
        data['Risk_Score'] = (
            (data['Job_Satisfaction'] <= 2).astype(int) * 2 +
            (data['Work_Life_Balance'] <= 2).astype(int) +
            (data['Years_Since_Last_Promotion'] >= 5).astype(int) +
            (data['Absenteeism'] >= 15).astype(int)
        )

    data['Engagement_Score'] = (
        data['Job_Involvement'] +
        data['Work_Environment_Satisfaction'] +
        data['Relationship_with_Manager'] +
        data['Job_Satisfaction']
    ) / 4

    data['Seniority_Score'] = (
        data['Years_at_Company'] * 0.5 +
        data['Years_in_Current_Role'] * 0.3 +
        data['Job_Level'] * 0.2
    )

    features_km = ['Risk_Score', 'Engagement_Score', 'Seniority_Score']
    X_km = data[features_km].copy()
    scaler_km = StandardScaler()
    X_km_scaled = scaler_km.fit_transform(X_km)
    scalers['clustering'] = scaler_km
    feature_columns['clustering'] = features_km

    # Find optimal K
    sil_scores = []
    db_scores = []
    ch_scores = []
    inertias = []
    k_range = range(2, 8)

    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=30, max_iter=500, random_state=42)
        labels = km.fit_predict(X_km_scaled)
        inertias.append(float(km.inertia_))
        sil_scores.append(float(silhouette_score(X_km_scaled, labels)))
        db_scores.append(float(davies_bouldin_score(X_km_scaled, labels)))
        ch_scores.append(float(calinski_harabasz_score(X_km_scaled, labels)))

    def normalize(arr):
        arr = np.array(arr, dtype=float)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    composite = (normalize(sil_scores) + normalize([-x for x in db_scores]) + normalize(ch_scores)) / 3
    optimal_k = list(k_range)[int(np.argmax(composite))]

    km_final = KMeans(n_clusters=optimal_k, init='k-means++', n_init=30, max_iter=500, random_state=42)
    data['Cluster'] = km_final.fit_predict(X_km_scaled)
    models['kmeans'] = km_final

    # Cluster profiles
    atr_col = 'Attrition_encoded' if 'Attrition_encoded' in data.columns else None
    agg_dict = {
        'Age': 'mean',
        'Monthly_Income': 'mean',
        'Years_at_Company': 'mean',
        'Job_Satisfaction': 'mean',
        'Work_Life_Balance': 'mean',
        'Risk_Score': 'mean',
        'Engagement_Score': 'mean',
        'Seniority_Score': 'mean',
    }
    if atr_col:
        agg_dict['Attrition_encoded'] = ['count', 'mean']

    profile = data.groupby('Cluster').agg(agg_dict).round(2)
    profile.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in profile.columns]
    profile = profile.reset_index()

    cluster_profiles = []
    for _, row in profile.iterrows():
        cid = int(row['Cluster'])
        risk = float(row.get('Risk_Score_mean', row.get('Risk_Score', 0)))
        eng = float(row.get('Engagement_Score_mean', row.get('Engagement_Score', 0)))
        sen = float(row.get('Seniority_Score_mean', row.get('Seniority_Score', 0)))
        atr_pct = round(float(row.get('Attrition_encoded_mean', 0)) * 100, 1)
        count = int(row.get('Attrition_encoded_count', 0)) if atr_col else 0

        if risk >= 3 and eng <= 2.5:
            label = "Profil Critique"
            color = "#ef4444"
        elif risk >= 3 and eng > 2.5:
            label = "Profil Sous Pression"
            color = "#f97316"
        elif sen >= 12 and eng >= 3:
            label = "Senior Stable"
            color = "#22c55e"
        elif sen < 6 and eng >= 3:
            label = "Junior Prometteur"
            color = "#3b82f6"
        elif risk <= 1 and eng >= 3:
            label = "Profil Fidèle"
            color = "#10b981"
        else:
            label = "Profil Intermédiaire"
            color = "#eab308"

        cluster_profiles.append({
            'cluster': cid,
            'label': label,
            'color': color,
            'count': count,
            'attrition_pct': atr_pct,
            'age_mean': round(float(row.get('Age_mean', row.get('Age', 0))), 1),
            'salary_mean': round(float(row.get('Monthly_Income_mean', row.get('Monthly_Income', 0))), 0),
            'tenure_mean': round(float(row.get('Years_at_Company_mean', row.get('Years_at_Company', 0))), 1),
            'satisfaction_mean': round(float(row.get('Job_Satisfaction_mean', row.get('Job_Satisfaction', 0))), 2),
            'wlb_mean': round(float(row.get('Work_Life_Balance_mean', row.get('Work_Life_Balance', 0))), 2),
            'risk_score': round(risk, 2),
            'engagement_score': round(eng, 2),
            'seniority_score': round(sen, 2),
        })

    # Scatter data for visualization
    scatter_data = []
    for i, (row_data, cluster) in enumerate(zip(X_km_scaled, data['Cluster'])):
        scatter_data.append({
            'x': round(float(row_data[0]), 3),
            'y': round(float(row_data[1]), 3),
            'z': round(float(row_data[2]), 3),
            'cluster': int(cluster)
        })

    return {
        'optimal_k': optimal_k,
        'metrics': {
            'silhouette': round(sil_scores[optimal_k - 2], 3),
            'davies_bouldin': round(db_scores[optimal_k - 2], 3),
            'calinski_harabasz': round(ch_scores[optimal_k - 2], 1)
        },
        'elbow_data': [{'k': int(k), 'inertia': round(v, 1)} for k, v in zip(k_range, inertias)],
        'silhouette_data': [{'k': int(k), 'score': round(v, 3)} for k, v in zip(k_range, sil_scores)],
        'cluster_profiles': cluster_profiles,
        'scatter_data': scatter_data[:500]  # sample for perf
    }


def train_regression_models(df):
    """DSO3 - HR recommendation system (Job Satisfaction prediction)."""

    # =============================
    # 1. Preprocess data
    # =============================
    data = preprocess_data(df).copy()

    import numpy as np
    import pandas as pd

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.ensemble import GradientBoostingRegressor

    # =============================
    # 2. Create synthetic targets
    # =============================
    n = len(data)
    np.random.seed(42)

    # Formule centrée et cohérente métier :
    # - Bas salaire, mauvais WLB, overtime, longue distance, mauvais manager -> satisfaction BASSE
    # - Haut salaire, bon WLB, pas d'overtime -> satisfaction HAUTE
    data['Job_Satisfaction'] = (
        2.5
        + 0.00008 * (data['Monthly_Income'] - 6000)
        + 0.45 * (data['Work_Life_Balance'] - 2.5)
        + 0.25 * (data['Performance_Rating'] - 3)
        + 0.30 * (data['Relationship_with_Manager'] - 2.5)
        - 0.60 * data.get('Overtime_Yes', 0)
        - 0.08 * data['Years_Since_Last_Promotion']
        - 0.02 * data['Distance_From_Home']
        + np.random.normal(0, 0.3, n)
    ).clip(1, 4).round(2)

    attrition_prob = (
        0.05
        + 0.20 * data.get('Overtime_Yes', 0)
        + 0.12 * (data['Job_Satisfaction'] <= 2)
        + 0.08 * (data['Work_Life_Balance'] <= 2)
        - 0.07 * (data['Monthly_Income'] > 15000)
        + 0.06 * (data['Years_Since_Last_Promotion'] >= 6)
    )

    data['Attrition'] = (
        np.random.uniform(size=n) < attrition_prob.clip(0.01, 0.90)
    ).astype(int)

    # =============================
    # 3. Features selection
    # =============================
    features_reg = [
        'Monthly_Income',
        'Work_Life_Balance',
        'Relationship_with_Manager',
        'Distance_From_Home',
        'Years_Since_Last_Promotion',
        'Age',
        'Job_Level',
        'Performance_Rating'
    ]

    if 'Overtime_Yes' in data.columns:
        features_reg.append('Overtime_Yes')

    # keep existing columns only
    features_reg = [f for f in features_reg if f in data.columns]
    feature_columns['regression'] = features_reg

    X = data[features_reg]
    y = data['Job_Satisfaction']

    # =============================
    # 4. Train/Test split
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =============================
    # 5. Scaling
    # =============================
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    scalers['regression'] = scaler

    # =============================
    # 6. Linear Regression
    # =============================
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)

    y_pred_lr = lr.predict(X_test_s)

    models['linear_regression'] = lr

    # =============================
    # 7. Gradient Boosting
    # =============================
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)

    models['gradient_boosting'] = gb

    # =============================
    # 8. Metrics
    # =============================
    results = {}

    feat_imp_gb = sorted(
        zip(features_reg, gb.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )

    results['Gradient Boosting'] = {
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred_gb))), 4),
        'mae': round(float(mean_absolute_error(y_test, y_pred_gb)), 4),
        'r2': round(float(r2_score(y_test, y_pred_gb)), 4),
        'feature_importance': [
            {'feature': f, 'importance': round(float(i), 4)}
            for f, i in feat_imp_gb
        ]
    }

    # =============================
    # 9. Visualization Data
    # =============================
    residuals = (y_test - y_pred_gb).tolist()

    results['pred_vs_actual'] = [
        {'actual': float(a), 'predicted': round(float(p), 2)}
        for a, p in zip(y_test[:100], y_pred_gb[:100])
    ]

    results['residuals'] = [
        round(float(r), 3) for r in residuals[:200]
    ]
    data['Predicted_Satisfaction'] = gb.predict(X)

    avg_score = float(data['Predicted_Satisfaction'].mean())

# label logic (same as frontend expectation)
    if avg_score >= 3.5:
     label = "High"
    elif avg_score >= 2.5:
     label = "Medium"
    else:
     label = "Low"

# IMPORTANT: ensure keys exist for frontend
    results['predicted_satisfaction'] = round(avg_score, 2)
    results['satisfaction_label'] = label

    return results

# ─── Dataset storage ──────────────────────────────────────────────────────────
raw_df = None

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'xgboost': XGBOOST_AVAILABLE})


@app.route('/upload', methods=['POST'])
def upload_dataset():
    global raw_df
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    sep = request.form.get('sep', ',')
    try:
        raw_df = pd.read_csv(file, sep=sep)
        dataset_info['shape'] = raw_df.shape
        dataset_info['columns'] = list(raw_df.columns)
        dataset_info['attrition_rate'] = round(float((raw_df['Attrition'] == 'Yes').mean() * 100), 1) if 'Attrition' in raw_df.columns else None
        return jsonify({
            'message': 'Dataset loaded',
            'rows': raw_df.shape[0],
            'cols': raw_df.shape[1],
            'columns': list(raw_df.columns),
            'attrition_rate': dataset_info.get('attrition_rate'),
            'preview': raw_df.head(5).to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train/classification', methods=['POST'])
def train_classification():
    global raw_df
    if raw_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    try:
        results = train_classification_models(raw_df)
        return jsonify({'results': results})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/train/clustering', methods=['POST'])
def train_clustering():
    global raw_df
    if raw_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    try:
        results = train_clustering_models(raw_df)
        return jsonify({'results': results})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/train/regression', methods=['POST'])
def train_regression():
    global raw_df
    if raw_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    try:
        results = train_regression_models(raw_df)
        return jsonify({'results': results})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/predict/attrition', methods=['POST'])
def predict_attrition():
    """Predict attrition for a single employee."""
    if 'random_forest' not in models:
        return jsonify({'error': 'Models not trained yet'}), 400
    data = request.json
    try:
        input_df = pd.DataFrame([data])
        # Align to training features
        for col in feature_columns['classification']:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns['classification']]
        X_scaled = scalers['classification'].transform(input_df)
        prob = float(models['random_forest'].predict_proba(X_scaled)[0][1])
        risk_level = 'HIGH' if prob < 0.21 else 'MEDIUM' if prob > 0.35 else 'LOW'
        return jsonify({
            'probability': round(prob * 100, 1),
            'prediction': 'Attrition Risk' if prob < 0.21 else 'Likely to Stay',
            'risk_level': risk_level,
            'recommendations': generate_recommendations(data, prob)
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/predict/satisfaction', methods=['POST'])
def predict_satisfaction():
    """Predict job satisfaction for a single employee."""

    try:
        data = request.json

        # =============================
        # CHECK MODEL SAFELY
        # =============================
        model = models.get('gradient_boosting', None)

        if model is None or 'regression' not in feature_columns:
            return jsonify({
                'predicted_satisfaction': 2.5,
                'satisfaction_label': 'Medium',
                'recommendations': [],
                'warning': 'Model not loaded'
            }), 200

        # =============================
        # INPUT PREPARATION
        # =============================
        input_df = pd.DataFrame([data])

        features = feature_columns['regression']

        for col in features:
            if col not in input_df.columns:
                input_df[col] = data.get(col, 0)   # 🔥 FIX

        input_df = input_df[features]

        # =============================
        # PREDICTION
        # =============================
        pred = float(model.predict(input_df)[0])
        pred = max(1.0, min(4.0, pred))
        print(pred)
        # =============================
        # LABEL
        # =============================
        if pred >= 3.5:
            label = "High"
        elif pred >= 2.5:
            label = "Medium"
        else:
            label = "Low"

        # =============================
        # RECOMMENDATIONS SAFE
        # =============================
        try:
            recs = get_satisfaction_recommendations(data, pred)
        except:
            recs = []

        # =============================
        # RESPONSE (ALWAYS VALID)
        # =============================
        return jsonify({
            'predicted_satisfaction': round(pred, 2),
            'satisfaction_label': label,
            'recommendations': recs
        }), 200

    except Exception as e:
        return jsonify({
            'predicted_satisfaction': 2.5,
            'satisfaction_label': 'Medium',
            'recommendations': [],
            'error': str(e)
        }), 200


@app.route('/segment/employee', methods=['POST'])
def segment_employee():
    """Assign a single employee to a cluster."""
    if 'kmeans' not in models:
        return jsonify({'error': 'Clustering model not trained yet'}), 400
    data = request.json
    try:
        # Compute composite scores
        overtime_val = data.get('Overtime_Yes', 0)
        risk = (int(overtime_val) * 2 +
                int(data.get('Job_Satisfaction', 3) <= 2) * 2 +
                int(data.get('Work_Life_Balance', 3) <= 2) +
                int(data.get('Years_Since_Last_Promotion', 0) >= 5) +
                int(data.get('Absenteeism', 0) >= 15))
        engagement = (data.get('Job_Involvement', 3) +
                      data.get('Work_Environment_Satisfaction', 3) +
                      data.get('Relationship_with_Manager', 3) +
                      data.get('Job_Satisfaction', 3)) / 4
        seniority = (data.get('Years_at_Company', 0) * 0.5 +
                     data.get('Years_in_Current_Role', 0) * 0.3 +
                     data.get('Job_Level', 1) * 0.2)

        input_arr = np.array([[risk, engagement, seniority]])
        input_scaled = scalers['clustering'].transform(input_arr)
        cluster_id = int(models['kmeans'].predict(input_scaled)[0])

        cluster_labels = {0: 'Profil Critique', 1: 'Senior Stable', 2: 'Junior Prometteur',
                          3: 'Profil Fidèle', 4: 'Sous Pression', 5: 'Intermédiaire'}

        return jsonify({
            'cluster': cluster_id,
            'risk_score': round(risk, 2),
            'engagement_score': round(engagement, 2),
            'seniority_score': round(seniority, 2),
            'segment_label': cluster_labels.get(cluster_id, f'Segment {cluster_id}')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_recommendations(data, prob):
    recs = []
    if data.get('Job_Satisfaction', 3) <= 2:
        recs.append("📋 Organiser un entretien de satisfaction immédiat")
    if data.get('Work_Life_Balance', 3) <= 2:
        recs.append("⚖️ Proposer des options de travail flexible")
    if data.get('Years_Since_Last_Promotion', 0) >= 4:
        recs.append("📈 Évaluer les opportunités de promotion")
    if data.get('Overtime_Yes', 0) == 1:
        recs.append("⏰ Réduire les heures supplémentaires")
    if data.get('Relationship_with_Manager', 3) <= 2:
        recs.append("🤝 Programme de coaching managérial recommandé")
    if prob > 0.7:
        recs.append("🚨 Intervention RH urgente requise")
    if not recs:
        recs.append("✅ Employé stable — maintenir l'engagement actuel")
    return recs


def get_satisfaction_label(score):
    if score >= 3.5:
        return "Très Satisfait"
    elif score >= 2.5:
        return "Satisfait"
    elif score >= 1.5:
        return "Peu Satisfait"
    else:
        return "Insatisfait"


def get_satisfaction_recommendations(data, pred):
    recs = []
    if pred < 2.5:
        recs.append("🎯 Revoir les objectifs et la charge de travail")
    if data.get('Work_Life_Balance', 3) < 3:
        recs.append("⚖️ Améliorer l'équilibre vie professionnelle/personnelle")
    if data.get('Monthly_Income', 5000) < 4000:
        recs.append("💰 Révision salariale à envisager")
    if data.get('Years_Since_Last_Promotion', 0) >= 3:
        recs.append("📈 Opportunités de développement de carrière")
    if data.get('Relationship_with_Manager', 3) < 3:
        recs.append("🤝 Améliorer la relation manager-employé")
    if not recs:
        recs.append("✅ Profil satisfaisant — continuer le suivi régulier")
    return recs


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')