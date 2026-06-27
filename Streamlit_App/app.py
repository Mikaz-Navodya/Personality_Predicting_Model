import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set up page configuration
st.set_page_config(
    page_title="Personality Predictor Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------------------
# 1. RESOURCE LOADING & CACHING
# -------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Loads the preprocessor and registry of trained models securely."""
    artifacts = {
        "preprocessor": None,
        "models": {},
        "target_mapping": {0: "Extrovert", 1: "Introvert"},
        "errors": []
    }
    
    # Load Preprocessor
    if os.path.exists("preprocessor.pkl"):
        try:
            artifacts["preprocessor"] = joblib.load("preprocessor.pkl")
        except Exception as e:
            artifacts["errors"].append(f"Failed to load preprocessor: {e}")
    else:
        artifacts["errors"].append("Missing core file: 'preprocessor.pkl'")
        
    # Model Registry Configuration
    model_files = {
        "K-Nearest Neighbours": "model_k_nearest_neighbours.pkl",
        "Random Forest": "model_random_forest.pkl",
        "Logistic Regression": "model_logistic_regression.pkl",
        "Gradient Boosting": "model_gradient_boosting.pkl",
        "Support Vector Machine": "model_support_vector_machine.pkl"
    }
    
    for display_name, file_name in model_files.items():
        if os.path.exists(file_name):
            try:
                artifacts["models"][display_name] = joblib.load(file_name)
            except Exception as e:
                artifacts["errors"].append(f"Failed to load model '{display_name}': {e}")
        else:
            artifacts["errors"].append(f"Missing model artifact: '{file_name}'")
            
    return artifacts

# Initialize artifacts
registry = load_artifacts()

# -------------------------------------------------------------------------
# 2. TOP-LEFT HEALTH BADGE (Sidebar completely wiped)
# -------------------------------------------------------------------------
health_col, __spacer = st.columns([1.5, 4.5]) 

with health_col:
    if registry["errors"]:
        st.error(f"⚠️ **Artifact Health:** {len(registry['errors'])} Missing")
        for err in registry["errors"]:
            st.caption(f"└ {err}")
    else:
        st.success("🟢 **Artifact Health:** 100% Active")

st.markdown("---")
# -------------------------------------------------------------------------
# 3. MAIN WORKSPACE (TAB NAVIGATION)
# -------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Real-time Prediction Workspace", "📊 Model Architecture Leaderboard"])


# =========================================================================
# TAB 1: PREDICTION VIEW
# =========================================================================
with tab1:
    st.title("Personality Type Analysis Interface")
    st.markdown(
        "Input personal behavioral metrics below to infer a user's baseline classification "
        "(`Extrovert` vs. `Introvert`) across our tuned machine learning suite."
    )
    

    if os.path.exists("decoration.png"):
        st.image("decoration.png", use_container_width=True)
    
    st.markdown("---")
    
    if registry["errors"]:
        st.warning("⚠️ **Inference Locked:** Please restore the missing model files indicated in the top-left corner.")
        st.stop()

    with st.form("inference_input_form"):
        col1, col2 = st.columns(2)
        
        # ---> [CHANGE 2]: DESCRIPTIVE HINTS UPGRADED <---
        with col1:
            st.subheader("Numerical Behavioral Metrics")
            
            time_spent_alone = st.slider(
                "Time Spent Alone (Hours/Day)", 0, 11, 4, 
                help="Average daily hours spent in solitary activities (reading, solo gaming, personal hobbies). Baseline dataset range: 0 to 11 hours."
            )
            social_attendance = st.slider(
                "Social Event Attendance", 0, 10, 4, 
                help="Self-reported monthly frequency of voluntarily attending parties, meetups, or group outings. 0 = Absolute isolation; 10 = Constant social presence."
            )
            going_outside = st.slider(
                "Going Outside Frequency", 0, 7, 3, 
                help="Number of days per week the subject leaves their residence for non-mandatory, leisure-based reasons."
            )
            friends_circle = st.slider(
                "Active Friends Circle Size", 0, 15, 6, 
                help="The headcount of close, active confidants the subject maintains regular, bidirectional communication with."
            )
            post_frequency = st.slider(
                "Social Media Post Frequency", 0, 10, 3, 
                help="Weekly broadcasting volume across public digital networks (Instagram, X, TikTok, etc.)."
            )
            
        with col2:
            st.subheader("Psychological Indicators")
            
            stage_fear = st.selectbox(
                "Experiences Stage / Public Speaking Fear?", ["Yes", "No"], index=0,
                help="Select 'Yes' if the subject manifests acute acute physiological or psychological distress when speaking to an audience."
            )
            drained_socializing = st.selectbox(
                "Experiences Post-Socialization Fatigue?", ["Yes", "No"], index=0,
                help="Select 'Yes' if the subject experiences a pronounced 'social battery drain' requiring quiet isolation to recover baseline energy."
            )
            
            st.markdown("---")
            st.subheader("Model")
            selected_model_name = st.selectbox(
                "Routing Classifier", 
                list(registry["models"].keys()),
                help="Select which mathematical model weights will be used to resolve the final boundary decision."
            )
            
        submit_btn = st.form_submit_button("Predict", type="primary")

    if submit_btn:
        input_data = pd.DataFrame([{
            'Time_spent_Alone': float(time_spent_alone),
            'Stage_fear': stage_fear,
            'Social_event_attendance': float(social_attendance),
            'Going_outside': float(going_outside),
            'Drained_after_socializing': drained_socializing,
            'Friends_circle_size': float(friends_circle),
            'Post_frequency': float(post_frequency)
        }])
        
        try:
            processed_features = registry["preprocessor"].transform(input_data)
            active_model = registry["models"][selected_model_name]
            prediction_encoded = active_model.predict(processed_features)[0]
            
            probabilities = None
            if hasattr(active_model, "predict_proba"):
                probabilities = active_model.predict_proba(processed_features)[0]
            elif hasattr(active_model, "decision_function"):
                decision = active_model.decision_function(processed_features)[0]
                probabilities = [1 / (1 + np.exp(decision)), 1 / (1 + np.exp(-decision))]

            decoded_prediction = registry["target_mapping"][prediction_encoded]
            
            st.markdown("---")
            st.subheader("Inference Resolution")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                if decoded_prediction == "Extrovert":
                    st.success(f"### Predicted Class:\n# **{decoded_prediction}**")
                else:
                    st.info(f"### Predicted Class:\n# **{decoded_prediction}**")
                    
            with res_col2:
                if probabilities is not None:
                    prob_df = pd.DataFrame({
                        'Class': ['Extrovert', 'Introvert'],
                        'Probability': [probabilities[0], probabilities[1]]
                    })
                    st.bar_chart(data=prob_df, x='Class', y='Probability', color="#440154")
                    st.caption(f"Model Confidence: Extrovert ({probabilities[0]:.1%}) | Introvert ({probabilities[1]:.1%})")
                    
        except Exception as e:
            st.error(f"Inference Pipeline Failure: {e}")


# =========================================================================
# TAB 2: LEADERBOARD VIEW
# =========================================================================
with tab2:
    st.title("Model Training & Architecture Benchmark")
    st.markdown("Independent test-set evaluation metrics extracted during the training lifecycle.")
    st.markdown("---")
    
    leaderboard_data = pd.DataFrame([
        {"Rank": 1, "Model Architecture": "K-Nearest Neighbours", "CV F1": 0.9257, "Test Accuracy": 0.9165, "Test F1 (Macro)": 0.9156, "Optimized Hyperparameters": "{'n_neighbors': 7, 'weights': 'uniform'}"},
        {"Rank": 2, "Model Architecture": "Random Forest", "CV F1": 0.9268, "Test Accuracy": 0.9145, "Test F1 (Macro)": 0.9136, "Optimized Hyperparameters": "{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 50}"},
        {"Rank": 3, "Model Architecture": "Logistic Regression", "CV F1": 0.9288, "Test Accuracy": 0.9145, "Test F1 (Macro)": 0.9136, "Optimized Hyperparameters": "{'C': 0.1}"},
        {"Rank": 4, "Model Architecture": "Gradient Boosting", "CV F1": 0.9293, "Test Accuracy": 0.9145, "Test F1 (Macro)": 0.9136, "Optimized Hyperparameters": "{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}"},
        {"Rank": 5, "Model Architecture": "Support Vector Machine", "CV F1": 0.9293, "Test Accuracy": 0.9145, "Test F1 (Macro)": 0.9136, "Optimized Hyperparameters": "{'C': 0.1, 'kernel': 'rbf'}"}
    ]).set_index("Rank")
    
    st.dataframe(leaderboard_data, use_container_width=True)


    st.markdown("---")
    st.subheader("📈 Training & Diagnostic Visualizations")
    st.caption("Rendered directly from the data-benchmarking Jupyter Notebook outputs.")
    
    graph_row1_col1, graph_row1_col2 = st.columns(2)
    with graph_row1_col1:
        if os.path.exists("correlation_matrix.png"):
            st.image("correlation_matrix.png", caption="Fig 1.0: Feature Correlation Matrix", use_container_width=True)
                   
    with graph_row1_col2:
        if os.path.exists("class_balance.png"):
            st.image("class_balance.png", caption="Fig 1.1: Target Class Imbalance Check", use_container_width=True)
            
    if os.path.exists("confusion_matrices.png"):
        st.image("confusion_matrices.png", caption="Fig 1.2: 2x3 Confusion Matrix Showdown Grid", use_container_width=True)

    if os.path.exists("model_f1_accuracy.png"):
        st.image("model_f1_accuracy.png", caption="Figure 1.3: Model Benachmark", use_container_width=True)
