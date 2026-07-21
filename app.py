import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="NSCLC Radiomics Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"]{
    background:#111827;
}
[data-testid="stSidebar"] *{
    color:white;
}
.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 5px 20px rgba(0,0,0,.08);
}
.title{
    font-size:38px;
    font-weight:bold;
}
.subtitle{
    color:gray;
    margin-bottom:20px;
}
</style>
""",unsafe_allow_html=True)

MODEL_DIR=Path("model")

@st.cache_resource
def load_model():
    model=joblib.load(MODEL_DIR/"model.pkl")
    scaler=joblib.load(MODEL_DIR/"scaler.pkl")
    selector=joblib.load(MODEL_DIR/"selector.pkl")
    feature_columns=joblib.load(MODEL_DIR/"feature_columns.pkl")
    label_encoder=joblib.load(MODEL_DIR/"label_encoder.pkl")
    return model,scaler,selector,feature_columns,label_encoder

model,scaler,selector,feature_columns,label_encoder=load_model()

# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:
    st.title("🫁 NSCLC Dashboard")
    st.caption("Machine Learning Prototype")

    st.divider()

    st.subheader("Model Status")
    st.success("✔ XGBoost Loaded")
    st.success("✔ Scaler Loaded")
    st.success("✔ Feature Selector Loaded")
    st.success("✔ Label Encoder Loaded")

    st.divider()

    st.subheader("Model Information")

    info=pd.DataFrame({
        "Property":[
            "Algorithm",
            "Task",
            "Input",
            "Output",
            "Selected Feature"
        ],
        "Value":[
            "XGBoost",
            "Multi-class Classification",
            "Radiomics CSV",
            "NSCLC Subtype",
            "225"
        ]
    })

    st.dataframe(
        info,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.caption("Universitas Gunadarma")
    st.caption("Radiomics-Based NSCLC Classification")

# ======================================================
# HEADER
# ======================================================

st.markdown('<div class="title">🫁 NSCLC Radiomics Classification</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prototype Deployment using XGBoost and Radiomics Features</div>',unsafe_allow_html=True)

# ======================================================
# METRICS
# ======================================================

c1,c2,c3,c4=st.columns(4)

c1.metric(
    "Algorithm",
    "XGBoost"
)

c2.metric(
    "Accuracy",
    "62%"
)

c3.metric(
    "Classes",
    "3"
)

c4.metric(
    "Selected Features",
    "225"
)

st.divider()

# ======================================================
# MAIN TABS
# ======================================================

tab1,tab2,tab3=st.tabs([
    "📄 Dataset",
    "🤖 Prediction",
    "📊 Analytics"
])

# ======================================================
# TAB 1
# ======================================================

with tab1:

    st.subheader("Upload Radiomics Dataset")

    uploaded_file=st.file_uploader(
        "Choose CSV File",
        type="csv",
        help="Upload radiomics feature dataset."
    )

    if uploaded_file is not None:

        try:
            df=pd.read_csv(uploaded_file)

            st.toast(
                "Dataset loaded successfully.",
                icon="✅"
            )

        except Exception as e:

            st.error(e)
            st.stop()

        left,right=st.columns([3,1])

        with left:
            st.write("### Dataset Preview")
            st.dataframe(
                df.head(10),
                use_container_width=True
            )

        with right:
            st.write("### Dataset Summary")

            st.metric(
                "Rows",
                len(df)
            )

            st.metric(
                "Columns",
                len(df.columns)
            )

            st.metric(
                "Missing Values",
                int(df.isna().sum().sum())
            )

            st.metric(
                "Duplicate Rows",
                int(df.duplicated().sum())
            )

# ======================================================
# TAB 2 - PREDICTION
# ======================================================

with tab2:

    st.subheader("NSCLC Prediction")

    if uploaded_file is None:
        st.info("Please upload a radiomics dataset in the Dataset tab first.")
        st.stop()

    required_cols={"patient_id","label"}

    if not required_cols.issubset(df.columns):
        st.error("Dataset must contain 'patient_id' and 'label' columns.")
        st.stop()

    patient_id=df["patient_id"]

    X=df.drop(columns=["patient_id","label"],errors="ignore")

    missing_features=[col for col in feature_columns if col not in X.columns]
    extra_features=[col for col in X.columns if col not in feature_columns]

    col1,col2=st.columns(2)

    with col1:
        st.metric("Required Features",len(feature_columns))

    with col2:
        st.metric("Dataset Features",X.shape[1])

    if missing_features:

        st.error(f"{len(missing_features)} required features are missing.")

        with st.expander("Show Missing Features"):
            st.write(missing_features)

        st.stop()

    if extra_features:

        st.warning(f"{len(extra_features)} unused features detected.")

        with st.expander("Show Extra Features"):
            st.write(extra_features)

    X=X[feature_columns]

    st.success("Dataset validation completed.")

    st.divider()

    if st.button("🚀 Start Prediction",type="primary",use_container_width=True):

        progress=st.progress(0,text="Initializing...")

        with st.spinner("Running preprocessing..."):

            progress.progress(20,text="Scaling features...")
            X_scaled=scaler.transform(X)

            progress.progress(45,text="Selecting features...")
            X_selected=selector.transform(X_scaled)

            progress.progress(70,text="Running XGBoost model...")
            prediction=model.predict(X_selected)

            progress.progress(90,text="Calculating confidence...")

            try:
                probability=model.predict_proba(X_selected)
                confidence=np.max(probability,axis=1)*100
            except:
                confidence=np.repeat(np.nan,len(prediction))

            prediction=label_encoder.inverse_transform(prediction)

        progress.progress(100,text="Prediction completed.")

        result=pd.DataFrame({
            "Patient ID":patient_id,
            "Prediction":prediction,
            "Confidence (%)":np.round(confidence,2)
        })

        st.session_state["result"]=result

        st.success("Prediction finished successfully.")

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        csv=result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Prediction Result",
            data=csv,
            file_name="prediction_result.csv",
            mime="text/csv",
            use_container_width=True
        )

# ======================================================
# TAB 3 - ANALYTICS
# ======================================================

with tab3:

    st.subheader("Prediction Analytics")

    if "result" not in st.session_state:
        st.info("Run prediction first.")
        st.stop()

    result=st.session_state["result"]

    col1,col2,col3=st.columns(3)

    col1.metric(
        "Total Patient",
        len(result)
    )

    col2.metric(
        "Average Confidence",
        f"{result['Confidence (%)'].mean():.2f}%"
    )

    col3.metric(
        "Highest Confidence",
        f"{result['Confidence (%)'].max():.2f}%"
    )

    st.divider()

    left,right=st.columns([1,1])

    with left:

        st.markdown("#### Prediction Distribution")

        prediction_count=(
            result["Prediction"]
            .value_counts()
            .reset_index()
        )

        prediction_count.columns=[
            "Subtype",
            "Total"
        ]

        fig=px.pie(
            prediction_count,
            names="Subtype",
            values="Total",
            hole=.45
        )

        fig.update_layout(
            height=450,
            margin=dict(l=10,r=10,t=30,b=10)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown("#### Confidence Score")

        fig=px.bar(
            result,
            x="Patient ID",
            y="Confidence (%)",
            color="Prediction"
        )

        fig.update_layout(
            height=450,
            xaxis_title=None,
            yaxis_title="Confidence (%)",
            showlegend=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.markdown("### Search Patient")

    patient=st.selectbox(
        "Patient ID",
        result["Patient ID"]
    )

    patient_result=result[
        result["Patient ID"]==patient
    ]

    c1,c2,c3=st.columns(3)

    c1.metric(
        "Patient",
        patient_result.iloc[0]["Patient ID"]
    )

    c2.metric(
        "Prediction",
        patient_result.iloc[0]["Prediction"]
    )

    c3.metric(
        "Confidence",
        f"{patient_result.iloc[0]['Confidence (%)']:.2f}%"
    )

    st.divider()

    st.markdown("### Prediction Summary")

    summary=(
        result.groupby("Prediction")
        .agg(
            Total=("Prediction","count"),
            Avg_Confidence=("Confidence (%)","mean")
        )
        .reset_index()
    )

    summary["Avg_Confidence"]=summary[
        "Avg_Confidence"
    ].round(2)

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    with st.expander("View Prediction Result"):

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

# ======================================================
# FOOTER
# ======================================================

st.divider()

col1,col2=st.columns([3,1])

with col1:

    st.caption(
        """
        **NSCLC Radiomics Classification Prototype**

        This application implements an XGBoost model for predicting
        NSCLC histopathological subtype based on radiomics features
        extracted from CT images.

        Workflow:
        Radiomics Features → StandardScaler →
        SelectKBest → XGBoost → Prediction
        """
    )

with col2:

    st.info(
        """
**Model**

XGBoost

Version 1.0
"""
    )

# ======================================================
# SYSTEM STATUS
# ======================================================

with st.expander("⚙ System Status"):

    status=pd.DataFrame({
        "Component":[
            "Model",
            "Scaler",
            "Feature Selector",
            "Label Encoder"
        ],
        "Status":[
            "Loaded",
            "Loaded",
            "Loaded",
            "Loaded"
        ]
    })

    st.dataframe(
        status,
        hide_index=True,
        use_container_width=True
    )

# ======================================================
# RESET BUTTON
# ======================================================

col1,col2,col3=st.columns([1,1,5])

with col1:

    if st.button("🔄 Reset"):

        if "result" in st.session_state:
            del st.session_state["result"]

        st.cache_resource.clear()

        st.success("Application has been reset.")

        st.rerun()

# ======================================================
# WARNING
# ======================================================

if "result" in st.session_state:

    low_confidence=st.session_state["result"][
        st.session_state["result"]["Confidence (%)"]<60
    ]

    if len(low_confidence)>0:

        st.warning(
            f"{len(low_confidence)} prediction(s) have confidence below 60%."
        )

# ======================================================
# DATASET QUALITY CHECK
# ======================================================

if uploaded_file is not None:

    with st.expander("📋 Dataset Quality Report"):

        report=pd.DataFrame({

            "Item":[
                "Rows",
                "Columns",
                "Missing Values",
                "Duplicate Rows",
                "Required Features"
            ],

            "Value":[
                len(df),
                len(df.columns),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                len(feature_columns)
            ]

        })

        st.dataframe(
            report,
            hide_index=True,
            use_container_width=True
        )

# ======================================================
# ABOUT PROJECT
# ======================================================

with st.expander("📖 About This Project"):

    st.markdown("""
This application is developed as a research prototype for undergraduate thesis.

### Classification Model
- XGBoost Classifier

### Feature Extraction
- PyRadiomics

### Feature Engineering
- Correlation Filtering
- StandardScaler
- SelectKBest

### Dataset
TCIA NSCLC-Radiomics Dataset

### Classes
- Adenocarcinoma
- Large Cell Carcinoma
- Squamous Cell Carcinoma

### Deployment
Streamlit
""")

# ======================================================
# COPYRIGHT
# ======================================================

st.markdown(
    """
---
<center>

Developed by **Salsa Tashfiyatul Qolbi**

Bachelor of Computer Science

Universitas Gunadarma

2026

</center>
""",
    unsafe_allow_html=True
)
