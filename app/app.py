
import streamlit as st
import pandas as pd
import joblib
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="SmartHire",
    page_icon="💼",
    layout="wide"
)

st.title("💼 SmartHire")
st.subheader("Resume-to-Job Matching & Career Guidance Engine")

st.write(
    "Analyze your resume, predict your career category, "
    "find suitable jobs, and identify skill gaps."
)

BASE_DIR = "/content/SmartHire_Final_Backup"

# ============================================
# SKILL LIST
# ============================================

SKILL_LIST = [
    "python", "java", "c++", "c", "javascript",
    "html", "css", "sql", "mysql", "mongodb",
    "oracle", "postgresql", "pandas", "numpy",
    "scikit-learn", "tensorflow", "keras",
    "pytorch", "machine learning", "deep learning",
    "data science", "data analysis", "nlp",
    "power bi", "tableau", "excel",
    "aws", "azure", "gcp", "docker",
    "kubernetes", "git", "github", "linux",
    "spark", "hadoop", "hive", "etl",
    "flask", "django", "spring", "react",
    "angular", "selenium", "testing",
    "automation testing", "devops",
    "jenkins", "terraform",
    "cyber security", "network security"
]

def extract_skills(text):
    text = str(text).lower()
    found = []

    for skill in SKILL_LIST:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


# ============================================
# RESUME INPUT
# ============================================

resume_text = st.text_area(
    "📄 Paste your resume here:",
    height=300,
    placeholder="Paste your resume text here..."
)


# ============================================
# ANALYZE
# ============================================

if st.button("🚀 Analyze Resume", use_container_width=True):

    if not resume_text.strip():
        st.warning("Please enter your resume text first.")
        st.stop()

    with st.spinner("Analyzing your resume..."):

        try:
            # Load models only when Analyze is clicked
            classifier = joblib.load(
                os.path.join(BASE_DIR, "models", "resume_classifier.pkl")
            )

            resume_tfidf = joblib.load(
                os.path.join(BASE_DIR, "models", "resume_tfidf.pkl")
            )

            job_tfidf = joblib.load(
                os.path.join(BASE_DIR, "models", "job_tfidf.pkl")
            )

            jobs = pd.read_csv(
                os.path.join(
                    BASE_DIR,
                    "data",
                    "processed",
                    "naukri_jobs_processed.csv"
                )
            )

            # ------------------------------------
            # Career prediction
            # ------------------------------------

            resume_vector = resume_tfidf.transform([resume_text])

            predicted_category = classifier.predict(
                resume_vector
            )[0]

            st.success(
                f"🎯 Predicted Career Category: **{predicted_category}**"
            )

            # ------------------------------------
            # Job recommendation
            # ------------------------------------

            job_vector = job_tfidf.transform([resume_text])

            job_matrix = job_tfidf.transform(
                jobs["job_text"]
            )

            similarity_scores = cosine_similarity(
                job_vector,
                job_matrix
            ).flatten()

            results = jobs.copy()

            results["Match Score (%)"] = (
                similarity_scores * 100
            )

            results = results.sort_values(
                "Match Score (%)",
                ascending=False
            )

            top_jobs = results.head(10)

            # ------------------------------------
            # Display recommendations
            # ------------------------------------

            st.header("💼 Top 10 Job Recommendations")

            display_columns = [
                column for column in [
                    "jobtitle",
                    "company",
                    "joblocation_address",
                    "skills",
                    "Match Score (%)"
                ]
                if column in top_jobs.columns
            ]

            st.dataframe(
                top_jobs[display_columns],
                use_container_width=True
            )

            # ------------------------------------
            # Skill Gap Analysis
            # ------------------------------------

            st.header("🧠 Skill Gap Analysis")

            resume_skills = set(
                extract_skills(resume_text)
            )

            top_jobs_text = " ".join(
                top_jobs["job_text"].astype(str)
            )

            required_skills = set(
                extract_skills(top_jobs_text)
            )

            matched_skills = sorted(
                resume_skills & required_skills
            )

            missing_skills = sorted(
                required_skills - resume_skills
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Resume Skills",
                    len(resume_skills)
                )

            with col2:
                st.metric(
                    "Matched Skills",
                    len(matched_skills)
                )

            with col3:
                st.metric(
                    "Missing Skills",
                    len(missing_skills)
                )

            st.write("### ✅ Skills You Have")

            if resume_skills:
                st.write(
                    ", ".join(sorted(resume_skills))
                )
            else:
                st.write("No skills detected.")

            st.write("### 🎯 Matching Skills")

            if matched_skills:
                st.write(
                    ", ".join(matched_skills)
                )
            else:
                st.write("No matching skills detected.")

            st.write("### ❌ Skills You May Need")

            if missing_skills:
                st.write(
                    ", ".join(missing_skills)
                )
            else:
                st.success(
                    "Great! No major missing skills detected."
                )

            st.success(
                "🎉 SmartHire analysis completed successfully!"
            )

        except Exception as e:
            st.error("Unable to analyze the resume.")
            st.exception(e)
