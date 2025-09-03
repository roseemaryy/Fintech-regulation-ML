import streamlit as st
from main import FintechRegulationML

st.title("Fintech Regulation ML Dashboard")

if st.button("Run Full Analysis"):
    project = FintechRegulationML()
    project.run_complete_analysis()
    st.success("✅ Analysis Completed! See console for details")
