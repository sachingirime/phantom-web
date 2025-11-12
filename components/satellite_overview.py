from datetime import datetime
import streamlit as st

# Satellite Overview Component
def satellite_overview():
    st.markdown("<h2 style='text-align: center;'>Satellite Technologies for Methane Detection</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Methane is a potent greenhouse gas that significantly contributes to climate change. Monitoring methane emissions is crucial for understanding and mitigating its impact on the environment. Satellite technologies have emerged as powerful tools for detecting and quantifying methane emissions from various sources. Below are some of the leading satellite technologies used in this field:
    """)

    # Satellite Technologies
    satellites = [
        {
            "name": "Carbon Mapper",
            "image": "assets/images/satellites/carbon-mapper.jpg",
            "description": "Carbon Mapper is designed to identify and quantify methane emissions from various sources, providing critical data for climate action."
        },
        {
            "name": "MethaneSat",
            "image": "assets/images/satellites/methanesat.jpg",
            "description": "MethaneSat aims to provide global coverage of methane emissions, enabling better tracking and management of this potent greenhouse gas."
        },
        {
            "name": "GHGSat",
            "image": "assets/images/satellites/ghgsat.jpg",
            "description": "GHGSat uses advanced satellite technology to monitor greenhouse gas emissions from industrial facilities, offering insights into emission sources."
        },
        {
            "name": "AVIRIS-NG",
            "image": "assets/images/satellites/aviris-ng.jpg",
            "description": "AVIRIS-NG is a hyperspectral imaging technology that captures detailed spectral information, aiding in the detection of methane and other gases."
        }
    ]

    for satellite in satellites:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(satellite["image"], use_column_width=True)
        with col2:
            st.markdown(f"**{satellite['name']}**")
            st.markdown(satellite["description"])

    st.markdown("""
    ### Importance of Monitoring Methane Emissions
    Methane is over 25 times more effective at trapping heat in the atmosphere than carbon dioxide over a 100-year period. Reducing methane emissions is essential for achieving climate goals and improving air quality. Satellite technologies play a vital role in providing accurate and timely data to inform policy decisions and emission reduction strategies.
    """)

    st.markdown("""
    ### Current Technologies for Measuring Emissions
    Various technologies are employed to measure methane emissions, including:
    - **Hyperspectral Imaging**: Captures detailed spectral data to identify gas concentrations.
    - **Lidar Detection**: Uses laser pulses to measure distances and detect gas emissions.
    - **Ground Sensors**: Provide localized measurements of methane concentrations.

    These technologies complement satellite observations, providing a comprehensive understanding of methane emissions and their sources.
    """)

# Call the satellite overview function to render the component
satellite_overview()