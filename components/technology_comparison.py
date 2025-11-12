from streamlit import st

# Title for the technology comparison section
st.title("Current Technologies for Measuring Methane Emissions")

# Introduction
st.markdown("""
Methane is a potent greenhouse gas that significantly contributes to climate change. Accurate measurement of methane emissions is crucial for effective monitoring and mitigation strategies. This section compares various technologies currently used for measuring methane emissions, highlighting their features, advantages, and limitations.
""")

# Technology Comparison Table
st.subheader("Technology Comparison")

# Create a table to compare technologies
technology_data = {
    "Technology": [
        "Hyperspectral Imaging",
        "Lidar Detection",
        "Ground Sensors",
        "Carbon Mapper",
        "MethaneSat",
        "GHGSat"
    ],
    "Description": [
        "Uses spectral imaging to detect methane concentrations from the air.",
        "Employs laser technology to measure methane levels and distances.",
        "Utilizes sensors placed on the ground to monitor methane emissions directly.",
        "A satellite-based system designed specifically for mapping methane emissions.",
        "A satellite dedicated to monitoring methane emissions globally.",
        "Provides satellite-based monitoring of greenhouse gases, including methane."
    ],
    "Advantages": [
        "High spatial resolution and sensitivity.",
        "Can cover large areas quickly.",
        "Direct measurement of emissions.",
        "High accuracy and detailed mapping.",
        "Global coverage with frequent revisits.",
        "Comprehensive greenhouse gas monitoring."
    ],
    "Limitations": [
        "Requires clear atmospheric conditions.",
        "Limited by range and atmospheric conditions.",
        "Limited spatial coverage.",
        "High operational costs.",
        "Dependent on satellite positioning.",
        "Less frequent data updates."
    ]
}

# Display the technology comparison table
st.table(technology_data)

# Images for each technology
st.subheader("Visual Representation of Technologies")

# Display images
col1, col2, col3 = st.columns(3)

with col1:
    st.image("assets/images/technology/hyperspectral-imaging.jpg", caption="Hyperspectral Imaging")

with col2:
    st.image("assets/images/technology/lidar-detection.jpg", caption="Lidar Detection")

with col3:
    st.image("assets/images/technology/ground-sensors.jpg", caption="Ground Sensors")

# Satellite Technologies
st.subheader("Satellite Technologies for Methane Monitoring")

# Display satellite images
satellite_col1, satellite_col2, satellite_col3 = st.columns(3)

with satellite_col1:
    st.image("assets/images/satellites/carbon-mapper.jpg", caption="Carbon Mapper")

with satellite_col2:
    st.image("assets/images/satellites/methanesat.jpg", caption="MethaneSat")

with satellite_col3:
    st.image("assets/images/satellites/ghgsat.jpg", caption="GHGSat")

# Conclusion
st.markdown("""
The advancement of technology in methane detection is crucial for addressing climate change. Each technology has its unique strengths and weaknesses, making them suitable for different applications. By leveraging these technologies, we can enhance our understanding of methane emissions and develop effective strategies for mitigation.
""")