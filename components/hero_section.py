from streamlit import markdown, image

def hero_section():
    # Cover Image
    markdown("<h1 style='text-align: center;'>Welcome to PHANTOM</h1>", unsafe_allow_html=True)
    image("assets/images/cover/hero-banner.jpg", use_column_width=True)

    # Importance of Methane
    markdown("""
    <h2 style='text-align: center;'>Why Methane Matters</h2>
    <p style='text-align: center;'>
        Methane is a potent greenhouse gas with a global warming potential over 25 times greater than CO2 over a 100-year period. 
        It is responsible for approximately 25% of current global warming. 
        Understanding and monitoring methane emissions is crucial for climate change mitigation.
    </p>
    """, unsafe_allow_html=True)

    # Current Technologies for Measuring Emissions
    markdown("<h2 style='text-align: center;'>Current Technologies for Measuring Methane Emissions</h2>", unsafe_allow_html=True)
    markdown("""
    <div style='display: flex; justify-content: center; gap: 20px;'>
        <div style='text-align: center;'>
            <h3>Hyperspectral Imaging</h3>
            <img src='assets/images/technology/hyperspectral-imaging.jpg' width='200'/>
        </div>
        <div style='text-align: center;'>
            <h3>Lidar Detection</h3>
            <img src='assets/images/technology/lidar-detection.jpg' width='200'/>
        </div>
        <div style='text-align: center;'>
            <h3>Ground Sensors</h3>
            <img src='assets/images/technology/ground-sensors.jpg' width='200'/>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Satellite Technologies
    markdown("<h2 style='text-align: center;'>Satellite Technologies for Methane Monitoring</h2>", unsafe_allow_html=True)
    markdown("""
    <div style='display: flex; justify-content: center; gap: 20px;'>
        <div style='text-align: center;'>
            <h3>Carbon Mapper</h3>
            <img src='assets/images/satellites/carbon-mapper.jpg' width='200'/>
        </div>
        <div style='text-align: center;'>
            <h3>MethaneSat</h3>
            <img src='assets/images/satellites/methanesat.jpg' width='200'/>
        </div>
        <div style='text-align: center;'>
            <h3>GHGSat</h3>
            <img src='assets/images/satellites/ghgsat.jpg' width='200'/>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Conclusion
    markdown("""
    <h2 style='text-align: center;'>Join Us in the Fight Against Climate Change</h2>
    <p style='text-align: center;'>
        By leveraging advanced technologies and data, we can better understand and mitigate methane emissions.
    </p>
    """, unsafe_allow_html=True)