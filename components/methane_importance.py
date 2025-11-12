from streamlit import markdown, image, container

def display_methane_importance():
    with container():
        # Cover Image
        image("assets/images/cover/hero-banner.jpg", use_column_width=True)

        # Importance of Methane
        markdown("""
        ## Importance of Methane
        Methane (CH₄) is a potent greenhouse gas that has a significant impact on climate change. It is over 25 times more effective than carbon dioxide at trapping heat in the atmosphere over a 100-year period. Understanding methane emissions is crucial for mitigating climate change and protecting the environment.

        ### Key Facts:
        - Methane is responsible for approximately 30% of the global warming observed since the pre-industrial era.
        - Major sources of methane emissions include agriculture (especially livestock), landfills, natural gas production, and wetlands.
        - Reducing methane emissions is one of the most effective strategies for slowing climate change in the short term.

        ![Climate Warming Impact](assets/images/methane-impact/climate-warming.jpg)

        ### Atmospheric Concentration
        The concentration of methane in the atmosphere has been rising steadily, contributing to global warming and climate change.

        ![Atmospheric Concentration](assets/images/methane-impact/atmospheric-concentration.jpg)

        ### Emission Sources
        Understanding the various sources of methane emissions is essential for developing effective mitigation strategies.

        ![Emission Sources](assets/images/methane-impact/emission-sources.jpg)
        """)

        # Current Technologies for Measuring Emissions
        markdown("""
        ## Current Technologies for Measuring Methane Emissions
        Several advanced technologies are currently being utilized to monitor and measure methane emissions effectively:

        - **Hyperspectral Imaging**: This technology captures a wide spectrum of light to identify methane concentrations in the atmosphere.
        ![Hyperspectral Imaging](assets/images/technology/hyperspectral-imaging.jpg)

        - **Lidar Detection**: Lidar (Light Detection and Ranging) uses laser pulses to measure distances and detect methane emissions from various sources.
        ![Lidar Detection](assets/images/technology/lidar-detection.jpg)

        - **Ground Sensors**: These sensors are deployed on the ground to provide real-time data on methane concentrations.
        ![Ground Sensors](assets/images/technology/ground-sensors.jpg)
        """)

        # Satellite Technologies
        markdown("""
        ## Satellite Technologies for Methane Monitoring
        Satellite technologies play a crucial role in monitoring methane emissions on a global scale. Here are some key satellite missions:

        - **Carbon Mapper**: This satellite mission aims to provide high-resolution data on methane emissions, helping to identify and quantify sources.
        ![Carbon Mapper](assets/images/satellites/carbon-mapper.jpg)

        - **MethaneSat**: Designed to monitor methane emissions globally, MethaneSat will provide critical data for climate action.
        ![MethaneSat](assets/images/satellites/methanesat.jpg)

        - **GHGSat**: This satellite technology focuses on monitoring greenhouse gas emissions from industrial facilities worldwide.
        ![GHGSat](assets/images/satellites/ghgsat.jpg)

        - **AVIRIS-NG**: A hyperspectral imaging satellite that provides detailed information on methane concentrations in the atmosphere.
        ![AVIRIS-NG](assets/images/satellites/aviris-ng.jpg)
        """)

# Call the function to display the content
display_methane_importance()