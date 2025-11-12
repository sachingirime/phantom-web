from streamlit import st
import folium
from streamlit_folium import st_folium

def create_interactive_map(data):
    # Create a base map
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=7)

    # Add markers for each emission site
    for idx, row in data.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Emission Rate: {row['Q_kg_hr']} kg/hr",
            icon=folium.Icon(color='blue')
        ).add_to(m)

    return m

def display_methane_importance():
    st.markdown("""
    ## Importance of Methane
    Methane (CH4) is a potent greenhouse gas with a global warming potential many times greater than carbon dioxide over a short time frame. It is responsible for a significant portion of climate change and has various sources, including agriculture, landfills, and fossil fuel extraction. Understanding and monitoring methane emissions is crucial for mitigating climate change impacts.
    """)

def display_current_technologies():
    st.markdown("""
    ## Current Technologies for Measuring Methane Emissions
    - **Hyperspectral Imaging**: Utilizes advanced imaging techniques to detect methane concentrations from aerial platforms.
    - **Lidar Detection**: Employs laser-based technology to measure methane levels in the atmosphere.
    - **Ground Sensors**: Deploys sensors on the ground to monitor methane emissions in real-time.
    """)

def display_satellite_images():
    st.markdown("""
    ## Satellite Technologies for Methane Detection
    - **Carbon Mapper**: A satellite-based system designed to monitor methane emissions globally.
      ![Carbon Mapper](assets/images/satellites/carbon-mapper.jpg)
      
    - **MethaneSat**: Focuses on providing high-resolution data on methane emissions from various sources.
      ![MethaneSat](assets/images/satellites/methanesat.jpg)
      
    - **GHGSat**: Offers satellite imagery to monitor greenhouse gas emissions, including methane.
      ![GHGSat](assets/images/satellites/ghgsat.jpg)
    """)

def interactive_map_component(data):
    st.header("Interactive Methane Emission Map")
    map_object = create_interactive_map(data)
    st_folium(map_object, width=700, height=500)

# Main function to render the interactive map component
def render_interactive_map(data):
    display_methane_importance()
    display_current_technologies()
    display_satellite_images()
    interactive_map_component(data)