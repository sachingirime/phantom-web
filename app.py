# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime
# import numpy as np
# import folium
# import folium.plugins
# from streamlit_folium import st_folium
# from pyproj import Transformer
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import rasterio
# from rasterio.enums import Resampling
# import os
# import pickle
# import cv2
# from PIL import Image
# import base64

# # Page config
# st.set_page_config(
#     page_title="PHANTOM - Methane Detection Platform", 
#     layout="wide", 
#     page_icon="🛰️",
#     initial_sidebar_state="collapsed"
# )

# def get_base64_image(image_path):
#     """Convert image to base64 for CSS background"""
#     try:
#         with open(image_path, "rb") as img_file:
#             return base64.b64encode(img_file.read()).decode()
#     except:
#         return None

# # Enhanced CSS with VERY LARGE FONTS, reduced gaps, and !important color fix
# st.markdown("""
# <style>
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     .stDeployButton {visibility: hidden;}
    
#     /* Main app styling */
#     .stApp {
#         background: #ffffff;
#     }
    
#     /* Typography */
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
#     html, body, [class*="css"] {
#         font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
#     }
    
#     /* Hero section (dark background, white text - OK) */
#     .hero-section {
#         position: relative;
#         text-align: center;
#         padding: 5rem 2rem 4rem 2rem; 
#         background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.84) 100%);
#         color: white;
#         margin: -2rem -2rem 1rem -2rem; 
#         background-size: cover;
#         background-position: center;
#     }
    
#     .hero-title {
#         font-size: 10rem !important; 
#         font-weight: 300;
#         margin-bottom: 1.5rem;
#         letter-spacing: -0.03em;
#         text-shadow: 0 4px 12px rgba(0,0,0,0.5);
#     }
    
#     .hero-acronym {
#         font-size: 3rem !important; 
#         font-weight: 200;
#         margin-bottom: 2.5rem;
#         opacity: 0.95;
#         letter-spacing: 0.08em;
#         text-transform: uppercase;
#     }
    
#     .hero-subtitle {
#         font-size: 2rem !important; 
#         font-weight: 600 !important;
#         margin: 0 auto 4rem auto;
#         line-height: 1.8;
#         opacity: 0.98;
#     }
    
#     .hero-stats {
#         display: flex;
#         justify-content: center;
#         gap: 6rem;
#         margin-top: 5rem;
#     }
    
#     .hero-stat {
#         text-align: center;
#     }
    
#     .hero-stat-value {
#         font-size: 5rem; 
#         font-weight: 800;
#         display: block;
#         margin-bottom: 1rem;
#         text-shadow: 0 2px 8px rgba(0,0,0,0.3);
#     }
    
#     .hero-stat-label {
#         font-size: 2.5rem; 
#         opacity: 0.95;
#         text-transform: uppercase;
#         letter-spacing: 0.15em;
#         font-weight: 500;
#     }
    
#     /* Section container */
#     .section-container {
#         max-width: 1400px;
#         margin: 0 auto;
#         padding: 0 2rem;
#     }
    
#     /* --- FIX FOR WHITE TEXT --- */
    
#     /* Section styling */
#     .section-header {
#         font-size: 5rem !important; 
#         font-weight: 800;
#         color: #0f172a !important; /* <--- FIX */
#         text-align: center;
#         margin-bottom: 2rem;
#         letter-spacing: -0.02em;
#     }
    
#     .section-subheader {
#         font-size: 3rem !important; 
#         color: #475569 !important; /* <--- FIX */
#         text-align: center;
#         margin-bottom: 2rem; 
#         margin-left: auto;
#         margin-right: auto;
#         line-height: 1.9;
#         font-weight: 400;
#     }
    
#     /* Subsection headers */
#     .subsection-title {
#         font-size: 3.5rem !important; 
#         font-weight: 700;
#         color: #0f172a !important; /* <--- FIX */
#         margin: 1.5rem 0 1.5rem 0; 
#         text-align: center;
#     }
    
#     .subsection-intro {
#         font-size: 2.2rem !important; 
#         color: #64748b !important; /* <--- FIX */
#         text-align: center;
#         margin-bottom: 2rem; 
#         line-height: 1.8;
#     }
    
#     /* What We Do section */
#     .what-we-do-section {
#         background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
#         padding: 1.5rem 0 1.5rem 0; 
#         margin: 0.5rem 0; 
#     }
    
#     .feature-card {
#         background: white;
#         padding: 3.5rem 3rem;
#         border-radius: 24px;
#         box-shadow: 0 8px 16px rgba(0,0,0,0.08);
#         border: 1px solid #e2e8f0;
#         transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
#         height: 100%; 
#     }
    
#     .feature-card:hover {
#         transform: translateY(-8px);
#         box-shadow: 0 20px 40px rgba(0,0,0,0.15);
#         border-color: #2563eb;
#     }
    
#     .feature-icon {
#         font-size: 5rem;
#         margin-bottom: 2rem;
#         display: block;
#     }
    
#     .feature-title {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #0f172a !important; /* <--- FIX */
#         margin-bottom: 1.5rem;
#         letter-spacing: -0.01em;
#     }
    
#     .feature-text {
#         font-size: 2.5rem; 
#         color: #64748b !important; /* <--- FIX */
#         line-height: 1.9;
#         font-weight: 400;
#     }
    
#     /* Impact section */
#     .impact-section {
#         background: #ffffff;
#         padding: 1.5rem 0 1.5rem 0; 
#         margin: 0.5rem 0; 
#     }
    
#     .impact-card {
#         background: white;
#         padding: 3.5rem 3rem;
#         border-radius: 24px;
#         box-shadow: 0 8px 16px rgba(0,0,0,0.08);
#         border: 1px solid #e2e8f0;
#         transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
#         height: 100%; 
#     }
    
#     .impact-card:hover {
#         transform: translateY(-8px);
#         box-shadow: 0 20px 40px rgba(0,0,0,0.15);
#     }
    
#     .impact-number {
#         font-size: 6rem;
#         font-weight: 800;
#         color: #dc2626; /* No !important, color is fine */
#         margin-bottom: 1.5rem;
#         letter-spacing: -0.02em;
#     }
    
#     .impact-title {
#         font-size: 2.25rem;
#         font-weight: 700;
#         color: #0f172a !important; /* <--- FIX */
#         margin-bottom: 1.5rem;
#         letter-spacing: -0.01em;
#     }
    
#     .impact-text {
#         font-size: 2.5rem; 
#         color: #475569 !important; /* <--- FIX */
#         line-height: 1.9;
#         font-weight: 400;
#     }
    
#     /* Stats highlight (dark background, white text - OK) */
#     .stats-highlight {
#         background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
#         color: white;
#         padding: 5rem 4rem;
#         border-radius: 32px;
#         text-align: center;
#         margin: 1.5rem 0; 
#         box-shadow: 0 20px 40px rgba(30, 64, 175, 0.3);
#     }
    
#     .stats-grid {
#         display: grid;
#         grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
#         gap: 5rem;
#         margin-top: 4rem;
#     }
    
#     .stat-item {
#         text-align: center;
#     }
    
#     .stat-value {
#         font-size: 6rem;
#         font-weight: 900;
#         display: block;
#         margin-bottom: 1rem;
#         text-shadow: 0 2px 8px rgba(0,0,0,0.2);
#     }
    
#     .stat-label {
#         font-size: 1.75rem;
#         opacity: 0.95;
#         text-transform: uppercase;
#         letter-spacing: 0.15em;
#         font-weight: 500;
#     }
    
#     /* Content blocks */
#     .content-block {
#         background: #f8fafc;
#         padding: 3.5rem 3rem;
#         border-radius: 24px;
#         margin: 1.5rem 0; 
#         border-left: 6px solid #2563eb;
#     }
    
#     .content-block-title {
#         font-size: 2.75rem;
#         font-weight: 700;
#         color: #0f172a !important; /* <--- FIX */
#         margin-bottom: 1.5rem;
#     }
    
#     .content-block-text {
#         font-size: 2.5rem; 
#         color: #475569 !important; /* <--- FIX */
#         line-height: 1.9;
#         font-weight: 400;
#     }
    
#     /* Image containers */
#     .img-container {
#         border-radius: 20px;
#         overflow: hidden;
#         box-shadow: 0 12px 28px rgba(0,0,0,0.12);
#         margin: 1.5rem 0; 
#     }
    
#     .img-container img {
#         width: 100%;
#         height: auto;
#         display: block;
#     }
    
#     .img-caption {
#         font-size: 2rem; 
#         color: #64748b !important; /* <--- FIX */
#         text-align: center;
#         margin-top: 1.5rem;
#         font-style: italic;
#     }
    
#     /* List styling */
#     .custom-list {
#         font-size: 2.5rem; 
#         color: #475569 !important; /* <--- FIX */
#         line-height: 2.2;
#         margin: 2rem 0;
#     }
    
#     .custom-list li {
#         margin-bottom: 1.25rem;
#         padding-left: 0.5rem;
#     }
    
#     /* NO Divider */
#     .section-divider {
#         display: none; /* Removed gaps */
#     }
    
#     /* Responsive */
#     @media (max-width: 768px) {
#         .hero-title { font-size: 5rem; } 
#         .hero-acronym { font-size: 2.5rem; }
#         .hero-subtitle { font-size: 2.5rem; }
#         .section-header { font-size: 2.75rem; }
#         .section-subheader { font-size: 2.2rem; }
#         .feature-title, .impact-title { font-size: 1.75rem; }
#         .feature-text, .impact-text, .content-block-text { font-size: 1.6rem; } 
#     }
# </style>
# """, unsafe_allow_html=True)

# # Model classes
# class TransformerBottleneck(nn.Module):
#     def __init__(self, in_channels, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
#         super().__init__()
#         self.in_channels = in_channels
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=in_channels, nhead=nhead, dim_feedforward=dim_feedforward,
#             dropout=dropout, batch_first=True, activation=F.gelu
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.positional_embedding = nn.Parameter(torch.randn(1, 64*64, in_channels))
    
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_flat = x.flatten(2).permute(0, 2, 1)
#         x_pos = x_flat + self.positional_embedding
#         transformer_out = self.transformer_encoder(x_pos)
#         out = transformer_out.permute(0, 2, 1).view(B, C, H, W)
#         return out

# class UNetGenerator(nn.Module):
#     def __init__(self, in_ch=6, base=32):
#         super().__init__()
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(True),
#             nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True)
#         )
#         self.pool = nn.MaxPool2d(2)
#         self.enc2 = nn.Sequential(
#             nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(True),
#             nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(True)
#         )
#         self.transformer_bottleneck = TransformerBottleneck(in_channels=base*2)
#         self.up = nn.ConvTranspose2d(base*2, base, 2, stride=2)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(True),
#             nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True)
#         )
#         self.out = nn.Conv2d(base, 1, 1)
    
#     def forward(self, x):
#         x1 = self.enc1(x)
#         x2 = self.enc2(self.pool(x1))
#         x_bottleneck = self.transformer_bottleneck(x2)
#         u = self.up(x_bottleneck)
#         u = self.dec1(torch.cat([u, x1], dim=1))
#         return self.out(u)

# CONFIG = {
#     'model_path': 'data/generator_model_full_dataset_128.pth',
#     'test_root': '/hdd/starcop/STARCOP_test',
#     'plume_cache_dir': 'data/plumes',
#     'band_files': [
#         "TOA_AVIRIS_2004nm.tif", "TOA_AVIRIS_2109nm.tif", "TOA_AVIRIS_2310nm.tif",
#         "TOA_AVIRIS_2350nm.tif", "TOA_AVIRIS_2360nm.tif",
#     ],
#     'mf_file': 'weight_mag1c.tif',
#     'mf_scale': 0.2,
#     'detection_threshold': 0.5,
# }

# os.makedirs(CONFIG['plume_cache_dir'], exist_ok=True)

# @st.cache_resource
# def load_model():
#     try:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = UNetGenerator(in_ch=6).to(device)
#         model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device, weights_only=True))
#         model.eval()
#         return model, device
#     except Exception as e:
#         return None, None

# G, DEVICE = load_model()

# def get_crs_from_geotiff(folder_name):
#     folder_path = os.path.join(CONFIG['test_root'], folder_name)
#     band_path = os.path.join(folder_path, CONFIG['band_files'][0])
#     if os.path.exists(band_path):
#         try:
#             with rasterio.open(band_path) as src:
#                 return src.crs
#         except:
#             pass
#     return None

# def transform_utm_to_latlon(folder_name, center_x, center_y):
#     crs = get_crs_from_geotiff(folder_name)
#     if crs is None:
#         crs = "EPSG:32611"
#     try:
#         transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
#         lon, lat = transformer.transform(center_x, center_y)
#         return lat, lon
#     except:
#         return None, None

# def load_scene_data(folder_path, resize_to=128):
#     try:
#         swir = []
#         for bf in CONFIG['band_files']:
#             path = os.path.join(folder_path, bf)
#             with rasterio.open(path) as src:
#                 arr = src.read(1, out_shape=(resize_to, resize_to), 
#                              resampling=Resampling.bilinear).astype(np.float32)
#             swir.append(arr)
        
#         swir = np.stack(swir, axis=0)
#         for i in range(5):
#             lo, hi = swir[i].min(), swir[i].max()
#             swir[i] = (swir[i] - lo) / max(hi - lo, 1e-6)
        
#         mf_path = os.path.join(folder_path, CONFIG['mf_file'])
#         with rasterio.open(mf_path) as src:
#             mf = src.read(1, out_shape=(resize_to, resize_to), 
#                          resampling=Resampling.bilinear).astype(np.float32)
#         mf = np.nan_to_num(mf, nan=0.0, posinf=0.0, neginf=0.0)
#         mf = (mf - mf.min()) / max(mf.max() - mf.min(), 1e-6)
        
#         swir_tensor = torch.from_numpy(swir)
#         mf_tensor = torch.from_numpy(mf).unsqueeze(0)
        
#         return swir_tensor, mf_tensor
#     except Exception as e:
#         return None, None

# def precompute_plume_polygons(folder_name):
#     cache_file = os.path.join(CONFIG['plume_cache_dir'], f"{folder_name}.pkl")
    
#     if os.path.exists(cache_file):
#         with open(cache_file, 'rb') as f:
#             return pickle.load(f)
    
#     if G is None:
#         return None
    
#     folder_path = os.path.join(CONFIG['test_root'], folder_name)
#     swir, mf = load_scene_data(folder_path)
    
#     if swir is None:
#         return None
    
#     with torch.no_grad():
#         swir_batch = swir.unsqueeze(0).to(DEVICE)
#         mf_batch = mf.unsqueeze(0).to(DEVICE)
#         inp = torch.cat([swir_batch, CONFIG['mf_scale'] * mf_batch], dim=1)
#         pred_logits = G(inp)
#         pred_prob = torch.sigmoid(pred_logits).cpu().numpy()[0, 0]
    
#     pred_mask = (pred_prob > CONFIG['detection_threshold']).astype(np.uint8)
#     contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     plume_polygons = []
    
#     if len(contours) > 0:
#         band_path = os.path.join(folder_path, CONFIG['band_files'][0])
        
#         try:
#             with rasterio.open(band_path) as src:
#                 transform = src.transform
#                 crs = src.crs
#                 transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                
#                 for contour in contours:
#                     if len(contour) < 3:
#                         continue
                    
#                     polygon_coords = []
#                     for point in contour:
#                         x_pixel, y_pixel = point[0]
#                         scale_x = src.width / 128.0
#                         scale_y = src.height / 128.0
#                         x_pixel_full = x_pixel * scale_x
#                         y_pixel_full = y_pixel * scale_y
#                         utm_x, utm_y = rasterio.transform.xy(transform, y_pixel_full, x_pixel_full)
#                         lon, lat = transformer.transform(utm_x, utm_y)
#                         polygon_coords.append([lat, lon])
                    
#                     if len(polygon_coords) >= 3:
#                         plume_polygons.append(polygon_coords)
#         except Exception as e:
#             print(f"Error extracting polygons for {folder_name}: {e}")
    
#     plume_data = {
#         'polygons': plume_polygons,
#         'det_pixels': int(pred_mask.sum()),
#         'max_conf': float(pred_prob.max()),
#         'mean_conf': float(pred_prob[pred_mask > 0].mean()) if pred_mask.sum() > 0 else 0
#     }
    
#     with open(cache_file, 'wb') as f:
#         pickle.dump(plume_data, f)
    
#     return plume_data

# @st.cache_data
# def load_data():
#     try:
#         df = pd.read_csv('data/emission_quantification_results.csv')
        
#         coords = []
#         for idx, row in df.iterrows():
#             lat, lon = transform_utm_to_latlon(row['folder_name'], row['center_lon'], row['center_lat'])
#             coords.append({'latitude': lat, 'longitude': lon})
        
#         coords_df = pd.DataFrame(coords)
#         df['latitude'] = coords_df['latitude']
#         df['longitude'] = coords_df['longitude']
#         df = df.dropna(subset=['latitude', 'longitude'])
        
#         df['date'] = pd.to_datetime(df['folder_name'].str.extract(r'ang(\d{8})')[0], format='%Y%m%d', errors='coerce')
#         df['time'] = df['folder_name'].str.extract(r't(\d{6})')[0]
#         df['time_formatted'] = pd.to_datetime(df['time'], format='%H%M%S', errors='coerce').dt.strftime('%H:%M:%S')
#         df['date_formatted'] = df['date'].dt.strftime('%Y-%m-%d')
        
#         df['emission_category'] = df['Q_kg_hr'].apply(
#             lambda x: 'Critical' if x >= 100 else 'Standard'
#         )
        
#         return df
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return pd.DataFrame()

# @st.cache_data
# def precompute_all_polygons(_df):
#     if G is None:
#         return
#     for idx, row in _df.iterrows():
#         cache_file = os.path.join(CONFIG['plume_cache_dir'], f"{row['folder_name']}.pkl")
#         if not os.path.exists(cache_file):
#             precompute_plume_polygons(row['folder_name'])

# df = load_data()
# if len(df) > 0:
#     precompute_all_polygons(df)

# # ============================================================================
# # HERO SECTION
# # ============================================================================
# hero_img_path = "assets/images/cover/earth-from-space.jpg"
# hero_bg = get_base64_image(hero_img_path)

# if hero_bg:
#     st.markdown(f"""
#     <div class="hero-section" style="background-image: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.84) 100%), url('data:image/jpeg;base64,{hero_bg}'); background-size: cover; background-position: center;">
#         <h1 class="hero-title">PHANTOM</h1>
#         <p class="hero-acronym">Physics-Informed Hyperspectral Adversarial Network for Transformer-Optimized Methane Detection</p>
#         <p class="hero-subtitle">
#             Advanced AI Pipeline for Methane Emission Detection and Quantification<br/>
#             from Aerial and Satellite Hyperspectral Imagery
#         </p>
#         <div class="hero-stats">
#             <div class="hero-stat">
#                 <span class="hero-stat-value">4×</span>
#                 <span class="hero-stat-label">More Accurate</span>
#             </div>
#             <div class="hero-stat">
#                 <span class="hero-stat-value">11.43%</span>
#                 <span class="hero-stat-label">False Positive Rate</span>
#             </div>
#             <div class="hero-stat">
#                 <span class="hero-stat-value">97 kg/hr</span>
#                 <span class="hero-stat-label">Verified Min. Detection Rate</span>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
# else:
#     st.markdown("""
#     <div class="hero-section">
#         <h1 class="hero-title">PHANTOM</h1>
#         <p class="hero-acronym">Physics-Informed Hyperspectral Adversarial Network for Transformer-Optimized Methane Detection</p>
#         <p class="hero-subtitle">
#             Advanced AI Platform for Real-Time Methane Emission Detection<br/>
#             from Satellite Hyperspectral Imagery
#         </p>
#         <div class="hero-stats">
#             <div class="hero-stat">
#                 <span class="hero-stat-value">4×</span>
#                 <span class="hero-stat-label">More Accurate</span>
#             </div>
#             <div class="hero-stat">
#                 <span class="hero-stat-value">11.43%</span>
#                 <span class="hero-stat-label">False Positive Rate</span>
#             </div>
#             <div class="hero-stat">
#                 <span class="hero-stat-value">&lt;100ms</span>
#                 <span class="hero-stat-label">Processing Time</span>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# # ============================================================================
# # WHAT WE DO SECTION
# # ============================================================================
# st.markdown('<div class="what-we-do-section">', unsafe_allow_html=True)
# st.markdown('<div class="section-container">', unsafe_allow_html=True)

# st.markdown('<h1 class="section-header">What We Do</h1>', unsafe_allow_html=True)
# st.markdown('<p class="section-subheader">PHANTOM combines cutting-edge artificial intelligence with satellite remote sensing to detect and quantify methane emissions in real-time, enabling rapid response to climate threats.</p>', unsafe_allow_html=True)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown("""
#     <div class="feature-card">
#         <span class="feature-icon">🛰️</span>
#         <div class="feature-title">Satellite Detection</div>
#         <div class="feature-text">
#             We use data from airborne platforms and satellites to look at the Earth. Our system can spot the invisible 
#             signature of methane gas from space, even in complex landscapes.
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="feature-card">
#         <span class="feature-icon">🧠</span>
#         <div class="feature-title">AI-Powered Analysis</div>
#         <div class="feature-text">
#             Our smart AI is trained to find methane plumes. It's extremely good at finding real leaks 
#             and ignoring things that just *look* like leaks, which keeps errors low.
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="feature-card">
#         <span class="feature-icon">📊</span>
#         <div class="feature-title">Emission Quantification</div>
#         <div class="feature-text">
#             Once we find a plume, our system automatically calculates *how much* methane is leaking. 
#             We report this in kilograms per hour, giving you a clear, actionable number.
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown('</div>', unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ============================================================================
# # LIVE DETECTION MAP SECTION
# # ============================================================================
# st.markdown('<div class="section-container">', unsafe_allow_html=True)

# st.markdown('<h3 class="subsection-title">Emission Detection Map</h3>', unsafe_allow_html=True)
# st.markdown('<p class="subsection-intro">Interactive map displaying methane detections with AI-generated plume boundaries and precise emission quantification</p>', unsafe_allow_html=True)

# if len(df) > 0:
#     # *** MODIFICATION START ***
#     # Find the SECOND largest emitter to center on
#     df_sorted = df.sort_values(by='Q_kg_hr', ascending=False)
    
#     # Check if there are at least 2 data points, otherwise fall back to the first
#     if len(df_sorted) > 1:
#         center_row = df_sorted.iloc[2] # Center on the second largest
#     else:
#         center_row = df_sorted.iloc[0] # Fallback to the largest
        
#     center_lat = center_row['latitude']
#     center_lon = center_row['longitude']
#     zoom_level = 15 # Zoom in close

#     m = folium.Map(
#         location=[center_lat, center_lon],
#         zoom_start=zoom_level,
#         tiles='CartoDB positron', # <-- BACK TO ORIGINAL DEFAULT
#         prefer_canvas=True,
#         control_scale=True,
#         scrollWheelZoom=True
#     )

#     # Add the satellite map as an *alternative* layer, just like your original code
#     folium.TileLayer('Esri WorldImagery', name='Satellite', overlay=True, control=True).add_to(m)
#     # *** MODIFICATION END ***

#     def create_simple_marker(color, size=20):
#         svg = f"""
#         <svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
#             <circle cx="{size/2}" cy="{size/2}" r="{size/2-2}" fill="{color}" opacity="0.85" stroke="white" stroke-width="2"/>
#         </svg>
#         """
#         return svg

#     for idx, row in df.iterrows():
#         if row['emission_category'] == 'Critical':
#             color = '#dc2626'
#             size = 24
#         else:
#             color = '#059669'
#             size = 16
        
#         popup_html = f"""
#         <div style="font-family: Inter, sans-serif; width: 380px; padding: 20px; background: white;">
#             <div style="background: {color}; color: white; padding: 12px; margin: -20px -20px 15px -20px; border-radius: 8px 8px 0 0;">
#                 <h3 style="margin: 0; font-size: 1.1rem; font-weight: 600;">{row['emission_category']} Emission</h3>
#             </div>
            
#             <table style="width: 100%; font-size: 0.9rem; border-collapse: collapse;">
          
#                 <tr>
#                     <td style="padding: 8px 0; font-weight: 500; color: #64748b;">Detection Time</td>
#                     <td style="padding: 8px 0;">{row['date_formatted']} {row['time_formatted']}</td>
#                 </tr>
#                 <tr style="background: #f8fafc;">
#                     <td style="padding: 12px 8px; font-weight: 600;">Emission Rate</td>
#                     <td style="padding: 12px 8px; color: {color}; font-weight: 700; font-size: 1.1rem;">
#                         {row['Q_kg_hr']:.2f} kg/hr
#                     </td>
#                 </tr>
     
#                 <tr>
#                     <td style="padding: 8px 0; font-weight: 500; color: #64748b;">Wind Speed (10 m)</td>
#                     <td style="padding: 8px 0;">{row['U_10_ms']:.2f} m/s</td>
#                 </tr>
#             </table>
#         </div>
#         """
        
#         folium.Marker(
#             location=[row['latitude'], row['longitude']],
#             popup=folium.Popup(popup_html, max_width=400),
#             tooltip=f"{row['Q_kg_hr']:.1f} kg/hr",
#             icon=folium.DivIcon(html=create_simple_marker(color, size))
#         ).add_to(m)
        
#         plume_data = precompute_plume_polygons(row['folder_name'])
        
#         if plume_data and plume_data['polygons']:
#             for polygon_coords in plume_data['polygons']:
#                 folium.Polygon(
#                     locations=polygon_coords,
#                     color=color,
#                     weight=2,
#                     fill=True,
#                     fillColor=color,
#                     fillOpacity=0.25,
#                     popup=folium.Popup(popup_html, max_width=400)
#                 ).add_to(m)

#     legend_html = f'''
#     <div style="position: fixed; top: 10px; right: 10px; background: white; 
#                 border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; 
#                 font-family: Inter, sans-serif; font-size: 0.875rem; z-index: 9999;
#                 box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
#         <div style="font-weight: 600; margin-bottom: 10px; color: #0f172a;">Detection Legend</div>
#         <div style="margin: 8px 0; display: flex; align-items: center;">
#             <div style="width: 16px; height: 16px; background: #dc2626; border-radius: 50%; margin-right: 8px;"></div>
#             <span>Critical (≥100 kg/hr)</span>
#         </div>
#         <div style="margin: 8px 0; display: flex; align-items: center;">
#             <div style="width: 14px; height: 14px; background: #059669; border-radius: 50%; margin-right: 10px;"></div>
#             <span>Standard (<100 kg/hr)</span>
#         </div>
#         <hr style="margin: 10px 0; border: 0; border-top: 1px solid #e2e8f0;">
#         <div style="font-size: 0.8rem; color: #64748b;">
#             <div>Sites: {len(df)}</div>
#             <div>Total: {df['Q_kg_hr'].sum():.0f} kg/hr</div>
#         </div>
#     </div>
#     '''
#     m.get_root().html.add_child(folium.Element(legend_html))
#     folium.LayerControl().add_to(m)
    
#     st_folium(m, width=None, height=900, returned_objects=[])

#     # Key stats
#     st.markdown("<br/>", unsafe_allow_html=True)
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Detection Sites", len(df), help="Number of emission events detected")
#     with col2:
#         st.metric("Critical Emitters", len(df[df['emission_category'] == 'Critical']), 
#                  help="Sites exceeding 100 kg/hr threshold")
#     with col3:
#         st.metric("Total Emissions", f"{df['Q_kg_hr'].sum():.0f} kg/hr", 
#                  help="Sum of all quantified emission rates")
#     with col4:
#         st.metric("Median Rate", f"{df['Q_kg_hr'].median():.1f} kg/hr",
#                  help="Median emission rate across all sites")

# st.markdown('</div>', unsafe_allow_html=True)

# # ============================================================================
# # WHY METHANE CONCERNS MATTER
# # ============================================================================
# st.markdown('<div class="impact-section">', unsafe_allow_html=True)
# st.markdown('<div class="section-container">', unsafe_allow_html=True)

# st.markdown('<h1 class="section-header">Why Methane Concerns Matter</h1>', unsafe_allow_html=True)
# st.markdown('<p class="section-subheader">Methane is responsible for 30% of global warming since pre-industrial times. Detecting and reducing methane emissions is our fastest opportunity to slow climate change in the next decade.</p>', unsafe_allow_html=True)

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown("""
#     <div class="impact-card">
#         <div class="impact-number">80×</div>
#         <div class="impact-title">Warming Potential</div>
#         <div class="impact-text">
#             Methane traps 80 times more heat than CO₂ over a 20-year period, making it the most critical 
#             greenhouse gas for immediate climate action. 
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="impact-card">
#         <div class="impact-number">30%</div>
#         <div class="impact-title">Global Warming Contribution</div>
#         <div class="impact-text">
#             Responsible for approximately 30% of global temperature rise since the Industrial Revolution. 
#             Its potency makes it second only to carbon dioxide in overall climate impact.
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="impact-card">
#         <div class="impact-number">45%</div>
#         <div class="impact-title">Reduction Potential by 2030</div>
#         <div class="impact-text">
#             Cutting methane emissions by 45% by 2030 could prevent 0.3°C of warming by 2040. 
#             Our fastest and most cost-effective climate solution.
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# # Methane sources subsection
# st.markdown('<h3 class="subsection-title">Global Emission Sources</h3>', unsafe_allow_html=True)

# col1, col2 = st.columns(2)

# with col1:
#     # *** REORDERED: Text block now comes FIRST ***
#     st.markdown("""
#     <div class="content-block">
#         <div class="content-block-title">Major Emission Sources</div>
#         <div class="content-block-text">
#             According to the Environmental Protection Agency, total U.S. methane emissions in 2020 were 650.4 million metric tons of CO2 equivalent, which is comparable to the emissions from 140 million cars. The primary sources of these emissions were Natural Gas and Petroleum (32%) and Enteric Fermentation (27%). Other significant contributors included Landfills (17%), Manure Management (9%), Coal Mining (6%), and other miscellaneous sources (9%).
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # *** Image now comes SECOND ***
#     try:
#         if os.path.exists('assets/images/methane-impact/methane-sources.png'):
#             img = Image.open('assets/images/methane-impact/methane-sources.png')
#             st.markdown('<div class="img-container">', unsafe_allow_html=True)
#             st.image(img, use_column_width=True)
#             st.markdown('</div>', unsafe_allow_html=True)
#             st.markdown('<p class="img-caption">Global Methane Emission Sources by Sector</p>', unsafe_allow_html=True)
#     except:
#         pass

# with col2:
#     # *** REORDERED: Text block now comes FIRST ***
#     # *** TYPO FIX: Changed class_name to class ***
#     st.markdown("""
#     <div class="content-block">
#         <div class="content-block-title">Why Satellite Detection is Critical</div>
#         <div class="content-block-text">
#          Data from NASA illustrates a significant global warming trend, to which methane contributes approximately 30% of the forcing. As conventional ground-based monitoring covers less than 5% of global infrastructure, satellite detection is the only viable solution. This technology is critical for identifying the "super-emitters" responsible for over 50% of hotspot emissions, offering the most rapid path to mitigating near-term warming.
#                 </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # *** Image now comes SECOND ***
#     try:
#         if os.path.exists('assets/images/methane-impact/climate-impact.png'):
#             img = Image.open('assets/images/methane-impact/climate-impact.png')
#             st.markdown('<div class="img-container">', unsafe_allow_html=True)
#             st.image(img, use_column_width=True)
#             st.markdown('</div>', unsafe_allow_html=True)
#             st.markdown('<p class="img-caption">Methane\'s Contribution to Global Temperature Rise</p>', unsafe_allow_html=True)
#     except:
#         pass

# st.markdown('</div>', unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ============================================================================
# # PLATFORM CAPABILITIES
# # ============================================================================
# st.markdown('<div class="section-container">', unsafe_allow_html=True)

# st.markdown('<h2 class="section-header">Platform Performance</h2>', unsafe_allow_html=True)
# st.markdown('<p class="section-subheader"> AI-driven methane detection validated on real-world satellite data</p>', unsafe_allow_html=True)

# st.markdown(f"""
# <div class="stats-highlight">
#     <h2 style="margin-bottom: 3rem; font-size: 3.5rem; font-weight: 800;">Benchmark Results</h2>
#     <div class="stats-grid">
#         <div class="stat-item">
#             <span class="stat-value">100%</span>
#             <span class="stat-label">F1 Score (Critical)</span>
#         </div>
#         <div class="stat-item">
#             <span class="stat-value">96.65%</span>
#             <span class="stat-label">F1 Score (Standard)</span>
#         </div>
#         <div class="stat-item">
#             <span class="stat-value">11.43%</span>
#             <span class="stat-label">False Positive Rate</span>
#         </div>
#         <div class="stat-item">
#             <span class="stat-value">62.42%</span>
#             <span class="stat-label">AUPRC</span>
#         </div>
#     </div>
#     <p style="margin-top: 3rem; font-size: 1.5rem; opacity: 0.95; line-height: 1.9;">
#         Validated on NASA AVIRIS-NG hyperspectral imagery across  real-world emission events<br/>
#         Tested on diverse geographic locations and environmental conditions
#     </p>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("""
# <div class="content-block">
#     <div class="content-block-title">Technical Innovation</div>
#     <div class="content-block-text">
#         PHANTOM represents a breakthrough in methane detection technology by combining physics-informed neural networks 
#         with transformer-based architecture. Our system processes Short Wave Infrared spectral bands from hyperspectral sensors, applying 
#         matched filter preprocessing and adversarial training to achieve state-of-the-art performance. The platform delivers 
#      exceptional accuracy—outperforming existing detection methods by a factor of four.
#     </div>
# </div>
# """, unsafe_allow_html=True)

# st.markdown('</div>', unsafe_allow_html=True)

# # ============================================================================
# # FOOTER
# # ============================================================================
# st.markdown(f"""
# <div style='margin-top: 1.5rem; padding: 3rem 3rem; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
#             color: white; text-align: center;'>
#     <h2 style='font-size: 4rem; font-weight: 800; margin-bottom: 2rem; letter-spacing: -0.02em;'>
#         PHANTOM Platform
#     </h2>
#     <p style='font-size: 2rem; margin-bottom: 3rem; opacity: 0.95; line-height: 1.8;'>
# Advanced AI platform for global-scale methane emission monitoring and quantification
# Validated on NASA AVIRIS-NG hyperspectral satellite imagery    </p>
#     <div style='font-size: 1.5rem; opacity: 0.85; max-width: 1000px; margin: 0 auto 3rem auto; line-height: 2;'>
#         Developed at MESA Lab, University of California, Merced • Supporting global climate action
#     </div>
#     <div style='font-size: 1.125rem; opacity: 0.7; margin-top: 3rem;'>
#         Last Updated: {datetime.now().strftime("%B %d, %Y")} 
#     </div>
# </div>
# """, unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import folium
import folium.plugins
from streamlit_folium import st_folium
from pyproj import Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from rasterio.enums import Resampling
import os
import pickle
import cv2
from PIL import Image
import base64

# Page config
st.set_page_config(
    page_title="PHANTOM - Methane Detection Platform", 
    layout="wide", 
    page_icon="🛰️",
    initial_sidebar_state="collapsed"
)

def get_base64_image(image_path):
    """Convert image to base64 for CSS background"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Enhanced CSS with VERY LARGE FONTS, reduced gaps, and !important color fix
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Main app styling */
    .stApp {
        background: #ffffff;
    }
    
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hero section (dark background, white text - OK) */
    .hero-section {
        position: relative;
        text-align: center;
        padding: 5rem 2rem 4rem 2rem; 
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.84) 100%);
        color: white;
        margin: -2rem -2rem 1rem -2rem; 
        background-size: cover;
        background-position: center;
    }
    
    .hero-title {
        font-size: 10rem !important; 
        font-weight: 300;
        margin-bottom: 1.5rem;
        letter-spacing: -0.03em;
        text-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    
    .hero-acronym {
        font-size: 3rem !important; 
        font-weight: 200;
        margin-bottom: 2.5rem;
        opacity: 0.95;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    
    .hero-subtitle {
        font-size: 2rem !important; 
        font-weight: 600 !important;
        margin: 0 auto 4rem auto;
        line-height: 1.8;
        opacity: 0.98;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 6rem;
        margin-top: 5rem;
    }
    
    .hero-stat {
        text-align: center;
    }
    
    .hero-stat-value {
        font-size: 5rem; 
        font-weight: 800;
        display: block;
        margin-bottom: 1rem;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .hero-stat-label {
        font-size: 2.5rem; 
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 500;
    }
    
    /* Section container */
    .section-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* --- FIX FOR WHITE TEXT --- */
    
    /* Section styling */
    .section-header {
        font-size: 5rem !important; 
        font-weight: 800;
        color: #0f172a !important; /* <--- FIX */
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
    }
    
    .section-subheader {
        font-size: 3rem !important; 
        color: #475569 !important; /* <--- FIX */
        text-align: center;
        margin-bottom: 2rem; 
        margin-left: auto;
        margin-right: auto;
        line-height: 1.9;
        font-weight: 400;
    }
    
    /* Subsection headers */
    .subsection-title {
        font-size: 3.5rem !important; 
        font-weight: 700;
        color: #0f172a !important; /* <--- FIX */
        margin: 1.5rem 0 1.5rem 0; 
        text-align: center;
    }
    
    .subsection-intro {
        font-size: 2.2rem !important; 
        color: #64748b !important; /* <--- FIX */
        text-align: center;
        margin-bottom: 2rem; 
        line-height: 1.8;
    }
    
    /* What We Do section */
    .what-we-do-section {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        padding: 1.5rem 0 1.5rem 0; 
        margin: 0.5rem 0; 
    }
    
    .feature-card {
        background: white;
        padding: 3.5rem 3rem;
        border-radius: 24px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%; 
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        border-color: #2563eb;
    }
    
    .feature-icon {
        font-size: 5rem;
        margin-bottom: 2rem;
        display: block;
    }
    
    .feature-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f172a !important; /* <--- FIX */
        margin-bottom: 1.5rem;
        letter-spacing: -0.01em;
    }
    
    .feature-text {
        font-size: 2.5rem; 
        color: #64748b !important; /* <--- FIX */
        line-height: 1.9;
        font-weight: 400;
    }
    
    /* Impact section */
    .impact-section {
        background: #ffffff;
        padding: 1.5rem 0 1.5rem 0; 
        margin: 0.5rem 0; 
    }
    
    .impact-card {
        background: white;
        padding: 3.5rem 3rem;
        border-radius: 24px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%; 
    }
    
    .impact-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .impact-number {
        font-size: 6rem;
        font-weight: 800;
        color: #dc2626; /* No !important, color is fine */
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    .impact-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0f172a !important; /* <--- FIX */
        margin-bottom: 1.5rem;
        letter-spacing: -0.01em;
    }
    
    .impact-text {
        font-size: 2.5rem; 
        color: #475569 !important; /* <--- FIX */
        line-height: 1.9;
        font-weight: 400;
    }
    
    /* Stats highlight (dark background, white text - OK) */
    .stats-highlight {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
        color: white;
        padding: 5rem 4rem;
        border-radius: 32px;
        text-align: center;
        margin: 1.5rem 0; 
        box-shadow: 0 20px 40px rgba(30, 64, 175, 0.3);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 5rem;
        margin-top: 4rem;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-size: 6rem;
        font-weight: 900;
        display: block;
        margin-bottom: 1rem;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .stat-label {
        font-size: 1.75rem;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 500;
    }
    
    /* Content blocks */
    .content-block {
        background: #f8fafc;
        padding: 3.5rem 3rem;
        border-radius: 24px;
        margin: 1.5rem 0; 
        border-left: 6px solid #2563eb;
    }
    
    .content-block-title {
        font-size: 2.75rem;
        font-weight: 700;
        color: #0f172a !important; /* <--- FIX */
        margin-bottom: 1.5rem;
    }
    
    .content-block-text {
        font-size: 2.5rem; 
        color: #475569 !important; /* <--- FIX */
        line-height: 1.9;
        font-weight: 400;
    }
    
    /* Image containers */
    .img-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 12px 28px rgba(0,0,0,0.12);
        margin: 1.5rem 0; 
    }
    
    .img-container img {
        width: 100%;
        height: auto;
        display: block;
    }
    
    .img-caption {
        font-size: 2rem; 
        color: #64748b !important; /* <--- FIX */
        text-align: center;
        margin-top: 1.5rem;
        font-style: italic;
    }
    
    /* List styling */
    .custom-list {
        font-size: 2.5rem; 
        color: #475569 !important; /* <--- FIX */
        line-height: 2.2;
        margin: 2rem 0;
    }
    
    .custom-list li {
        margin-bottom: 1.25rem;
        padding-left: 0.5rem;
    }
    
    /* NO Divider */
    .section-divider {
        display: none; /* Removed gaps */
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title { font-size: 5rem; } 
        .hero-acronym { font-size: 2.5rem; }
        .hero-subtitle { font-size: 2.5rem; }
        .section-header { font-size: 2.75rem; }
        .section-subheader { font-size: 2.2rem; }
        .feature-title, .impact-title { font-size: 1.75rem; }
        .feature-text, .impact-text, .content-block-text { font-size: 1.6rem; } 
    }
</style>
""", unsafe_allow_html=True)

# Model classes
class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation=F.gelu
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_embedding = nn.Parameter(torch.randn(1, 64*64, in_channels))
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        x_pos = x_flat + self.positional_embedding
        transformer_out = self.transformer_encoder(x_pos)
        out = transformer_out.permute(0, 2, 1).view(B, C, H, W)
        return out

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=6, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(True)
        )
        self.transformer_bottleneck = TransformerBottleneck(in_channels=base*2)
        self.up = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True)
        )
        self.out = nn.Conv2d(base, 1, 1)
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x_bottleneck = self.transformer_bottleneck(x2)
        u = self.up(x_bottleneck)
        u = self.dec1(torch.cat([u, x1], dim=1))
        return self.out(u)

CONFIG = {
    'model_path': 'data/generator_model_full_dataset_128.pth',
    'test_root': '/hdd/starcop/STARCOP_test',
    'plume_cache_dir': 'data/plumes',
    'band_files': [
        "TOA_AVIRIS_2004nm.tif", "TOA_AVIRIS_2109nm.tif", "TOA_AVIRIS_2310nm.tif",
        "TOA_AVIRIS_2350nm.tif", "TOA_AVIRIS_2360nm.tif",
    ],
    'mf_file': 'weight_mag1c.tif',
    'mf_scale': 0.2,
    'detection_threshold': 0.5,
}

os.makedirs(CONFIG['plume_cache_dir'], exist_ok=True)

@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNetGenerator(in_ch=6).to(device)
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device, weights_only=True))
        model.eval()
        return model, device
    except Exception as e:
        return None, None

G, DEVICE = load_model()

def get_crs_from_geotiff(folder_name):
    """
    Get CRS from geotiff. If file doesn't exist (production),
    return default CRS based on data location.
    """
    folder_path = os.path.join(CONFIG['test_root'], folder_name)
    band_path = os.path.join(folder_path, CONFIG['band_files'][0])
    
    if os.path.exists(band_path):
        try:
            with rasterio.open(band_path) as src:
                return src.crs
        except Exception as e:
            print(f"Error reading CRS from {folder_name}: {e}")
    
    # Default CRS for Permian Basin AVIRIS-NG data (UTM Zone 13N)
    # Adjust this based on your actual data location
    return rasterio.crs.CRS.from_epsg(32613)

def transform_utm_to_latlon(folder_name, center_x, center_y):
    crs = get_crs_from_geotiff(folder_name)
    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(center_x, center_y)
        return lat, lon
    except:
        return None, None

def load_scene_data(folder_path, resize_to=128):
    try:
        swir = []
        for bf in CONFIG['band_files']:
            path = os.path.join(folder_path, bf)
            with rasterio.open(path) as src:
                arr = src.read(1, out_shape=(resize_to, resize_to), 
                             resampling=Resampling.bilinear).astype(np.float32)
            swir.append(arr)
        
        swir = np.stack(swir, axis=0)
        for i in range(5):
            lo, hi = swir[i].min(), swir[i].max()
            swir[i] = (swir[i] - lo) / max(hi - lo, 1e-6)
        
        mf_path = os.path.join(folder_path, CONFIG['mf_file'])
        with rasterio.open(mf_path) as src:
            mf = src.read(1, out_shape=(resize_to, resize_to), 
                         resampling=Resampling.bilinear).astype(np.float32)
        mf = np.nan_to_num(mf, nan=0.0, posinf=0.0, neginf=0.0)
        mf = (mf - mf.min()) / max(mf.max() - mf.min(), 1e-6)
        
        swir_tensor = torch.from_numpy(swir)
        mf_tensor = torch.from_numpy(mf).unsqueeze(0)
        
        return swir_tensor, mf_tensor
    except Exception as e:
        return None, None

def precompute_plume_polygons(folder_name):
    """
    Pre-compute plume polygons with lat/lon coordinates.
    Saves polygons directly in geographic coordinates (no CRS needed in production).
    """
    cache_file = os.path.join(CONFIG['plume_cache_dir'], f"{folder_name}.pkl")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    if G is None:
        print(f"Warning: Model not loaded for {folder_name}")
        return {
            'polygons': [],
            'det_pixels': 0,
            'max_conf': 0,
            'mean_conf': 0
        }
    
    folder_path = os.path.join(CONFIG['test_root'], folder_name)
    
    # Check if folder exists (only on local machine)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found for {folder_name}")
        return {
            'polygons': [],
            'det_pixels': 0,
            'max_conf': 0,
            'mean_conf': 0
        }
    
    swir, mf = load_scene_data(folder_path)
    
    if swir is None:
        print(f"Warning: Could not load scene data for {folder_name}")
        return {
            'polygons': [],
            'det_pixels': 0,
            'max_conf': 0,
            'mean_conf': 0
        }
    
    with torch.no_grad():
        swir_batch = swir.unsqueeze(0).to(DEVICE)
        mf_batch = mf.unsqueeze(0).to(DEVICE)
        inp = torch.cat([swir_batch, CONFIG['mf_scale'] * mf_batch], dim=1)
        pred_logits = G(inp)
        pred_prob = torch.sigmoid(pred_logits).cpu().numpy()[0, 0]
    
    pred_mask = (pred_prob > CONFIG['detection_threshold']).astype(np.uint8)
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plume_polygons = []
    
    if len(contours) > 0:
        band_path = os.path.join(folder_path, CONFIG['band_files'][0])
        
        try:
            with rasterio.open(band_path) as src:
                transform = src.transform
                crs = src.crs
                
                # *** KEY CHANGE: Transform to lat/lon immediately ***
                transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    
                    polygon_coords = []
                    for point in contour:
                        x_pixel, y_pixel = point[0]
                        # Scale from 128x128 to original resolution
                        scale_x = src.width / 128.0
                        scale_y = src.height / 128.0
                        x_pixel_full = x_pixel * scale_x
                        y_pixel_full = y_pixel * scale_y
                        
                        # Convert pixel to UTM
                        utm_x, utm_y = rasterio.transform.xy(transform, y_pixel_full, x_pixel_full)
                        
                        # *** CONVERT TO LAT/LON AND SAVE ***
                        lon, lat = transformer.transform(utm_x, utm_y)
                        polygon_coords.append([lat, lon])
                    
                    if len(polygon_coords) >= 3:
                        plume_polygons.append(polygon_coords)
                        
        except Exception as e:
            print(f"Error extracting polygons for {folder_name}: {e}")
    
    # Save data with lat/lon polygons (no CRS needed)
    plume_data = {
        'polygons': plume_polygons,  # Already in lat/lon format
        'det_pixels': int(pred_mask.sum()),
        'max_conf': float(pred_prob.max()),
        'mean_conf': float(pred_prob[pred_mask > 0].mean()) if pred_mask.sum() > 0 else 0
    }
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(plume_data, f)
    
    print(f"✓ Cached plume data for {folder_name} with {len(plume_polygons)} polygons")
    
    return plume_data

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/emission_quantification_results.csv')
        
        coords = []
        for idx, row in df.iterrows():
            # Get CRS (will use default if file doesn't exist)
            crs = get_crs_from_geotiff(row['folder_name'])
            
            try:
                transformer = Transformer.from_crs(
                    crs,  # Source CRS
                    "EPSG:4326",  # Target: WGS84 lat/lon
                    always_xy=True
                )
                lon, lat = transformer.transform(row['center_lon'], row['center_lat'])
                coords.append({'latitude': lat, 'longitude': lon})
            except Exception as e:
                print(f"Transform error for {row['folder_name']}: {e}")
                coords.append({'latitude': None, 'longitude': None})
        
        coords_df = pd.DataFrame(coords)
        df['latitude'] = coords_df['latitude']
        df['longitude'] = coords_df['longitude']
        df = df.dropna(subset=['latitude', 'longitude'])
        
        df['date'] = pd.to_datetime(df['folder_name'].str.extract(r'ang(\d{8})')[0], format='%Y%m%d', errors='coerce')
        df['time'] = df['folder_name'].str.extract(r't(\d{6})')[0]
        df['time_formatted'] = pd.to_datetime(df['time'], format='%H%M%S', errors='coerce').dt.strftime('%H:%M:%S')
        df['date_formatted'] = df['date'].dt.strftime('%Y-%m-%d')
        
        df['emission_category'] = df['Q_kg_hr'].apply(
            lambda x: 'Critical' if x >= 100 else 'Standard'
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def precompute_all_polygons(_df):
    if G is None:
        return
    for idx, row in _df.iterrows():
        cache_file = os.path.join(CONFIG['plume_cache_dir'], f"{row['folder_name']}.pkl")
        if not os.path.exists(cache_file):
            precompute_plume_polygons(row['folder_name'])

df = load_data()
if len(df) > 0:
    precompute_all_polygons(df)

# ============================================================================
# HERO SECTION
# ============================================================================
hero_img_path = "assets/images/cover/earth-from-space.jpg"
hero_bg = get_base64_image(hero_img_path)

if hero_bg:
    st.markdown(f"""
    <div class="hero-section" style="background-image: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.84) 100%), url('data:image/jpeg;base64,{hero_bg}'); background-size: cover; background-position: center;">
        <h1 class="hero-title">PHANTOM</h1>
        <p class="hero-acronym">Physics-Informed Hyperspectral Adversarial Network for Transformer-Optimized Methane Detection</p>
        <p class="hero-subtitle">
            Advanced AI Pipeline for Methane Emission Detection and Quantification<br/>
            from Aerial and Satellite Hyperspectral Imagery
        </p>
        <div class="hero-stats">
            <div class="hero-stat">
                <span class="hero-stat-value">4×</span>
                <span class="hero-stat-label">More Accurate</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-value">11.43%</span>
                <span class="hero-stat-label">False Positive Rate</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-value">97 kg/hr</span>
                <span class="hero-stat-label">Verified Min. Detection Rate</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">PHANTOM</h1>
        <p class="hero-acronym">Physics-Informed Hyperspectral Adversarial Network for Transformer-Optimized Methane Detection</p>
        <p class="hero-subtitle">
            Advanced AI Platform for Real-Time Methane Emission Detection<br/>
            from Satellite Hyperspectral Imagery
        </p>
        <div class="hero-stats">
            <div class="hero-stat">
                <span class="hero-stat-value">4×</span>
                <span class="hero-stat-label">More Accurate</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-value">11.43%</span>
                <span class="hero-stat-label">False Positive Rate</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-value">&lt;100ms</span>
                <span class="hero-stat-label">Processing Time</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# WHAT WE DO SECTION
# ============================================================================
st.markdown('<div class="what-we-do-section">', unsafe_allow_html=True)
st.markdown('<div class="section-container">', unsafe_allow_html=True)

st.markdown('<h1 class="section-header">What We Do</h1>', unsafe_allow_html=True)
st.markdown('<p class="section-subheader">PHANTOM combines cutting-edge artificial intelligence with satellite remote sensing to detect and quantify methane emissions in real-time, enabling rapid response to climate threats.</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🛰️</span>
        <div class="feature-title">Aerial and Satellite Detection</div>
        <div class="feature-text">
            We use data from airborne platforms and satellites to look at the Earth. Our system can spot the invisible 
            signature of methane gas from space, even in complex landscapes.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🧠</span>
        <div class="feature-title">AI-Powered Analysis</div>
        <div class="feature-text">
           PHANTOM is trained to find methane plumes. It's extremely good at finding real leaks 
            and ignoring background clusters and artifacts which are confounders to real emissions, which keeps errors low.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">📊</span>
        <div class="feature-title">Emission Quantification</div>
        <div class="feature-text">
            Once we find a plume, our system automatically quantifies using enhancement methods. 
            We report this in kilograms per hour, giving you a clear, actionable number.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# LIVE DETECTION MAP SECTION
# ============================================================================
st.markdown('<div class="section-container">', unsafe_allow_html=True)

st.markdown('<h3 class="subsection-title">Emission Detection Map</h3>', unsafe_allow_html=True)
st.markdown('<p class="subsection-intro">Interactive map displaying methane detections with AI-generated plume boundaries and precise emission quantification</p>', unsafe_allow_html=True)

if len(df) > 0:
    # Find the SECOND largest emitter to center on
    df_sorted = df.sort_values(by='Q_kg_hr', ascending=False)
    
    # Check if there are at least 2 data points, otherwise fall back to the first
    if len(df_sorted) > 1:
        center_row = df_sorted.iloc[2] # Center on the second largest
    else:
        center_row = df_sorted.iloc[0] # Fallback to the largest
        
    center_lat = center_row['latitude']
    center_lon = center_row['longitude']
    zoom_level = 15 # Zoom in close

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_level,
        tiles='CartoDB positron',
        prefer_canvas=True,
        control_scale=True,
        scrollWheelZoom=True
    )

    # Add the satellite map as an *alternative* layer
    folium.TileLayer('Esri WorldImagery', name='Satellite', overlay=True, control=True).add_to(m)

    def create_simple_marker(color, size=20):
        svg = f"""
        <svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
            <circle cx="{size/2}" cy="{size/2}" r="{size/2-2}" fill="{color}" opacity="0.85" stroke="white" stroke-width="2"/>
        </svg>
        """
        return svg

    for idx, row in df.iterrows():
        if row['emission_category'] == 'Critical':
            color = '#dc2626'
            size = 24
        else:
            color = '#059669'
            size = 16
        
        popup_html = f"""
        <div style="font-family: Inter, sans-serif; width: 380px; padding: 20px; background: white;">
            <div style="background: {color}; color: white; padding: 12px; margin: -20px -20px 15px -20px; border-radius: 8px 8px 0 0;">
                <h3 style="margin: 0; font-size: 1.1rem; font-weight: 600;">{row['emission_category']} Emission</h3>
            </div>
            
            <table style="width: 100%; font-size: 0.9rem; border-collapse: collapse;">
          
                <tr>
                    <td style="padding: 8px 0; font-weight: 500; color: #64748b;">Detection Time</td>
                    <td style="padding: 8px 0;">{row['date_formatted']} {row['time_formatted']}</td>
                </tr>
                <tr style="background: #f8fafc;">
                    <td style="padding: 12px 8px; font-weight: 600;">Emission Rate</td>
                    <td style="padding: 12px 8px; color: {color}; font-weight: 700; font-size: 1.1rem;">
                        {row['Q_kg_hr']:.2f} kg/hr
                    </td>
                </tr>
     
                <tr>
                    <td style="padding: 8px 0; font-weight: 500; color: #64748b;">Wind Speed (10 m)</td>
                    <td style="padding: 8px 0;">{row['U_10_ms']:.2f} m/s</td>
                </tr>
            </table>
        </div>
        """
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=400),
            tooltip=f"{row['Q_kg_hr']:.1f} kg/hr",
            icon=folium.DivIcon(html=create_simple_marker(color, size))
        ).add_to(m)
        
        plume_data = precompute_plume_polygons(row['folder_name'])
        
        if plume_data and plume_data['polygons']:
            for polygon_coords in plume_data['polygons']:
                folium.Polygon(
                    locations=polygon_coords,
                    color=color,
                    weight=2,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.25,
                    popup=folium.Popup(popup_html, max_width=400)
                ).add_to(m)

    legend_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; background: white; 
                border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; 
                font-family: Inter, sans-serif; font-size: 0.875rem; z-index: 9999;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-weight: 600; margin-bottom: 10px; color: #0f172a;">Detection Legend</div>
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <div style="width: 16px; height: 16px; background: #dc2626; border-radius: 50%; margin-right: 8px;"></div>
            <span>Critical (≥100 kg/hr)</span>
        </div>
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <div style="width: 14px; height: 14px; background: #059669; border-radius: 50%; margin-right: 10px;"></div>
            <span>Standard (<100 kg/hr)</span>
        </div>
        <hr style="margin: 10px 0; border: 0; border-top: 1px solid #e2e8f0;">
        <div style="font-size: 0.8rem; color: #64748b;">
            <div>Sites: {len(df)}</div>
            <div>Total: {df['Q_kg_hr'].sum():.0f} kg/hr</div>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    
    st_folium(m, width=None, height=900, returned_objects=[])

    # Key stats
    st.markdown("<br/>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Detection Sites", len(df), help="Number of emission events detected")
    with col2:
        st.metric("Critical Emitters", len(df[df['emission_category'] == 'Critical']), 
                 help="Sites exceeding 100 kg/hr threshold")
    with col3:
        st.metric("Total Emissions", f"{df['Q_kg_hr'].sum():.0f} kg/hr", 
                 help="Sum of all quantified emission rates")
    with col4:
        st.metric("Median Rate", f"{df['Q_kg_hr'].median():.1f} kg/hr",
                 help="Median emission rate across all sites")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# WHY METHANE CONCERNS MATTER
# ============================================================================
st.markdown('<div class="impact-section">', unsafe_allow_html=True)
st.markdown('<div class="section-container">', unsafe_allow_html=True)

st.markdown('<h1 class="section-header">Why Methane Concerns Matter</h1>', unsafe_allow_html=True)
st.markdown('<p class="section-subheader">Methane is responsible for 30% of global warming since pre-industrial times. Detecting and reducing methane emissions is our fastest opportunity to slow climate change in the next decade.</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="impact-card">
        <div class="impact-number">80×</div>
        <div class="impact-title">Warming Potential</div>
        <div class="impact-text">
            Methane traps 80 times more heat than CO₂ over a 20-year period, making it the most critical 
            greenhouse gas for immediate climate action. 
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="impact-card">
        <div class="impact-number">30%</div>
        <div class="impact-title">Global Warming Contribution</div>
        <div class="impact-text">
            Responsible for approximately 30% of global temperature rise since the Industrial Revolution. 
            Its potency makes it second only to carbon dioxide in overall climate impact.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="impact-card">
        <div class="impact-number">45%</div>
        <div class="impact-title">Reduction Potential by 2030</div>
        <div class="impact-text">
            Cutting methane emissions by 45% by 2030 could prevent 0.3°C of warming by 2040. 
            Our fastest and most cost-effective climate solution.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Methane sources subsection
st.markdown('<h3 class="subsection-title">Global Emission Sources</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="content-block">
        <div class="content-block-title">Major Emission Sources</div>
        <div class="content-block-text">
            According to the Environmental Protection Agency, total U.S. methane emissions in 2020 were 650.4 million metric tons of CO2 equivalent, which is comparable to the emissions from 140 million cars. The primary sources of these emissions were Natural Gas and Petroleum (32%) and Enteric Fermentation (27%). Other significant contributors included Landfills (17%), Manure Management (9%), Coal Mining (6%), and other miscellaneous sources (9%).
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        if os.path.exists('assets/images/methane-impact/methane-sources.png'):
            img = Image.open('assets/images/methane-impact/methane-sources.png')
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(img, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<p class="img-caption">Global Methane Emission Sources by Sector</p>', unsafe_allow_html=True)
    except:
        pass

with col2:
    st.markdown("""
    <div class="content-block">
        <div class="content-block-title">Why Satellite Detection is Critical</div>
        <div class="content-block-text">
         Data from NASA illustrates a significant global warming trend, to which methane contributes approximately 30% of the forcing. As conventional ground-based monitoring covers less than 5% of global infrastructure, satellite detection is the only viable solution. This technology is critical for identifying the "super-emitters" responsible for over 50% of hotspot emissions, offering the most rapid path to mitigating near-term warming.
                </div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        if os.path.exists('assets/images/methane-impact/climate-impact.png'):
            img = Image.open('assets/images/methane-impact/climate-impact.png')
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(img, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<p class="img-caption">Methane\'s Contribution to Global Temperature Rise</p>', unsafe_allow_html=True)
    except:
        pass

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PLATFORM CAPABILITIES
# ============================================================================
st.markdown('<div class="section-container">', unsafe_allow_html=True)

st.markdown('<h2 class="section-header">Platform Performance</h2>', unsafe_allow_html=True)
st.markdown('<p class="section-subheader"> AI-driven methane detection validated on real-world satellite data</p>', unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-highlight">
    <h2 style="margin-bottom: 3rem; font-size: 3.5rem; font-weight: 800;">Benchmark Results</h2>
    <div class="stats-grid">
        <div class="stat-item">
            <span class="stat-value">100%</span>
            <span class="stat-label">F1 Score (Critical)</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">96.65%</span>
            <span class="stat-label">F1 Score (Standard)</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">11.43%</span>
            <span class="stat-label">False Positive Rate</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">62.42%</span>
            <span class="stat-label">AUPRC</span>
        </div>
    </div>
    <p style="margin-top: 3rem; font-size: 1.5rem; opacity: 0.95; line-height: 1.9;">
        Validated on NASA AVIRIS-NG hyperspectral imagery across  real-world emission events<br/>
        Tested on diverse geographic locations and environmental conditions
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="content-block">
    <div class="content-block-title">Technical Innovation</div>
    <div class="content-block-text">
        PHANTOM represents a breakthrough in methane detection technology by combining physics-informed neural networks 
        with transformer-based architecture. Our system processes Short Wave Infrared spectral bands from hyperspectral sensors, applying 
        matched filter preprocessing and adversarial training to achieve state-of-the-art performance. The platform delivers 
     exceptional accuracy—outperforming existing detection methods by a factor of four.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown(f"""
<div style='margin-top: 1.5rem; padding: 3rem 3rem; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
            color: white; text-align: center;'>
    <h2 style='font-size: 4rem; font-weight: 800; margin-bottom: 2rem; letter-spacing: -0.02em;'>
        PHANTOM Platform
    </h2>
    <p style='font-size: 2rem; margin-bottom: 3rem; opacity: 0.95; line-height: 1.8;'>
Advanced AI platform for global-scale methane emission monitoring and quantification
Validated on NASA AVIRIS-NG hyperspectral satellite imagery    </p>
    <div style='font-size: 1.5rem; opacity: 0.85; max-width: 1000px; margin: 0 auto 3rem auto; line-height: 2;'>
        Developed at MESA Lab, University of California, Merced • Supporting global climate action
    </div>
    <div style='font-size: 1.125rem; opacity: 0.7; margin-top: 3rem;'>
        Last Updated: {datetime.now().strftime("%B %d, %Y")} 
    </div>
</div>
""", unsafe_allow_html=True)