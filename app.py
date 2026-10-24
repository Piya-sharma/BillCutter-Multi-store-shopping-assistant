import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import pandas as pd
import plotly.express as px
import os
import glob

# ---------------------------------------------------------
# 1. Page Config & CSS Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="BillCutter - Visual Price Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 25px;
        font-weight: 400;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 10px;
        color: #475569;
        font-weight: 600;
        border: 1px solid #E2E8F0;
        padding: 10px 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #10B981 !important;
        color: white !important;
        border: none !important;
    }
    /* Custom Button */
    .stButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

MAPPING_FILE = "Mapping_Database.xlsx"
PRODUCT_FILE = "Product_Database.xlsx"
IMAGE_DIR = "product_images"

# State Management
if "selected_product_info" not in st.session_state:
    st.session_state.selected_product_info = None

# ---------------------------------------------------------
# 2. Excel Database Loaders
# ---------------------------------------------------------
@st.cache_data
def load_excel_databases():
    if not os.path.exists(MAPPING_FILE) or not os.path.exists(PRODUCT_FILE):
        return None, None
    df_map = pd.read_excel(MAPPING_FILE)
    df_prod = pd.read_excel(PRODUCT_FILE)
    return df_map, df_prod

df_map, df_prod = load_excel_databases()

def get_product_details(query_str):
    if df_prod is None:
        return None
    
    clean_query = str(query_str).strip().lower()
    
    prod_row = df_prod[df_prod['Product_ID'].astype(str).str.strip().str.lower() == clean_query]
    if prod_row.empty:
        prod_row = df_prod[df_prod['Product_Name'].astype(str).str.strip().str.lower() == clean_query]
    if prod_row.empty:
        prod_row = df_prod[df_prod['Product_Name'].astype(str).str.lower().apply(lambda x: clean_query in x or x in clean_query)]

    if prod_row.empty:
        return None
    
    row = prod_row.iloc[0]
    platform_cols = ['Amazon', 'Flipkart', 'Croma', 'Vijay Sales', 'Myntra', 'Ajio']
    stores = {}
    for col in platform_cols:
        if col in row:
            val = row[col]
            if str(val).strip() != '-' and pd.notna(val):
                try:
                    stores[col] = int(float(val))
                except ValueError:
                    pass
    return {
        "product_id": row['Product_ID'],
        "title": f"{row['Brand']} {row['Product_Name']}",
        "category": row['Category'],
        "market_data": stores
    }

# ---------------------------------------------------------
# 3. AI Deep Learning Engine
# ---------------------------------------------------------
@st.cache_resource
def load_feature_extractor():
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    feature_extractor = model.features
    feature_extractor.eval()
    pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
    return feature_extractor, pooling

feat_model, pool_layer = load_feature_extractor()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(pil_image):
    tensor_img = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        features = feat_model(tensor_img)
        features = pool_layer(features)
        features = torch.flatten(features, 1)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features
# Modern & Aesthetic Bar Chart Builder
def build_price_chart(stores, best_store):
    df = pd.DataFrame(list(stores.items()), columns=["Store Platform", "Price Structure"])
    
    # Palette: Green for Best Deal, Modern Indigo/Violet shades for others
    color_palette = ['#6366F1', '#3B82F6', '#8B5CF6', '#EC4899', '#06B6D4', '#F59E0B']
    
    color_map = {}
    palette_idx = 0
    for store in df['Store Platform']:
        if store == best_store:
            color_map[store] = '#10B981'  # Vibrant Green
        else:
            color_map[store] = color_palette[palette_idx % len(color_palette)]
            palette_idx += 1

    fig = px.bar(
        df, x="Store Platform", y="Price Structure", text="Price Structure",
        color="Store Platform",
        color_discrete_map=color_map,
        title="📊 Multi-Store Real-Time Price Comparison"
    )
    
    fig.update_layout(
        showlegend=False, 
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=13, color="#334155"),
        title_font=dict(size=16, color="#1E293B", family="Inter, sans-serif"),
        yaxis=dict(title="Price (₹)", showgrid=True, gridcolor='#E2E8F0', zeroline=False),
        xaxis=dict(title="", showgrid=False),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    fig.update_traces(
        texttemplate='<b>₹%{text:,}</b>', 
        textposition='outside', 
        width=0.45,
        marker_line_width=0
    )
    
    return fig

def load_catalog_embeddings():
    catalog_embeddings = {}
    if not os.path.exists(IMAGE_DIR):
        return catalog_embeddings
        
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.*"))
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        ref_key = os.path.splitext(filename)[0].strip()
        try:
            img = Image.open(img_path).convert("RGB")
            emb = extract_features(img)
            catalog_embeddings[ref_key] = emb
        except Exception:
            pass
    return catalog_embeddings

catalog_embeddings = load_catalog_embeddings()

def find_best_visual_match(uploaded_image):
    if not catalog_embeddings:
        return None, 0.0
        
    input_emb = extract_features(uploaded_image)
    best_ref_key = None
    highest_similarity = -1.0
    
    for ref_key, ref_emb in catalog_embeddings.items():
        similarity = torch.mm(input_emb, ref_emb.t()).item()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_ref_key = ref_key
            
    return best_ref_key, highest_similarity

# ---------------------------------------------------------
# 4. Header & Clean Sidebar
# ---------------------------------------------------------
st.markdown("<div class='main-title'>🛍️ BillCutter </div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'> Multi-Store Shopping Assistant </div>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #475569; font-size: 0.95rem; margin-top: -15px; margin-bottom: 25px;'>"
    " A smart shopping assistant that recognizes products directly from uploaded images "
    "and compares prices across top e-commerce platforms to help you find the best deal."
    "</p>", 
    unsafe_allow_html=True
)
# Clean Sidebar - About Section
st.sidebar.markdown(" ℹ️ About : ")
st.sidebar.info(
    """
    **BillCutter** is an intelligent price comparison system powered by **MobileNetV2 Computer Vision Model**.
    
    * 📸 **Scans product image features**
    * 📊 **Compares multi-platform pricing**
    * 💰 **Instantly finds the best deal**
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("Created by **Piya** 🎓")

# ---------------------------------------------------------
# 5. Dual Mode Navigation Tabs
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📸 Visual AI Search Mode", "🔍 Manual Product Search Mode"])

# ---------------------------------------------------------
# TAB 1: VISUAL SEARCH MODE
# ---------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([4, 6], gap="large")
    
    with col_left:
        st.markdown("#### 📸 Image Recognition Scan")
        uploaded_file = st.file_uploader("Upload product image:", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image Preview", use_container_width=True)
            
            analyze_btn = st.button("✨ Scan & Compare Prices", key="btn_visual")
            
            if analyze_btn:
                best_key, match_score = find_best_visual_match(image)
                match_percentage = round(match_score * 100, 1)
                
                if best_key and match_percentage >= 50.0:
                    info = get_product_details(best_key)
                    if info:
                        st.session_state.selected_product_info = info
                        st.success(f"🎯 **Visual Match Found:** `{info['title']}` ({match_percentage}% Confidence)")
                    else:
                        st.error(f"Matched key: '{best_key}'")
st.error(f"Product DB loaded: {df_prod is not None}")
if df_prod is not None:
    st.write("Product names:", df_prod["Product_Name"].astype(str).tolist())
                else:
                    st.session_state.selected_product_info = None
                    st.error(f"⚠️ Low confidence match ({match_percentage}%). Reference image missing in catalog.")

    with col_right:
        selected_info = st.session_state.selected_product_info
        if selected_info:
            full_title = selected_info["title"]
            category = selected_info["category"]
            stores = selected_info["market_data"]
            
            st.markdown(f"### 🏷️ {full_title}")
            st.caption(f"Category: **{category}** | Product ID: **{selected_info['product_id']}**")
            st.markdown("---")
            
            if stores:
                best_store = min(stores, key=stores.get)
                lowest_price = stores[best_store]
                worst_store = max(stores, key=stores.get)
                highest_price = stores[worst_store]
                savings = highest_price - lowest_price
                
                st.success(f"🎉 **Lowest Price Found!** Buy on **{best_store}** to save **₹{savings:,}**")
                
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(label="🏆 Best Platform", value=best_store)
                with m_col2:
                    st.metric(label="📉 Target Price", value=f"₹{lowest_price:,}")
                with m_col3:
                    st.metric(label="💰 Max Savings", value=f"₹{savings:,}", delta="Best Deal")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                fig = build_price_chart(stores, best_store)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Upload an image on the left and click **Scan & Compare Prices**.")

# ---------------------------------------------------------
# TAB 2: MANUAL SEARCH MODE
# ---------------------------------------------------------
with tab2:
    st.markdown("#### 🔍 Direct Catalog Lookup")
    
    catalog_options = ["Select a product..."]
    if df_prod is not None:
        catalog_options += df_prod['Product_Name'].tolist()

    manual_choice = st.selectbox("Select Target Item From Database:", catalog_options)
    
    if manual_choice != "Select a product...":
        manual_info = get_product_details(manual_choice)
        
        if manual_info:
            full_title = manual_info["title"]
            category = manual_info["category"]
            stores = manual_info["market_data"]
            
            st.markdown(f"### 🏷️ {full_title}")
            st.caption(f"Category: **{category}** | Product ID: **{manual_info['product_id']}**")
            st.markdown("---")
            
            if stores:
                best_store = min(stores, key=stores.get)
                lowest_price = stores[best_store]
                worst_store = max(stores, key=stores.get)
                highest_price = stores[worst_store]
                savings = highest_price - lowest_price
                
                st.success(f"🎉 **Lowest Price Found!** Buy on **{best_store}** to save **₹{savings:,}**")
                
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(label="🏆 Best Platform", value=best_store)
                with m_col2:
                    st.metric(label="📉 Target Price", value=f"₹{lowest_price:,}")
                with m_col3:
                    st.metric(label="💰 Max Savings", value=f"₹{savings:,}", delta="Best Deal")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                fig = build_price_chart(stores, best_store)
                st.plotly_chart(fig, use_container_width=True)
