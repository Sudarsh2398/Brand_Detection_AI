import streamlit as st
import os
import tempfile
import cv2
from ultralytics import YOLO
import pandas as pd
import time
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
import boto3
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = os.getenv("YOLO_MODEL_PATH")

DB_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'database': os.getenv('PG_DB', 'brand_detection'),
    'user': os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASS'),
    'port': int(os.getenv('PG_PORT', 5432))
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="BrandVision AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Premium Custom CSS with Glassmorphism
# -------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark theme background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Animated gradient header */
    .hero-header {
        background: linear-gradient(270deg, #ff6b6b, #4ecdc4, #45b7d1, #f7b731);
        background-size: 800% 800%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 8s ease infinite;
        font-size: 4.5rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        padding: 1rem 0;
        letter-spacing: -2px;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.3rem;
        font-weight: 300;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }
    
    /* Glassmorphic containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Premium metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: rgba(255, 255, 255, 0.9);
    }
    
    [data-testid="stSidebar"] * {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: white !important;
    }
    
    /* Premium buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    /* Animated tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.9);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* File uploader - PURPLE GLASSMORPHISM */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%) !important;
        border: 2px dashed rgba(102, 126, 234, 0.5) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        backdrop-filter: blur(10px) !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.8) !important;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%) !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* File uploader drag drop area - DARK BACKGROUND */
    [data-testid="stFileUploader"] section {
        background: rgba(20, 20, 50, 0.7) !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }
    
    /* File uploader text - WHITE */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p {
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* Browse files button - PURPLE GRADIENT */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 10px;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        margin: 0.5rem 0 !important;
        padding: 1rem !important;
    }
    
    /* Chat message text - WHITE */
    .stChatMessage p,
    .stChatMessage div,
    .stChatMessage span,
    .stChatMessage li,
    .stChatMessage strong,
    .stChatMessage em,
    .stChatMessage code {
        color: white !important;
    }
    
    /* Chat input - DARK BACKGROUND */
    .stChatInput textarea {
        background: rgba(0, 0, 0, 0.5) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
    }
    
    .stChatInput textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stChatInput textarea:focus {
        background: rgba(0, 0, 0, 0.6) !important;
        border: 1px solid rgba(102, 126, 234, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stDataFrame table {
        color: white !important;
    }
    
    .stDataFrame th {
        background: rgba(102, 126, 234, 0.3) !important;
        color: white !important;
    }
    
    .stDataFrame td {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .stAlert * {
        color: white !important;
    }
    
    /* All text elements - WHITE */
    p, span, div, label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: 600 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white !important;
    }
    
    /* Text inputs - DARK BACKGROUND */
    .stTextInput>div>div>input {
        background: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput>div>div>input::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: rgba(102, 126, 234, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        background: rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Select boxes - DARK BACKGROUND */
    .stSelectbox>div>div {
        background: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
    }
    
    .stSelectbox label {
        color: white !important;
    }
    
    .stSelectbox input {
        color: white !important;
    }
    
    /* Dropdown menu */
    [data-baseweb="popover"] {
        background: rgba(20, 20, 40, 0.95) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    [role="listbox"] {
        background: rgba(20, 20, 40, 0.95) !important;
    }
    
    [role="option"] {
        background: rgba(20, 20, 40, 0.95) !important;
        color: white !important;
    }
    
    [role="option"]:hover {
        background: rgba(102, 126, 234, 0.5) !important;
        color: white !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        margin: 2rem 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Footer glow */
    .footer-glow {
        text-align: center;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        margin-top: 2rem;
    }
    
    .creator-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        margin-top: 0.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        color: white !important;
    }
    
    /* Code blocks */
    code {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #4ecdc4 !important;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
    
    pre {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
    }
    
    pre code {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# S3 Upload Function
# -------------------------
def upload_to_s3(file_path, bucket_name, s3_key):
    """Upload a file to an S3 bucket"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        s3_client.upload_file(file_path, bucket_name, s3_key)
        return f"https://{bucket_name}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    except ClientError as e:
        st.error(f"❌ S3 Upload failed: {e}")
        return None

# -------------------------
# Database Functions - IMPROVED
# -------------------------
def check_database_connection():
    """Check if database connection works and table exists"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'brand_detections'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        cursor.close()
        return True, table_exists
    except Exception as e:
        return False, False
    finally:
        if conn:
            conn.close()

def create_table():
    """Create detections table in PostgreSQL"""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS brand_detections (
                id SERIAL PRIMARY KEY,
                video_name VARCHAR(255),
                frame INTEGER,
                timestamp_s REAL,
                detected_logo_name VARCHAR(100),
                confidence REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                frame_width INTEGER,
                frame_height INTEGER,
                detection_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_name 
            ON brand_detections(video_name)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_detected_logo 
            ON brand_detections(detected_logo_name)
        """)
        
        conn.commit()
        st.session_state.db_initialized = True
        st.session_state.db_checked = False  # Force re-check
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def save_detections_to_db(video_name, detections_list):
    """Save all detections to PostgreSQL database"""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        data = [
            (
                video_name,
                d['Frame'],
                d['Timestamp (s)'],
                d['Detected_Logo_Name'],
                d['Confidence'],
                d.get('bbox_x1', 0),
                d.get('bbox_y1', 0),
                d.get('bbox_x2', 0),
                d.get('bbox_y2', 0),
                d.get('frame_width', 0),
                d.get('frame_height', 0)
            )
            for d in detections_list
        ]
        
        query = """
            INSERT INTO brand_detections 
            (video_name, frame, timestamp_s, detected_logo_name, confidence,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_width, frame_height)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        execute_batch(cur, query, data)
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Failed to save to database: {str(e)}")
        return False
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

@st.cache_data(ttl=60)
def get_all_data():
    """Get ALL data from table"""
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM brand_detections
            ORDER BY detection_datetime DESC
        """)
        
        results = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        
        return [dict(zip(colnames, row)) for row in results], colnames
    
    except Exception as e:
        return [], []
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@st.cache_data(ttl=300)
def get_table_info():
    """Get column names and data types"""
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'brand_detections'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        return columns
    
    except Exception as e:
        return []
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_database_stats():
    """Get database statistics"""
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM brand_detections")
        stats['total_rows'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT video_name) FROM brand_detections")
        stats['unique_videos'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT detected_logo_name) FROM brand_detections")
        stats['unique_brands'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confidence) FROM brand_detections")
        stats['avg_confidence'] = cursor.fetchone()[0] or 0
        
        return stats
    
    except Exception as e:
        return {}
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# -------------------------
# RAG Functions - OPTIMIZED
# -------------------------
def format_context_optimized(columns, all_data, max_samples=30):
    """Format database info into optimized context for AI"""
    if not all_data:
        return "No data available in the database. Please process some videos first."
    
    context = f"Brand Detection Database Summary\n"
    context += f"Total Rows: {len(all_data)}\n"
    context += f"Total Columns: {len(columns)}\n\n"
    
    context += "Columns:\n"
    for col_name, col_type in columns:
        context += f"  - {col_name} ({col_type})\n"
    
    # Add summary statistics
    df = pd.DataFrame(all_data)
    context += f"\nSummary Statistics:\n"
    
    if 'detected_logo_name' in df.columns:
        brand_counts = df['detected_logo_name'].value_counts()
        context += f"Brand Distribution (Top 10):\n"
        for brand, count in brand_counts.head(10).items():
            context += f"  - {brand}: {count} detections\n"
        
        if len(brand_counts) > 10:
            context += f"  ... and {len(brand_counts) - 10} more brands\n"
    
    if 'confidence' in df.columns:
        context += f"Confidence Stats: Min {df['confidence'].min():.2f}, Max {df['confidence'].max():.2f}, Avg {df['confidence'].mean():.2f}\n"
    
    if 'video_name' in df.columns:
        video_counts = df['video_name'].value_counts()
        context += f"Videos Analyzed: {len(video_counts)}\n"
    
    # Add limited sample data
    context += f"\nSample Data (first {min(max_samples, len(all_data))} of {len(all_data)} rows):\n"
    for i, row in enumerate(all_data[:max_samples], 1):
        context += f"Row {i}: "
        # Only include key fields to reduce token count
        key_fields = ['video_name', 'frame', 'timestamp_s', 'detected_logo_name', 'confidence']
        sample_data = []
        for key in key_fields:
            if key in row:
                sample_data.append(f"{key}: {row[key]}")
        context += " | ".join(sample_data) + "\n"
    
    if len(all_data) > max_samples:
        context += f"\n... and {len(all_data) - max_samples} more rows in database"
    
    return context

def ask_groq_optimized(question, context, max_retries=3):
    """Call Groq AI for RAG with optimized context and retry logic"""
    if not GROQ_API_KEY:
        return """❌ GROQ_API_KEY not found!

Get FREE API key:
1. Visit: https://console.groq.com/keys  
2. Sign up (no credit card needed)
3. Create API Key
4. Add to .env: GROQ_API_KEY=your_key"""
    
    for attempt in range(max_retries):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            
            system_prompt = """You are a brand detection analysis assistant. Answer questions based ONLY on the provided database context.

Capabilities:
- Identify brands in videos
- Calculate screen time and frequency
- Compare brand visibility
- Provide timing information (start, end, duration)
- Analyze confidence scores
- Identify patterns

Always cite specific data. If info isn't available, say so clearly.
Keep responses concise and focused on the data provided."""

            user_prompt = f"""Database Context:
{context}

Question: {question}

Provide a detailed but concise answer based on the data:"""
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                timeout=30
            )
            
            return completion.choices[0].message.content
        
        except Exception as e:
            error_msg = str(e)
            if "rate_limit_exceeded" in error_msg and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            elif "too large" in error_msg.lower() or "413" in error_msg:
                return "❌ The database is too large to analyze in one query. Please try a more specific question or use the Analytics tab for detailed insights."
            else:
                return f"❌ Error: {error_msg}"
    
    return "❌ Failed to get response after multiple attempts. Please try again later."

# -------------------------
# YOLO Model Loading
# -------------------------
@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

# -------------------------
# Hero Header
# -------------------------
st.markdown('<h1 class="hero-header">🎯 BrandVision AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Next-Generation Brand Detection • Powered by AI • Real-Time Analytics</p>', unsafe_allow_html=True)

# -------------------------
# Initialize Session State - IMPROVED
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pause" not in st.session_state:
    st.session_state.pause = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "db_checked" not in st.session_state:
    st.session_state.db_checked = False
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False
if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = False
if "all_data" not in st.session_state:
    st.session_state.all_data = []
if "columns" not in st.session_state:
    st.session_state.columns = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "current_video" not in st.session_state:
    st.session_state.current_video = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "detections_list" not in st.session_state:
    st.session_state.detections_list = []
if "video_info" not in st.session_state:
    st.session_state.video_info = {}

# -------------------------
# Auto-check database status
# -------------------------
if not st.session_state.db_checked:
    with st.spinner("🔍 Checking database connection..."):
        connection_ok, table_exists = check_database_connection()
        st.session_state.db_checked = True
        st.session_state.db_initialized = table_exists
        
        if not connection_ok:
            st.sidebar.error("❌ Database connection failed")
        elif not table_exists:
            st.sidebar.warning("⚠️ Database table not initialized")
        else:
            st.sidebar.success("✅ Database ready")

# -------------------------
# Auto-load data if database is ready
# -------------------------
if st.session_state.db_initialized and not st.session_state.db_loaded:
    with st.spinner("📡 Loading database data..."):
        columns = get_table_info()
        if columns:
            all_data, colnames = get_all_data()
            if all_data:
                st.session_state.columns = columns
                st.session_state.all_data = all_data
                st.session_state.db_loaded = True

# -------------------------
# Sidebar with Simplified Design
# -------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    
    # Database status
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🗄️ Database Status")
    
    if st.session_state.db_checked:
        if st.session_state.db_initialized:
            if st.session_state.db_loaded:
                data_count = len(st.session_state.all_data)
                st.success(f"✅ Ready ({data_count} records)")
            else:
                st.success("✅ Table Ready")
        else:
            st.warning("⚠️ Table Needed")
            
            if st.button("🚀 Initialize Database", use_container_width=True, type="primary"):
                if create_table():
                    st.success("✅ Database initialized!")
                    st.rerun()
    else:
        st.info("🔍 Checking...")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # AI Status
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🤖 AI Assistant")
    if GROQ_API_KEY:
        st.success("✅ Groq AI Ready")
    else:
        st.error("❌ API Key Missing")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Quick Actions
    st.markdown("### 🔄 Quick Actions")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.db_loaded = False
        st.session_state.db_checked = False
        st.rerun()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reset App", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.divider()
    
    # Example questions
    st.markdown("### 💡 Try Asking")
    st.markdown("""
    - What brands were detected?
    - Show high confidence detections
    - What's the average confidence?
    - Which brand appears most frequently?
    - Compare all brands
    """)

# -------------------------
# Main Tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📹 Video Processing",
    "💬 AI Assistant", 
    "📊 Analytics",
    "🔍 Data Explorer"
])

# -------------------------
# TAB 1: Video Processing - FIXED
# -------------------------
with tab1:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: white !important; font-weight: 700;">📹 Brand Detection Engine</h3>', unsafe_allow_html=True)
    
    # Database requirement check
    if not st.session_state.db_initialized:
        st.error("""
        🚫 Database not initialized!
        
        Please click **'Initialize Database'** in the sidebar first to set up the required database table.
        """)
        st.stop()
    
    model = load_model()
    if model:
        st.success("✅ YOLO model loaded successfully")
    else:
        st.error("❌ Failed to load YOLO model")
        st.stop()
    
    uploaded_file = st.file_uploader("📤 Upload Video File", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        video_name = uploaded_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        # Store video info in session state
        st.session_state.current_video = video_name
        
        # S3 Upload (optional)
        s3_url = None
        if S3_BUCKET_NAME and AWS_ACCESS_KEY_ID:
            with st.spinner("☁️ Uploading to S3..."):
                s3_key = f"uploads/{video_name}"
                s3_url = upload_to_s3(video_path, S3_BUCKET_NAME, s3_key)
                if s3_url:
                    st.success("✅ Uploaded to S3")
        
        # Video information
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Failed to open video")
            os.unlink(video_path)
            st.stop()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps <= 0:
            fps = 30
        
        # Store video info in session state
        st.session_state.video_info = {
            'total_frames': total_frames,
            'fps': fps,
            'frame_width': frame_width,
            'frame_height': frame_height,
            'video_path': video_path
        }
        
        # Display video info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{total_frames}</p><p class="metric-label">Frames</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{fps:.0f}</p><p class="metric-label">FPS</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{frame_width}</p><p class="metric-label">Width</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{frame_height}</p><p class="metric-label">Height</p></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Processing controls
        stframe = st.empty()
        progress_bar = st.progress(0, text="⏳ Ready to process...")
        
        # Control buttons - only show relevant ones based on processing state
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not st.session_state.is_processing:
                if st.button("🚀 Start Processing", type="primary", use_container_width=True, key="start_processing"):
                    st.session_state.is_processing = True
                    st.session_state.pause = False
                    st.session_state.detections_list = []
                    st.rerun()
            else:
                if st.button("⏹ Stop Processing", use_container_width=True, key="stop_processing"):
                    st.session_state.is_processing = False
                    st.session_state.pause = False
                    st.rerun()
        
        with col2:
            if st.session_state.is_processing and not st.session_state.pause:
                if st.button("⏸ Pause", use_container_width=True, key="pause_btn"):
                    st.session_state.pause = True
                    st.rerun()
        
        with col3:
            if st.session_state.is_processing and st.session_state.pause:
                if st.button("▶ Resume", use_container_width=True, key="resume_btn"):
                    st.session_state.pause = False
                    st.rerun()
        
        # Process video if processing is active
        if st.session_state.is_processing:
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                
                while cap.isOpened() and st.session_state.is_processing:
                    if st.session_state.pause:
                        if st.session_state.last_frame is not None:
                            stframe.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
                        time.sleep(0.1)
                        continue
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame with YOLO
                    results = model(frame)
                    annotated_frame = results[0].plot()
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st.session_state.last_frame = annotated_frame_rgb
                    
                    # Extract detections
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            label = model.names[cls_id]
                            timestamp = round(frame_count / fps, 2)
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            st.session_state.detections_list.append({
                                "Frame": frame_count,
                                "Timestamp (s)": timestamp,
                                "Detected_Logo_Name": label,
                                "Confidence": round(conf, 2),
                                "bbox_x1": int(x1),
                                "bbox_y1": int(y1),
                                "bbox_x2": int(x2),
                                "bbox_y2": int(y2),
                                "frame_width": frame_width,
                                "frame_height": frame_height
                            })
                    
                    # Display frame
                    stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                    
                    frame_count += 1
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress, text=f"Processing: {frame_count}/{total_frames}")
                
                # Processing complete or stopped
                cap.release()
                
                if frame_count >= total_frames - 1:  # Processing completed naturally
                    os.unlink(video_path)
                    st.session_state.processing_complete = True
                    st.session_state.is_processing = False
                    
                    # Save results
                    if st.session_state.detections_list:
                        df = pd.DataFrame(st.session_state.detections_list)
                        st.success("✅ Video processing completed!")
                        
                        with st.spinner("💾 Saving to database..."):
                            if save_detections_to_db(video_name, st.session_state.detections_list):
                                st.success(f"✅ {len(st.session_state.detections_list)} detections saved!")
                                # Clear cache and reload data
                                st.cache_data.clear()
                                st.session_state.db_loaded = False
                                st.session_state.db_checked = False
                            else:
                                st.error("❌ Failed to save to database")
                        
                        # Display results
                        st.markdown('<h3 style="color: white !important;">📊 Detection Results</h3>', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f'<div class="metric-card"><p class="metric-value">{len(df)}</p><p class="metric-label">Detections</p></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div class="metric-card"><p class="metric-value">{df["Detected_Logo_Name"].nunique()}</p><p class="metric-label">Brands</p></div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown(f'<div class="metric-card"><p class="metric-value">{frame_count}</p><p class="metric-label">Frames</p></div>', unsafe_allow_html=True)
                        with col4:
                            st.markdown(f'<div class="metric-card"><p class="metric-value">{df["Confidence"].mean():.2f}</p><p class="metric-label">Avg Conf</p></div>', unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.dataframe(df[['Frame', 'Timestamp (s)', 'Detected_Logo_Name', 'Confidence']], 
                                    use_container_width=True)
                        
                        # Brand summary
                        st.markdown('<h3 style="color: white !important;">📈 Brand Summary</h3>', unsafe_allow_html=True)
                        brand_summary = df.groupby('Detected_Logo_Name').agg({
                            'Confidence': ['count', 'mean'],
                            'Timestamp (s)': ['min', 'max']
                        }).round(2)
                        brand_summary.columns = ['Count', 'Avg Conf', 'First (s)', 'Last (s)']
                        brand_summary['Duration (s)'] = brand_summary['Last (s)'] - brand_summary['First (s)']
                        st.dataframe(brand_summary, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            f"{video_name}_detections.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ No brands detected in the video")
                else:
                    st.info("⏹ Processing stopped by user")
                    
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                cap.release()
                if os.path.exists(video_path):
                    os.unlink(video_path)
                st.session_state.is_processing = False
    else:
        st.info("👆 Upload a video to start brand detection")
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# TAB 2: AI Chat Assistant - FIXED
# -------------------------
with tab2:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: white !important; font-weight: 700;">💬 AI-Powered Database Q&A</h3>', unsafe_allow_html=True)
    
    if not st.session_state.db_initialized:
        st.warning("🚫 Please initialize the database first from the Video Processing tab")
    elif not st.session_state.db_loaded:
        st.info("📡 Loading database data...")
        # Try to load data
        columns = get_table_info()
        if columns:
            all_data, colnames = get_all_data()
            if all_data:
                st.session_state.columns = columns
                st.session_state.all_data = all_data
                st.session_state.db_loaded = True
                st.rerun()
            else:
                st.warning("💡 No data found. Process a video first!")
        else:
            st.error("❌ Database not properly initialized")
    else:
        st.success(f"✅ Database loaded: **{len(st.session_state.all_data)}** records")
        
        # Context size selector
        col1, col2 = st.columns([3, 1])
        with col2:
            context_size = st.selectbox(
                "Context Size",
                ["Small", "Optimized", "Medium"],
                index=1,
                help="Smaller context uses fewer tokens but may have less detail"
            )
        
        # Map context size to sample count
        size_map = {"Small": 20, "Optimized": 30, "Medium": 50}
        max_samples = size_map[context_size]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input - FIXED: No longer switches tabs
        if prompt := st.chat_input("Ask about brand detections..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing database..."):
                    context = format_context_optimized(
                        st.session_state.columns, 
                        st.session_state.all_data,
                        max_samples=max_samples
                    )
                    response = ask_groq_optimized(prompt, context)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# TAB 3: Analytics Dashboard
# -------------------------
with tab3:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: white !important; font-weight: 700;">📊 Analytics Dashboard</h3>', unsafe_allow_html=True)
    
    if not st.session_state.db_initialized:
        st.warning("🚫 Please initialize the database first")
    elif not st.session_state.db_loaded:
        st.info("📡 Loading analytics data...")
    elif not st.session_state.all_data:
        st.warning("💡 No data available. Process some videos first!")
    else:
        stats = get_database_stats()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{stats.get("total_rows", 0):,}</p><p class="metric-label">Total Rows</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{stats.get("unique_videos", 0)}</p><p class="metric-label">Videos</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{stats.get("unique_brands", 0)}</p><p class="metric-label">Brands</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{stats.get("avg_confidence", 0):.2f}</p><p class="metric-label">Avg Conf</p></div>', unsafe_allow_html=True)
        
        st.divider()
        
        df = pd.DataFrame(st.session_state.all_data)
        
        # Brand Distribution
        if 'detected_logo_name' in df.columns:
            st.markdown('<h3 style="color: white !important;">🏷️ Brand Distribution</h3>', unsafe_allow_html=True)
            brand_counts = df['detected_logo_name'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_bar = px.bar(
                    x=brand_counts.index, y=brand_counts.values,
                    labels={'x': 'Brand', 'y': 'Count'},
                    title='Brand Frequency',
                    color=brand_counts.values,
                    color_continuous_scale='plasma'
                )
                fig_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                fig_pie = px.pie(
                    values=brand_counts.values,
                    names=brand_counts.index,
                    title='Brand Distribution',
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                fig_pie.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        st.divider()
        
        # Confidence Distribution
        if 'confidence' in df.columns:
            st.markdown('<h3 style="color: white !important;">⭐ Confidence Distribution</h3>', unsafe_allow_html=True)
            fig_hist = px.histogram(
                df, x='confidence', nbins=20,
                title='Confidence Scores',
                labels={'confidence': 'Confidence', 'count': 'Frequency'},
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# TAB 4: Database Explorer
# -------------------------
with tab4:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: white !important; font-weight: 700;">🔍 Database Explorer</h3>', unsafe_allow_html=True)
    
    if not st.session_state.db_initialized:
        st.warning("🚫 Please initialize the database first")
    elif not st.session_state.db_loaded:
        st.info("📡 Loading database explorer...")
    elif not st.session_state.all_data:
        st.warning("💡 No data available. Process some videos first!")
    else:
        df = pd.DataFrame(st.session_state.all_data)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            if 'detected_logo_name' in df.columns:
                brands = ['All'] + list(df['detected_logo_name'].unique())
                selected_brand = st.selectbox("Filter by Brand", brands)
        
        with col2:
            if 'video_name' in df.columns:
                videos = ['All'] + list(df['video_name'].unique())
                selected_video = st.selectbox("Filter by Video", videos)
        
        # Apply filters
        filtered_df = df.copy()
        if selected_brand != 'All':
            filtered_df = filtered_df[filtered_df['detected_logo_name'] == selected_brand]
        if selected_video != 'All':
            filtered_df = filtered_df[filtered_df['video_name'] == selected_video]
        
        st.info(f"Showing {len(filtered_df)} of {len(df)} rows")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Data",
            csv,
            f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Premium Footer
# -------------------------
st.divider()
st.markdown(f"""
<div class="footer-glow">
    <p>🚀 Powered by YOLOv11 • Groq AI • AWS S3 • PostgreSQL</p>
    <p>⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <div class="creator-badge">💎 Crafted by SAI SUDHARSAN S G</div>
</div>
""", unsafe_allow_html=True)