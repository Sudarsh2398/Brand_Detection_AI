

🎯 Jio Hotstar AdVison & Analytics(BrandVision AI) - Brand Detection & Analytics Platform

A sophisticated AI-powered platform for real-time brand logo detection in videos, featuring advanced analytics, database storage, and intelligent Q&A capabilities.

🌟 Features

🎯 Core Capabilities

Real-time Brand Detection: YOLOv11 model for accurate logo detection
Video Processing: Support for MP4, AVI, MOV, MKV formats
Database Integration: PostgreSQL for structured data storage
AI-Powered Analytics: Groq AI for intelligent data insights
Cloud Storage: AWS S3 integration for video storage
Interactive Dashboard: Beautiful Streamlit interface with glassmorphism design

📊 Analytics & Insights

Brand Frequency Analysis: Track brand appearances over time
Confidence Metrics: Detailed confidence score analysis
Temporal Analysis: Brand appearance timing and duration
Interactive Visualizations: Plotly charts and graphs
Export Capabilities: CSV download for further analysis

💬 Intelligent Assistant

Natural Language Q&A: Ask questions about your brand detection data
RAG Implementation: Retrieval-Augmented Generation for accurate responses
Context-Aware: Understands brand relationships and patterns
Multi-turn Conversations: Maintains conversation context

🚀 Quick Start

Prerequisites

Python 3.8+
PostgreSQL 15+
AWS Account (for S3 - optional)
Groq API Key (free tier available)

Installation

Clone the repository

bash
git clone <repository-url>
cd brandvision-ai

Install dependencies

bash
pip install -r requirements.txt
Set up environment variables
Create a .env file with the following variables:

env
# Database Configuration
PG_HOST=localhost
PG_DB=brand_detection
PG_USER=postgres
PG_PASS=your_password
PG_PORT=5432

# YOLO Model Path
YOLO_MODEL_PATH=sportsModel.pt

# Groq AI Configuration
GROQ_API_KEY=your_groq_api_key

# AWS S3 Configuration (Optional)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-south-1
S3_BUCKET_NAME=your_bucket_name
Set up PostgreSQL Database

sql
CREATE DATABASE brand_detection;
Run the application

bash
streamlit run app.py
📁 Project Structure
text
brandvision-ai/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── sportsModel.pt        # YOLO model weights
└── README.md            # Project documentation

🛠️ Technical Architecture

Backend Stack

Streamlit: Web application framework
YOLOv11: Object detection model for brand logos
PostgreSQL: Relational database for detection data
Groq AI: LLM for natural language processing
AWS S3: Cloud storage for video files
OpenCV: Video processing and frame extraction

Frontend Features

Glassmorphism UI: Modern, translucent design elements
Real-time Processing: Live video analysis with progress tracking
Interactive Charts: Plotly-based data visualizations
Responsive Design: Works on desktop and mobile devices

📊 Application Tabs

1. 📹 Video Processing
Upload and process video files
Real-time brand detection visualization
Progress tracking and controls
Results export to database and CSV

2. 💬 AI Assistant
Natural language queries about brand data
Intelligent insights and analysis
Context-aware responses
Multi-turn conversation support

3. 📊 Analytics Dashboard
Brand frequency and distribution charts
Confidence score analysis
Temporal pattern recognition
Interactive data visualizations

4. 🔍 Data Explorer
Raw database data browsing
Advanced filtering capabilities
Data export functionality
Real-time data exploration

🔧 Configuration

Database Setup
Install PostgreSQL 15+
Create database: CREATE DATABASE brand_detection;

The application will automatically create required tables

Groq AI Setup

Visit Groq Console
Sign up for free account
Generate API key
Add to .env file

AWS S3 Setup (Optional)

Create S3 bucket
Generate IAM credentials with S3 access
Configure bucket name and region in .env

🎮 Usage Guide
Processing Videos
Navigate to Video Processing tab
Upload video file (MP4, AVI, MOV, MKV)
Click "Start Processing" to begin analysis
Monitor real-time detection results
View and export detection data

Asking Questions
Go to AI Assistant tab

Ask natural language questions like:
"What brands were detected?"
"Show high confidence detections"
"Which brand appears most frequently?"
"Compare brand visibility across videos"

Analyzing Data
Use Analytics Dashboard for visual insights
Explore raw data in Data Explorer
Apply filters for specific brands or videos
Download data for external analysis

📈 Model Performance
Model: YOLOv11 (custom-trained on brand logos)
Input Resolution: Adaptive to video source
Processing Speed: Real-time on supported hardware
Accuracy: High precision for trained brand classes

🔒 Security Features
Environment variable configuration
Secure database connections
Optional S3 encryption
No sensitive data in frontend

🌐 Deployment Options
Local Development
bash
streamlit run app.py
Production Deployment
Streamlit Cloud: One-click deployment

AWS EC2: Scalable cloud deployment
Docker: Containerized deployment
Heroku: Platform-as-a-service

📝 API Documentation
Groq AI Integration
python
# RAG Implementation
def ask_groq_optimized(question, context):
    # Context-aware responses
    # Automatic retry logic
    # Token optimization
Database Schema
sql
brand_detections (
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
    detection_datetime TIMESTAMP
)
🐛 Troubleshooting

Common Issues

Database Connection Failed
Check PostgreSQL service is running
Verify credentials in .env
Ensure database exists
Model Loading Error
Verify sportsModel.pt exists
Check file permissions
Ensure compatible PyTorch version
Groq API Errors
Verify API key in .env
Check internet connection
Monitor rate limits
S3 Upload Issues
Verify AWS credentials
Check bucket permissions
Confirm region configuration
Performance Optimization
Use GPU for faster processing
Optimize database indexes
Enable S3 multipart uploads for large files
Adjust frame sampling rate for longer videos

🤝 Contributing
Fork the repository

👨‍💻 Author
SAI SUDHARSAN S G

🙏 Acknowledgments
YOLO community for object detection models

Streamlit for amazing web app framework
Groq for high-performance AI inference
PostgreSQL team for robust database solutions
AWS for cloud infrastructure services

📞 Support
For support and questions:
Check Troubleshooting section
Open an issue on GitHub
Contact the maintainer

⭐ Star this repo if you find it helpful!

Built with ❤️ using Python, Streamlit, and AI technologies
