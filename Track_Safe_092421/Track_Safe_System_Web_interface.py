import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import sqlite3
import pickle
import face_recognition
from PIL import Image
import tempfile
import math
from ultralytics import YOLO
import hashlib
import base64

class UserAuthentication:
    def __init__(self, database_folder="user_auth_db"):
        self.database_folder = database_folder
        self.auth_db_path = os.path.join(database_folder, "user_authentication.db")
        self.initialize_auth_database()
        
    def initialize_auth_database(self):
        """Initialize the user authentication database"""
        os.makedirs(self.database_folder, exist_ok=True)
        
        conn = sqlite3.connect(self.auth_db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
                full_name TEXT,
                profile_image_path TEXT,
                registration_date TEXT NOT NULL,
                is_admin BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create admin user if not exists
        cursor.execute('SELECT username FROM users WHERE username = "admin"')
        if not cursor.fetchone():
            admin_hash = self.hash_password("admin123")  # Default admin password
            cursor.execute('''
                INSERT INTO users 
                (username, password_hash, email, full_name, registration_date, is_admin)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', ("admin", admin_hash, "admin@tracksafe.com", "Admin User", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), True))
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash a password for storing"""
        salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
        pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), 
                                     salt, 100000)
        pwdhash = base64.b64encode(pwdhash).decode('ascii')
        return f"{salt.decode('ascii')}${pwdhash}"
    
    def verify_password(self, stored_password, provided_password):
        """Verify a stored password against one provided by user"""
        salt, pwdhash = stored_password.split('$')
        new_hash = hashlib.pbkdf2_hmac('sha512', 
                                      provided_password.encode('utf-8'), 
                                      salt.encode('ascii'), 
                                      100000)
        new_hash = base64.b64encode(new_hash).decode('ascii')
        return pwdhash == new_hash
    
    def register_user(self, username, password, email, full_name, profile_image=None):
        """Register a new user"""
        if self.user_exists(username):
            return False, "Username already exists"
        
        password_hash = self.hash_password(password)
        registration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Handle profile image
        profile_image_path = None
        if profile_image is not None:
            profile_image_path = self.save_profile_image(username, profile_image)
        
        conn = sqlite3.connect(self.auth_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users 
                (username, password_hash, email, full_name, profile_image_path, registration_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, password_hash, email, full_name, profile_image_path, registration_date))
            
            conn.commit()
            return True, "Registration successful"
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
        finally:
            conn.close()
    
    def save_profile_image(self, username, profile_image):
        """Save profile image to disk with proper format handling"""
        os.makedirs(os.path.join(self.database_folder, "profile_images"), exist_ok=True)
        image_path = os.path.join(self.database_folder, "profile_images", f"{username}.jpg")
        
        try:
            img = Image.open(profile_image)
            
            # Convert RGBA to RGB if needed
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # Paste using alpha channel as mask
                img = background
            
            # Ensure image is in RGB mode before saving as JPEG
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.save(image_path, format='JPEG', quality=95)
            return image_path
            
        except Exception as e:
            st.error(f"Error saving profile image: {str(e)}")
            return None
    
    def update_profile_image(self, username, profile_image):
        """Update user's profile image"""
        if not self.user_exists(username):
            return False, "User not found"
        
        profile_image_path = self.save_profile_image(username, profile_image)
        if not profile_image_path:
            return False, "Failed to save profile image"
        
        conn = sqlite3.connect(self.auth_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE users SET profile_image_path = ? WHERE username = ?
            ''', (profile_image_path, username))
            
            conn.commit()
            return True, "Profile image updated successfully"
        except Exception as e:
            return False, f"Failed to update profile image: {str(e)}"
        finally:
            conn.close()
    
    def user_exists(self, username):
        """Check if a username exists"""
        conn = sqlite3.connect(self.auth_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def authenticate_user(self, username, password):
        """Authenticate a user"""
        conn = sqlite3.connect(self.auth_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT password_hash, is_admin FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result is None:
            return False, "User not found", False
        
        stored_password, is_admin = result
        if self.verify_password(stored_password, password):
            return True, "Authentication successful", is_admin
        else:
            return False, "Incorrect password", False
    
    def get_user_info(self, username):
        """Get user information"""
        conn = sqlite3.connect(self.auth_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, email, full_name, profile_image_path, registration_date, is_admin 
            FROM users WHERE username = ?
        ''', (username,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result is None:
            return None
        
        return {
            'username': result[0],
            'email': result[1],
            'full_name': result[2],
            'profile_image_path': result[3],
            'registration_date': result[4],
            'is_admin': result[5]
        }

class EnhancedTrackSafeSystem:
    def __init__(self, username=None):
        # Initialize with username-specific values
        self.username = username
        self.database_folder = f"user_databases/{username}_db" if username else "track_safe_db"
        self.database_name = f"{username}_db" if username else "track_safe_db"
        self.db_path = os.path.join(self.database_folder, f"{self.database_name}.db")
        self.known_encodings = {}
        self.workers_data = {}
        self.available_jobs = ["manager", "engineer", "technician", "supervisor", "operator"]
        self.recognition_threshold = 0.6
        self.ppe_model = None
        self.person_model = None
        self.ppe_model_path = "the main model.pt"
        self.person_model_path = "yolov8n.pt"
        self.classNames = ['Boot', 'Face-Protector', 'Gloves', 'Helmet', 'Safety-Glasses', 'Vest', 'Normal-Glasses']
        self.REQUIRED_SAFETY_ITEMS = ['Helmet', 'Vest', 'Safety-Glasses', 'Gloves', 'Boot', 'Face-Protector']
        self.log_columns = [
            'Date', 'Time', 'Name', 'ID', 'Position', 'Salary',
            'Helmet', 'Vest', 'Safety-Glasses', 'Gloves', 'Boot',
            'Face-Protector', 'Non_Safety_Glasses', 'Status'
        ]
        self.log_file = os.path.join(self.database_folder, "track_safe_log.csv")
        self.last_logged_state = {}
        self.last_beep_time = 0
        self.beep_interval = 5.0
        self.frame_skip = 2
        self.frame_count = 0

        # Ensure the database folder exists
        os.makedirs(self.database_folder, exist_ok=True)
        
        # Create log file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=self.log_columns).to_csv(self.log_file, index=False)

        # Initialize database
        self.initialize_database()

    def initialize_database(self):
        """Initialize or connect to database"""
        try:
            os.makedirs(self.database_folder, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create workers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workers (
                    id TEXT PRIMARY KEY,
                    first_name TEXT NOT NULL,
                    middle_name TEXT,
                    last_name TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    salary REAL NOT NULL,
                    job TEXT NOT NULL,
                    folder_name TEXT NOT NULL,
                    registration_date TEXT NOT NULL,
                    face_encodings BLOB
                )
            ''')
            
            # Create jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    job_name TEXT PRIMARY KEY
                )
            ''')
            
            # Insert default jobs
            for job in self.available_jobs:
                cursor.execute('INSERT OR IGNORE INTO jobs (job_name) VALUES (?)', (job,))
            
            conn.commit()
            conn.close()
            
            # Load existing data
            self.load_data()
            
        except Exception as e:
            st.error(f"Database initialization failed: {str(e)}")

    def load_data(self):
        """Load workers data and encodings from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM workers')
            workers = cursor.fetchall()
            
            for worker in workers:
                worker_id, first_name, middle_name, last_name, full_name, salary, job, folder_name, reg_date, encodings_blob = worker
                
                self.workers_data[worker_id] = {
                    'first_name': first_name,
                    'middle_name': middle_name or '',
                    'last_name': last_name,
                    'full_name': full_name,
                    'salary': salary,
                    'job': job,
                    'folder': folder_name,
                    'registration_date': reg_date
                }
                
                if encodings_blob:
                    encodings = pickle.loads(encodings_blob)
                    self.known_encodings[worker_id] = encodings
            
            cursor.execute('SELECT job_name FROM jobs')
            jobs = cursor.fetchall()
            self.available_jobs = [job[0] for job in jobs]
            
            conn.close()
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    def save_worker_to_database(self, worker_id, worker_data, encodings):
        """Save worker data to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            encodings_blob = pickle.dumps(encodings)
            
            cursor.execute('''
                INSERT OR REPLACE INTO workers 
                (id, first_name, middle_name, last_name, full_name, salary, job, folder_name, registration_date, face_encodings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                worker_id,
                worker_data['first_name'],
                worker_data['middle_name'],
                worker_data['last_name'],
                worker_data['full_name'],
                worker_data['salary'],
                worker_data['job'],
                worker_data['folder'],
                worker_data['registration_date'],
                encodings_blob
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Error saving worker to database: {str(e)}")
            return False

    def generate_worker_id(self):
        """Generate unique 3-digit worker ID"""
        if not self.workers_data:
            return "001"
        
        existing_ids = [int(worker_id) for worker_id in self.workers_data.keys()]
        max_id = max(existing_ids)
        new_id = max_id + 1
        return f"{new_id:03d}"

    def initialize_ppe_models(self):
        """Initialize PPE detection models with proper error handling"""
        try:
            # Check if model files exist
            if not os.path.exists(self.ppe_model_path):
                st.error(f"PPE model file '{self.ppe_model_path}' not found")
                return False
            if not os.path.exists(self.person_model_path):
                st.error(f"Person detection model file '{self.person_model_path}' not found")
                return False
                
            with st.spinner("Loading PPE detection model..."):
                self.ppe_model = YOLO(self.ppe_model_path)
                
            with st.spinner("Loading person detection model..."):
                self.person_model = YOLO(self.person_model_path)
                
            return True
        except Exception as e:
            st.error(f"Error loading PPE models: {str(e)}")
            return False

    def recognize_face(self, face_encoding):
        """Enhanced face recognition with confidence scoring"""
        best_match_id = None
        best_distance = float('inf')
        
        for worker_id, encodings_list in self.known_encodings.items():
            for encoding in encodings_list:
                distance = np.linalg.norm(face_encoding - encoding)
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = worker_id
        
        if best_distance < self.recognition_threshold:
            confidence = 1 - best_distance
            return best_match_id, confidence
        
        return None, 0.0

    def should_log_worker(self, worker_id, equipment_status, face_protector_status, non_safety_detected):
        """Determine if worker should be logged based on state changes"""
        current_state = {
            'equipment': {item: equipment_status[item]['worn'] for item in self.REQUIRED_SAFETY_ITEMS},
            'face_protector': face_protector_status['worn'],
            'non_safety': non_safety_detected
        }
        
        prev_state = self.last_logged_state.get(worker_id)
        
        if prev_state is None or prev_state != current_state:
            self.last_logged_state[worker_id] = current_state
            return True
        
        return False

    def log_worker_status(self, worker_info, worker_id, equipment_status, face_protector_status, non_safety_detected, all_worn, any_worn):
        """Log worker status to CSV file"""
        try:
            now = datetime.now()
            
            log_entry = {
                'Date': now.strftime("%Y-%m-%d"),
                'Time': now.strftime("%H:%M:%S"),
                'Name': worker_info['full_name'].replace('_', ' ').title(),
                'ID': worker_id,
                'Position': worker_info['job'].title(),
                'Salary': f"${worker_info['salary']:.2f}",
                'Helmet': 'Yes' if equipment_status['Helmet']['worn'] else 'No',
                'Vest': 'Yes' if equipment_status['Vest']['worn'] else 'No',
                'Safety-Glasses': 'Yes' if equipment_status['Safety-Glasses']['worn'] else 'No',
                'Gloves': 'Yes' if equipment_status['Gloves']['worn'] else 'No',
                'Boot': 'Yes' if equipment_status['Boot']['worn'] else 'No',
                'Face-Protector': 'Yes' if face_protector_status['worn'] else 'No',
                'Non_Safety_Glasses': 'Yes' if non_safety_detected else 'No',
                'Status': 'Compliant' if all_worn else 'Partial' if any_worn else 'Non-Compliant'
            }
            
            # Create DataFrame from the log entry
            log_df = pd.DataFrame([log_entry])
            
            # Check if file exists to determine if we need headers
            file_exists = os.path.isfile(self.log_file)
            
            # Append to CSV
            log_df.to_csv(
                self.log_file,
                mode='a',
                header=not file_exists,
                index=False
            )
            
        except Exception as e:
            st.error(f"Error logging worker status: {str(e)}")
            print(f"Logging error: {str(e)}")

class TrackSafeWebApp:
    def __init__(self):
        self.auth = UserAuthentication()
        self.system = None
        self.setup_ui()
        
    def setup_ui(self):
        """Configure Streamlit UI settings"""
        st.set_page_config(
            page_title="Track_Safe - Worker Safety Monitoring",
            page_icon="🛡️",
            layout="wide"
        )
        
        st.markdown("""
        <style>
            .main {
                padding: 2rem;
            }
            .stButton>button {
                width: 100%;
            }
            .stAlert {
                padding: 1rem;
            }
            .sidebar .sidebar-content {
                padding: 1rem;
            }
            .camera-feed {
                border-radius: 10px;
                border: 2px solid #ccc;
            }
            .logo {
                text-align: center;
                margin-bottom: 2rem;
            }
            .logo h1 {
                color: #2c3e50;
            }
            .profile-image {
                width: 100px;
                height: 100px;
                border-radius: 50%;
                object-fit: cover;
                margin: 0 auto;
                display: block;
            }
            .auth-container {
                max-width: 500px;
                margin: 0 auto;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        </style>
        """, unsafe_allow_html=True)
    
    def show_auth_pages(self):
        """Show authentication pages (login/signup)"""
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'current_auth_page' not in st.session_state:
            st.session_state.current_auth_page = "login"
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'is_admin' not in st.session_state:
            st.session_state.is_admin = False
        
        # If user is logged in, show the main app
        if st.session_state.get('logged_in', False):
            self.system = EnhancedTrackSafeSystem(st.session_state.username)
            self.run()
            return
        
        # Show auth pages if not logged in
        st.markdown("""
        <div class="logo">
            <h1>Track_Safe 🛡️</h1>
            <h3>Worker Safety Monitoring System</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            
            # Auth page selection tabs
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            
            with tab1:
                with st.form("login_form"):
                    st.subheader("Login to Your Account")
                    
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    
                    submitted = st.form_submit_button("Login")
                    
                    if submitted:
                        if not username or not password:
                            st.error("Please enter both username and password")
                        else:
                            success, message, is_admin = self.auth.authenticate_user(username, password)
                            if success:
                                st.session_state.logged_in = True
                                st.session_state.username = username
                                st.session_state.is_admin = is_admin
                                st.session_state.current_auth_page = "login"
                                st.rerun()
                            else:
                                st.error(message)
            
            with tab2:
                with st.form("signup_form"):
                    st.subheader("Create New Account")
                    
                    username = st.text_input("Choose a Username")
                    password = st.text_input("Choose a Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    email = st.text_input("Email (optional)")
                    full_name = st.text_input("Full Name (optional)")
                    profile_image = st.file_uploader("Profile Image (optional)", type=['jpg', 'jpeg', 'png'])
                    
                    submitted = st.form_submit_button("Create Account")
                    
                    if submitted:
                        if not username or not password or not confirm_password:
                            st.error("Please fill in all required fields")
                        elif password != confirm_password:
                            st.error("Passwords do not match")
                        else:
                            success, message = self.auth.register_user(username, password, email, full_name, profile_image)
                            if success:
                                st.success(message)
                                st.session_state.current_auth_page = "login"
                                st.rerun()
                            else:
                                st.error(message)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def show_profile_page(self):
        """Show user profile page"""
        st.title("👤 User Profile")
        
        user_info = self.auth.get_user_info(st.session_state.username)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if user_info['profile_image_path'] and os.path.exists(user_info['profile_image_path']):
                st.image(user_info['profile_image_path'], width=150, caption="Profile Image")
            else:
                st.image("https://via.placeholder.com/150", width=150, caption="Default Profile")
            
            new_image = st.file_uploader("Update Profile Image", type=['jpg', 'jpeg', 'png'])
            if new_image:
                success, message = self.auth.update_profile_image(st.session_state.username, new_image)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        with col2:
            st.subheader(user_info['full_name'] or st.session_state.username)
            st.write(f"**Username:** {user_info['username']}")
            st.write(f"**Email:** {user_info['email'] or 'Not provided'}")
            st.write(f"**Registered since:** {user_info['registration_date']}")
            
            if st.session_state.is_admin:
                st.success("**Account Type:** Administrator")
            else:
                st.info("**Account Type:** Standard User")
            
            if st.button("Logout", type="primary"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.is_admin = False
                self.system = None
                st.rerun()
    
    def run(self):
        """Main application runner"""
        # Initialize session state variables
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Dashboard"
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
        
        # Sidebar for navigation and profile
        with st.sidebar:
            st.title("Navigation")
            
            # Show profile info at the top
            user_info = self.auth.get_user_info(st.session_state.username)
            if user_info['profile_image_path'] and os.path.exists(user_info['profile_image_path']):
                st.image(user_info['profile_image_path'], width=80, caption=user_info['username'])
            else:
                st.image("https://via.placeholder.com/80", width=80, caption=user_info['username'])
            
            st.write(f"**{user_info['full_name'] or user_info['username']}**")
            
            if st.button("View Profile"):
                st.session_state.current_page = "Profile"
            
            st.markdown("---")
            
            # Standard pages available to all users
            page_options = ["Dashboard", "Worker Registration", "Worker Database", 
                           "Safety Monitoring", "Safety Logs"]
            
            # Add admin-only pages if user is admin
            if st.session_state.is_admin:
                page_options.append("System Settings")
            
            selected_page = st.radio("Menu", page_options, 
                                    index=page_options.index(st.session_state.current_page) 
                                    if st.session_state.current_page in page_options else 0)
            
            if selected_page != st.session_state.current_page:
                st.session_state.current_page = selected_page
                st.rerun()
            
            st.markdown("---")
            
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.is_admin = False
                self.system = None
                st.rerun()
        
        # Route to the selected page
        if st.session_state.current_page == "Dashboard":
            self.show_dashboard()
        elif st.session_state.current_page == "Worker Registration":
            self.show_registration()
        elif st.session_state.current_page == "Worker Database":
            self.show_database()
        elif st.session_state.current_page == "Safety Monitoring":
            self.show_monitoring()
        elif st.session_state.current_page == "Safety Logs":
            self.show_logs()
        elif st.session_state.current_page == "System Settings":
            self.show_settings()
        elif st.session_state.current_page == "Profile":
            self.show_profile_page()
    
    def show_dashboard(self):
        """Show the main dashboard"""
        st.markdown("""
        <div class="logo">
            <h1>Track_Safe 🛡️</h1>
            <h3>Worker Safety Monitoring System</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Registered Workers", len(self.system.workers_data))
        
        with col2:
            try:
                if os.path.exists(self.system.log_file):
                    logs = pd.read_csv(self.system.log_file)
                    compliant = logs[logs['Status'] == 'Compliant'].shape[0]
                    st.metric("Compliant Entries", compliant)
                else:
                    st.metric("Compliant Entries", 0)
            except:
                st.metric("Compliant Entries", 0)
        
        with col3:
            try:
                if os.path.exists(self.system.log_file):
                    logs = pd.read_csv(self.system.log_file)
                    non_compliant = logs[logs['Status'] != 'Compliant'].shape[0]
                    st.metric("Safety Violations", non_compliant)
                else:
                    st.metric("Safety Violations", 0)
            except:
                st.metric("Safety Violations", 0)
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("Quick Actions")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("🚀 Start Safety Monitoring"):
                st.session_state.current_page = "Safety Monitoring"
                st.rerun()
        
        with action_col2:
            if st.button("👤 Register New Worker"):
                st.session_state.current_page = "Worker Registration"
                st.rerun()
        
        with action_col3:
            if st.button("📊 View Safety Logs"):
                st.session_state.current_page = "Safety Logs"
                st.rerun()
        
        st.markdown("---")
        
        # Recent safety violations
        st.subheader("Recent Safety Violations")
        try:
            if os.path.exists(self.system.log_file):
                logs = pd.read_csv(self.system.log_file)
                violation_logs = logs[logs['Status'] != 'Compliant'].sort_values(
                    ['Date', 'Time'], ascending=False).head(5)
                st.dataframe(violation_logs)
            else:
                st.info("No safety violations recorded yet")
        except:
            st.info("No safety violations recorded yet")

    def show_registration(self):
        """Show worker registration interface"""
        st.title("👤 Worker Registration")
        
        with st.form("worker_registration"):
            st.subheader("Worker Information")
            
            # Name input
            name_col1, name_col2 = st.columns(2)
            with name_col1:
                first_name = st.text_input("First Name*", "")
            with name_col2:
                last_name = st.text_input("Last Name*", "")
            
            middle_name = st.text_input("Middle Name (optional)", "")
            
            # Job and salary
            job_col, salary_col = st.columns(2)
            with job_col:
                job = st.selectbox("Job Position*", self.system.available_jobs)
            with salary_col:
                salary = st.number_input("Salary*", min_value=0.0, step=100.0)
            
            # Face capture
            st.subheader("Face Capture")
            st.info("Please capture multiple images of the worker's face from different angles")
            
            # Webcam capture
            img_file_buffer = st.camera_input("Take a photo for registration")
            
            submitted = st.form_submit_button("Register Worker")
            
            if submitted:
                if not first_name or not last_name:
                    st.error("Please fill in all required fields (marked with *)")
                    return
                
                if not img_file_buffer:
                    st.error("Please capture at least one photo of the worker's face")
                    return
                
                # Process registration
                with st.spinner("Registering worker..."):
                    try:
                        # Create worker data dictionary
                        full_name = f"{first_name}_{middle_name}_{last_name}" if middle_name else f"{first_name}_{last_name}"
                        worker_id = self.system.generate_worker_id()
                        folder_name = f"{self.system.database_name}_{full_name}_{worker_id}"
                        worker_folder = os.path.join(self.system.database_folder, folder_name)
                        
                        os.makedirs(worker_folder, exist_ok=True)
                        
                        # Save captured image
                        img = Image.open(img_file_buffer)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        image_filename = f"{self.system.database_name}_{full_name}_{worker_id}_{timestamp}.jpg"
                        image_path = os.path.join(worker_folder, image_filename)
                        
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(image_path)
                        
                        # Generate face encoding
                        img_array = np.array(img)
                        rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        
                        # Detect face locations first
                        face_locations = face_recognition.face_locations(rgb_img)
                        if not face_locations:
                            st.error("No face detected in the captured image")
                            return
                        
                        # Then get encodings for detected faces
                        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                        
                        if not face_encodings:
                            st.error("Could not generate face encodings")
                            return
                        
                        # Create worker data
                        worker_data = {
                            'first_name': first_name,
                            'middle_name': middle_name,
                            'last_name': last_name,
                            'full_name': full_name,
                            'salary': salary,
                            'job': job,
                            'folder': folder_name,
                            'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Save to database
                        if self.system.save_worker_to_database(worker_id, worker_data, face_encodings):
                            # Update local data
                            self.system.workers_data[worker_id] = worker_data
                            self.system.known_encodings[worker_id] = face_encodings
                            
                            st.success(f"Worker {full_name.replace('_', ' ')} successfully registered with ID: {worker_id}")
                        else:
                            st.error("Failed to save worker to database")
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")

    def show_database(self):
        """Show worker database"""
        st.title("📋 Worker Database")
        
        if not self.system.workers_data:
            st.info("No workers registered yet")
            return
        
        # Search and filter
        search_col, filter_col = st.columns(2)
        with search_col:
            search_term = st.text_input("Search by name or ID")
        with filter_col:
            filter_job = st.selectbox("Filter by job", ["All"] + self.system.available_jobs)
        
        # Display workers in a table
        workers_list = []
        for worker_id, data in self.system.workers_data.items():
            workers_list.append({
                "ID": worker_id,
                "Full Name": data['full_name'].replace('_', ' ').title(),
                "First Name": data['first_name'].title(),
                "Middle Name": data['middle_name'].title() if data['middle_name'] else "",
                "Last Name": data['last_name'].title(),
                "Job": data['job'].title(),
                "Salary": f"${data['salary']:.2f}",
                "Registered": data['registration_date']
            })
        
        df = pd.DataFrame(workers_list)
        
        # Apply filters
        if search_term:
            df = df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False)).any(axis=1)]
        if filter_job != "All":
            df = df[df['Job'] == filter_job.title()]
        
        st.dataframe(df)
        
        # Export options
        st.download_button(
            label="📥 Export to CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='ppe_log.csv',
            mime='text/csv'
        )

    def show_monitoring(self):
        """Show real-time safety monitoring interface"""
        st.title("👷 Real-time Safety Monitoring")
        
        if not self.system.workers_data:
            st.warning("No workers registered yet. Please register workers first.")
            return
        
        # Initialize models if not already done
        if not hasattr(self.system, 'ppe_model') or self.system.ppe_model is None:
            with st.spinner("Loading safety detection models..."):
                if not self.system.initialize_ppe_models():
                    st.error("Failed to load safety detection models")
                    return
        
        # Monitoring controls
        col1, col2 = st.columns(2)
        with col1:
            start_monitoring = st.button("▶ Start Monitoring")
        with col2:
            stop_monitoring = st.button("⏹ Stop Monitoring")
        
        # Camera feed placeholder
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Run monitoring if active
        if start_monitoring:
            st.session_state.monitoring_active = True
        
        if stop_monitoring:
            st.session_state.monitoring_active = False
        
        if st.session_state.get('monitoring_active', False):
            cap = cv2.VideoCapture(0)
            
            try:
                while st.session_state.get('monitoring_active', False):
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture video")
                        break
                    
                    # Process frame
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Face recognition
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    # PPE detection
                    results = self.system.ppe_model(frame)
                    
                    # Initialize equipment status
                    equipment_status = {
                        item: {'worn': False, 'confidence': 0} 
                        for item in self.system.REQUIRED_SAFETY_ITEMS
                    }
                    face_protector_status = {'worn': False, 'confidence': 0}
                    non_safety_detected = False
                    
                    # Process PPE results
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            cls = int(box.cls[0])
                            label = f"{self.system.classNames[cls]} {conf:.2f}"
                            
                            if cls < len(self.system.classNames):
                                class_name = self.system.classNames[cls]
                                
                                if class_name in equipment_status:
                                    equipment_status[class_name]['worn'] = True
                                    equipment_status[class_name]['confidence'] = conf
                                elif class_name == 'Face-Protector':
                                    face_protector_status['worn'] = True
                                    face_protector_status['confidence'] = conf
                                elif class_name == 'Normal-Glasses':
                                    non_safety_detected = True
                    
                    # Check if all required equipment is worn
                    all_worn = all(equipment_status[item]['worn'] for item in self.system.REQUIRED_SAFETY_ITEMS)
                    any_worn = any(equipment_status[item]['worn'] for item in self.system.REQUIRED_SAFETY_ITEMS)
                    
                    # Process each detected face
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        worker_id, confidence = self.system.recognize_face(face_encoding)
                        
                        # Draw face box
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        
                        if worker_id:
                            worker_info = self.system.workers_data[worker_id]
                            label = f"{worker_info['full_name']} ({confidence:.2f})"
                            cv2.putText(frame, label, (left, top-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # Log worker status if needed
                            if self.system.should_log_worker(worker_id, equipment_status, face_protector_status, non_safety_detected):
                                self.system.log_worker_status(
                                    worker_info, worker_id, equipment_status, 
                                    face_protector_status, non_safety_detected, 
                                    all_worn, any_worn
                                )
                    
                    # Draw PPE detections
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            cls = int(box.cls[0])
                            
                            if cls < len(self.system.classNames):
                                label = f"{self.system.classNames[cls]} {conf:.2f}"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Display in Streamlit
                    camera_placeholder.image(frame, channels="BGR", use_column_width=True)
                    
                    # Add brief delay to allow UI updates
                    time.sleep(0.1)
                    
            finally:
                cap.release()
                cv2.destroyAllWindows()
        else:
            # Show static camera feed when not monitoring
            camera_placeholder.image("https://via.placeholder.com/640x480?text=Monitoring+Not+Active", 
                                   use_column_width=True)

    def show_logs(self):
        """Show safety compliance logs with better error handling"""
        st.title("📝 Safety Compliance Logs")
        
        try:
            if os.path.exists(self.system.log_file):
                try:
                    logs = pd.read_csv(self.system.log_file)
                    
                    if logs.empty:
                        st.info("No safety logs recorded yet")
                        return
                    
                    # Ensure all required columns exist
                    for col in self.system.log_columns:
                        if col not in logs.columns:
                            logs[col] = "Unknown"
                    
                    # Date range filter
                    date_col1, date_col2 = st.columns(2)
                    with date_col1:
                        start_date = st.date_input("Start date", value=pd.to_datetime(logs['Date']).min())
                    with date_col2:
                        end_date = st.date_input("End date", value=pd.to_datetime(logs['Date']).max())
                    
                    # Convert to datetime for comparison
                    logs['Date'] = pd.to_datetime(logs['Date'])
                    filtered_logs = logs[(logs['Date'].dt.date >= start_date) & 
                                        (logs['Date'].dt.date <= end_date)]
                    
                    # Additional filters
                    name_filter = st.text_input("Filter by name")
                    if name_filter:
                        filtered_logs = filtered_logs[filtered_logs['Name'].str.contains(name_filter, case=False)]
                    
                    status_filter = st.selectbox("Filter by status", ["All"] + list(filtered_logs['Status'].unique()))
                    if status_filter != "All":
                        filtered_logs = filtered_logs[filtered_logs['Status'] == status_filter]
                    
                    # Show logs
                    st.dataframe(filtered_logs.sort_values(['Date', 'Time'], ascending=False))
                    
                    # Statistics
                    st.subheader("Compliance Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Entries", len(filtered_logs))
                    
                    with col2:
                        compliant = len(filtered_logs[filtered_logs['Status'] == 'Compliant'])
                        st.metric("Compliant", compliant)
                    
                    with col3:
                        violations = len(filtered_logs) - compliant
                        st.metric("Violations", violations)
                    
                    # Export button
                    st.download_button(
                        label="📥 Export Logs to CSV",
                        data=filtered_logs.to_csv(index=False).encode('utf-8'),
                        file_name='safety_logs_export.csv',
                        mime='text/csv'
                    )
                except pd.errors.EmptyDataError:
                    st.error("The log file is empty or corrupted")
                except Exception as e:
                    st.error(f"Error reading log file: {str(e)}")
            else:
                st.info("No safety logs recorded yet")
                
        except Exception as e:
            st.error(f"Error loading safety logs: {str(e)}")

    def show_settings(self):
        """Show system settings (admin only)"""
        if not st.session_state.is_admin:
            st.warning("You must be an administrator to access system settings")
            return
        
        st.title("⚙️ System Settings")
        
        with st.expander("Database Settings"):
            st.subheader("Database Management")
            
            current_db = st.text_input("Current Database", value=self.system.database_name)
            
            if st.button("Update Database"):
                if current_db and current_db != self.system.database_name:
                    self.system.database_name = current_db
                    self.system.database_folder = current_db
                    self.system.db_path = os.path.join(current_db, f"{current_db}.db")
                    self.system.load_data()
                    st.success("Database updated successfully")
                else:
                    st.warning("Please enter a valid database name")
            
            if st.button("Initialize New Database"):
                new_db = st.text_input("New Database Name", value="track_safe_db_new")
                if st.button("Confirm Create"):
                    try:
                        os.makedirs(new_db, exist_ok=True)
                        self.system.database_name = new_db
                        self.system.database_folder = new_db
                        self.system.db_path = os.path.join(new_db, f"{new_db}.db")
                        self.system.initialize_database()
                        self.system.load_data()
                        st.success(f"New database '{new_db}' created successfully")
                    except Exception as e:
                        st.error(f"Failed to create database: {str(e)}")
        
        with st.expander("Safety Equipment Settings"):
            st.subheader("Required Safety Equipment")
            
            # Allow admin to modify required equipment
            current_equipment = st.text_area(
                "Current Required Equipment (one per line)",
                value="\n".join(self.system.REQUIRED_SAFETY_ITEMS)
            )
            
            if st.button("Update Equipment List"):
                new_equipment = [item.strip() for item in current_equipment.split("\n") if item.strip()]
                self.system.REQUIRED_SAFETY_ITEMS = new_equipment
                st.success("Safety equipment list updated")
        
        with st.expander("Model Settings"):
            st.subheader("PPE Detection Models")
            
            # Model upload
            st.info("Upload custom PPE detection models")
            ppe_model = st.file_uploader("PPE Model (YOLO format)", type=['pt'])
            person_model = st.file_uploader("Person Detection Model (YOLO format)", type=['pt'])
            
            if st.button("Update Models"):
                if ppe_model:
                    with open("custom_ppe_model.pt", "wb") as f:
                        f.write(ppe_model.getbuffer())
                    self.system.ppe_model = None  # Force reload
                
                if person_model:
                    with open("custom_person_model.pt", "wb") as f:
                        f.write(person_model.getbuffer())
                    self.system.person_model = None  # Force reload
                
                st.success("Models updated. They will be loaded on next monitoring session")

if __name__ == "__main__":
    app = TrackSafeWebApp()
    app.show_auth_pages()