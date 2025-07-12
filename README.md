
# 🛡️ TrackSafe – AI-Powered Worker Safety Monitoring System

TrackSafe is a real-time, AI-powered safety monitoring system designed for factories and industrial environments. It ensures that workers wear essential Personal Protective Equipment (PPE) by combining **YOLOv8 detection** and **facial recognition**, logging violations, and alerting in real-time – with or without a web dashboard.

---

## 🚀 Project Overview

TrackSafe monitors factory workers using a camera-based system, identifying each worker via face recognition and checking if they’re wearing all required PPE items (Helmet, Vest, Gloves, Glasses, etc.).

Key features:

- ✅ Real-time PPE detection using YOLOv8
- ✅ Face recognition to identify each worker
- ✅ Logs each violation to both **CSV** and **SQLite**
- ✅ Alerts via **buzzer sound** (Windows)
- ✅ Interactive **Streamlit Web Dashboard** *(optional)*
- ✅ Offline system – runs without internet
- ✅ Fully built in **Python**

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `Track_Safe_System` | CLI-based version of the system (no web UI) |
| `Track_Safe_System_Web_interface` | Web-based Streamlit version of the system |

---

## 🧠 Core Technologies

- **YOLOv8** – PPE & person detection
- **face_recognition** + OpenCV – Worker identification
- **SQLite3** – Persistent database storage
- **CSV** – Logging violations
- **Streamlit** – Web dashboard
- **winsound** – Alerts (Windows only)
- **cvzone, PIL, pandas, numpy, etc.** – Support libraries

---

## ⚙️ Installation & Requirements

1. **Clone the repo**

```bash
git clone https://github.com/Mirinda-web-ai/Track_Safe.git
cd Track_Safe
```

2. **Create a virtual environment (optional)**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure to manually download and include:
> - Your **YOLO PPE model** (`Track_Safe_Model`)
> - The **YOLO person model** (`yolov8n.pt`)

Put them in the root folder or adjust the paths (see below).

---

## 🧪 How to Run

### 🖥️ CLI Version

> File: `Track_Safe_System`

```bash
python Track_Safe_System.py
```

You’ll be prompted to:
- Enter a database name (used for saving worker info and logs)
- Register new workers
- Start monitoring directly from terminal

### 🌐 Web Dashboard (Streamlit)

> File: `Track_Safe_System_Web_interface`

```bash
streamlit run Track_Safe_System_Web_interface
```

From the browser, you’ll be able to:
- Log in / Sign up
- Register workers (via camera input)
- Start monitoring
- Browse safety logs and worker data

---

## 📌 Changing Model Paths

### 1. **For CLI version** – `Track_Safe_System`

Search for:

```python
self.ppe_model = YOLO("best (1).pt")
self.person_model = YOLO("yolov8n.pt")
```

To change the model path, modify **Line 690** inside the method `initialize_ppe_models()`.

---

### 2. **For Web version** – `Track_Safe_System_Web_interface`

Search for:

```python
self.ppe_model_path = "the main model.pt"
self.person_model_path = "yolov8n.pt"
```

To change the model path, modify **Line 222** inside the `EnhancedTrackSafeSystem` class constructor.

---

## 💡 Notes

- The system supports **automatic registration prevention** via face verification.
- Alerts are only triggered if required PPE items are **not worn**, avoiding noise spam.
- Workers' data is stored with their **face encodings** and job information.
- Compatible with **Windows** (for buzzer alert via `winsound`). For other OS, replace or disable alert mechanism.

---

## 📊 Logs & Database

- **CSV Logs**: All violations are saved to `track_safe_log.csv`.
- **SQLite Database**: Worker data is stored under the chosen database name (`<your_db_name>.db`).

---

## 🔒 Web Version Security

- Includes **user authentication system** (login/signup)
- Admin panel access
- Profile image management
- Each user has their own database storage (`user_databases/`)

---

## 📸 Sample Screenshot

Track_Safe > Project_Demo 

---

## 🧑‍💻 Developed By

Developed by the TrackSafe Capstone Team – Mobica IATS (2025)

- Engineer Malk Younes – Capstone Supervisor
- Yassin Ahmed – Team Member
- Youssef Ahmed – Team Member
- Yassin Badr – Team Member
- Mariam Khaled – Team Member
- Khadija Haitham – Team Member


---

## 📬 Contact

For feedback, contributions, or collaboration:  
📧 [yassinahmed20202009@gmail.com]  
📧 [malkyounes71@gmail.com]

---
