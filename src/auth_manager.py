import os
import sqlite3
import datetime
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "algi_auth.db")

def get_db_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes SQLite schema for users, isolated farm parameters, and system audit trails.
    Pre-seeds default admin and sample farm operator.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
        farm_name TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    # Isolated Farm Parameters per User
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_farm_data (
        username TEXT PRIMARY KEY,
        temperature REAL,
        ph REAL,
        co2 REAL,
        light REAL,
        nitrate REAL,
        iron REAL,
        phosphate REAL,
        ammonia REAL,
        do REAL,
        turbidity REAL,
        manganese REAL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (username) REFERENCES users(username)
    )
    """)

    # Seed Admin if not present
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        admin_hash = generate_password_hash("Admin@Algi2026")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, farm_name, created_at) VALUES (?, ?, ?, ?, ?)",
            ("admin", admin_hash, "admin", "Central Command & Validation Core", now)
        )

    # Seed Sample Farm Operator if not present
    cursor.execute("SELECT id FROM users WHERE username = 'operator_alpha'")
    if not cursor.fetchone():
        user_hash = generate_password_hash("Farm@1234")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, farm_name, created_at) VALUES (?, ?, ?, ?, ?)",
            ("operator_alpha", user_hash, "user", "Alpha Spirulina Bioreactor Station", now)
        )
        # Pre-seed default parameters for operator_alpha
        cursor.execute("""
            INSERT OR REPLACE INTO user_farm_data 
            (username, temperature, ph, co2, light, nitrate, iron, phosphate, ammonia, do, turbidity, manganese, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "operator_alpha", 26.5, 7.8, 550.0, 5200.0, 18.0, 0.45, 1.5, 0.02, 7.6, 14.5, 0.05, now
        ))

    conn.commit()
    conn.close()

def verify_login(username, password):
    """
    Verifies user credentials.
    Returns (True, user_dict) or (False, error_message).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return False, "User ID not found in system."

    if not check_password_hash(user["password_hash"], password):
        return False, "Invalid password. Access denied."

    user_dict = {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "farm_name": user["farm_name"],
        "created_at": user["created_at"]
    }
    return True, user_dict

def create_user(username, password, role="user", farm_name="Commercial Algae Farm"):
    """
    Creates a new user with hashed password. Admin only.
    """
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        pw_hash = generate_password_hash(password)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, farm_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, pw_hash, role, farm_name.strip(), now)
        )
        # Initialize default farm parameters for new operator
        cursor.execute("""
            INSERT OR REPLACE INTO user_farm_data 
            (username, temperature, ph, co2, light, nitrate, iron, phosphate, ammonia, do, turbidity, manganese, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username, 26.0, 7.5, 450.0, 5000.0, 18.0, 0.45, 1.5, 0.02, 7.5, 15.0, 0.05, now
        ))
        conn.commit()
        return True, f"User '{username}' provisioned successfully."
    except sqlite3.IntegrityError:
        return False, f"User ID '{username}' already exists."
    except Exception as e:
        return False, f"Database error: {str(e)}"
    finally:
        conn.close()

def change_user_password(username, old_password, new_password, confirm_password):
    """
    Allows a user to change their password with double verification.
    """
    if new_password != confirm_password:
        return False, "New passwords do not match."
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters long."

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return False, "User not found."

    if not check_password_hash(user["password_hash"], old_password):
        conn.close()
        return False, "Current password verification failed."

    new_hash = generate_password_hash(new_password)
    cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, username))
    conn.commit()
    conn.close()
    return True, "Password changed successfully."

def list_all_users():
    """
    Returns list of all provisioned users for admin inspection.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, farm_name, created_at FROM users ORDER BY id ASC")
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users

def save_user_farm_params(username, metrics):
    """
    Saves isolated farm parameters for a specific user into SQLite.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT OR REPLACE INTO user_farm_data 
        (username, temperature, ph, co2, light, nitrate, iron, phosphate, ammonia, do, turbidity, manganese, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        float(metrics.get("Temperature", 26.0)),
        float(metrics.get("pH", 7.5)),
        float(metrics.get("CO2", 450.0)),
        float(metrics.get("Light", 5000.0)),
        float(metrics.get("Nitrate", 18.0)),
        float(metrics.get("Iron", 0.45)),
        float(metrics.get("Phosphate", 1.5)),
        float(metrics.get("Ammonia", 0.02)),
        float(metrics.get("DO", 7.5)),
        float(metrics.get("Turbidity", 15.0)),
        float(metrics.get("Manganese", 0.05)),
        now
    ))
    conn.commit()
    conn.close()

def load_user_farm_params(username):
    """
    Loads isolated farm parameters for a specific user from SQLite.
    Falls back to optimal defaults if none saved yet.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_farm_data WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "Temperature": float(row["temperature"]),
            "pH": float(row["ph"]),
            "CO2": float(row["co2"]),
            "Light": float(row["light"]),
            "Nitrate": float(row["nitrate"]),
            "Iron": float(row["iron"]),
            "Phosphate": float(row["phosphate"]),
            "Ammonia": float(row["ammonia"]),
            "DO": float(row["do"]),
            "Turbidity": float(row["turbidity"]),
            "Manganese": float(row["manganese"])
        }
    return {
        "Temperature": 26.0,
        "pH": 7.5,
        "CO2": 450.0,
        "Light": 5000.0,
        "Nitrate": 18.0,
        "Iron": 0.45,
        "Phosphate": 1.5,
        "Ammonia": 0.02,
        "DO": 7.5,
        "Turbidity": 15.0,
        "Manganese": 0.05
    }
