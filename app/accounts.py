"""Local account and customer-order storage for the Streamlit prototype."""
import hashlib
import hmac
import os
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from src import config as C

DB_PATH = Path(os.environ.get("NEXUS_APP_DB", C.OUT_DIR / "app_data.sqlite3"))
PASSWORD_ITERATIONS = 310_000


@contextmanager
def _connect(db_path=DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_db(db_path=DB_PATH):
    with _connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('customer', 'admin')),
                customer_id TEXT,
                salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                age INTEGER,
                gender TEXT,
                region TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS customer_orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                product_id TEXT NOT NULL,
                category TEXT NOT NULL,
                unit_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                discount REAL NOT NULL,
                payment_method TEXT NOT NULL,
                order_date TEXT NOT NULL,
                total_amount REAL NOT NULL,
                delivery_time_days INTEGER NOT NULL DEFAULT 5,
                shipping_cost REAL NOT NULL DEFAULT 6,
                profit_margin REAL NOT NULL DEFAULT 0,
                customer_age INTEGER NOT NULL DEFAULT 35,
                customer_gender TEXT NOT NULL DEFAULT 'Other',
                region TEXT NOT NULL DEFAULT 'Central',
                returned INTEGER NOT NULL DEFAULT 0,
                return_date TEXT
            );
            CREATE INDEX IF NOT EXISTS customer_orders_customer_date
                ON customer_orders(customer_id, order_date DESC);
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        _ensure_columns(conn, "users", {
            "age": "INTEGER", "gender": "TEXT", "region": "TEXT",
        })
        _ensure_columns(conn, "customer_orders", {
            "product_id": "TEXT NOT NULL DEFAULT ''",
            "delivery_time_days": "INTEGER NOT NULL DEFAULT 5",
            "shipping_cost": "REAL NOT NULL DEFAULT 6",
            "profit_margin": "REAL NOT NULL DEFAULT 0",
            "customer_age": "INTEGER NOT NULL DEFAULT 35",
            "customer_gender": "TEXT NOT NULL DEFAULT 'Other'",
            "region": "TEXT NOT NULL DEFAULT 'Central'",
            "returned": "INTEGER NOT NULL DEFAULT 0",
            "return_date": "TEXT",
        })
        conn.execute(
            "INSERT OR IGNORE INTO app_state (key, value) VALUES ('last_matured_count', '0')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO app_state (key, value) VALUES ('last_attempt_count', '0')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO app_state (key, value) VALUES ('training_status', 'waiting')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO app_state (key, value) VALUES ('data_version', '0')"
        )


def _ensure_columns(conn, table, columns):
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for column, definition in columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _password_hash(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )


def register_user(email, display_name, password, role, admin_code="", db_path=DB_PATH,
                  age=None, gender=None, region=None):
    email = email.strip().lower()
    display_name = display_name.strip()
    if "@" not in email or not display_name:
        raise ValueError("Enter a valid email address and your name.")
    if len(password) < 10:
        raise ValueError("Use a password with at least 10 characters.")
    if role not in ("customer", "admin"):
        raise ValueError("Choose a valid account type.")
    if role == "customer" and (
        age is None or age < 18 or age > 100 or gender not in C.GENDERS or region not in C.REGIONS
    ):
        raise ValueError("Enter a valid age, gender, and region.")
    if role == "admin":
        expected_code = os.environ.get("NEXUS_ADMIN_INVITE_CODE", "")
        if not expected_code or not hmac.compare_digest(admin_code, expected_code):
            raise ValueError("Admin registration requires the configured admin invite code.")

    salt = secrets.token_bytes(16)
    customer_id = f"LOCAL-{uuid.uuid4().hex[:12].upper()}" if role == "customer" else None
    try:
        with _connect(db_path) as conn:
            conn.execute(
                """INSERT INTO users
                   (email, display_name, role, customer_id, salt, password_hash, age, gender, region)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (email, display_name, role, customer_id, salt, _password_hash(password, salt),
                 age, gender, region),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("An account with this email already exists.") from exc
    return {
        "email": email, "display_name": display_name, "role": role, "customer_id": customer_id,
        "age": age, "gender": gender, "region": region,
    }


def authenticate(email, password, db_path=DB_PATH):
    with _connect(db_path) as conn:
        user = conn.execute(
            """SELECT email, display_name, role, customer_id, age, gender, region, salt, password_hash
               FROM users WHERE email = ?""",
            (email.strip().lower(),),
        ).fetchone()
    if user is None or not hmac.compare_digest(_password_hash(password, user["salt"]), user["password_hash"]):
        return None
    return {key: user[key] for key in (
        "email", "display_name", "role", "customer_id", "age", "gender", "region"
    )}


def create_order(customer_id, product_name, category, unit_price, quantity, discount,
                 payment_method, order_date, db_path=DB_PATH, delivery_time_days=5,
                 shipping_cost=6.0, profit_margin=None, age=35, gender="Other", region="Central"):
    name = product_name.strip()
    if (
        not name or category not in C.CATEGORIES or payment_method not in C.PAYMENTS
        or unit_price <= 0 or quantity < 1 or not 0 <= discount <= 1
        or delivery_time_days < 1 or shipping_cost < 0
    ):
        raise ValueError("Enter a product name, positive price and quantity, and a valid discount.")
    order_id = f"ORD-{uuid.uuid4().hex[:12].upper()}"
    total = round(unit_price * quantity * (1 - discount), 2)
    estimated_profit = total * 0.1 if profit_margin is None else profit_margin
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO customer_orders
               (order_id, customer_id, product_name, product_id, category, unit_price, quantity, discount,
                payment_method, order_date, total_amount, delivery_time_days, shipping_cost,
                profit_margin, customer_age, customer_gender, region)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_id, customer_id, name, f"LOCAL-P-{uuid.uuid4().hex[:10].upper()}",
             category, unit_price, quantity, discount, payment_method, order_date.isoformat(),
             total, delivery_time_days, shipping_cost, estimated_profit, age, gender, region),
        )
        _bump_data_version(conn)
    return order_id


def _bump_data_version(conn):
    conn.execute(
        """INSERT INTO app_state (key, value) VALUES ('data_version', '1')
           ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"""
    )


def mark_order_returned(customer_id, order_id, db_path=DB_PATH):
    with _connect(db_path) as conn:
        result = conn.execute(
            """UPDATE customer_orders SET returned = 1, return_date = date('now')
               WHERE customer_id = ? AND order_id = ? AND returned = 0""",
            (customer_id, order_id),
        )
        if result.rowcount != 1:
            raise ValueError("Order was not found or has already been returned.")
        _bump_data_version(conn)


def get_orders(customer_id, db_path=DB_PATH):
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT order_id, product_name, category, unit_price, quantity, discount,
                      payment_method, order_date, total_amount, returned, return_date
               FROM customer_orders WHERE customer_id = ? ORDER BY order_date DESC, order_id DESC""",
            (customer_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_admin_orders(db_path=DB_PATH):
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT o.order_id, o.customer_id, u.display_name, u.email, o.product_name,
                      o.category, o.unit_price, o.quantity, o.discount, o.payment_method,
                      o.order_date, o.total_amount, o.returned, o.return_date
               FROM customer_orders o JOIN users u ON u.customer_id = o.customer_id
               ORDER BY o.order_date DESC, o.order_id DESC"""
        ).fetchall()
    return [dict(row) for row in rows]


def get_training_orders(db_path=DB_PATH):
    initialize_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT order_id, customer_id, product_id, category, unit_price AS price, discount,
                      quantity, payment_method, order_date, delivery_time_days, region,
                      CASE WHEN returned = 1 THEN 'Yes' ELSE 'No' END AS returned,
                      total_amount, shipping_cost, profit_margin, customer_age, customer_gender
               FROM customer_orders ORDER BY customer_id, order_date, order_id"""
        ).fetchall()
    return [dict(row) for row in rows]


def get_matured_order_count(db_path=DB_PATH):
    with _connect(db_path) as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n FROM customer_orders
               WHERE date(order_date) <= date('now', '-180 days')"""
        ).fetchone()
    return int(row["n"])


def get_matured_return_count(db_path=DB_PATH):
    with _connect(db_path) as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n FROM customer_orders
               WHERE returned = 1 AND date(order_date) <= date('now', '-180 days')"""
        ).fetchone()
    return int(row["n"])


def claim_retraining(matured_count, matured_return_count=0, minimum_new_records=100, db_path=DB_PATH):
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        state = {row["key"]: row["value"] for row in conn.execute(
            "SELECT key, value FROM app_state WHERE key IN "
            "('last_matured_count', 'last_attempt_count', 'last_return_event_count', "
            "'last_attempt_return_count', 'training_status')"
        )}
        status = state.get("training_status", "waiting")
        baseline = max(
            int(state.get("last_matured_count", "0")),
            int(state.get("last_attempt_count", "0")),
        )
        return_baseline = max(
            int(state.get("last_return_event_count", "0")),
            int(state.get("last_attempt_return_count", "0")),
        )
        enough_new_orders = matured_count - baseline >= minimum_new_records
        enough_new_returns = matured_return_count - return_baseline >= minimum_new_records
        if status in ("queued", "running") or not (enough_new_orders or enough_new_returns):
            return False
        conn.execute(
            """INSERT INTO app_state (key, value) VALUES ('training_status', 'queued')
               ON CONFLICT(key) DO UPDATE SET value = excluded.value"""
        )
        conn.execute(
            """INSERT INTO app_state (key, value) VALUES ('last_attempt_count', ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (str(matured_count),),
        )
        conn.execute(
            """INSERT INTO app_state (key, value) VALUES ('last_attempt_return_count', ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (str(matured_return_count),),
        )
        conn.execute(
            """INSERT INTO app_state (key, value) VALUES ('training_error', '')
               ON CONFLICT(key) DO UPDATE SET value = excluded.value"""
        )
    return True


def update_state(values, db_path=DB_PATH):
    with _connect(db_path) as conn:
        conn.executemany(
            """INSERT INTO app_state (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            [(key, str(value)) for key, value in values.items()],
        )


def get_state(key, db_path=DB_PATH, default=""):
    with _connect(db_path) as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else default


def set_state(key, value, db_path=DB_PATH):
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
