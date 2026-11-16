import os
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from app import accounts
from src.features import load_data


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "accounts.sqlite3"
        accounts.initialize_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_customer_registration_and_authentication(self):
        account = accounts.register_user(
            "shopper@example.com", "Shopper", "a-long-test-password", "customer",
            db_path=self.db_path, age=35, gender="Female", region="West",
        )
        self.assertTrue(account["customer_id"].startswith("LOCAL-"))
        self.assertEqual(
            accounts.authenticate("SHOPPER@example.com", "a-long-test-password", self.db_path),
            account,
        )
        self.assertIsNone(accounts.authenticate("shopper@example.com", "wrong-password", self.db_path))

    def test_orders_are_saved_and_scoped_to_customer(self):
        account = accounts.register_user(
            "shopper@example.com", "Shopper", "a-long-test-password", "customer",
            db_path=self.db_path, age=35, gender="Female", region="West",
        )
        order_id = accounts.create_order(
            account["customer_id"], "Canvas bag", "Fashion", 20, 2, .1,
            "Credit Card", date(2026, 11, 16), self.db_path,
        )
        orders = accounts.get_orders(account["customer_id"], self.db_path)
        self.assertEqual(orders[0]["order_id"], order_id)
        self.assertEqual(orders[0]["total_amount"], 36)
        self.assertEqual(accounts.get_orders("another-customer", self.db_path), [])
        accounts.mark_order_returned(account["customer_id"], order_id, self.db_path)
        self.assertEqual(accounts.get_orders(account["customer_id"], self.db_path)[0]["returned"], 1)
        self.assertEqual(accounts.get_training_orders(self.db_path)[0]["returned"], "Yes")
        self.assertEqual(accounts.get_admin_orders(self.db_path)[0]["order_id"], order_id)

    def test_live_orders_load_with_model_feature_columns(self):
        order = {
            "order_id": "ORD-LIVE-1", "customer_id": "LOCAL-1", "product_id": "LOCAL-P-1",
            "category": "Fashion", "price": 25.0, "discount": 0.1, "quantity": 2,
            "payment_method": "Wallet", "order_date": "2026-11-16", "delivery_time_days": 5,
            "region": "West", "returned": "Yes", "total_amount": 45.0, "shipping_cost": 6.0,
            "profit_margin": 4.5, "customer_age": 35, "customer_gender": "Female",
        }
        with patch("app.accounts.get_training_orders", return_value=[order]):
            data = load_data(include_live=True)
        local = data[data.customer_id == "LOCAL-1"].iloc[0]
        self.assertEqual(local.price, 25.0)
        self.assertEqual(local.is_returned, 1)
        self.assertEqual(str(local.order_date.date()), "2026-11-16")

    def test_retraining_waits_for_threshold_and_claims_only_once(self):
        self.assertFalse(accounts.claim_retraining(99, minimum_new_records=100, db_path=self.db_path))
        self.assertTrue(accounts.claim_retraining(100, minimum_new_records=100, db_path=self.db_path))
        self.assertFalse(accounts.claim_retraining(200, minimum_new_records=100, db_path=self.db_path))

    def test_return_labels_become_trainable_when_orders_mature(self):
        account = accounts.register_user(
            "shopper@example.com", "Shopper", "a-long-test-password", "customer",
            db_path=self.db_path, age=35, gender="Female", region="West",
        )
        order_id = accounts.create_order(
            account["customer_id"], "Wool scarf", "Fashion", 20, 1, 0,
            "Credit Card", date.today(), self.db_path,
        )
        accounts.mark_order_returned(account["customer_id"], order_id, self.db_path)
        self.assertEqual(accounts.get_matured_return_count(self.db_path), 0)
        with accounts._connect(self.db_path) as conn:
            conn.execute(
                "UPDATE customer_orders SET order_date = ? WHERE order_id = ?",
                (date(2025, 1, 1).isoformat(), order_id),
            )
        self.assertEqual(accounts.get_matured_return_count(self.db_path), 1)
        self.assertFalse(accounts.claim_retraining(
            0, matured_return_count=99, minimum_new_records=100, db_path=self.db_path
        ))
        self.assertTrue(accounts.claim_retraining(
            0, matured_return_count=100, minimum_new_records=100, db_path=self.db_path
        ))

    def test_admin_registration_requires_an_invite_code(self):
        with patch.dict(os.environ, {"NEXUS_ADMIN_INVITE_CODE": ""}):
            with self.assertRaisesRegex(ValueError, "invite code"):
                accounts.register_user(
                    "admin@example.com", "Admin", "a-long-test-password", "admin",
                    "anything", self.db_path,
                )
        with patch.dict(os.environ, {"NEXUS_ADMIN_INVITE_CODE": "private-code"}):
            account = accounts.register_user(
                "admin@example.com", "Admin", "a-long-test-password", "admin",
                "private-code", self.db_path,
            )
        self.assertEqual(account["role"], "admin")
        self.assertIsNone(account["customer_id"])

    def test_existing_local_order_database_migrates_in_place(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        conn = sqlite3.connect(legacy_path)
        try:
            conn.executescript("""
                CREATE TABLE users (
                    email TEXT PRIMARY KEY, display_name TEXT NOT NULL, role TEXT NOT NULL,
                    customer_id TEXT, salt BLOB NOT NULL, password_hash BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE customer_orders (
                    order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, product_name TEXT NOT NULL,
                    category TEXT NOT NULL, unit_price REAL NOT NULL, quantity INTEGER NOT NULL,
                    discount REAL NOT NULL, payment_method TEXT NOT NULL, order_date TEXT NOT NULL,
                    total_amount REAL NOT NULL
                );
            """)
        finally:
            conn.close()
        accounts.initialize_db(legacy_path)
        conn = sqlite3.connect(legacy_path)
        try:
            user_columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
            order_columns = {row[1] for row in conn.execute("PRAGMA table_info(customer_orders)")}
        finally:
            conn.close()
        self.assertIn("age", user_columns)
        self.assertIn("returned", order_columns)
        self.assertIn("product_id", order_columns)


if __name__ == "__main__":
    unittest.main()
