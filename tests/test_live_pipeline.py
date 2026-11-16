import unittest

import pandas as pd

from src.pipeline import _order_frames, _training_dates


class LivePipelineTests(unittest.TestCase):
    def test_live_orders_move_temporal_splits_to_latest_mature_window(self):
        df = pd.DataFrame({
            "customer_id": ["LOCAL-1"],
            "order_date": [pd.Timestamp("2026-11-16")],
        })
        train_dates, test_date = _training_dates(df)
        self.assertEqual(test_date, "2026-05-20")
        self.assertEqual(train_dates[-1], "2025-11-21")

    def test_order_masks_separate_training_and_recent_live_orders(self):
        dates = pd.to_datetime([
            "2025-09-01", "2025-10-01", "2026-02-01", "2026-06-01", "2026-11-16",
        ])
        df = pd.DataFrame({
            "customer_id": ["LOCAL-1"] * 5,
            "order_date": dates,
            "category": ["Fashion", "Beauty", "Home", "Grocery", "Sports"],
            "price": [25.0] * 5,
            "quantity": [1] * 5,
            "discount": [0.0] * 5,
            "delivery_time_days": [5] * 5,
            "shipping_cost": [6.0] * 5,
            "total_amount": [25.0] * 5,
            "profit_margin": [3.0] * 5,
            "customer_age": [35] * 5,
            "customer_gender": ["Female"] * 5,
            "region": ["West"] * 5,
            "payment_method": ["Wallet"] * 5,
            "is_returned": [0, 0, 1, 0, 0],
            "is_discounted": [0, 0, 0, 0, 0],
            "gross_amount": [25.0] * 5,
        })
        frames = _order_frames(df)
        ret_train, ret_test = frames["masks"]["ret"]
        self.assertEqual(int(ret_train.sum()), 2)
        self.assertEqual(int(ret_test.sum()), 1)
        self.assertTrue(ret_test[2])
        self.assertFalse(ret_train[3] or ret_test[3])
        self.assertFalse(ret_train[4] or ret_test[4])
        self.assertEqual(frames["splits"]["order_test_start"], "2026-05-20")
        self.assertEqual(int(frames["masks"]["nx"][1].sum()), 1)


if __name__ == "__main__":
    unittest.main()
