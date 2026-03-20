import csv
import os
from datetime import datetime


class PlateStorage:
    def __init__(self, csv_path="data/plates.csv"):
        self.csv_path = csv_path
        self.recent_saves = {}

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["plate_number", "timestamp", "image_path"])

    def save_plate(self, plate_text: str, image_path: str = ""):
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        last_time = self.recent_saves.get(plate_text)
        if last_time and (now - last_time).total_seconds() < 15:
            return False

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([plate_text, now_str, image_path])

        self.recent_saves[plate_text] = now
        return True