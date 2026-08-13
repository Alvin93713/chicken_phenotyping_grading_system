"""建立範例雞隻資料。"""

from __future__ import annotations

import random

from database import add_chicken, get_chicken, init_db


def create_sample_data(count=20, seed=42):
    """建立 C001 到 C020 的模擬表型資料。"""
    init_db()
    random.seed(seed)
    inserted = 0

    for index in range(1, count + 1):
        chicken_id = f"C{index:03d}"
        if get_chicken(chicken_id):
            continue

        weight_g = round(random.uniform(2400, 4200), 1)
        comb_area_cm2 = None
        shank_width_cm = None
        shank_length_cm = None
        add_chicken(chicken_id, weight_g, comb_area_cm2, shank_width_cm, shank_length_cm)
        inserted += 1

    return inserted


if __name__ == "__main__":
    total = create_sample_data()
    print(f"已建立 {total} 筆範例資料。")
