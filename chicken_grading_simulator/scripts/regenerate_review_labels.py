"""Create missing YOLO-seg labels for reviewed cases captured before labels were saved.

Re-runs the current comb/shank model on each case's original photo with the case's own
confidence threshold (same classes and settings as live inference) and writes the label next to
the result image (<result>.txt), which is where the fine-tuning export looks for it.

Usage:
    python scripts/regenerate_review_labels.py            # only images that have no label yet
    python scripts/regenerate_review_labels.py --dry-run  # show what would be written
    python scripts/regenerate_review_labels.py --force    # also overwrite existing labels

If the model finds nothing on a photo, no label is written: an empty label would mean
"nothing to detect", which is wrong for an image a person reviewed. Such images stay in
to_relabel/ for manual labelling.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import product_review  # noqa: E402
import training_labels  # noqa: E402

REVIEWED = ("正確", "不正確", "無法判斷")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite labels that already exist")
    parser.add_argument("--dry-run", action="store_true", help="list the work without running the model")
    args = parser.parse_args(argv)

    import cv2
    import numpy as np
    import torch
    from dual_camera import model_for, target_classes

    device = 0 if torch.cuda.is_available() else "cpu"
    written = skipped = failed = 0
    for case in product_review.list_cases():
        for role in ("comb", "shank"):
            if case[f"{role}_review_state"] not in REVIEWED:
                continue
            source = case["source_paths"].get(role, "")
            result = case["result_paths"].get(role, "")
            name = f"{case['case_id']} {role:<5}"
            if not result:
                print(f"{name}  skip: no result image recorded")
                skipped += 1
                continue
            label = training_labels.label_path_for(result)
            if label.is_file() and not args.force:
                print(f"{name}  skip: label already exists")
                skipped += 1
                continue
            if not source or not Path(source).is_file():
                print(f"{name}  FAIL: original photo missing ({source})")
                failed += 1
                continue
            threshold = float(case["confidence_threshold"] or 0.25)
            if args.dry_run:
                print(f"{name}  would write {label}")
                continue

            model = model_for(role)
            frame = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
            prediction = model.predict(frame, conf=threshold, classes=target_classes(model, role),
                                       retina_masks=True, device=device, verbose=False)[0]
            if prediction.masks is None or prediction.boxes is None or len(prediction.boxes) == 0:
                print(f"{name}  no detection at conf={threshold:.2f}: left for manual labelling "
                      f"(review: {case[f'{role}_review_state']})")
                failed += 1
                continue
            if training_labels.write_result_labels(prediction, result) is None:
                print(f"{name}  FAIL: could not write {label}")
                failed += 1
                continue
            training_labels.save_class_names(product_review.REVIEW_DIR, role, model.names)
            confidences = [round(float(v), 4) for v in prediction.boxes.conf.detach().cpu().tolist()]
            recorded = case["role_confidences"].get(role)
            print(f"{name}  wrote {label.name}: {len(confidences)} object(s), conf {confidences} "
                  f"(recorded at review: {recorded})")
            written += 1
    print(f"\nwritten {written}, skipped {skipped}, not labelled {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
