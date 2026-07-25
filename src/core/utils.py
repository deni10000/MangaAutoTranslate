from typing import List, Tuple, Optional
import numpy as np


def merge_boxes(
    items: List[Tuple[int, int, int, int, Optional[np.ndarray]]]
) -> List[Tuple[int, int, int, int, Optional[np.ndarray]]]:
    if not items:
        return []

    merged = [[item[0], item[1], item[2], item[3], item[4]] for item in items]
    changed = True

    while changed:
        changed = False
        new_merged = []
        skip_indices = set()

        for i in range(len(merged)):
            if i in skip_indices:
                continue

            box_a = merged[i]

            for j in range(i + 1, len(merged)):
                if j in skip_indices:
                    continue

                box_b = merged[j]

                cx_a = (box_a[0] + box_a[2]) / 2.0
                cy_a = (box_a[1] + box_a[3]) / 2.0
                cx_b = (box_b[0] + box_b[2]) / 2.0
                cy_b = (box_b[1] + box_b[3]) / 2.0

                a_in_b = (box_b[0] <= cx_a <= box_b[2]) and (box_b[1] <= cy_a <= box_b[3])
                b_in_a = (box_a[0] <= cx_b <= box_a[2]) and (box_a[1] <= cy_b <= box_a[3])

                if a_in_b or b_in_a:
                    box_a[0] = min(box_a[0], box_b[0])
                    box_a[1] = min(box_a[1], box_b[1])
                    box_a[2] = max(box_a[2], box_b[2])
                    box_a[3] = max(box_a[3], box_b[3])

                    if box_a[4] is not None and box_b[4] is not None:
                        box_a[4] = np.maximum(box_a[4], box_b[4])
                    elif box_a[4] is None:
                        box_a[4] = box_b[4]

                    skip_indices.add(j)
                    changed = True

            new_merged.append(box_a)

        merged = new_merged

    return [(b[0], b[1], b[2], b[3], b[4]) for b in merged]