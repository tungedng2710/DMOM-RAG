#!/usr/bin/env python3
import csv
from pathlib import Path


def transform(src_path: str = 'data/dmom_data.csv', dst_path: str = 'data/dmom_data_context.csv') -> int:
    src = Path(src_path)
    dst = Path(dst_path)

    rows_out = []
    with src.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            no = (row.get('no') or '').strip()
            q = (row.get('input') or '').strip()
            out = (row.get('output') or '').strip()
            manual = (row.get('Manually review') or '').strip()
            ref = (row.get('Reference') or '').strip()

            answer = manual if manual else out
            context = f"question: {q}\nanswer: {answer}\nreference: {ref}"

            rows_out.append({'no': no, 'context': context})

    with dst.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['no', 'context'])
        writer.writeheader()
        writer.writerows(rows_out)

    return len(rows_out)


if __name__ == '__main__':
    n = transform()
    print(f"Wrote {n} rows to data/dmom_data_context.csv")

