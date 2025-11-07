# -*- coding: utf-8 -*-
import os, csv, argparse, json, re

def load_jsonl(path, encoding='utf-8'):
    data = []
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data

def norm_text(s):
    if s is None:
        return ""
    s = s.replace('\ufeff','').replace('\x00',' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def rows_from_records(recs, label_field='label_desc', text_field='sentence', allow_empty_label=False):
    rows = []
    for r in recs:
        txt = norm_text(r.get(text_field, ""))
        lab = r.get(label_field, None)
        if not allow_empty_label and (lab is None or lab == ""):
            continue
        if allow_empty_label and (lab is None or lab == ""):
            lab = "unknown"
        rows.append((str(lab), txt))
    return rows

def write_csv(rows, out_csv, dedup=True):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    seen = set()
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label','text'])
        for lab, txt in rows:
            key = (lab, txt)
            if dedup and key in seen:
                continue
            seen.add(key)
            w.writerow([lab, txt])
    return out_csv, len(seen)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--out_dir', type=str, default=None)
    ap.add_argument('--label_field', type=str, default='label_desc')
    ap.add_argument('--text_field', type=str, default='sentence')
    ap.add_argument('--include_test', action='store_true')
    args = ap.parse_args()

    train_p = os.path.join(args.dir, 'train.json')
    dev_p   = os.path.join(args.dir, 'dev.json')
    test_p  = os.path.join(args.dir, 'test.json')

    if args.out:
        rows = []
        if os.path.exists(train_p):
            rows += rows_from_records(load_jsonl(train_p), args.label_field, args.text_field, allow_empty_label=False)
        if os.path.exists(dev_p):
            rows += rows_from_records(load_jsonl(dev_p), args.label_field, args.text_field, allow_empty_label=False)
        if args.include_test and os.path.exists(test_p):
            rows += rows_from_records(load_jsonl(test_p), args.label_field, args.text_field, allow_empty_label=True)
        out_csv, n = write_csv(rows, args.out, dedup=True)
        print(f"已生成：{out_csv}  共 {n} 行")
        return

    if not args.out_dir:
        print("请指定 --out 或 --out_dir 之一"); return

    if os.path.exists(train_p):
        rows = rows_from_records(load_jsonl(train_p), args.label_field, args.text_field, allow_empty_label=False)
        out_csv, n = write_csv(rows, os.path.join(args.out_dir, 'train.csv'), dedup=True)
        print(f"已生成：{out_csv}  共 {n} 行")
    if os.path.exists(dev_p):
        rows = rows_from_records(load_jsonl(dev_p), args.label_field, args.text_field, allow_empty_label=False)
        out_csv, n = write_csv(rows, os.path.join(args.out_dir, 'dev.csv'), dedup=True)
        print(f"已生成：{out_csv}  共 {n} 行")
    if args.include_test and os.path.exists(test_p):
        rows = rows_from_records(load_jsonl(test_p), args.label_field, args.text_field, allow_empty_label=True)
        out_csv, n = write_csv(rows, os.path.join(args.out_dir, 'test.csv'), dedup=True)
        print(f"已生成：{out_csv}  共 {n} 行（无标签以 unknown 占位）")

if __name__ == '__main__':
    main()
