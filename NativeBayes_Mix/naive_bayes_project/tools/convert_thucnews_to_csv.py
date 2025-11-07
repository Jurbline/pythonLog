# -*- coding: utf-8 -*-
import os, csv, argparse, sys, re

def norm_text(s, strip_newlines=True, collapse_spaces=True, max_chars=None):
    s = s.replace('\ufeff','').replace('\x00',' ')
    if strip_newlines:
        s = s.replace('\r', ' ').replace('\n', ' ')
    if collapse_spaces:
        s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    if max_chars and len(s) > max_chars:
        s = s[:max_chars]
    return s

def iter_category_files(root, exts=('.txt',), recursive=True):
    for cat in sorted(os.listdir(root)):
        cat_path = os.path.join(root, cat)
        if not os.path.isdir(cat_path):
            continue
        if recursive:
            for dirpath, _, files in os.walk(cat_path):
                for fn in files:
                    if exts and not fn.lower().endswith(exts):
                        continue
                    yield cat, os.path.join(dirpath, fn)
        else:
            for fn in os.listdir(cat_path):
                if exts and not fn.lower().endswith(exts):
                    continue
                yield cat, os.path.join(cat_path, fn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--classes', type=str, default=None)
    ap.add_argument('--limit_per_class', type=int, default=None)
    ap.add_argument('--min_chars', type=int, default=2)
    ap.add_argument('--max_chars', type=int, default=None)
    ap.add_argument('--encoding', type=str, default='utf-8')
    ap.add_argument('--exts', type=str, default='.txt')
    args = ap.parse_args()

    allow = None
    if args.classes:
        allow = set([c.strip() for c in args.classes.split(',') if c.strip()])
    exts = tuple([e.strip().lower() for e in args.exts.split(',') if e.strip()])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    written = 0
    per_class = {}

    with open(args.out, 'w', encoding='utf-8', newline='') as fout:
        w = csv.writer(fout)
        w.writerow(['label', 'text'])

        for lab, fpath in iter_category_files(args.root, exts=exts, recursive=True):
            if allow and lab not in allow:
                continue
            cnt = per_class.get(lab, 0)
            if args.limit_per_class and cnt >= args.limit_per_class:
                continue
            data = None
            for enc in ('utf-8', 'utf-8-sig', args.encoding, 'gb18030', 'latin-1'):
                try:
                    with open(fpath, 'r', encoding=enc, errors='strict') as f:
                        data = f.read()
                    break
                except Exception:
                    data = None
                    continue
            if data is None:
                print(f"[WARN] 读取失败: {fpath}", file=sys.stderr)
                continue
            text = norm_text(data, max_chars=args.max_chars)
            if len(text) < args.min_chars:
                continue
            w.writerow([lab, text])
            per_class[lab] = cnt + 1
            written += 1

    sys.stdout.write(f"完成：{args.out}  共 {written} 行；类别统计：{per_class}\n")

if __name__ == '__main__':
    main()
