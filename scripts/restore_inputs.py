"""Restore the archived derived inputs and verify every SHA-256 checksum."""
from pathlib import Path
import argparse, hashlib, json, shutil, zipfile

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('download_dir',type=Path)
    ap.add_argument('--target',type=Path,default=ROOT)
    ap.add_argument('--link-large',action='store_true',help='Optional local verification: hard-link immutable h5ad files on the same filesystem.')
    a=ap.parse_args(); target=a.target.resolve();target.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.download_dir/'data_manifest.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(a.download_dir/'analysis_inputs_and_source_tables.zip') as z:
        for name in z.namelist():
            dest=(target/name).resolve()
            if not dest.is_relative_to(target):raise ValueError('Unsafe archive member: '+name)
        z.extractall(target)
    for row in manifest['files']:
        dest=(target/row['path']).resolve()
        if not dest.is_relative_to(target):raise ValueError('Unsafe manifest path')
        if dest.suffix=='.h5ad':
            src=a.download_dir/dest.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if sha(dest)!=row['sha256']:raise ValueError('Existing different input: '+str(dest))
            elif a.link_large:dest.hardlink_to(src.resolve())
            else:shutil.copy2(src,dest)
        if not dest.exists() or dest.stat().st_size!=row['bytes'] or sha(dest)!=row['sha256']:
            raise ValueError('Checksum/size mismatch: '+str(dest))
    print('INPUT RESTORATION PASSED:',len(manifest['files']),'verified files',flush=True)
if __name__=='__main__':main()
