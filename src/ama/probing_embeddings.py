"""Offline native ama eval helper using a pinned local encoder; no network."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input');parser.add_argument('output');parser.add_argument('--encoder',required=True)
    args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    from sentence_transformers import SentenceTransformer
    source=json.loads(Path(args.input).read_text())
    texts=sorted({t for group in source for t in group if isinstance(t,str) and t.strip()})
    model=SentenceTransformer(args.encoder,device='cpu',local_files_only=True)
    embeddings=model.encode(texts,normalize_embeddings=True,show_progress_bar=False,batch_size=64)
    indexed=dict(zip(texts,embeddings))
    scores=[]
    for group in source:
        if len(group)!=20 or any(not isinstance(t,str) or not t.strip() for t in group):
            scores.append(None);continue
        values=np.array([indexed[t] for t in group]);matrix=values@values.T
        scores.append(float(np.clip(np.mean(matrix[np.triu_indices(20,k=1)]),0,1)))
    Path(args.output).write_text(json.dumps({'scores':scores,'unique_texts':len(texts),
        'encoder':args.encoder,'method':'mean upper-triangle cosine, normalized embeddings; clipped [0,1]'})+'\n')


if __name__=='__main__':main()
