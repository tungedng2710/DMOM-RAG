import os
import chromadb
path = os.path.abspath('data/chroma_dmom')
client = chromadb.PersistentClient(path=path)
coll = client.get_collection('dmom_qa')
res = coll.query(query_texts=['vai trò của cô đỡ thôn bản'], n_results=3)
print('IDs:', res.get('ids'))
print('Docs snippet:', [ (d or '')[:].replace('\n',' ') for d in (res.get('documents') or [[]])[0] ])
