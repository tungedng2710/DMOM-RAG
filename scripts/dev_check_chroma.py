import os
import chromadb

path = os.path.abspath('data/chroma_dmom')
client = chromadb.PersistentClient(path=path)
print('Collections:', [c.name for c in client.list_collections()])
coll = client.get_collection('dmom_qa')
print('Count:', coll.count())
try:
    ef = getattr(coll, 'embedding_function', None)
    print('Has EF:', bool(ef))
except Exception as e:
    print('EF check error:', e)
res = coll.query(query_texts=['vai trò của cô đỡ thôn bản'], n_results=3)
print('IDs:', res.get('ids'))
