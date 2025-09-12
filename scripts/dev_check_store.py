from tonrag.vectorstore import ChromaStore
from tonrag.config import settings
print('DIR:', settings.chroma_dir, 'COL:', settings.collection_name)
store = ChromaStore()
res = store.query_text('vai trò của cô đỡ thôn bản', top_k=3)
print('len:', len(res))
for i,h in enumerate(res,1):
    print(i, h['id'], (h.get('distance')))
    print((h['document'] or '')[:100].replace('\n',' '))
