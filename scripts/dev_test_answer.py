from tonrag.rag import RAGPipeline
r = RAGPipeline()
res = r.answer('vai trò của cô đỡ thôn bản', top_k=3)
print('Answer:', (res['answer'] or '')[:300])
print('Contexts:', len(res['contexts']))
print('First ctx:', (res['contexts'][0]['document'] or '')[:120].replace('\n',' '))
