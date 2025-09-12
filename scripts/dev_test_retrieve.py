from tonrag.rag import RAGPipeline

if __name__ == "__main__":
    r = RAGPipeline()
    res = r.retrieve('vai trò của cô đỡ thôn bản', top_k=3)
    print('Retrieved', len(res))
    for i,h in enumerate(res,1):
        print(i, h['id'], h.get('distance'))
        print((h['document'] or '')[:200].replace('\n',' '))
        print('---')

