from RAG_Bot.txt.txt_rag import retrieve_documents

results, section = retrieve_documents('ipc section 11', k=5)

print('section', section)
print('results', len(results))
print('---')

for i,(d,s) in enumerate(results):
    print(i, 'score', s, 'title', d.metadata.get('title'), 'section', d.metadata.get('section_number'), 'article', d.metadata.get('article_number'))
    print(d.page_content[:180].replace('\n',' '))
