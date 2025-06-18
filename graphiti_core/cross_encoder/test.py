from modelscope.pipelines import pipeline

# 创建文本重排序管道
rerank_pipeline = pipeline(
    task='text-ranking',
    model='BAAI/bge-reranker-v2-m3'
)

# 示例输入
query = "example query"
documents = [
    "first document content",
    "second document content",
    "third document content"
]

# 执行重排序
result = rerank_pipeline(query, documents)
print(result)