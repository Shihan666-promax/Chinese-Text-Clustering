import os
import json
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from gensim.models import Word2Vec
import warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# 加载JSON文件
def load_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                paper = json.load(f)
                data.append(paper)
    return data

# 文本预处理：分词、去除停用词，使用Word2Vec进行词嵌入
def preprocess_data(data):
    all_texts = []
    for paper in data:
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        keywords = paper.get('keywords', '')
        # 合并标题、摘要和关键词
        text = title + ' ' + abstract + ' ' + keywords
        all_texts.append(text)
    
    # 分词处理
    tokenized_texts = [word_tokenize(text.lower()) for text in all_texts]
    return tokenized_texts, all_texts

# 使用Word2Vec进行词嵌入
def train_word2vec_model(tokenized_texts):
    model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    return model

# 获取每个文本的词向量表示（平均词向量）
def get_text_vectors(tokenized_texts, model):
    text_vectors = []
    for tokens in tokenized_texts:
        vectors = []
        for word in tokens:
            if word in model.wv:
                vectors.append(model.wv[word])
        if vectors:
            text_vectors.append(np.mean(vectors, axis=0))
        else:
            text_vectors.append(np.zeros(100))  
    return np.array(text_vectors)

def kmeans(X, k, max_iters=100):
    n_samples, n_features = X.shape
    # 随机选择K个初始中心点
    centroids = X[np.random.choice(range(n_samples), k, replace=False)]
    for _ in range(max_iters):
        # 计算每个点到各个中心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        # 计算新的质心
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# 计算不同K值的SSE（误差平方和）以选择最佳K
def calculate_sse(X, max_k=10):
    sse = []
    for k in range(1, max_k + 1):
        labels, centroids = kmeans(X, k)
        sse.append(np.sum((X - centroids[labels])**2))
    return sse

# 绘制不同K值下的SSE图
def plot_sse(X, max_k=10):
    sse = calculate_sse(X, max_k)
    plt.figure(figsize=(8, 6))  
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.title('SSE for different k values')
    plt.show()

# 层次聚类（可视化树状图）
def hierarchical_clustering(X):
    linked = linkage(X, 'ward')
    plt.figure(figsize=(10, 7))  
    dendrogram(linked)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

# 聚类评估
def evaluate_clustering(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    print(f"轮廓系数: {silhouette_avg:.4f}")
    return silhouette_avg

def save_results(data, labels, output_file='clustering_results.json'):
    for idx, label in enumerate(labels):
        data[idx]['cluster'] = int(label)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    
    folder_path = 'cluster_data'
    data = load_data(folder_path)
    tokenized_texts, all_texts = preprocess_data(data)
    
    # 训练Word2Vec模型
    word2vec_model = train_word2vec_model(tokenized_texts)
    
    # 获取文本的词向量
    text_vectors = get_text_vectors(tokenized_texts, word2vec_model)

    # 标准化文本向量
    scaler = StandardScaler()
    text_vectors = scaler.fit_transform(text_vectors)

    # 绘制SSE图，选择合适的K值
    plot_sse(text_vectors, max_k=10)
    
    k = 3  # 根据SSE图选择K值
    kmeans_labels, kmeans_centroids = kmeans(text_vectors, k)
    evaluate_clustering(text_vectors, kmeans_labels)
    
    # 执行层次聚类并绘制树状图
    hierarchical_clustering(text_vectors)

    # 保存聚类结果
    save_results(data, kmeans_labels, output_file='kmeans_clustering_results.json')

if __name__ == "__main__":
    main()