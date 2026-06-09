import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def main():
    ratings_path = "datasets/ml-1m/ratings.dat"
    movies_path = "datasets/ml-1m/movies.dat"
    
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        print("Dataset files not found.")
        return

    # Load ratings & movies
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(ratings_path, sep='::', names=r_cols, encoding='latin-1', engine='python')
    
    m_cols = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(movies_path, sep='::', names=m_cols, encoding='latin-1', engine='python')
    movie_dict = movies.set_index('movie_id')['title'].to_dict()

    # Chọn một nhóm nhỏ gồm 3 users có chung sở thích để đồ thị được kết nối và đẹp mắt
    # Tìm phim phổ biến nhất để tìm các users đã đánh giá nó
    popular_movies = ratings['movie_id'].value_counts()
    top_movie_id = popular_movies.index[0] # Phim phổ biến nhất (thường là Toy Story hoặc Star Wars)
    
    # Lấy 3 users bất kỳ đã rate phim phổ biến này
    selected_users = ratings[ratings['movie_id'] == top_movie_id]['user_id'].head(3).tolist()
    
    # Lấy các đánh giá của 3 users này (giới hạn tối đa 4 phim khác nhau cho mỗi user để đồ thị không bị rối)
    sub_ratings = []
    for u in selected_users:
        u_ratings = ratings[ratings['user_id'] == u].head(5) # Lấy tối đa 5 phim đầu tiên của user này
        sub_ratings.append(u_ratings)
    sub_ratings = pd.concat(sub_ratings).drop_duplicates(subset=['user_id', 'movie_id'])

    # Khởi tạo đồ thị NetworkX
    G = nx.Graph()

    # Chuẩn bị nhãn và phân nhóm các node
    user_nodes = [f"User {u}" for u in selected_users]
    
    movie_ids = sub_ratings['movie_id'].unique()
    movie_nodes = [movie_dict.get(m_id, f"Movie {m_id}") for m_id in movie_ids]
    
    G.add_nodes_from(user_nodes, bipartite=0)
    G.add_nodes_from(movie_nodes, bipartite=1)

    # Thêm các cạnh nối (User -> Movie) kèm trọng số là điểm đánh giá (Rating)
    edges = []
    for _, row in sub_ratings.iterrows():
        u_node = f"User {int(row['user_id'])}"
        m_node = movie_dict.get(int(row['movie_id']), f"Movie {int(row['movie_id'])}")
        rating = int(row['rating'])
        G.add_edge(u_node, m_node, weight=rating)
        edges.append((u_node, m_node, rating))

    # Vẽ đồ thị lưỡng phân (Bipartite Layout)
    plt.figure(figsize=(14, 8), dpi=150)
    
    # Đặt vị trí cột trái cho User, cột phải cho Movie
    pos = {}
    # Sắp xếp tọa độ Y cho User nodes
    for idx, u_node in enumerate(user_nodes):
        pos[u_node] = np.array([0, idx - (len(user_nodes) - 1) / 2.0])
        
    # Sắp xếp tọa độ Y cho Movie nodes
    for idx, m_node in enumerate(movie_nodes):
        pos[m_node] = np.array([1, (idx - (len(movie_nodes) - 1) / 2.0) * 0.4])

    # Vẽ các node User (Màu xanh lam nhạt, hình tròn)
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=user_nodes, 
        node_color='#3498db', 
        node_shape='o', 
        node_size=1200, 
        alpha=0.9
    )
    
    # Vẽ các node Movie (Màu vàng cam nhạt, hình vuông)
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=movie_nodes, 
        node_color='#e67e22', 
        node_shape='s', 
        node_size=1500, 
        alpha=0.9
    )

    # Vẽ nhãn cho các Node
    # Đối với User: nhãn lệch sang trái
    user_labels = {u: u for u in user_nodes}
    nx.draw_networkx_labels(G, pos, labels=user_labels, font_size=10, font_weight='bold')
    
    # Đối với Movie: nhãn lệch sang phải một chút để dễ đọc
    movie_labels = {m: m for m in movie_nodes}
    # Vẽ nhãn Movie với font size nhỏ hơn để tránh tràn chữ
    nx.draw_networkx_labels(G, pos, labels=movie_labels, font_size=8, font_weight='bold')

    # Vẽ các cạnh kết nối, độ đậm nhạt tương ứng với điểm Rating (1-5 sao)
    for u, m, w in edges:
        width = w * 0.8
        alpha = 0.3 + (w / 5.0) * 0.6
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=[(u, m)], 
            width=width, 
            alpha=alpha, 
            edge_color='#7f8c8d'
        )

    # Thêm chú thích cho đồ thị
    plt.title("Đồ Thị Lưỡng Phân Tương Tác User-Item (MovieLens 1M Subgraph)", fontsize=14, weight='bold', pad=20)
    plt.axis('off')
    
    # Vẽ chú thích bên lề
    plt.plot([], [], 'o', color='#3498db', markersize=10, label='Người dùng (User Node)')
    plt.plot([], [], 's', color='#e67e22', markersize=10, label='Phim ảnh (Item Node)')
    plt.plot([], [], '-', color='#7f8c8d', linewidth=2, label='Tương tác (Độ dày ~ Số sao đánh giá)')
    plt.legend(loc='lower center', ncol=3, frameon=True, shadow=True, borderpad=1)
    
    plt.tight_layout()
    output_path = "checkpoints/user_item_graph.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Bipartite graph visualization saved to: {output_path}")

if __name__ == '__main__':
    main()
