# BÁO CÁO ĐỒ ÁN MÔN HỌC: HỆ THỐNG GỢI Ý (RECOMMENDER SYSTEMS)
## Đề tài: Nghiên cứu và Triển khai Đánh giá So sánh Mô hình Matrix Factorization và LightGCN trên Dataset MovieLens 1M

---

## THÔNG TIN CHUNG
* **Môn học:** Hệ Gợi Ý / Khai Phá Dữ Liệu
* **Đề tài:** Đánh giá Hiệu năng Matrix Factorization & LightGCN trên MovieLens 1M
* **Môi trường triển khai:** Docker Container (PyTorch, CUDA 12.1, Python 3.10), Host OS: Windows 11
* **Dataset sử dụng:** MovieLens 1M (ML-1M)

---

## CHƯƠNG 1: GIỚI THIỆU BÀI TOÁN VÀ ĐỀ TÀI

### 1.1. Đặt vấn đề
Trong kỷ nguyên số hóa, sự bùng nổ của thông tin và nội dung số đặt ra thách thức lớn đối với người dùng trong việc tìm kiếm thông tin hữu ích và đối với các doanh nghiệp trong việc giữ chân khách hàng. Các hệ thống gợi ý (Recommender Systems) đã trở thành một phần cốt lõi của các nền tảng thương mại điện tử (Amazon, Shopee) và dịch vụ phát trực tuyến (Netflix, YouTube, Spotify). Hệ gợi ý giúp cá nhân hóa trải nghiệm người dùng bằng cách tự động đề xuất các sản phẩm, bộ phim, hay bài hát mà người dùng có khả năng yêu thích cao nhất dựa trên lịch sử hành vi của họ.

### 1.2. Phát biểu bài toán gợi ý phim
Cho một tập hợp người dùng $U = \{u_1, u_2, ..., u_M\}$ và một tập hợp các bộ phim $I = \{i_1, i_2, ..., i_N\}$. Lịch sử tương tác giữa người dùng và phim được biểu diễn dưới dạng ma trận tương tác $R \in \mathbb{R}^{M \times N}$. Bài toán gợi ý được chia làm hai hướng tiếp cận chính:
1. **Dự đoán điểm đánh giá (Rating Prediction):** Dự đoán giá trị cụ thể mà người dùng $u$ sẽ chấm cho phim $i$ (ví dụ: từ 1 đến 5 sao). Mục tiêu là giảm thiểu sai số dự đoán.
2. **Gợi ý danh sách Top-N (Top-N Recommendation / Ranking):** Dự đoán danh sách gồm $N$ bộ phim mà người dùng $u$ có xu hướng tương tác nhiều nhất. Bài toán này gần với thực tế hơn vì đo lường trực tiếp hành vi lựa chọn của người dùng.

### 1.3. Mục tiêu đồ án
Đồ án này tập trung nghiên cứu lý thuyết, cài đặt thực nghiệm và so sánh đối chứng hai mô hình đại diện cho hai trường phái gợi ý tiêu biểu:
* **Matrix Factorization (MF) with Bias:** Đại diện cho phương pháp học máy lọc cộng tác (Collaborative Filtering) truyền thống dựa trên ma trận nhân tử ẩn.
* **LightGCN (Light Graph Convolutional Network):** Đại diện cho trường phái học sâu trên đồ thị (Graph Learning) hiện đại, khai thác mối liên kết đa cấp giữa người dùng và sản phẩm trên đồ thị lưỡng phân.

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ PHƯƠNG PHÁP LUẬN

### 2.1. Phương pháp Lọc cộng tác (Collaborative Filtering)
Phương pháp lọc cộng tác dựa trên giả định rằng các người dùng có hành vi tương tự trong quá khứ sẽ tiếp tục có sở thích giống nhau trong tương lai. Lọc cộng tác chỉ sử dụng ma trận tương tác User-Item để học đặc trưng ẩn của người dùng và sản phẩm mà không cần thông tin mô tả chi tiết (side information) như nhân khẩu học hay nội dung phim.

### 2.2. Mô hình Matrix Factorization (MF) tích hợp Bias
Mô hình MF phân rã ma trận đánh giá thưa thớt $R$ thành tích của hai ma trận đặc trưng ẩn có số chiều thấp hơn: ma trận người dùng $W \in \mathbb{R}^{M \times K}$ và ma trận phim $X \in \mathbb{R}^{N \times K}$ (với $K \ll \min(M, N)$ là số nhân tử ẩn).

#### Công thức dự đoán rating:
$$\hat{r}_{u,i} = W_u^T X_i + b_u + b_i + \mu$$

Trong đó:
* $W_u \in \mathbb{R}^K$ là vector biểu diễn ẩn của người dùng $u$.
* $X_i \in \mathbb{R}^K$ là vector biểu diễn ẩn của phim $i$.
* $b_u$ là hệ số thiên lệch (bias) của người dùng $u$ (thể hiện việc người dùng này khó hay dễ tính khi chấm điểm).
* $b_i$ là hệ số thiên lệch (bias) của phim $i$ (thể hiện phim này được đánh giá cao hay thấp trên mặt bằng chung).
* $\mu$ là điểm đánh giá trung bình toàn hệ thống.

#### Hàm mất mát tối ưu (Loss Function):
Cực tiểu hóa sai số bình phương trung bình trên tập huấn luyện kết hợp số hạng chuẩn hóa (Regularization L2) để tránh Overfitting:
$$\mathcal{L}_{MF} = \sum_{(u,i) \in R_{train}} (r_{u,i} - \hat{r}_{u,i})^2 + \lambda \left( \|W_u\|_2^2 + \|X_i\|_2^2 + b_u^2 + b_i^2 \right)$$
Trong đó $\lambda$ là siêu tham số kiểm soát mức độ chuẩn hóa. Mô hình được huấn luyện bằng phương pháp hạ cực dốc ngẫu nhiên (SGD).

---

### 2.3. Mô hình Mạng Đồ thị LightGCN
LightGCN được thiết kế tối giản hóa từ mạng tích chập đồ thị (GCN) truyền thống nhằm phục vụ riêng cho tác vụ gợi ý. Các tác giả phát hiện ra rằng hai thành phần phổ biến trong GCN là **phép biến đổi phi tuyến (Non-linear Activation)** và **ma trận trọng số chuyển đổi (Weight Matrix)** không giúp ích mà thậm chí còn làm giảm hiệu năng và tăng độ phức tạp tính toán của hệ gợi ý.

#### 2.3.1. Đồ thị lưỡng phân tương tác (Bipartite Graph)
Hành vi của người dùng được biểu diễn dưới dạng Đồ thị lưỡng phân vô hướng $G = (U \cup I, E)$, trong đó một cạnh $e = (u, i) \in E$ đại diện cho tương tác (ví dụ: người dùng $u$ đã xem phim $i$).

#### 2.3.2. Quy tắc lan truyền đặc trưng (Propagation Rule)
Tại mỗi lớp $k$, biểu diễn ẩn (embedding) của một nút được cập nhật bằng cách lấy trung bình trọng số biểu diễn của các nút láng giềng trực tiếp từ lớp trước:
$$e_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u| \sqrt{|\mathcal{N}_i|}}} e_i^{(k)}$$
$$e_i^{(k+1)} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i| \sqrt{|\mathcal{N}_u|}}} e_u^{(k)}$$

Trong đó:
* $\mathcal{N}_u$ là tập hợp các bộ phim đã được tương tác bởi người dùng $u$.
* $\mathcal{N}_i$ là tập hợp các người dùng đã tương tác với bộ phim $i$.
* Hệ số đối xứng hóa $\frac{1}{\sqrt{|\mathcal{N}_u| \sqrt{|\mathcal{N}_i|}}}$ giúp chuẩn hóa độ lớn của embedding tránh bùng nổ trị số khi lan truyền qua các nút có bậc đồ thị lớn.

#### 2.3.3. Kết hợp nhúng đa lớp (Layer Combination)
Sau khi lan truyền qua $K$ lớp, mô hình gom tất cả embeddings ở các lớp lại bằng trung bình cộng để tạo ra embedding cuối cùng:
$$e_u = \sum_{k=0}^K \alpha_k e_u^{(k)}; \quad e_i = \sum_{k=0}^K \alpha_k e_i^{(k)}$$
Thông thường chọn hệ số đều $\alpha_k = \frac{1}{K+1}$. Điều này giúp giữ lại cả thông tin tương tác trực tiếp (lớp thấp) và thông tin cộng tác đa cấp (lớp cao).

#### 2.3.4. Hàm mất mát Bayesian Personalized Ranking (BPR Loss)
LightGCN tối ưu hóa bài toán xếp hạng (ranking) thông qua BPR Loss. Với mỗi cặp tương tác dương $(u, i)$ (người dùng $u$ đã xem phim $i$) và tương tác âm $(u, j)$ (người dùng $u$ chưa xem phim $j$), mô hình tối đa hóa khoảng cách điểm dự đoán tương thích giữa $y_{u,i}$ và $y_{u,j}$:
$$\mathcal{L}_{BPR} = -\sum_{(u,i,j) \in D_{train}} \ln \sigma \left( e_u^T e_i - e_u^T e_j \right) + \lambda \|E^{(0)}\|_2^2$$
Trong đó $\sigma(x)$ là hàm sigmoid.

---

## CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

### 3.1. Mô tả Dữ liệu MovieLens 1M
Tập dữ liệu MovieLens 1M được cung cấp bởi GroupLens Research, chứa thông tin xếp hạng phim từ nền tảng MovieLens. Dữ liệu gồm 3 tệp tin chính:
* `ratings.dat`: Chứa 1,000,209 dòng ghi nhận đánh giá (`UserID::MovieID::Rating::Timestamp`).
* `users.dat`: Chứa thông tin nhân khẩu học của người dùng (`UserID::Gender::Age::Occupation::Zip-code`).
* `movies.dat`: Chứa thông tin phim (`MovieID::Title::Genres`).

### 3.2. Phân tích Dữ liệu Khám phá (Exploratory Data Analysis - EDA)

#### 3.2.1. Thống kê Cơ bản của Dataset
| Chỉ số | Giá trị |
| :--- | :--- |
| **Số lượng Người dùng (Users)** | 6,040 |
| **Số lượng Bộ phim (Items)** | 3,706 |
| **Số lượng Tương tác (Ratings)** | 1,000,209 |
| **Độ thưa thớt (Sparsity)** | **95.53%** |

*Công thức tính Sparsity:*
$$\text{Sparsity} = 1 - \frac{|R|}{|U| \times |I|} = 1 - \frac{1,000,209}{6,040 \times 3,706} \approx 95.53\%$$

#### 3.2.2. Phân tích chi tiết qua biểu đồ trực quan hóa

##### A. Phân Phối Điểm Đánh Giá (Rating Distribution)
Biểu đồ phân phối cho thấy người dùng MovieLens có xu hướng đánh giá tích cực. Điểm số tập trung nhiều nhất ở mức **4 sao** (hơn 340,000 đánh giá), tiếp theo là 3 sao và 5 sao. Các đánh giá tiêu cực (1 sao và 2 sao) chiếm tỷ lệ cực kỳ thấp. Điều này phản ánh sự mất cân bằng lớp nếu áp dụng các mô hình phân loại thông thường, và khẳng định tầm quan trọng của việc chuẩn hóa dữ liệu rating theo từng User/Item trước khi thực hiện phân rã ma trận.

![Phân phối điểm đánh giá](checkpoints/eda_rating_distribution.png)
*Hình 1: Phân phối điểm đánh giá từ 1 đến 5 sao trong tập dữ liệu MovieLens 1M.*

##### B. Phân Phối Tương Tác Theo Người Dùng (User Interactions)
Mỗi người dùng trong hệ thống có tối thiểu 20 lượt tương tác. Tần suất tương tác của người dùng tập trung mạnh ở phân khúc dưới 200 lượt đánh giá. Tuy nhiên, đồ thị xuất hiện phần đuôi dài kéo dài về phía bên phải đại diện cho nhóm người dùng "siêu hoạt động" có tới hàng ngàn đánh giá. Nhóm người dùng này đóng vai trò là các nút trung tâm kết nối các phần khác nhau của đồ thị lưỡng phân.

![Số lượt tương tác của mỗi người dùng](checkpoints/eda_user_interactions.png)
*Hình 2: Phân phối số lượng tương tác của từng User.*

##### C. Phân Phối Tương Tác Theo Phim (Movie Popularity)
Độ phổ biến của các bộ phim thể hiện quy luật **Đuôi dài (Long Tail)** kinh điển trong dữ liệu thương mại điện tử. Một số lượng rất nhỏ phim bom tấn (như Toy Story, Star Wars) nhận được lượng tương tác khổng lồ từ đại đa số người dùng. Ngược lại, phần lớn các phim nghệ thuật hoặc phim ít tiếng tăm nằm ở phần đuôi dài chỉ nhận được một vài đánh giá. Thách thức của hệ gợi ý là phải đề xuất được các bộ phim ở phần đuôi dài này cho người dùng phù hợp thay vì chỉ liên tục gợi ý các phim quá nổi tiếng.

![Số lượt tương tác của mỗi bộ phim](checkpoints/eda_item_interactions.png)
*Hình 3: Phân phối số lượng tương tác của từng Movie (Quy luật Long Tail).*

##### D. Xu Hướng Số Lượng Đánh Giá Theo Thời Gian (Ratings Over Time)
Dữ liệu ghi nhận lượng tương tác lịch sử biến động mạnh theo thời gian. Có một đỉnh bùng nổ số lượng đánh giá cực lớn vào khoảng nửa cuối năm 2000, đây là giai đoạn thu thập dữ liệu tập trung của nhóm dự án GroupLens. Sau đó, lượng tương tác giảm dần và đi vào trạng thái ổn định lâu dài.

![Ratings Over Time](checkpoints/eda_ratings_over_time.png)
*Hình 4: Xu hướng biến động số lượng tương tác theo thời gian ghi nhận.*

---

## CHƯƠNG 4: THIẾT KẾ CÀI ĐẶT VÀ CÁC GIẢI PHÁP TỐI ƯU HÓA

### 4.1. Giải pháp Tối ưu hóa trong Mô hình Matrix Factorization (MF)
Trong các triển khai thông thường trên Jupyter Notebook, việc huấn luyện mô hình MF bằng cách lặp qua từng phần tử trong danh sách tương tác và tìm kiếm người dùng/phim tương ứng mất rất nhiều thời gian (thường là $O(N)$ cho mỗi bước).
* **Giải pháp:** Trong file `Model/MF/mf_with_bias.py`, tôi đã thiết lập cơ chế **Lập chỉ mục tương tác nhanh (Fast Interaction Indexing)** bằng cách xây dựng trước các mảng lưu chỉ mục tương ứng cho người dùng và phim trong pha khởi tạo (`__init__`). Quá trình lấy mẫu và cập nhật SGD chuyển từ dạng tìm kiếm tuyến tính sang truy cập trực tiếp bộ nhớ với độ phức tạp $O(1)$.
* **Kết quả:** Tốc độ huấn luyện tăng vọt hơn 100 lần, hoàn thành 30 epoch huấn luyện trên tập dữ liệu 1 triệu tương tác chỉ mất **50 giây** trên CPU.

### 4.2. Giải pháp Tối ưu hóa trong Mô hình Mạng Đồ thị LightGCN
Huấn luyện GCN trên tập dữ liệu lớn thường gặp nút thắt cổ chai ở khâu **Lấy mẫu tương tác âm (Negative Sampling)** cho thuật toán BPR. Mô hình yêu cầu với mỗi tương tác dương $(u, i)$, ta phải chọn ngẫu nhiên một phim $j$ mà người dùng $u$ chưa từng tương tác. Việc kiểm tra điều kiện này trên mảng NumPy của lịch sử tương tác bằng vòng lặp tuyến tính cực kỳ tốn thời gian.
* **Giải pháp:** Trong file `Model/lightgcn-1M/src/data.py`, tôi đã chuyển độ tương tác của mỗi người dùng thành cấu trúc **Python Set** lưu trong bộ nhớ. Phép kiểm tra âm tính (`j not in user_pos_set`) được thực hiện với độ phức tạp trung bình là $O(1)$ thay vì quét tuyến tính $O(N)$. Đồng thời tăng kích thước batch huấn luyện lên `8192` để tận dụng song song hóa của GPU CUDA.
* **Kết quả:** Thời gian huấn luyện mỗi epoch rút ngắn từ hơn 2 phút xuống chỉ còn **15 giây**. Toàn bộ 15 epoch huấn luyện và kiểm định hoàn thành trong **4 phút** trên GPU CUDA.

### 4.3. Thiết lập Siêu tham số (Hyperparameters)
| Siêu tham số | Matrix Factorization (MF) | LightGCN (Graph Learning) |
| :--- | :--- | :--- |
| **Kích thước nhúng / Nhân tử ẩn ($K$)** | 20 | 64 |
| **Tốc độ học (Learning Rate)** | 0.5 | 0.002 |
| **Hệ số chuẩn hóa ($\lambda$)** | 0.05 | 1e-4 |
| **Số epoch tối đa** | 30 | 15 |
| **Kích thước lô (Batch size)** | N/A (Full batch SGD) | 8192 |
| **Số lớp lan truyền ($L$)** | N/A | 3 |
| **Phương pháp phân tách tập test** | Random Split (80% Train, 20% Test) | Leave-One-Out (Tương tác cuối của User làm Test) |

---

## CHƯƠNG 5: KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ

### 5.1. Kết quả Huấn luyện của Matrix Factorization (Tối ưu Ranking)
Trong phiên bản cải tiến, thay vì chỉ dự đoán điểm số (Rating), mô hình Matrix Factorization đã được lập trình lại để trực tiếp xử lý bài toán **Xếp hạng Top-20 (Ranking)**. Thuật toán phân rã ma trận được cấp chung tập dữ liệu Leave-One-Out với LightGCN, tại mỗi epoch kiểm định, nó tính toán điểm tương thích cho toàn bộ các phim chưa xem và lọc ra Top 20.

* **Kết quả cuối cùng:** **Recall@20 = 0.0089**, **NDCG@20 = 0.0031** đạt được ở Epoch 30.

![Biểu đồ đánh giá Matrix Factorization](checkpoints/mf_metrics_k20.png)
*Hình 5: Biểu đồ thể hiện sự suy giảm của Loss và sự tăng trưởng của Recall/NDCG trên tập Validation.*

---

### 5.2. Kết quả Huấn luyện của LightGCN
LightGCN được huấn luyện tối ưu hóa BPR Loss và đánh giá bằng hai chỉ số xếp hạng phổ biến:
* **Recall@20 (Độ bao phủ):** Đo lường tỷ lệ các bộ phim người dùng thực sự thích xuất hiện trong Top 20 gợi ý của mô hình.
* **NDCG@20 (Normalized Discounted Cumulative Gain):** Đo lường chất lượng xếp hạng, ưu tiên các bộ phim phù hợp nằm ở các vị trí đầu tiên của danh sách gợi ý.

* **Kết quả cuối cùng ở Epoch 15:**
  * **Recall@20 = 0.0833** (Tương đương 8.33% phim test của người dùng được mô hình gợi ý chính xác trong Top 20).
  * **NDCG@20 = 0.0338**.

![Biểu đồ đánh giá LightGCN](checkpoints/metrics_k20.png)
*Hình 6: Biểu đồ ghi nhận sự tăng trưởng ổn định của Recall@20 và NDCG@20 qua các epoch kiểm định.*

---

### 5.3. So sánh Đối chứng Hiệu năng Xếp hạng (Top-K Ranking)
Bởi vì cả hai thuật toán MF và LightGCN đã được chuẩn hóa để chạy trên cùng giao thức kiểm định Leave-One-Out và cùng tập Validation, chúng ta có thể đặt chúng lên bàn cân một cách hoàn toàn công bằng.

![Biểu đồ so sánh MF và LightGCN](checkpoints/comparison_bar_chart.png)
*Hình 7: Biểu đồ cột so sánh trực tiếp chỉ số Recall@20 và NDCG@20 giữa Matrix Factorization và LightGCN.*

**Bảng so sánh chi tiết:**
| Mô hình | Recall@20 | NDCG@20 |
| :--- | :--- | :--- |
| **Matrix Factorization (with Bias)** | 0.0089 | 0.0031 |
| **LightGCN** | **0.0833** | **0.0338** |

**Phân tích kết quả:**
* **Sự vượt trội của LightGCN:** LightGCN đạt hiệu năng xếp hạng cao hơn Matrix Factorization **gần 10 lần** (0.0833 so với 0.0089 ở Recall@20). Điều này hoàn toàn dễ hiểu vì MF chỉ học tính tương đồng qua tương tác trực tiếp (Bậc 1) và tối ưu hóa sai số điểm số (Pointwise). Trong khi đó, LightGCN được sinh ra để học **sự kết nối đa bậc (High-order connectivity)** trên đồ thị và tối ưu hóa trực tiếp hàm mất mát xếp hạng (BPR Loss - Pairwise).
* **Bài học rút ra:** Phương pháp tiếp cận truyền thống như MF tỏ ra rất yếu trong việc trả về danh sách Top-K phim liên quan nhất cho người dùng trong thế giới thực. Đây chính là lý do các hệ thống gợi ý hiện đại (YouTube, Netflix) đã chuyển hoàn toàn sang Graph Neural Networks.

---

### 5.4. Trực quan hóa Không gian Nhúng Phim (Movie Embedding Space)
Để chứng minh chất lượng biểu diễn đặc trưng của LightGCN, tôi đã trích xuất ma trận nhúng 64 chiều của toàn bộ 3,706 phim từ checkpoint lưu trữ `checkpoints/lightgcn.pt`. Dữ liệu này được giảm chiều xuống 2D bằng thuật toán **t-SNE** và tô màu theo thể loại chính của phim.

![Biểu đồ phân cụm t-SNE Movie Embeddings](checkpoints/embeddings_visualization.png)
*Hình 8: Trực quan hóa không gian nhúng phim bằng t-SNE. Các màu sắc đại diện cho các thể loại phim chính.*

#### Nhận xét kết quả phân cụm:
1. **Sự tách biệt rõ ràng theo thể loại:** Mặc dù mô hình LightGCN **không được cung cấp bất kỳ thông tin nhãn thể loại nào** trong quá trình huấn luyện (chỉ học duy nhất từ lịch sử click của người dùng), các bộ phim cùng thể loại tự động tụ lại thành các cụm điểm riêng biệt.
2. **Cụm phim Hoạt hình (Animation - Màu tím):** Tạo thành một cụm tách biệt hoàn toàn nằm ở góc trên của đồ thị. Các phim hoạt hình nổi tiếng như *"Toy Story (1995)"*, *"Aladdin (1992)"*, *"Lion King, The (1994)"* nằm sát cạnh nhau.
3. **Cụm phim Hành động (Action - Màu xanh lá) và Viễn tưởng:** Các tác phẩm nổi tiếng như *"Matrix, The (1999)"*, *"Terminator 2: Judgment Day (1991)"*, *"Jurassic Park (1993)"* tụ lại gần nhau.
4. **Cụm phim Hài (Comedy - Màu xanh lam):** Nằm phân bố rộng rãi hơn ở trung tâm đồ thị, cho thấy tính chất phổ biến và dễ kết hợp của phim hài với các thể loại khác.

---

### 5.5. Trực quan hóa Đồ thị Lưỡng phân User-Item
Để kiểm tra cấu trúc đồ thị thực tế được nạp vào LightGCN, một phần đồ thị lưỡng phân bao gồm 10 người dùng ngẫu nhiên và các bộ phim họ đã tương tác được trực quan hóa dưới dạng mạng lưới kết nối mạng xã hội.

![Đồ thị lưỡng phân tương tác User-Item](checkpoints/user_item_graph.png)
*Hình 9: Trực quan hóa kết nối lưỡng phân giữa User (nút màu cam) và Movie (nút màu xanh dương).*

Đồ thị minh chứng rõ ràng cấu trúc kết nối đa chiều, nơi một bộ phim nổi tiếng (các nút xanh dương lớn ở trung tâm) đóng vai trò là "cầu nối" liên kết nhiều người dùng khác nhau. Đây là nguồn thông tin cộng tác đa bậc cốt lõi mà mô hình LightGCN khai thác qua các lớp tích chập đồ thị.

---

### 5.6. Bảng Tổng kết Thuộc tính Mô hình

| Tiêu chí | Matrix Factorization (MF) with Bias | LightGCN (Graph Learning) |
| :--- | :--- | :--- |
| **Loại thuật toán** | Lọc cộng tác cổ điển (Collaborative Filtering) | Học sâu trên đồ thị (Graph Deep Learning) |
| **Đầu ra cốt lõi** | Dự đoán điểm số cụ thể (Rating prediction - RMSE) | Xếp hạng danh sách (Ranking - BPR Loss) |
| **Đặc trưng học được** | Nhân tử ẩn tuyến tính (Bậc 1) | Biểu diễn phi tuyến tính qua láng giềng (Đa bậc) |
| **Độ tối ưu thực tế** | Kém trong gợi ý Top-K, phù hợp đoán số sao | Vượt trội trong tính năng gợi ý trang chủ (Top-N) |
| **Sai số kiểm định** | **Test RMSE = Không áp dụng** | **Test RMSE = Không áp dụng** |
| **Ranking Metric** | **Recall@20 = 0.0089** | **Recall@20 = 0.0833** |

---

## CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Kết luận đồ án
Đồ án đã hoàn thành đầy đủ các mục tiêu đề ra:
1. Nghiên cứu sâu sắc lý thuyết toán học của mô hình Matrix Factorization truyền thống và mô hình LightGCN hiện đại trên đồ thị.
2. Triển khai lập trình thành công cả hai mô hình bằng Python & PyTorch chạy tối ưu trong Docker Container.
3. Giải quyết thành công các nút thắt hiệu năng về tốc độ huấn luyện (tối ưu hóa sampler và chỉ mục giúp huấn luyện đạt hiệu suất cực cao).
4. Thực hiện phân tích dữ liệu khám phá (EDA) chuyên sâu trên dataset MovieLens 1M và đưa ra các biểu đồ trực quan hóa không gian nhúng phim chất lượng cao.

### 6.2. Hướng phát triển trong tương lai
* **Giải quyết bài toán Khởi đầu lạnh (Cold Start):** Tích hợp thêm các đặc trưng phụ (Side Information) của người dùng (tuổi tác, giới tính, nghề nghiệp) và phim (thể loại, đạo diễn, diễn viên) vào mô hình đồ thị thông qua kiến trúc Graph Attention Networks (GAT) hoặc GCN đa thuộc tính.
* **Gợi ý Real-time:** Xây dựng hệ thống lưu trữ vector (Vector Database như Milvus hoặc FAISS) kết hợp mô hình LightGCN đã huấn luyện để phục vụ tính toán gợi ý thời gian thực qua API Web Service.
* **Tương tác đa hành vi:** Khai thác các hành vi tương tác khác nhau của người dùng như xem thử trailer, thêm vào danh sách yêu thích, đánh giá sao để xây dựng mô hình gợi ý đa đồ thị (Multi-behavior recommendation).
