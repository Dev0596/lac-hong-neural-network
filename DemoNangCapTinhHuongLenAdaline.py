import csv                      # Thư viện đọc file CSV thuần (không cần pandas)
import numpy as np              # Thư viện tính toán ma trận/vector
import matplotlib.pyplot as plt # Thư viện vẽ đồ thị

# ==========================================
# BƯỚC 1 & 2: ĐỌC VÀ KIỂM TRA DỮ LIỆU THÔ
# ==========================================
# [ĐẦU VÀO] Danh sách tên 5 đặc trưng (feature), THỨ TỰ này quyết định
# thứ tự cột của X và thứ tự 5 trọng số w sau khi học.
FEATURE_NAMES = [
    'khoang_cach_km',            # x1: khoảng cách giao hàng (km)
    'so_don_tai_xe',             # x2: số đơn tài xế đang mang
    'troi_mua',                  # x3: trời mưa (0/1)
    'gio_cao_diem',              # x4: giờ cao điểm (0/1)
    'thoi_gian_chuan_bi_phut'    # x5: thời gian chuẩn bị món (phút)
]

X_list = []   # Danh sách chứa các dòng feature (mỗi dòng = 1 đơn hàng, 5 số)
t_list = []   # Danh sách chứa nhãn/target t của từng đơn hàng

# ==========================================
# BƯỚC 3 NẠP DỮ LIỆU ĐẦU VÀO
# ==========================================
with open('du_lieu_giao_muon_50000.csv', mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)   # Mỗi dòng được đọc thành dict {tên_cột: giá_trị_chuỗi}
    for row in reader:
        # Xử lý: ép 5 giá trị chuỗi sang float, xếp theo đúng thứ tự FEATURE_NAMES
        X_list.append([
            float(row['khoang_cach_km']),
            float(row['so_don_tai_xe']),
            float(row['troi_mua']),
            float(row['gio_cao_diem']),
            float(row['thoi_gian_chuan_bi_phut'])
        ])
        # [ĐẦU VÀO - NHÃN] t = khả năng giao muộn (giá trị cần dự đoán)
        t_list.append(float(row['kha_nang_giao_muon']))

X_raw = np.array(X_list, dtype=float)  # Ma trận feature thô, shape (50000, 5)
t_raw = np.array(t_list, dtype=float)  # Vector target thô, shape (50000,)

print(f"-> Đọc dữ liệu thành công mà không cần pandas!")
print(f"Kích thước X_raw: {X_raw.shape}")  # Kỳ vọng: (50000, 5)
print(f"Kích thước t_raw: {t_raw.shape}")  # Kỳ vọng: (50000,)

# ==========================================
# BƯỚC 4: TÁCH TRAIN/TEST (80% TRAIN, 20% TEST)
# ==========================================
np.random.seed(42)                      # Cố định seed -> kết quả xáo trộn lặp lại được
indices = np.arange(len(X_raw))         # Mảng chỉ số 0,1,...,49999
np.random.shuffle(indices)              # Xáo trộn ngẫu nhiên chỉ số (tránh dữ liệu theo thứ tự)

train_size = int(0.8 * len(X_raw))      # 0.8 * 50000 = 40000 mẫu train
train_idx, test_idx = indices[:train_size], indices[train_size:]  # 40000 chỉ số train, 10000 chỉ số test

# Xử lý: lấy các dòng theo chỉ số đã xáo trộn
X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]  # shape (40000,5) và (10000,5)
t_train, t_test = t_raw[train_idx], t_raw[test_idx]          # shape (40000,) và (10000,)

# ==========================================
# BƯỚC 5: CHUẨN HOÁ FEATURE (Z-score)
# ==========================================
# Công thức: x_scaled = (x - mean) / std
# CHỈ tính mean/std từ tập TRAIN (tránh rò rỉ thông tin từ test).
X_mean = X_train_raw.mean(axis=0)   # Vector 5 số: trung bình từng cột (feature)
X_std = X_train_raw.std(axis=0)     # Vector 5 số: độ lệch chuẩn từng cột

X_train = (X_train_raw - X_mean) / X_std  # Train đã chuẩn hoá: mỗi cột có mean≈0, std≈1
X_test = (X_test_raw - X_mean) / X_std    # Test dùng ĐÚNG mean/std của train
# Lý do chuẩn hoá: các feature có thang đo rất khác nhau (km, phút, 0/1),
# nếu không chuẩn hoá thì gradient descent hội tụ chậm/lệch và
# trọng số không so sánh được với nhau.

# ==========================================
# BƯỚC 6: CÀI ĐẶT THUẬT TOÁN ADALINE (WIDROW-HOFF / LMS)
# ==========================================
class AdalineLMS:
    def __init__(self, eta=0.1, epochs=30, random_state=42):
        self.eta = eta                      # Tốc độ học η (learning rate) = 0.1
        self.epochs = epochs                # Số lần duyệt toàn bộ tập train = 30
        self.random_state = random_state    # Seed để khởi tạo trọng số lặp lại được

    def fit(self, X, t, X_val=None, t_val=None):
        # [ĐẦU VÀO] X: (N,5) feature đã chuẩn hoá | t: (N,) nhãn
        #           X_val, t_val: tập test dùng để theo dõi MSE (không dùng để cập nhật trọng số)
        rgen = np.random.RandomState(self.random_state)

        # [TRỌNG SỐ] Khởi tạo w ~ Normal(0, 0.01²), vector 5 phần tử (mỗi feature 1 trọng số)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # [TRỌNG SỐ] Bias b khởi tạo = 0
        self.b_ = np.float64(0.0)

        self.mse_train_ = []     # Lưu MSE train qua từng epoch (để vẽ đồ thị)
        self.mse_val_ = []       # Lưu MSE test qua từng epoch
        n_samples = len(X)       # N = 40000

        for epoch in range(self.epochs):
            # [CÔNG THỨC 1 - ĐẦU RA TUYẾN TÍNH] net = w^T·x + b  (tính cho cả N mẫu cùng lúc)
            # X (N,5) · w (5,) -> vector (N,)
            net_train = np.dot(X, self.w_) + self.b_

            # [CÔNG THỨC 2 - SAI SỐ] e = t - net   (vector N phần tử)
            errors = t - net_train

            # Quy tắc Cập nhật Widrow-Hoff (Batch Gradient Descent)
            # Hàm mất mát: J = (1/N) * Σ 0.5 * e²
            # Gradient:   ∂J/∂w = -(1/N) * Xᵀ·e     ∂J/∂b = -(1/N) * Σe
            # [CÔNG THỨC 3 - CẬP NHẬT w] w ← w + η · (Xᵀ·e) / N
            self.w_ += self.eta * np.dot(X.T, errors) / n_samples
            # [CÔNG THỨC 4 - CẬP NHẬT b] b ← b + η · (Σe) / N
            self.b_ += self.eta * errors.sum() / n_samples

            # [CÔNG THỨC 5 - MSE] J = mean(0.5 · e²)
            # Lưu ý: e ở đây tính bằng w, b TRƯỚC khi cập nhật trong epoch này.
            mse_t = np.mean(0.5 * errors**2)
            self.mse_train_.append(mse_t)

            # Nếu có tập kiểm tra thì tính MSE trên đó (chỉ để theo dõi, KHÔNG học từ nó)
            if X_val is not None and t_val is not None:
                net_val = np.dot(X_val, self.w_) + self.b_       # net của tập test
                mse_v = np.mean(0.5 * (t_val - net_val)**2)      # MSE test (dùng w,b SAU cập nhật)
                self.mse_val_.append(mse_v)

        return self   # Trả về chính model để có thể gọi tiếp

model = AdalineLMS(eta=0.1, epochs=30, random_state=42)  # Tạo model với các siêu tham số
model.fit(X_train, t_train, X_test, t_test)              # Huấn luyện: học w_, b_ từ tập train

# ==========================================
# BƯỚC 7: ĐÁNH GIÁ KẾT QUẢ VÀ IN TRỌNG SỐ
# ==========================================
net_test = np.dot(X_test, model.w_) + model.b_   # [ĐẦU RA] Dự đoán net cho 10000 mẫu test
errors_test = t_test - net_test                  # Sai số e = t - net của từng mẫu test
abs_errors_test = np.abs(errors_test)            # Sai số tuyệt đối |e|

# argsort: sắp chỉ số theo |e| tăng dần -> [-3:] lấy 3 chỉ số lớn nhất -> [::-1] đảo để lớn nhất đứng đầu
top_3_idx = np.argsort(abs_errors_test)[-3:][::-1]

print("\n=============== KẾT QUẢ TRỌNG SỐ SẢN PHẨM SAU KHI HỌC ĐƯỢC ===============")
# [ĐẦU RA - TRỌNG SỐ] In từng cặp (tên feature, trọng số). w>0: feature tăng -> nguy cơ trễ tăng.
# Lưu ý: w học trên dữ liệu ĐÃ CHUẨN HOÁ, nên so sánh độ lớn giữa các w là hợp lý,
# nhưng không phải "tác động trên 1 km" hay "1 phút" thô.
for name, w in zip(FEATURE_NAMES, model.w_):
    print(f"Trọng số {name:25s}: {w:.4f}")
print(f"Bias b                           : {model.b_:.4f}")  # b ≈ mức dự đoán khi mọi feature ở giá trị trung bình

print("\n=============== TOP 3 MẪU CÓ SAI SỐ TUYỆT ĐỐI |e| LỚN NHẤT TRÊN TEST ===============")
for idx in top_3_idx:
    print(f"Mẫu #{idx}: Feature thô = {X_test_raw[idx]}")   # In feature gốc (chưa chuẩn hoá) cho dễ đọc
    print(f"  Target t = {t_test[idx]:.4f} | Dự đoán net = {net_test[idx]:.4f} | Sai số e = {errors_test[idx]:.4f}")

# ==========================================
# BƯỚC 8: DỰ ĐOÁN THỬ CHO MẪU MỚI
# ==========================================
def predict_one(record_dict, model_w, model_b, mean_vec, std_vec, threshold=0.5):
    # [ĐẦU VÀO] record_dict: 1 đơn hàng dạng dict | model_w, model_b: trọng số đã học
    #           mean_vec, std_vec: mean/std của TRAIN | threshold: ngưỡng quyết định 0.5

    # Lấy giá trị theo đúng thứ tự FEATURE_NAMES -> vector 5 phần tử
    x_raw = np.array([record_dict[col] for col in FEATURE_NAMES], dtype=float)
    # Chuẩn hoá bằng mean/std của train (PHẢI giống lúc huấn luyện)
    x_scaled = (x_raw - mean_vec) / std_vec
    # [CÔNG THỨC] net = w^T·x_scaled + b
    net = float(np.dot(x_scaled, model_w) + model_b)
    # [ĐẦU RA] (điểm net, True nếu net >= 0.5 -> cảnh báo trễ)
    return net, net >= threshold

# Đơn hàng mới cần dự đoán (giá trị thô)
sample_test = {
    'khoang_cach_km': 12,            # 12 km
    'so_don_tai_xe': 7,              # tài xế đang có 7 đơn
    'troi_mua': 1,                   # có mưa
    'gio_cao_diem': 1,               # giờ cao điểm
    'thoi_gian_chuan_bi_phut': 60    # chuẩn bị 60 phút
}

net_score, is_warning = predict_one(sample_test, model.w_, model.b_, X_mean, X_std)
print(f"\n=== CHẠY THỬ ĐƠN HÀNG MỚI ===")
print(f"Điểm nguy cơ giao trễ (net): {net_score:.4f}")
print(f"Quyết định                 : {'CẢNH BÁO TRỄ' if is_warning else 'BÌNH THƯỜNG'}")

# ==========================================
# VẼ ĐƯỜNG HỌC
# ==========================================
plt.figure(figsize=(8, 5))
# Trục x: epoch 1..30 | Trục y: MSE. Đường đi xuống và phẳng dần = mô hình hội tụ.
plt.plot(range(1, len(model.mse_train_) + 1), model.mse_train_, marker='o', label='MSE Train')
plt.plot(range(1, len(model.mse_val_) + 1), model.mse_val_, marker='s', label='MSE Test')
plt.title('Đường học Adaline (MSE theo Epoch) - Cảnh báo giao hàng trễ')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()