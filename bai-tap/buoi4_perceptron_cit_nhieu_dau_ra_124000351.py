"""Bài tập mở rộng Buổi 4 — ba Perceptron song song cho C / I / T.

File này nối tiếp starter ``buoi4_perceptron_cit.py``.  Ở bài trước, một
Perceptron trả lời câu hỏi nhị phân ``có phải C không?``.  Ở đây, sinh viên
ghép ba node có cùng đầu vào nhưng có bộ trọng số và bias riêng:

    P_C(x), P_I(x), P_T(x) -> chọn node có score lớn nhất.

Hoàn thiện TODO 1–4 rồi chạy:

    python bai-tap/buoi4_perceptron_cit_nhieu_dau_ra.py

Các assert trong ``run_checks`` là chuẩn đối chiếu cho một epoch và cho bộ
trọng số sau khi huấn luyện hội tụ.  Không cần dùng PyTorch; chỉ cần NumPy.
"""

from __future__ import annotations

import numpy as np


LABELS = ("C", "I", "T")
X = np.array(
    [
        [1, 0, 1, 1, 0],  # C
        [0, 0, 0, 0, 1],  # I
        [0, 0, 1, 0, 1],  # T
    ],
    dtype=float,
)

# Mỗi hàng là mục tiêu one-vs-rest của một mẫu C/I/T.
Y = np.eye(len(LABELS), dtype=int)

ETA = 0.5
W0 = np.zeros((len(LABELS), X.shape[1]), dtype=float)
B0 = np.full(len(LABELS), -0.25, dtype=float)


def predict(
    x: np.ndarray, W: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    net = W @ x + b
    y_hat = (net >= 0).astype(int)
    return net, y_hat

    raise NotImplementedError("Hoàn thiện TODO 1 và TODO 2 trong predict().")


def update(
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dự đoán rồi cập nhật cả ba node theo quy tắc Perceptron.

    ``error`` có ba phần tử, một phần tử cho mỗi node.  Chỉ feature đang bật
    trong ``x`` mới làm hàng tương ứng của ``W`` thay đổi.
    """

    net, y_hat = predict(x, W, b)

    error = y - y_hat
    w_new = W + eta * error[:, None]* x
    b_new = b + eta * error
    return net, y_hat, error, w_new, b_new 
    raise NotImplementedError("Hoàn thiện TODO 3 và TODO 4 trong update().")


def train_one_epoch(
    eta: float = ETA,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Đi qua C, I, T đúng một lần và lưu lịch sử từng lượt."""

    W = W0.copy()
    b = B0.copy()
    history: list[dict[str, object]] = []

    for label, x_i, y_i in zip(LABELS, X, Y):
        net, y_hat, error, W, b = update(x_i, y_i, W, b, eta)
        history.append(
            {
                "mau": label,
                "net": net.copy(),
                "y_hat": y_hat.copy(),
                "error": error.copy(),
                "W_sau_luot": W.copy(),
                "b_sau_luot": b.copy(),
            }
        )

    return W, b, history


def fit(
    epochs: int = 20, eta: float = ETA
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Huấn luyện nhiều epoch, trả về W, b và số node sai mỗi epoch."""

    W = W0.copy()
    b = B0.copy()
    mistakes_per_epoch: list[int] = []

    for _ in range(epochs):
        mistakes = 0
        for x_i, y_i in zip(X, Y):
            net, y_hat = predict(x_i, W, b)
            error = y_i - y_hat
            if np.any(error != 0):
                mistakes += 1
            W = W + eta * error[:, None] * x_i
            b = b + eta * error
        mistakes_per_epoch.append(mistakes)
        if mistakes == 0:
            break

    return W, b, mistakes_per_epoch


def predict_class(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> str:
    """Chọn lớp có ``net`` lớn nhất (bước đọc output, không phải node mới)."""

    net, _ = predict(x, W, b)
    return LABELS[int(np.argmax(net))]


def run_checks() -> None:
    """Kiểm tra bảng tính tay và phần mở rộng nhiều đầu ra."""

    W_epoch, b_epoch, history = train_one_epoch()
    assert len(history) == 3, "Một epoch phải đi qua đủ C, I, T."
    assert np.array_equal(history[0]["error"], [1, 0, 0])
    assert np.array_equal(history[1]["error"], [-1, 1, 0])
    assert np.array_equal(history[2]["error"], [0, -1, 1])
    assert np.allclose(
        W_epoch,
        [
            [0.5, 0.0, 0.5, 0.5, -0.5],
            [0.0, 0.0, -0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.5],
        ],
    )
    assert np.allclose(b_epoch, [-0.25, -0.25, 0.25])

    W_final, b_final, mistakes = fit()
    assert mistakes[-1] == 0, "Mạng chưa có epoch nào không còn node sai."
    assert np.array_equal(
        [predict_class(x_i, W_final, b_final) for x_i in X],
        np.array(LABELS),
    )
    assert mistakes == [3, 3, 2, 1, 0], (
        "Lịch sử số mẫu sai thay đổi; hãy kiểm tra thứ tự C/I/T và quy tắc cập nhật."
    )


if __name__ == "__main__":
    run_checks()
    W_final, b_final, mistakes = fit()
    print("Số mẫu sai theo epoch:", mistakes)
    print("W cuối:\n", W_final)
    print("b cuối:", b_final)
    print("Dự đoán:", [predict_class(x_i, W_final, b_final) for x_i in X])
    print("Đã vượt qua toàn bộ kiểm tra mở rộng Buổi 4.")