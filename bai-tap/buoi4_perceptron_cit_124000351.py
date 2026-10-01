"""Buổi 4 — Perceptron nhận biết chữ C từ feature C/I/T.

Quy trình học: dự đoán bằng tay -> hoàn thiện 3 TODO -> chạy kiểm tra -> giải thích.
File này là starter dành cho sinh viên, không kèm lời giải.
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
Y = np.array([1, 0, 0], dtype=int)  # C so với không-phải-C

ETA = 0.5
W0 = np.zeros(X.shape[1], dtype=float)
B0 = -0.25


def predict(x: np.ndarray, w: np.ndarray, b: float) -> tuple[float, int]:
    """Trả về net và nhãn dự đoán theo ngưỡng net >= 0."""
    net = float(np.dot(w, x) + b)
    y_hat = 1 if net >= 0 else 0
    return net, y_hat
    raise NotImplementedError("Hoàn thiện TODO 1 và TODO 2 trong predict().")


def update(
    x: np.ndarray,
    y: int,
    w: np.ndarray,
    b: float,
    eta: float,
) -> tuple[float, int, int, np.ndarray, float]:
    """Thực hiện một lượt dự đoán và cập nhật Perceptron."""
    net, y_hat = predict(x, w, b)

    error = int(y - y_hat)
    w_new = w + eta * error * x
    b_new = b + eta * error

    return net, y_hat, error, w_new, b_new

def train_one_epoch(eta: float = ETA) -> tuple[np.ndarray, float, list[dict[str, object]]]:
    """Chạy đúng một epoch theo thứ tự C, I, T và lưu lịch sử từng lượt."""
    w = W0.copy()
    b = B0
    history: list[dict[str, object]] = []

    for label, x_i, y_i in zip(LABELS, X, Y):
        net, y_hat, error, w, b = update(x_i, int(y_i), w, b, eta)
        history.append(
            {
                "mau": label,
                "net": net,
                "y_hat": y_hat,
                "error": error,
                "w_sau_luot": w.copy(),
                "b_sau_luot": b,
            }
        )

    return w, b, history


def run_checks() -> None:
    """Các kiểm tra chỉ chạy được sau khi hoàn thiện đủ ba TODO."""
    net, y_hat = predict(X[0], W0, B0)
    assert np.isclose(net, -0.25), "Kiểm tra lại phép tính net của mẫu C."
    assert y_hat == 0, "Kiểm tra lại điều kiện ngưỡng net >= 0."

    w_final, b_final, history = train_one_epoch()
    assert len(history) == 3, "Một epoch phải đi qua đủ C, I, T."
    assert [row["error"] for row in history] == [1, -1, 0], (
        "Sai số từng lượt chưa khớp bảng tính tay."
    )
    assert np.allclose(w_final, [0.5, 0.0, 0.5, 0.5, -0.5]), (
        "Trọng số sau một epoch chưa đúng."
    )
    assert np.isclose(b_final, -0.25), "Bias sau một epoch chưa đúng."


if __name__ == "__main__":
    run_checks()
    _, _, rows = train_one_epoch()
    for row in rows:
        print(row)
    print("Đã vượt qua toàn bộ kiểm tra Buổi 4.")