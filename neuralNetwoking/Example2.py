import numpy as np
# Hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Đạo hàm hàm sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Lớp neural network
class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # Mô hình layer ví dụ [2,2,1]
        self.layers = layers
        # Hệ số learning rate
        self.alpha = alpha
        # Tham số W, b
        self.W = []
        self.b = []
        # np.random.randn(layers[i], layers[i+1]): Đây là phần tạo ra ma trận trọng số (w_) cho layer thứ i. np.random.randn tạo ra các giá trị 
        # ngẫu nhiên từ phân phối chuẩn (phân phối Gaussian có trung bình 0 và độ lệch chuẩn 1). layers[i] là số lượng node ở layer thứ i, 
        # và layers[i+1] là số lượng node ở layer tiếp theo (i+1). Do đó, w_ sẽ có kích thước là (layers[i], layers[i+1]).

        # np.zeros((layers[i+1], 1)): Đây là phần tạo ra vector bias (b_) cho layer thứ i. np.zeros tạo ra một vector toàn số 0 với kích thước là
        #  (layers[i+1], 1). Số lượng phần tử trong vector này bằng với số lượng node ở layer tiếp theo.

        # self.W.append(w_/layers[i]) và self.b.append(b_): Trọng số w_ được chia cho layers[i]. Ý nghĩa của việc này là để điều chỉnh tỷ lệ trọng 
        # số ban đầu dựa trên số lượng node ở layer trước đó. Điều này giúp kiểm soát sự lan truyền ngược hiệu ứng gradient và có thể làm cho quá trình học hiệu quả hơn.
        
        # # Khởi tạo các tham số ở mỗi layer
        for i in range(0, len(layers)-1):
            w_ = np.random.randn(layers[i], layers[i+1])
            b_ = np.zeros((layers[i+1], 1))
            self.W.append(w_/layers[i])
            self.b.append(b_)

    #  Tóm tắt mô hình neural network
    def __repr__(self):
        return "Neural network [{}]".format("-".join(str(l) for l in self.layers))

    # Train mô hình với dữ liệu
    def fit_partial(self, x, y):
        A = [x]
        # # quá trình feedforward

        # out = A[-1]: Biến out được khởi tạo với giá trị của layer cuối cùng trong danh sách A, tức là giá trị đầu vào x. A[-1] là cách Python lấy phần tử cuối cùng của danh sách.
        out = A[-1]
        for i in range(0, len(self.layers) - 1):
            # out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T)): Đây là bước quan trọng của quá trình feedforward. Công thức này tính giá trị đầu ra của mỗi layer bằng cách sử dụng hàm sigmoid trên tổ hợp tuyến tính của giá trị đầu ra từ layer trước, trọng số (self.W[i]), và bias (self.b[i]).

            # np.dot(out, self.W[i]): Tích vô hướng giữa giá trị đầu ra từ layer trước (out) và ma trận trọng số của layer hiện tại (self.W[i]).
            # (self.b[i].T): Vector bias của layer hiện tại (self.b[i]), được chuyển vị để có cùng chiều dài với out.
            out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
            A.append(out)
        for i in A:
            print('A')
            print(i)
        for b in self.b:
            print('B')
            print(b)
        # quá trình backpropagation
        y = y.reshape(-1, 1)  # Correct indentation

        dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]
        dW = []
        db = []

        for i in reversed(range(0, len(self.layers)-1)):
            dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivative(A[i+1]))
            db_ = np.sum(dA[-1] * sigmoid_derivative(A[i+1]),axis=  0).reshape(-1, 1)
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i+1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)
        
        # # Đảo ngược dW, db
        dW = dW[::-1]
        db = db[::-1]

        # Gradient descent
        for i in range(0, len(self.layers)-1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]

    def fit(self, X, y, epochs=20, verbose=10):
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print("Epoch {}, loss {}".format(epoch, loss))
    # Dự đoán
    def predict(self, X):
        for i in range(0, len(self.layers) - 1):
            X = sigmoid(np.dot(X, self.W[i]) + (self.b[i].T))
        return X

    # Tính loss function
    def calculate_loss(self, X, y):
        y_predict = self.predict(X)
        return -(np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict)))


# Example usage:
# Define your X and y
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Reshape y to be a column vector
y = y.reshape(-1, 1)

# Create a NeuralNetwork object
nn = NeuralNetwork(layers=[2, 2, 1], alpha=0.1)

# Train the model
nn.fit(X, y, epochs=1000, verbose=100)

# Make predictions
predictions = nn.predict(X)
print("Predictions:", predictions)
