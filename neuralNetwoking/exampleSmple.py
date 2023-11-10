import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Hàm để khởi tạo trọng số mạng
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
    weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1
    return weights_input_hidden, weights_hidden_output

# Hàm để huấn luyện mạng neural network
def train_neural_network(X, y, epochs, learning_rate):
    input_size = X.shape[1]
    hidden_size = 4  # Số lượng node trong hidden layer
    output_size = 1

    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        predicted_output = sigmoid(output_layer_input)

        # Tính lỗi
        error = y - predicted_output

        # Backpropagation
        output_error = error * sigmoid_derivative(predicted_output)
        hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

        # Cập nhật trọng số
        weights_hidden_output += hidden_layer_output.T.dot(output_error) * learning_rate
        weights_input_hidden += X.T.dot(hidden_layer_error) * learning_rate

        # In ra giá trị lỗi sau mỗi epoch
        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

    return weights_input_hidden, weights_hidden_output

# Dữ liệu đầu vào và đầu ra mẫu
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Huấn luyện mô hình
trained_weights_input_hidden, trained_weights_hidden_output = train_neural_network(X, y, epochs=60000, learning_rate=0.1)

# Kiểm thử mô hình
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
hidden_layer_input_test = np.dot(test_data, trained_weights_input_hidden)
hidden_layer_output_test = sigmoid(hidden_layer_input_test)
output_layer_input_test = np.dot(hidden_layer_output_test, trained_weights_hidden_output)
predicted_output_test = sigmoid(output_layer_input_test)

print("Predictions after training:")
print(predicted_output_test)
