from data.load_mnist import load_data
from model.layers import Linear, Activation, BatchNormalization
from model.loss import SoftmaxCrossEntropy
from model.optimize import GradientDescent
from model.net import Sequential, DataLoader


# -------------------------------
# Load Dataset and Prerequisites
#---------------------------------
X_train, y_train, X_test, y_test = load_data(normalize=True)

input_dim = X_train.shape[-1]
hidden_dim = 128
output_dim = y_train.shape[-1]

epochs = 200
lr = 0.1
batch_size = 100


#---------------------------
# Instantiate Model Classes
#---------------------------
data_load = DataLoader(X_train, y_train, batch_size=batch_size, drop_last=True, shuffle=True)

models = Sequential([
    Linear(input_dim, hidden_dim),
    Activation('relu'),
    BatchNormalization(hidden_dim),
    Linear(hidden_dim, output_dim)
]) 
criterion = SoftmaxCrossEntropy()
optimize = GradientDescent(lr)


#---------------------
# Write Training Loop
#----------------------
def main():
    for ep in range(epochs):
        for batch_i, (X_batch, y_batch) in enumerate(data_load):
            # Execute Forward Pass
            output = models.forward(X_batch.T)

            # Calculate Loss Function
            loss = criterion.forward(output, y_batch.T)

            # Derive Loss and Initiate Backpropagation
            loss_grad = criterion.backward()

            # Propagate gradients backward
            grad = models.backward(loss_grad)

            # Store learnable parameters
            parameters = models.get_params()

            # Update parameters
            optimize.step(parameters)


        if ep % 20 == 0:
            print(f"Epochs {ep}     Loss {loss.round(4)}")


if __name__ == "__main__":
    main()