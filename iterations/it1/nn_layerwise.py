import numpy as np

class Layer():
    """Class representing the layers that make up a neural network"""
    
    def __init__(self, layer_size: int, is_input_layer: bool):
        """Initializes an instance of the Layer class"""
        
        self.is_input_layer = is_input_layer
        self.layer_size = layer_size

        self.weighted_inputs = None
        self.activation_values = None
        self.der_node_values = None

        self.weights = None
        self.biases = None

    def init_weights(self, min, max, weights_count, method="He"):
        """Initializes the weights for current layer"""

        if method=="He": # He initialization specifically for ReLU
            std_dev = np.sqrt(2 / weights_count)
            self.weights = np.random.standard_normal((self.layer_size, weights_count)) * std_dev
            
            self.biases = np.zeros(self.layer_size)
        else: # Initialization by uniformly sampling between minimum and maximum value
            self.weights = np.random.uniform(min, max, (self.layer_size, weights_count))
            self.biases = np.random.uniform(min, max, self.layer_size)


class NeuralNetwork():
    """Class representing the neural network containing layer objects"""

    def __init__(self, layer_dimensions: list):
        """Initialize an instance of the Neural Network class"""
        
        self.layers = []
        self.best_val_mse = 0.0
        self.training_epochs = 0

        self.weight_min = -1.0
        self.weight_max = 1.0

        self.loss_history = []

        # Initialize layers
        for i in range(len(layer_dimensions)):
            is_input_layer = i == 0
            self.layers.append(Layer(layer_dimensions[i], is_input_layer))

        # Initialize weights
        prev_layer_size = 0
        for layer in self.layers:
            if not layer.is_input_layer:
                layer.init_weights(self.weight_min, self.weight_max, prev_layer_size)

            prev_layer_size = layer.layer_size

    def activation(self, inputs : np.ndarray) -> np.ndarray:
        """The leaky Rectified Linear Unit function acts as our activation function"""

        return np.where(inputs < 0, 0.01*inputs, inputs)

    
    def activation_derivative(self, inputs : np.ndarray) -> np.ndarray:
        """The derivative of the leaky Rectified Linear Unit function"""

        return np.where(inputs < 0, 0.01, 1)
    
    def loss(self, prediction : float, real_value : float) -> float:
        """Mean squared error acts as our loss function"""

        return (prediction - real_value)**2
    
    def loss_derivative(self, predictions : np.ndarray, real_value : float) -> np.ndarray:
        """The derivative of the mean squared error function"""

        return 2*(predictions - real_value)
    
    def forward_pass(self, input_values : np.ndarray) -> np.ndarray:
        """Pass the input values (a feature vector) through all layers of the neural network while storing all relevant computed values in the corresponding layers for backpropagation"""

        activation_values = input_values

        for layer in self.layers:
            if layer.is_input_layer:
                # Save the values of the input vector as the activation values of the input layer
                layer.activation_values = activation_values
            else:
                # Calculate weighted inputs of current layer
                wi = layer.weights @ activation_values + layer.biases
                # Save weighted inputs for backpropagation
                layer.weighted_inputs = wi

                # Calculate activation values by running the weighted inputs through our activation function
                if layer == self.layers[-1]: # Do not run final output through the activation function, as this would skew results
                    activation_values = wi
                else:
                    activation_values = self.activation(wi)
                # Save activation value for backpropagation
                layer.activation_values = activation_values
        
        return self.layers[-1].activation_values

    def backpropagation(self, input_values : list, real_value : float, learning_rate : float, lam : float):
        """Implementation of the backpropagation algorithm"""
        
        # Start with a forward pass to generate all the required values for the backward pass
        self.forward_pass(input_values)

        # Go backwards through the layers, starting at the output layer
        for i in range(len(self.layers))[1:][::-1]: # do not include input layer, as it has no weights
            cur_layer = self.layers[i]

            # Get the derivative of the weighted inputs with respect to the weights (-> the activation values of the previous layer)
            der_wi_wrt_weights = self.layers[i-1].activation_values

            # Calculate gradients and update weights of each layer based on them
            if i == len(self.layers) - 1:
                # The following two partial derivatives are only calculated at the output layer and reused in future layers
                der_cost_wrt_act = self.loss_derivative(cur_layer.activation_values, real_value)
                der_act_wrt_wi = self.activation_derivative(cur_layer.weighted_inputs)
                
                cur_layer.der_node_values = der_cost_wrt_act * der_act_wrt_wi
            else:
                der_act_wrt_wi = self.activation_derivative(cur_layer.weighted_inputs)

                # Re-use node values
                next_layer = self.layers[i+1]
                der_wi_wrt_act = np.sum(next_layer.der_node_values[:, None] * next_layer.weights, axis=0)

                # Save new node values
                cur_layer.der_node_values = der_wi_wrt_act * der_act_wrt_wi

            # Calculate the derivative of the weighted inputs with respect to the bias (i.e. the bias gradient)
            bias_gradient = 1 * cur_layer.der_node_values
            
            # Now calculate the weight gradients
            gradients = np.outer(cur_layer.der_node_values, der_wi_wrt_weights)
            
            # Clip the gradients to prevent overflow
            np.clip(gradients, -1e3, 1e3, out=gradients)

            # And finally update the weights using the gradient
            self.update_weights(cur_layer, gradients, bias_gradient, learning_rate, lam)

    def update_weights(self, layer : Layer, weights_gradients : np.ndarray, bias_gradient : np.ndarray, learning_rate : float, lam : float):
        """Updates the given layers weights and biases using the given gradients"""
        
        layer.weights -= learning_rate * (weights_gradients + lam * layer.weights)
        layer.biases = layer.biases - bias_gradient * learning_rate

    def get_weights(self) -> list:
        """Returns the weights of the current state of the neural network"""
        
        weights = []
        for i in range(len(self.layers)):
            weights.append(self.layers[i].weights)

        return weights

    def set_weights(self, weights):
        """Sets the weights of the neural network to a given set of weigths"""
        
        for i in range(len(self.layers)):
            self.layers[i].weights = weights[i]

    def get_full_mse(self, X : np.ndarray, y : np.ndarray) -> float:
        """Returns the average MSE for a set of training examples"""
        
        # Get predictions of the model for all given input data
        predictions = []
        for j in range(len(X)):
            prediction = self.forward_pass(X[j])[0]
            predictions.append(prediction)
        predictions = np.array(predictions)

        # Calculate loss (MSE)
        mse = np.mean((predictions - y) ** 2)

        return mse

    def train(self, X : np.ndarray, y : np.ndarray, epochs : int, learning_rate : float, X_val : np.ndarray, y_val : np.ndarray, 
              patience : int, max_epochs : int, lam : float):
        """Training function that performs backward propagation for a specified or unspeciefied number of epochs"""

        if epochs == 0:
            # Unspecified number of epochs -> early stopping + patience
            best_model = None
            current_model = None
            patience_counter = 0
            for i in range(max_epochs + 10): # is never reached because we stop the loop when max_epochs is reached
                # Train the model for one epoch
                for j in range(len(X)):
                    self.backpropagation(X[j], y[j], learning_rate, lam)

                # Get mse of the model for the validation set
                current_model = [self.get_full_mse(X_val, y_val), self.get_weights()]

                # Keep track of loss
                self.loss_history.append(current_model[0])

                # Keep track of whether or not the model improved compared to the best model
                if i == 0:
                    best_model = current_model
                else:
                    # If there was improvement of the current model compared to our best model, reset the patience counter, otherwise increase it by 1
                    if (best_model[0] - current_model[0] > 0): 
                        best_model = current_model
                        patience_counter = 0
                    else:
                        patience_counter += 1
                
                # Stop training if validation loss doesn't improve for a certain number of epochs (patience) or a specified maximum number of epochs is reached
                if (patience_counter >= patience or i >= max_epochs):
                    # Keep track of the best MSE value achieved during training of this model and after how many epochs that value was achieved
                    self.best_val_mse = best_model[0]
                    self.training_epochs = i-patience

                    # Set the weights of the model back to that of the best version
                    self.set_weights(best_model[1])

                    # Stop training
                    break
        else:
            # Specified number of epochs
            for i in range(epochs):
                se = 0
                
                for j in range(len(X)):
                    self.backpropagation(X[j], y[j], learning_rate, lam)

                    # Calculate MSE (assumes there is just one output neuron)
                    se += self.loss(self.layers[-1].activation_values[0], y[j])

                mse = se / len(X)

                self.loss_history.append(se)

                # if (i % 20 == 0): print(f"MSE at epoch {i}: {se}") # Print SE to get value more easier to interpret by eye

            self.best_val_mse = mse
            self.training_epochs = epochs

