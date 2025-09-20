import numpy as np

class Layer():
    """Class representing the layers that make up a neural network
    
    Attributes
    ----------
    is_input_layer : bool
        Whether or not the layer is the input layer
    layer_size : int
        Number of nodes in the layer
    weighted_inputs: np.ndarray
        The weighted inputs of all neurons in this layer, before activation - determined during forward propagation
    activation_values : np.ndarray
        The activation values of all neurons in this layer - determined during forward propagation
    self.der_node_values : np.ndarray
        The products of partial derivatives relevant to the calculation of gradients - determined during backward propagation
    weights : np.ndarray
        The weights between the neurons of the previous layer and the neurons of this layer
    biases : np.ndarray
        The biases of the neurons of this layer

    Methods
    -------
    init_weights(self, min, max, weights_count, method="He")
        Initializes the weights for current layer
    """
    
    def __init__(self, layer_size: int, is_input_layer: bool):
        """Initializes an instance of the Layer class

        Parameters
        ----------
        layer_size : int
            Number of nodes this layer should contain
        is_input_layer : bool
            Whether or not the layer is the input layer
        """
        
        self.is_input_layer = is_input_layer
        self.layer_size = layer_size

        self.weighted_inputs = None
        self.activation_values = None
        self.der_node_values = None

        self.weights = None
        self.biases = None

    def init_weights(self, min : float, max : float, weights_count : int, method : str ="He"):
        """Initializes the weights for current layer

        If not specified otherwise, He initialization is used.

        Parameters
        ----------
        min : float
            The minimal value a weight is able to take during uniform weight initialization
        max : float
            The maximal value a weight is able to take during uniform weight initialization
        weights_count : int
            How many weights should be created per neuron in this layer
        method : str
            Indicates which weight initialization should be used. Default is 'He'.
        """

        if method=="He": # He initialization specifically for ReLU
            std_dev = np.sqrt(2 / weights_count)
            self.weights = np.random.standard_normal((self.layer_size, weights_count)) * std_dev
            
            self.biases = np.zeros(self.layer_size)
        else: # Initialization by uniformly sampling between minimum and maximum value
            self.weights = np.random.uniform(min, max, (self.layer_size, weights_count))
            self.biases = np.random.uniform(min, max, self.layer_size)


class NeuralNetwork():
    """Class representing the neural network containing layer objects
    
    This class implements all of the functions required for training our neural network, most notably backpropagation.

    Attributes
    ----------
    layers : list[Layer]
        List of layer objects associated with this neural network
    best_val_mse : float
        The best mean squared error value achieved during training of the model
    training_epochs : int
        Number of epochs the model was trained for
    weight_min : float
        The smallest value a weight should be able to take during uniform initialization
    weight_max : float
        The biggest value a weight should be able to take during uniform initialization

    Methods
    -------
    activation(self, inputs : np.ndarray) -> np.ndarray
        The leaky Rectified Linear Unit function acts as our activation function

    activation_derivative(self, inputs : np.ndarray) -> np.ndarray
        The derivative of the leaky Rectified Linear Unit function

    loss(self, prediction : float, real_value : float) -> float
        Mean squared error acts as our loss function

    loss_derivative(self, predictions : np.ndarray, real_value : float) -> np.ndarray
        The derivative of the mean squared error function
    
    forward_pass(self, input_values : np.ndarray) -> np.ndarray
        Pass the input values (a feature vector) through all layers of the neural network while storing all relevant computed values in the corresponding layers for backpropagation

    backpropagation(self, input_values : list, real_value : float, learning_rate : float, lam : float)
        Implementation of the backpropagation algorithm

    update_weights(self, layer : Layer, weights_gradients : np.ndarray, bias_gradient : np.ndarray, learning_rate : float, lam : float)
        Updates the given layers weights and biases using the given gradients

    get_weights(self) -> list
        Returns the weights of the current state of the neural network

    set_weights(self, weights)
        Sets the weights of the neural network to a given set of weigths

    get_full_mse(self, X : np.ndarray, y : np.ndarray) -> float
        Returns the average MSE for a set of training examples

    train(self, X : np.ndarray, y : np.ndarray, epochs : int, learning_rate : float, X_val : np.ndarray, y_val : np.ndarray, patience : int, max_epochs : int, lam : float)
        Training function that performs backward propagation for a specified or unspeciefied number of epochs
    """

    def __init__(self, layer_dimensions: list):
        """Initialize an instance of the Neural Network class

        Instantiates the layers based on the specified layer dimensions. Then initializes the weights associated with each layer.

        Parameters
        ----------
        layer_dimensions : list
            Indicates how many neurons each layer should contain
        """
        
        self.layers = []
        self.best_val_mse = 0.0
        self.training_epochs = 0

        self.weight_min = -1.0
        self.weight_max = 1.0

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
        """The leaky Rectified Linear Unit function acts as our activation function
        
        Parameters
        ----------
        inputs : np.ndarray
            The values that the function should be applied to

        Returns
        -------
        Returns a modified input array with all negative values multiplied by 0.01
        """

        return np.where(inputs < 0, 0.01*inputs, inputs)

    
    def activation_derivative(self, inputs : np.ndarray) -> np.ndarray:
        """The derivative of the leaky Rectified Linear Unit function
        
        Parameters
        ----------
        inputs : np.ndarray
            The values that the function should be applied to
        
        Returns
        -------
        Returns a modified input array where all negative values are set to 0.01 and all positive ones are set to 1
        """

        return np.where(inputs < 0, 0.01, 1)
    
    def loss(self, prediction : float, real_value : float) -> float:
        """Mean squared error acts as our loss function
        
        Parameters
        ----------
        prediction : float
            The predicted value for the current feature vector
        real_value : float
            The ground truth value corresponding to the current feature vector
        
        Return
        ------
        Returns the mean squared error
        """

        return (prediction - real_value)**2
    
    def loss_derivative(self, predictions : np.ndarray, real_value : float) -> np.ndarray:
        """The derivative of the mean squared error function
        
        Parameters
        ----------
        predictions : np.ndarray
            The predicted values for the current feature vector (if there are multiple output neurons)
        real_value : float
            The ground truth value corresponding to the current feature vector

        Return
        ------
        Returns the derivative of the mean squared error for all predictions
        """

        return 2*(predictions - real_value)
    
    def forward_pass(self, input_values : np.ndarray) -> np.ndarray:
        """Pass the input values (a feature vector) through all layers of the neural network while storing all relevant computed values in the corresponding layers for backpropagation

        Parameters
        ----------
        input_values : np.ndarray
            The feature vector of the current training example

        Returns
        -------
        activation_values : np.ndarray
            The activation values of the output neurons
        """

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
        """Implementation of the backpropagation algorithm

        Notation:
        der - derivative
        wrt - with respect to
        act - activation
        wi - weighted input

        Parameters
        ----------
        input_values : np.array
            Feature vector of the current training example
        real_value : float
            Ground truth target value of the current training example
        learning_rate : float
            The learning rate. Governs how big the change in the weights should be for one update
        lam : float
            Lambda. Governs how strongly weigths are regularized.
        """
        
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
        """Updates the given layers weights and biases using the given gradients

        Here, we implement L2 regularization to penalize large weights.
        
        Parameters
        ----------
        layer : Layer
            The layer whose weights should be updated
        weights_gradients : np.ndarray
            The weight gradients that should be used in updating the current layers weights
        bias_gradient : np.array
            The bias gradient that should be used in updating the current layers biases
        learning_rate : float
            The learning rate. Governs how big the change in the weights should be for one update.
        lam : float
            Lambda. Governs how strongly weigths are regularized.
        """
        
        layer.weights -= learning_rate * (weights_gradients + lam * layer.weights)
        layer.biases = layer.biases - bias_gradient * learning_rate

    def get_weights(self) -> list:
        """Returns the weights of the current state of the neural network
        
        Returns
        -------
        weights : list[np.array]
           The weights of the current state of the neural network
        """
        
        weights = []
        for i in range(len(self.layers)):
            weights.append(self.layers[i].weights)

        return weights

    def set_weights(self, weights):
        """Sets the weights of the neural network to a given set of weigths

        Parameters
        ----------
        weights : list[np.array]
            A list containing the weights for all layers in the neural network, ordered from first hidden layer to output layer
        """
        
        for i in range(len(self.layers)):
            self.layers[i].weights = weights[i]

    def get_full_mse(self, X : np.ndarray, y : np.ndarray) -> float:
        """Returns the average MSE for a set of training examples

        Parameters
        ----------
        X : np.ndarray
            The array of feature vectors
        y : np.array
            The array of ground truth values
        
        Returns
        -------
        mse : float
            The MSE for all predictions
        """
        
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
        """Training function that performs backward propagation for a specified or unspeciefied number of epochs

        Parameters
        ----------
        X : np.ndarray
            The array of feature vectors of all training examples
        y : np.ndarray
            The array of ground truths for all training examples
        epochs : int
            Number of epochs the model should be trained for. If 0, this indicates that the number of epochs should be chosen automatically, using early stopping and patience.
        learning_rate : float
            The learning rate. Governs how big the change in the weights should be for one update.
        X_val : np.ndarray
            The array of feature vectors of all validation examples
        y_val : np.ndarray
            The array of ground truths for all validation examples
        patience : int
            Determines the patience value, i.e. for how many epochs the model has to train without improvement to the best MSE value, before stopping
        max_epochs : int
            Determines the maximum amount of epochs the model should be trained for
        lam : float
            Lambda. Governs how strongly weigths are regularized.
        """

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
                    self.backpropagation(X[j], y[j], learning_rate)

                    # Calculate MSE (assumes there is just one output neuron)
                    se += self.loss(self.layers[-1].activation_values[0], y[j])

                mse = se / len(X)

                # if (i % 20 == 0): print(f"MSE at epoch {i}: {se}") # Print SE to get value more easier to interpret by eye

            self.best_val_mse = mse
            self.training_epochs = epochs

