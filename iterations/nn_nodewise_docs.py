import numpy as np

class NodeNodeNN():
    """Class representing the individual nodes that make up a layer
    
    Attributes 
    ----------
    weighted_input: float
        The weighted input of the neuron, before activation- determined during forward propagation
    activation_value : float
        The activation value of the neuron - determined during forward propagation
    self.der_node_value : float
        The product of some partial derivatives relevant to the calculation of gradients - determined during backward propagation
    weights : np.array
        The weights between the neurons of the previous layer and this neuron
    weight_min : float
        The minimal value a weight is able to take during weight initialization
    weight_max : float
        The maximal value a weight is able to take during weight initialization        

    Methods
    -------
    init_weights(self, weights_count: int)
        Uniformly samples n=weights_count weights between weight_min and weight_max
    """

    def __init__(self):
        """Initializes an instance of the Node class"""

        self.weighted_input = 0.0
        self.activation_value = 0.0
        self.der_node_value = 0.0
        self.weights = []

        self.weight_min = -1.0
        self.weight_max = 1.0

    def init_weights(self, weights_count: int):
        """Uniformly samples n=weights_count weights between weight_min and weight_max
        
        Parameters
        ----------
        weights_count : int
            The number of weights required for this neuron. This depends on the size of the previous layer.
        """

        self.weights = np.random.uniform(self.weight_min, self.weight_max, weights_count)

class LayerNodeNN():
    """Class representing the layers that make up a neural network
    
    Attributes
    ----------
    is_input_layer : bool
        Whether or not the layer is the input layer
    layer_size : int
        Number of nodes in the layer
    nodes : list[Node]
        List of node objects associated with this layer of the neural network
    """

    def __init__(self, layer_size: int, is_input_layer: bool):
        """Initializes an instance of the Layer class and all the nodes associated with it

        Parameters
        ----------
        layer_size : int
            Number of nodes this layer should contain
        is_input_layer : bool
            Whether or not the layer is the input layer
        """

        self.is_input_layer = is_input_layer
        self.layer_size = layer_size
        self.nodes = []

        for _ in range(layer_size):
            self.nodes.append(NodeNodeNN())

class NeuralNetworkNodeNN():
    """Class representing the neural network, containing layer objects with node objects
    
    This class implements all of the functions required for training our neural network, most notably backpropagation.

    Attributes
    ----------
    layers : list[Layer]
        List of layer objects associated with this neural network
    best_val_mse : float
        The best mean squared error value achieved during training of the model
    training_epochs : int
        Number of epochs the model was trained for

    Methods
    -------
    activation(self, value: float) -> float
        Implements the Rectified Linear Unit function as our activation function

    activation_derivative(self, value: float) -> float
        The derivative of the Rectified Linear Unit function

    loss(self, prediction: float, value: float) -> float
        Mean squared error acts as our loss function

    loss_derivative(self, prediction: float, value: float) -> float
        The derivative of the mean squared error function

    forward_pass(self, input_values : list) -> float
        Pass the input values (a feature vector) through all layers of the neural network while storing all relevant computed values in the corresponding nodes

    backpropagation(self, input_values : list, real_value : float, learning_rate : float)
        Implementation of the backpropagation algorithm

    update_weights(self, node : Node, gradient : list, learning_rate : float)
        Takes the node whose weights should be updated, the gradient they should be updated with and the learning rate

    get_weights(self) -> dict
        Returns the weights of the current state of the neural network

    set_weights(self, weights : dict)
        Sets the weights of the neural network to a given set of weigths

    get_full_mse(self, X : list[list], y : list) -> float
        Returns the mean MSE for multiple training examples

    train(self, X : list[list], y : list, epochs : int, learning_rate : float, X_val : list[list], y_val : list, patience : int)
        Training function that performs backward propagation for a specified or unspeciefied number of epochs
    """

    def __init__(self, layer_dimensions: list):
        """Initialize an instance of the Neural Network class

        Instantiates the layers based on the specified layer dimensions, thereby also instantiating each layer's neurons.
        Initializes the weights associated with each hidden / output layer neuron.

        Parameters
        ----------
        layer_dimensions : list
            How many nodes each layer of the network should contain
        """

        self.layers = []
        self.best_val_mse = 0.0
        self.training_epochs = 0

        # Initialize layers
        for i in range(len(layer_dimensions)):
            is_input_layer = i == 0
            self.layers.append(LayerNodeNN(layer_dimensions[i], is_input_layer))

        # Initialize weights
        prev_layer_size = 0
        for layer in self.layers:
            if not layer.is_input_layer:

                for node in layer.nodes:
                    node.init_weights(prev_layer_size + 1) # + 1 for bias
            
            prev_layer_size = layer.layer_size
    

    def activation(self, value: float) -> float:
        """The Rectified Linear Unit function acts as our activation function
        
        Parameters
        ----------
        value : float
            The value that the function should be applied to

        Returns
        -------
        Returns either 0 if the value is negative, or the value itself if it is non-negative
        """

        if value < 0:
            return 0
        
        return value
    
    def activation_derivative(self, value: float) -> float:
        """The derivative of the Rectified Linear Unit function
        
        Parameters
        ----------
        value : float
            The value that the function should be applied to
        
        Returns
        -------
        Returns either 0 if the value is negative, or 1 if it is non-negative
        """

        if value < 0:
            return 0
        
        return 1

    def loss(self, prediction: float, value: float) -> float:
        """Mean squared error acts as our loss function
        
        Parameters
        ----------
        prediction : float
            The predicted value of the current training example
        value : float
            The ground truth value for the current training example
        
        Return
        ------
        Returns the mean squared error
        """

        return (prediction - value)**2
    
    def loss_derivative(self, prediction: float, value: float) -> float:
        """The derivative of the mean squared error function
        
        Parameters
        ----------
        prediction : float
            The predicted value of the current training example
        value : float
            The ground truth value for the current training example

        Return
        ------
        Returns the derivative of the mean squared error
        """

        return 2*(prediction - value)
    
    def forward_pass(self, input_values : list) -> float:
        """Pass the input values (a feature vector) through all layers of the neural network while storing all relevant computed values in the corresponding nodes

        Parameters
        ----------
        input_values : np.array
            The feature vector of the current training example

        Returns
        -------
        activation_value : float
            The activation value of the first output neuron (assumes there is only one, for regression).
        """

        activation_values = input_values

        for layer in self.layers:
            if layer.is_input_layer:
                # Save the feature values of the input layer nodes as their activation values
                for i in range(len(layer.nodes)):
                    layer.nodes[i].activation_value = activation_values[i]
            else:
                # Calculate weighted inputs of each neuron in current layer
                for i in range(len(layer.nodes)):
                    wi = layer.nodes[i].weights[:-1]@activation_values + layer.nodes[i].weights[-1] # Add the last weight seperately, as it denotes the bias

                    # Save weighted inputs for backpropagation
                    layer.nodes[i].weighted_input = wi

                # Calculate activation values by running the weighted inputs through our activation function
                activation_values = []
                for i in range(len(layer.nodes)):
                    activation_value = self.activation(layer.nodes[i].weighted_input)
                    activation_values.append(activation_value)

                    # Save activation value for backpropagation
                    layer.nodes[i].activation_value = activation_value
        
        return self.layers[-1].nodes[0].activation_value

    def backpropagation(self, input_values : list, real_value : float, learning_rate : float):
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
            The learning rate. Governs how big the change in the weights should be for one update.
        """

        # Start with a forward pass to generate all the required values for the backward pass
        self.forward_pass(input_values)

        # Go backwards through the layers, starting at the output layer
        for i in range(len(self.layers))[1:][::-1]: # start at first hidden layer to skip input layer as there are no weights there
            cur_layer = self.layers[i]

            # Calculate the derivative of the weighted input with respect to the weights (-> the activation value of each of the previous layer's nodes)
            der_wi_wrt_weights = []
            prev_layer = self.layers[i-1]
            prev_layer_nodes = prev_layer.nodes
            for k in range(len(prev_layer_nodes)):
                der_wi_wrt_weights.append(prev_layer_nodes[k].activation_value)
            der_wi_wrt_weights = np.array(der_wi_wrt_weights)

            # Calculate gradients and update weights of each node in current layer based on them
            for j in range(len(cur_layer.nodes)):
                cur_node = cur_layer.nodes[j]

                # The following two partial derivatives are only calculated at the output layer and reused in future layers
                if i == len(self.layers) - 1:
                    # Calculate the derivative of the cost with respect to the activation
                    der_cost_wrt_act = self.loss_derivative(cur_node.activation_value, real_value)

                    # Calculate the derivative of the activation with respect to the weighted input
                    der_act_wrt_wi = self.activation_derivative(cur_node.weighted_input)

                    cur_node.der_node_value = der_cost_wrt_act * der_act_wrt_wi
                else:
                    # Update the value for the "der_node_value" of each node in the current layer, using the node values from the subsequent layer
                    der_wi_wrt_act = 0
                    der_act_wrt_wi = self.activation_derivative(cur_node.weighted_input)

                    next_layer = self.layers[i+1]
                    next_layer_nodes = next_layer.nodes

                    for k in range(len(next_layer_nodes)):
                        cur_next_node = next_layer_nodes[k]

                        der_wi_wrt_act += cur_next_node.der_node_value * cur_next_node.weights[j]
                    
                    cur_node.der_node_value = der_wi_wrt_act * der_act_wrt_wi

                # Calculate the derivative of the weighted input with respect to the bias
                der_wi_wrt_bias = 1 * cur_node.der_node_value
                
                # Now calculate the full gradient
                gradient = der_wi_wrt_weights * (cur_node.der_node_value)
                gradient = np.append(gradient, der_wi_wrt_bias)

                # And finally update the weights using the gradient
                self.update_weights(cur_node, gradient, learning_rate)

    def update_weights(self, node : NodeNodeNN, gradient : list, learning_rate : float):
        """Takes the node whose weights should be updated, the gradient they should be updated with and the learning rate
        
        Parameters
        ----------
        node : Node
            The node whose weights should be updated
        gradient : np.array
            The gradient that should be used in updating the current nodes weights
        learning_rate : float
            The learning rate. Governs how big the change in the weights should be for one update.
        """

        node.weights = node.weights - gradient * learning_rate

    def get_weights(self) -> dict:
        """Returns the weights of the current state of the neural network
        
        Returns
        -------
        weights : dict[str, np.array]
           The weights of the current state of the neural network
        """

        weights = {}

        # Go through all layers and all nodes and save the weights of the nodes with the corresponding index
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].nodes)):
                weights[f"{i}{j}"] = self.layers[i].nodes[j].weights

        return weights

    def set_weights(self, weights : dict):
        """Sets the weights of the neural network to a given set of weigths

        Parameters
        ----------
        weights : dict[str, np.array]
            A dict containing the weights for all nodes in the neural network
        """

        # Go through all layers and all nodes and set the weights of the current node to the weights with the corresponding index
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].nodes)):
                self.layers[i].nodes[j].weights = weights[f"{i}{j}"]

    def get_full_mse(self, X : list[list], y : list) -> float:
        """Returns the mean MSE for multiple training examples

        Parameters
        ----------
        X : list[list]
            The array of feature vectors
        y : list
            The array of ground truth values
        
        Returns
        -------
        val_mse : float
            The average MSE across all examples
        """

        # Get predictions of the model for all given input data
        predictions = []
        for j in range(len(X)):
            prediction = self.forward_pass(X[j])
            predictions.append(prediction)
        predictions = np.array(predictions) # np.array for quicker loss calculation

        # Calculate loss (MSE)
        avg_mse = np.mean((predictions - y) ** 2)

        return avg_mse

    def train(self, X : list[list], y : list, epochs : int, learning_rate : float, X_val : list[list], y_val : list, patience : int):
        """Training function that performs backward propagation for a specified or unspeciefied number of epochs

        Parameters
        ----------
        X : list[np.array]
            The list of feature vectors of all training examples
        y : list
            The list of ground truths for all training examples
        epochs : int
            Number of epochs the model should be trained for. If 0, this indicates that the number of epochs should be chosen automatically, using early stopping and patience.
        learning_rate : float
            The learning rate. Governs how big the change in the weights should be for one update.
        X_val : list[np.array]
            The list of feature vectors of all validation examples
        y_val : list
            The list of ground truths for all validation examples
        patience : int
            Determines the patience value, i.e. for how many epochs the model has to train without improvement to the best MSE value, before stopping
        """

        if epochs == 0:
            # Unspecified number of epochs (early stopping + patience)
            best_model = None
            current_model = None
            improvements = []
            for i in range(10000):
                # Train the model for one epoch
                for j in range(len(X)):
                    self.backpropagation(X[j], y[j], learning_rate)

                # Get mse of the model for the validation set
                val_mse = self.get_full_mse(X_val, y_val)

                # Keep track of past mse values and weights of the model at that point for patience
                current_model = [val_mse, self.get_weights()]

                # Keep track of whether or not the model improved compared to the best model
                if i == 0:
                    best_model = current_model
                    improvements.append(True)
                else:
                    if (best_model[0] - current_model[0] > 0.5):
                        best_model = current_model
                        improvements.append(True)
                    else:
                        improvements.append(False)

                # Stop training if validation loss doesn't improve for a certain number of epochs (patience)
                if not any(improvements[-patience:]):
                    # Reset the weights of the current model to be that of the best model
                    self.set_weights(best_model[1])

                    # Keep track of the best MSE value achieved during training of this model and after how many epochs that value was achieved
                    self.best_val_mse = best_model[0]
                    self.training_epochs = i-patience

                    print(f"✅ Finished training for model with final MSE of {self.best_val_mse} in {i} epochs. Best MSE was achieved in epoch {self.training_epochs}")

                    # Stop training
                    break
        else:
            # Specified number of epochs
            for i in range(epochs):
                mse = 0
                
                for j in range(len(X)):
                    self.backpropagation(X[j], y[j], learning_rate)

                    # Calculate MSE (assumes that there is only one output neuron)
                    mse += self.loss(self.layers[-1].nodes[0].activation_value, y[j])

                mse /= len(X)

            self.best_val_mse = mse
            self.training_epochs = epochs



