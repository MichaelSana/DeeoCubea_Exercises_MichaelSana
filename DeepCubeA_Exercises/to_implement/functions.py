from typing import List
from numpy import testing
from torch import nn, tensor
import numpy as np
from torch._C import device
from environments.environment_abstract import Environment, State
import torch



class nnet_model(nn.Module):
    def __init__(self):
        super(nnet_model, self).__init__()
        self.linearLayer1 = nn.Linear(81, 40)
        self.reluLayer1 = nn.ReLU()
        self.linearLayer2 = nn.Linear(40, 21)
        self.reluLayer2 = nn.ReLU()
        self.linearLayer3 = nn.Linear(21, 1)
        self.reluLayer3 = nn.ReLU()

    def forward(self, x):
        x = self.linearLayer1(x)
        x = self.reluLayer1(x)
        x = self.linearLayer2(x)
        x = self.reluLayer2(x)
        x = self.linearLayer3(x)
        x = self.reluLayer3(x)
        return x
def get_nnet_model() -> nn.Module:
    """ Get the neural network model
    @return: neural network model
    """
    return nnet_model()
pass

def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray, batch_size: int, num_itrs: int,
               train_itr: int):
    
    #Create tensors for inputs and outputs
    input_tensor = tensor(states_nnet)
    target = tensor(outputs)

    #Define criterion function
    criterion = nn.MSELoss()

    #Establish batch size
    batchsize = 100

    #Make tensors of float value
    input_tensor = torch.from_numpy(states_nnet).float()
    target = torch.from_numpy(outputs).float()

    #Define optimizer function
    optimizer = torch.optim.SGD(nnet.parameters(), lr = 0.01)

    #Create dataLoader
    data = torch.utils.data.TensorDataset(input_tensor, target)
    data_loader = torch.utils.data.DataLoader(data, batch_size = batchsize, shuffle = True, num_workers = 4)

    #training loop
    for epoch, data in enumerate(data_loader, 0):

                #Slip information in data variable
                inputs, outputs = data

                #Get the nnet output for the inputs variable
                nnetOutput = nnet(inputs)
                optimizer.zero_grad()

                #Plug in the loss function
                loss = criterion(nnetOutput, outputs)
                loss.backward()
                optimizer.step()  
pass


def value_iteration(nnet, device, env: Environment, states: List[State]) -> List[float]:
    env = nnet

    qtable = np.random.rand()

    epochs = 100
    epsilon = 0.08
    done = False
    for i in range(epochs):
        #state, reward, done = env.reset()
        steps = 0

        while not done:
            
            print('epoch #', i+1, epochs)
            
            steps +=1
            states = tensor(states).float()
            if np.random.uniform() <epsilon:
                action = env.randomAction()

            else:
                action = qtable[states].index(max(qtable[states]))
            
            next_state, reward, done = env.step(action)

            qtable[states][action] = reward + 0.1 * max(qtable[next_state])

            state = next_state

            epsilon -= 0.1 * epsilon

            print("\nDone in", steps, "steps".format(steps))



    pass
