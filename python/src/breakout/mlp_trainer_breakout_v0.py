import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from skimage.color import rgb2gray

class MlpAgent(nn.Module):
    def __init__(self, **kwargs):
        super(MlpAgent, self).__init__()

        self.layer0 = nn.Sequential(

            nn.Linear(28000, 10000),
            nn.ReLU(True),
            # nn.Dropout(0.2),

            nn.Linear(10000, 1000),
            nn.ReLU(True),
            # nn.Dropout(0.2),

            nn.Linear(1000, 128),
            nn.ReLU(True),
            # nn.Dropout(0.2),

            nn.Linear(128, 3),
            nn.ReLU(True),
            # nn.Dropout(0.2),

            nn.Softmax()

        )

    def forward(self, x):
        x = self.layer0(x)
        return x

def init_model():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("# Init model with device=" + str(device))

    model = MlpAgent().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.MSELoss()

    return model, optimizer, criterion, device

def train_step(model, optimizer, criterion, device, t_obs, t_reward):

    if t_reward == 0:
        t_reward = 0.001

    t_obs = rgb2gray(t_obs)

    t_obs = t_obs[25:200, :]

    # plt.imshow(t_obs)
    # plt.show()

    c_loss = (1/t_reward) * 1000000
    c_loss = torch.tensor(c_loss, requires_grad=True)
    c_input = torch.Tensor(t_obs.reshape(-1)).float().to(device)

    # print(c_loss)

    c_loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    n_step = model(c_input)
    n_step = n_step.detach().numpy()

    n_step = np.argmax(n_step, axis=0)

    next_step = n_step

    if next_step == 0:
        return 0
    elif next_step == 1:
        return 2
    elif next_step == 2:
        return 3
    else:
        print("ERROR: Wrong return value")
        exit(-1)
