import numpy as np
import torch
import torch.nn.functional as F


def one_hot_encode(arr: np.ndarray, n_labels: int) -> np.ndarray:

    # Initialize the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.0

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


train_on_gpu = torch.cuda.is_available()


def predict(net: torch.nn.Module,
            char: str,
            h: tuple = None,
            top_k: int = None) -> tuple:
    """
    Given a character, predict the next character.
    Returns the predicted character and the hidden state.
    """

    # Tensor inputs
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if train_on_gpu:
        inputs = inputs.cuda()

    # Detach hidden state from history
    h = tuple([each.data for each in h])
    # Get the output of the model
    out, h = net(inputs, h)

    # Get the character probabilities
    p = F.softmax(out, dim=1).data
    if train_on_gpu:
        p = p.cpu()  # move to cpu

    # Get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # Select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p / p.sum())

    # Return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h


def sample(net: torch.nn.Module,
           size: int,
           prime: str = "Начало",
           top_k: int = None):

    if train_on_gpu:
        net.cuda()
    else:
        net.cpu()

    net.eval()  # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return "".join(chars)
