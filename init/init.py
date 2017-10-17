"""
Here are defined the basic methods to initialize a CNN in a data driven way.
For initializing complex architecture or using more articulated stuff (e.g LDA
has two functions) one should implement his own init function.
"""

# Utils
import logging

import numpy as np
import torch
from sklearn.feature_extraction.image import extract_patches_2d

# Init tools
import util.lda as lda


def get_patches(X, y, kernel_size):
    all_patches, labels = [], []
    for image, label in zip(X, y):
        image = np.transpose(image, axes=[1, 2, 0])
        patches = extract_patches_2d(image, kernel_size, max_patches=0.5)
        all_patches.append(np.transpose(patches, axes=[0, 3, 1, 2]))
        labels.append(np.repeat(label, len(patches)))

    # Flatten them
    all_patches = np.array([sample for minibatch in all_patches for sample in minibatch])
    labels = np.array([sample for minibatch in labels for sample in minibatch])
    return all_patches, labels


def init(model, data_loader, *args, **kwargs):
    """
    Initialize a standard CNN composed by convolutional layer followed by fully
    connected layers.
    :param model:
        the network to initialize
    :param data:
        the dataloader to take the data from
    :param args:
        parameters for the function. In particular:
            > num_points: integer
            specifies how many points should be used to compute the data-driven
            initialization
    :param kwargs:
        parameters for the function.
    :return:
        nothing, the parameters of the network are modified in place
    """

    # Collect initial data
    logging.debug('Collect initial data')
    X = []
    y = []
    for i, (input, target) in enumerate(data_loader, 1):
        X.append(torch.autograd.Variable(input))
        y.append(torch.autograd.Variable(target))
        if i * data_loader.batch_size >= kwargs['num_points']:
            break

    ###############################################################################################
    # Iterate over all layers
    logging.info('Iterate over all layers')
    for index, module in enumerate(model.children()):
        logging.info('Layer: {}'.format(index))

        #######################################################################
        # Get the patches of data
        logging.info('Get the patches of data')

        # Generate exhaustive patches for each input
        if index != len(list(model.children())) - 1:
            # Get kernel size of current layer
            kernel_size = module[0].kernel_size

            patches, labels = get_patches(
                np.array([element.data.numpy() for minibatch in X for element in minibatch]),
                np.squeeze(minibatches_to_matrix(y)),
                kernel_size=kernel_size
            )

        #######################################################################
        # Compute data-driven parameters
        logging.info('Compute data-driven parameters')
        if index != len(list(model.children())) - 1:
            logging.info('LDA Transform')
            W, B = lda.transform(
                X=patches.reshape(patches.shape[0], -1),
                y=labels
            )
        else:
            logging.info('LDA Discriminants')
            W, B = lda.discriminants(
                X=minibatches_to_matrix(X),
                y=np.squeeze(minibatches_to_matrix(y))
            )

        #######################################################################
        # Reshape / Crop the parameters matrix to the proper size
        if 'conv' in str(type(list(module.children())[0])):
            # The T belongs to the reshape operation! It is NOT transposing the input! It is necessary to select columns
            W = W.T.reshape(W.shape[0], module[0].in_channels, kernel_size[0], kernel_size[1])[:module[0].out_channels]
            B = B[:module[0].out_channels]
        else:
            W = W / max(np.max(np.abs(B)), np.max(np.abs(W)))
            B = B / max(np.max(np.abs(B)), np.max(np.abs(W)))

        # Assign parameters
        logging.info('Assign parameters')
        module[0].weight.data = torch.Tensor(W)
        module[0].bias.data = torch.Tensor(B)

        # If the layer is not convolutional then flatten the data because
        # we assume it is a fully connected one
        # if 'conv' not in str(type(list(model.children())[index+1][0])):
        if index == len(list(model.children())) - 1:
            logging.info('Flattening input')
            for i, minibatch in enumerate(X):
                X[i] = X[i].view(X[i].size(0), -1)

        # Forward pass
        logging.info('Forward pass')
        for i, minibatch in enumerate(X):
            X[i] = module(X[i])


def minibatches_to_matrix(X):
    """
    Flattens the a list of matrices of shape[[minibatch, dim_1, ..., dim_n], [minibatch, dim_1, ..., dim_n] ...] such
    that it becomes [minibatch * len(list), dim_1 * dim_2 ... *dim_n]
    :param X: list of matrices
    :return: flattened matrix
    """
    return np.array([sample.data.view(-1).numpy() for minibatch in X for sample in minibatch])


"""
def fuck_i_deleted_something_here():
    data = [item for item in train_loader]
    X, y = np.vstack([item[0].numpy() for item in data]), np.concatenate([item[1].numpy() for item in data])
    del data
    X, y = X[:num_points_lda], y[:num_points_lda]
    le.fit(y)

    # Compute LDA for 1st Conv Layer
    X_5x5 = X[:, :, 11 - 2:11 + 3, 11 - 2:11 + 3]
    X_flat = X_5x5.reshape(X_5x5.shape[0], np.prod(np.array(X_5x5.shape[1:])))
    L = transform(X_flat, le.transform(y)).T
    weights = model.conv1.weight.data.numpy()
    for i in range(len(weights)):
        weights[i] = L[:, i].reshape(3, 5, 5)
    model.conv1.weight.data = torch.from_numpy(weights).float()
    model.conv1.bias.data = torch.from_numpy(np.zeros(model.conv1.bias.data.size())).float()

    outputs = []
    for i in range(0, len(X), args.batch_size):
        outputs.append(model.ss(model.conv1(V(torch.from_numpy(X[i:i + args.batch_size]).float()))))
        # outputs.append([model(torch.autograd.Variable(torch.from_numpy(X[i:i + args.batch_size]).float()))[1][2]])

    X_3x3 = np.vstack([item.data.numpy() for item in outputs])[:, :, 3 - 1:3 + 2, 3 - 1:3 + 2]

    # X_3x3 = model.conv1(V(torch.from_numpy(X).float())).data.numpy()[:,:,3-1:3+2,3-1:3+2]
    X_flat = X_3x3.reshape(X_3x3.shape[0], np.prod(np.array(X_3x3.shape[1:])))
    L = transform(X_flat, le.transform(y)).T
    weights = model.conv2.weight.data.numpy()
    for i in range(len(weights)):
        weights[i] = L[:, i].reshape(24, 3, 3)
    model.conv2.weight.data = torch.from_numpy(weights).float()
    model.conv2.bias.data = torch.from_numpy(np.zeros(model.conv2.bias.data.size())).float()

    outputs = []
    for i in range(0, len(X), args.batch_size):
        outputs.append(
            model.ss(model.conv2(model.ss(model.conv1(V(torch.from_numpy(X[i:i + args.batch_size]).float()))))))
        # outputs.append([model(torch.autograd.Variable(torch.from_numpy(X[i:i + args.batch_size]).float()))[1][1]])
    X_3x3 = np.vstack([item.data.numpy() for item in outputs])  # [:,:,3-1:3+2,3-1:3+2]

    # X_3x3 = model.conv2(model.conv1(V(torch.from_numpy(X).float()))).data.numpy()
    X_flat = X_3x3.reshape(X_3x3.shape[0], np.prod(np.array(X_3x3.shape[1:])))
    L = transform(X_flat, le.transform(y)).T
    weights = model.conv3.weight.data.numpy()
    for i in range(len(weights)):
        weights[i] = L[:, i].reshape(48, 3, 3)
    model.conv3.weight.data = torch.from_numpy(weights).float()
    model.conv3.bias.data = torch.from_numpy(np.zeros(model.conv3.bias.data.size())).float()

    outputs = []
    for i in range(0, len(X), args.batch_size):
        outputs.append(model.ss(model.conv3(
            model.ss(model.conv2(model.ss(model.conv1(V(torch.from_numpy(X[i:i + args.batch_size]).float()))))))))
        # outputs.append([model(torch.autograd.Variable(torch.from_numpy(X[i:i + args.batch_size]).float()))[0][0]])

    X_fc = np.squeeze(np.vstack([item.data.numpy() for item in outputs]))
    # X_fc = model.fc(model.conv3(model.conv2(model.conv1(V(torch.from_numpy(X).float())))).view(len(X),-1)).data.numpy()

    W, C = discriminants(X_fc, le.transform(y))
    # import pdb; pdb.set_trace()
    W = W.T
    C = C.T
    weights = np.zeros(model.fc.weight.data.numpy().shape)
    bias = np.zeros(model.fc.bias.data.numpy().shape)

    for i in range(len(W)):
        weights[i] = W[i, :]
    for i in range(len(C)):
        bias[i] = C[i]
    model.fc.weight.data = torch.from_numpy(weights).float()
    model.fc.bias.data = torch.from_numpy(bias).float()

    print("LDA initialization took {} seconds".format(time.time() - t))

    return model
    """
