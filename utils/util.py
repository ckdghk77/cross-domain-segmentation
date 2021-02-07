import torch
import numpy as np
import torch.nn.functional as F


def resize(input, new_shape) :

    if len(input.shape) == 2 :
        input = input.unsqueeze(0).unsqueeze(0)
    elif len(input.shape) == 3 :
        input = input.unsqueeze(0)

    reshape_input = F.interpolate(input,
                  size=new_shape,
                  mode='nearest').squeeze();

    return reshape_input

def save_ckpt(cur_itrs, model, optimizer, scheduler, best_score, path):
    """ save current model
    """
    torch.save({
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
    }, path)
    print("Model saved as %s" % path)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def add_prefix(inputs, prefix):
    """Add prefix for dict.
    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.
    Returns:
        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs

