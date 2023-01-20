def normalize_for_vgg(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div(255.0)
    return (batch - mean) / std


def gram_matrix_batch(batch):
    batch_size, channels, height, width = batch.shape
    features = batch.view(batch_size, channels, height * width)
    return features.bmm(features.transpose(1, 2)) / (channels * height * width)
