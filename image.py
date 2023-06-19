import multiprocessing
from tqdm import tqdm

def process_block(imageData, imageGrayscale, imageBlockRGB, i, j, blockDimension, isThisRGBImage):
    imageBlockGrayscale = imageData.crop((i, j, i + blockDimension, j + blockDimension))
    imageBlock = Blocks.Blocks(imageBlockGrayscale, imageBlockRGB, i, j, blockDimension)
    return imageBlock.computeBlock()

def process_image(imageData, imageGrayscale, blockDimension, isThisRGBImage):
    imageWidthOverlap = imageData.width - blockDimension
    imageHeightOverlap = imageData.height - blockDimension

    featuresContainer = []

    if isThisRGBImage:
        imageBlocksRGB = [
            imageData.crop((i, j, i + blockDimension, j + blockDimension))
            for i in range(0, imageWidthOverlap + 1)
            for j in range(0, imageHeightOverlap + 1)
        ]
    else:
        imageBlocksRGB = [None] * ((imageWidthOverlap + 1) * (imageHeightOverlap + 1))

    with multiprocessing.Pool() as pool:
        for i, j in tqdm([(i, j) for i in range(0, imageWidthOverlap + 1) for j in range(0, imageHeightOverlap + 1)], ncols=80):
            features = pool.apply_async(process_block, (imageData, imageGrayscale, imageBlocksRGB[i*(imageHeightOverlap+1)+j], i, j, blockDimension, isThisRGBImage))
            featuresContainer.append(features.get())

    return featuresContainer
