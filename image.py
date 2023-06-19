def compute(self):
    """
    To compute the characteristic features of image block
    :return: None
    """
    print("Step 2 of 4: Computing feature vectors")

    image_width_overlap = self.image_width - self.block_dimension
    image_height_overlap = self.image_height - self.block_dimension

    if self.is_this_rgb_image:
        for i in range(0, image_width_overlap + 1):
            for j in range(0, image_height_overlap + 1):
                image_block_rgb = self.image_data.crop((i, j, i + self.block_dimension, j + self.block_dimension))
                image_block_grayscale = self.image_grayscale.crop((i, j, i + self.block_dimension, j + self.block_dimension))
                image_block = Blocks.Blocks(image_block_grayscale, image_block_rgb, i, j, self.block_dimension)
                self.features_container.addBlock(image_block.computeBlock())
    else:
        for i in range(image_width_overlap + 1):
            for j in range(image_height_overlap + 1):
                image_block_grayscale = self.image_data.crop((i, j, i + self.block_dimension, j + self.block_dimension))
                image_block = Blocks.Blocks(image_block_grayscale, None, i, j, self.block_dimension)
                self.features_container.addBlock(image_block.computeBlock())
