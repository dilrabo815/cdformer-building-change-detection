import numpy as np

class Tiler:
    """
    Utility to split large images into smaller overlapping patches for inference.
    Reassembles them dynamically using a Gaussian weighting scheme to eliminate block-border artifacts.
    """
    def __init__(self, image_shape, tile_size=256, overlap=64):
        self.image_shape = image_shape
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        
        self.h, self.w = image_shape[:2]
        self.tiles = []
        
        for y in range(0, self.h, self.stride):
            for x in range(0, self.w, self.stride):
                y1 = min(y, self.h - self.tile_size)
                x1 = min(x, self.w - self.tile_size)
                if y1 < 0: y1 = 0
                if x1 < 0: x1 = 0
                y2 = min(y1 + self.tile_size, self.h)
                x2 = min(x1 + self.tile_size, self.w)
                self.tiles.append((x1, y1, x2, y2))
                
        # To merge tiles back smoothly
        self.weight_mask = self._create_weight_mask((self.tile_size, self.tile_size))
        self.sum_preds = np.zeros((self.h, self.w), dtype=np.float32)
        self.sum_weights = np.zeros((self.h, self.w), dtype=np.float32)

    def _create_weight_mask(self, shape):
        # 2D Gaussian mask
        x = np.linspace(-1, 1, shape[1])
        y = np.linspace(-1, 1, shape[0])
        x, y = np.meshgrid(x, y)
        d = np.sqrt(x*x + y*y)
        sigma, mu = 0.5, 0.0
        weight = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        # Normalize to max 1
        weight = weight / np.max(weight)
        return weight

    def get_tiles_coords(self):
        return self.tiles

    def crop(self, image, box):
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]

    def add_prediction(self, box, pred_tile):
        x1, y1, x2, y2 = box
        th, tw = pred_tile.shape
        self.sum_preds[y1:y1+th, x1:x1+tw] += pred_tile * self.weight_mask[:th, :tw]
        self.sum_weights[y1:y1+th, x1:x1+tw] += self.weight_mask[:th, :tw]

    def reassemble(self):
        # Prevent division by zero
        safe_weights = np.where(self.sum_weights == 0, 1.0, self.sum_weights)
        final_prediction = self.sum_preds / safe_weights
        return final_prediction
