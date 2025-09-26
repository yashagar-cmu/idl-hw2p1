import numpy as np


class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        self.index_mapper = lambda i : i*self.upsampling_factor

    def get_indices(self, in_shape):
        original_indices = np.arange(in_shape)
        final_indices = self.index_mapper(original_indices).astype(int)
        return final_indices

    def forward(self, A: np.ndarray):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        bs, inc, w_in = A.shape
        w_out = self.upsampling_factor*(w_in-1) + 1

        Z = np.zeros((bs, inc, w_out), dtype = A.dtype)
        Z[..., self.get_indices(w_in)] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        return dLdZ[..., ::self.upsampling_factor]


class Downsample1d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.w_in = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        _, _, self.w_in = A.shape

        # w_out = w_in//self.downsampling_factor + (1 if w_in%self.downsampling_factor!=0 else 1)

        Z = A[..., ::self.downsampling_factor]
        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        assert self.w_in is not None
        bs, inc, _ = dLdZ.shape

        upsampled = Upsample1d(self.downsampling_factor).forward(dLdZ)

        dLdA = np.zeros((bs, inc, self.w_in), dtype=dLdZ.dtype)
        dLdA[..., :upsampled.shape[-1]] = upsampled
        return dLdA

class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        self.up1d = Upsample1d(upsampling_factor)

    def forward(self, A: np.ndarray):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        bs, inc, hin, win = A.shape

        hout = self.upsampling_factor*(hin-1) + 1
        wout = self.upsampling_factor*(win-1) + 1

        Z = np.zeros(shape=(bs, inc, hout, wout), dtype = A.dtype)
        print(f"{self.up1d.get_indices(hin)=}, {self.up1d.get_indices(win)=}")
        Z[..., self.up1d.get_indices(hin)[:, None], self.up1d.get_indices(win)[None, :]] = A
        
        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        return dLdZ[..., ::self.upsampling_factor, ::self.upsampling_factor]  # TODO


class Downsample2d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.hin = None
        self.win = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        self.hin, self.win = A.shape[-2:]
        return A[..., ::self.downsampling_factor, ::self.downsampling_factor]

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        assert self.hin is not None and self.win is not None

        bs, inc, _, _ = dLdZ.shape

        dLdA = np.zeros(shape=(bs, inc, self.hin, self.win), dtype = dLdZ.dtype)

        upped = Upsample2d(self.downsampling_factor).forward(dLdZ)
        dLdA[..., :upped.shape[-2], :upped.shape[-1]] = upped

        return dLdA
