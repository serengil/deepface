# pylint: disable=unnecessary-pass


class ImgNotFound(ValueError):
    """Raised when the input image is not found or cannot be loaded."""

    pass


class PathNotFound(ValueError):
    """Raised when the input path is not found."""

    pass


class FaceNotDetected(ValueError):
    """Raised when no face is detected in the input image."""

    pass


class SpoofDetected(ValueError):
    """Raised when a spoofed face is detected in the input image."""

    pass


class EmptyDatasource(ValueError):
    """Raised when the provided data source is empty."""

    pass


class DimensionMismatchError(ValueError):
    """Raised when the dimensions of the input do not match the expected dimensions."""

    pass


class InvalidEmbeddingsShapeError(ValueError):
    """Raised when the shape of the embeddings is invalid."""

    pass


class DataTypeError(ValueError):
    """Raised when the input data type is incorrect."""

    pass


class UnimplementedError(ValueError):
    """Raised when a requested feature is not implemented."""

    pass


class DuplicateEntryError(ValueError):
    """Raised when a duplicate entry is found in the database."""

    pass
