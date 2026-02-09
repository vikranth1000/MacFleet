"""Tensor serialization utilities for MacFleet.

Uses raw bytes for efficient tensor transfer, not pickle or protobuf.
Protocol follows Section 9.2 of DESIGN.md:
- 16-byte header: msg_type (4B), dtype (4B), n_dims (4B), payload_size (4B)
- Shape: n_dims * 4 bytes
- Payload: raw tensor bytes
"""

import struct
from enum import IntEnum
from typing import Optional

import numpy as np
import torch


class MessageType(IntEnum):
    """Message type codes for tensor channel."""
    TENSOR_GRADIENT = 0x01
    TENSOR_WEIGHTS = 0x02
    TENSOR_ACTIVATIONS = 0x03
    COMPRESSED_GRADIENT = 0x04


# Mapping from torch dtype to integer code
DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.float64: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.int16: 5,
    torch.int8: 6,
    torch.uint8: 7,
    torch.bfloat16: 8,
}

CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}

# Mapping from torch dtype to numpy dtype
TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    # Note: bfloat16 needs special handling as numpy doesn't support it natively
}

# Header format: msg_type (I), dtype (I), n_dims (I), payload_size (I)
HEADER_FORMAT = "!IIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 16 bytes


def tensor_to_bytes(
    tensor: torch.Tensor,
    msg_type: MessageType = MessageType.TENSOR_GRADIENT,
) -> bytes:
    """Serialize a tensor to raw bytes.

    The tensor is first moved to CPU if necessary, then converted to
    contiguous numpy array and serialized with a header containing
    metadata for reconstruction.

    Args:
        tensor: PyTorch tensor to serialize.
        msg_type: Message type for the tensor channel protocol.

    Returns:
        Bytes containing header + shape + raw tensor data.
    """
    # Ensure tensor is on CPU and contiguous
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    tensor = tensor.contiguous()

    # Get tensor metadata
    dtype = tensor.dtype
    shape = tensor.shape
    n_dims = len(shape)

    # Handle bfloat16 specially - convert to float16 for transfer
    if dtype == torch.bfloat16:
        tensor = tensor.to(torch.float16)
        dtype = torch.float16

    # Convert to numpy and get raw bytes
    np_dtype = TORCH_TO_NUMPY.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    np_array = tensor.numpy()
    payload = np_array.tobytes()
    payload_size = len(payload)

    # Pack header
    dtype_code = DTYPE_TO_CODE[dtype]
    header = struct.pack(HEADER_FORMAT, msg_type, dtype_code, n_dims, payload_size)

    # Pack shape (each dimension as 4-byte unsigned int)
    shape_bytes = struct.pack(f"!{n_dims}I", *shape)

    return header + shape_bytes + payload


def bytes_to_tensor(
    data: bytes,
    device: Optional[str] = None,
) -> tuple[torch.Tensor, MessageType]:
    """Deserialize bytes to a tensor.

    Args:
        data: Bytes containing header + shape + raw tensor data.
        device: Target device for the tensor (e.g., "mps", "cpu").
                If None, tensor stays on CPU.

    Returns:
        Tuple of (tensor, message_type).
    """
    # Unpack header
    header = data[:HEADER_SIZE]
    msg_type, dtype_code, n_dims, payload_size = struct.unpack(HEADER_FORMAT, header)
    msg_type = MessageType(msg_type)

    # Unpack shape
    shape_start = HEADER_SIZE
    shape_end = shape_start + n_dims * 4
    shape = struct.unpack(f"!{n_dims}I", data[shape_start:shape_end])

    # Get payload
    payload_start = shape_end
    payload = data[payload_start:payload_start + payload_size]

    # Convert to tensor
    dtype = CODE_TO_DTYPE[dtype_code]
    np_dtype = TORCH_TO_NUMPY[dtype]
    np_array = np.frombuffer(payload, dtype=np_dtype).reshape(shape)

    # Create tensor (numpy array is a view, so copy for safety)
    tensor = torch.from_numpy(np_array.copy())

    # Move to target device if specified
    if device is not None and device != "cpu":
        tensor = tensor.to(device)

    return tensor, msg_type


def get_tensor_byte_size(tensor: torch.Tensor) -> int:
    """Get the serialized byte size of a tensor.

    Args:
        tensor: PyTorch tensor.

    Returns:
        Total byte size including header and shape.
    """
    n_dims = len(tensor.shape)
    element_size = tensor.element_size()
    payload_size = tensor.numel() * element_size
    return HEADER_SIZE + n_dims * 4 + payload_size


def serialize_compressed_gradient(
    indices: torch.Tensor,
    values: torch.Tensor,
    original_numel: int,
    original_dtype: torch.dtype,
) -> bytes:
    """Serialize a compressed gradient (Top-K sparse + FP16).

    Format from Section 9.2:
    - Header (16 bytes)
    - Compression metadata (12 bytes): orig_numel (4B), topk_count (4B), orig_dtype (4B)
    - Indices (topk_count * 4 bytes) - int32
    - Values (topk_count * 2 bytes) - float16

    Args:
        indices: Int32 tensor of selected indices.
        values: Float16 tensor of selected values.
        original_numel: Original number of elements in the gradient.
        original_dtype: Original dtype of the gradient.

    Returns:
        Serialized bytes.
    """
    # Ensure tensors are on CPU and correct dtypes
    indices = indices.cpu().to(torch.int32).contiguous()
    values = values.cpu().to(torch.float16).contiguous()

    topk_count = indices.numel()
    indices_bytes = indices.numpy().tobytes()
    values_bytes = values.numpy().tobytes()

    # Total payload is indices + values
    payload_size = len(indices_bytes) + len(values_bytes)

    # Pack header
    header = struct.pack(
        HEADER_FORMAT,
        MessageType.COMPRESSED_GRADIENT,
        DTYPE_TO_CODE.get(original_dtype, 0),
        0,  # n_dims unused for compressed
        payload_size,
    )

    # Pack compression metadata
    orig_dtype_code = DTYPE_TO_CODE.get(original_dtype, 0)
    metadata = struct.pack("!III", original_numel, topk_count, orig_dtype_code)

    return header + metadata + indices_bytes + values_bytes


def deserialize_compressed_gradient(
    data: bytes,
    device: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.dtype]:
    """Deserialize a compressed gradient.

    Args:
        data: Serialized compressed gradient bytes.
        device: Target device for tensors.

    Returns:
        Tuple of (indices, values, original_numel, original_dtype).
    """
    # Unpack header
    header = data[:HEADER_SIZE]
    msg_type, _, _, payload_size = struct.unpack(HEADER_FORMAT, header)

    if msg_type != MessageType.COMPRESSED_GRADIENT:
        raise ValueError(f"Expected COMPRESSED_GRADIENT, got {msg_type}")

    # Unpack compression metadata
    metadata_start = HEADER_SIZE
    metadata_end = metadata_start + 12
    original_numel, topk_count, orig_dtype_code = struct.unpack(
        "!III", data[metadata_start:metadata_end]
    )
    original_dtype = CODE_TO_DTYPE.get(orig_dtype_code, torch.float32)

    # Unpack indices
    indices_start = metadata_end
    indices_size = topk_count * 4
    indices_bytes = data[indices_start:indices_start + indices_size]
    indices = torch.from_numpy(
        np.frombuffer(indices_bytes, dtype=np.int32).copy()
    )

    # Unpack values
    values_start = indices_start + indices_size
    values_size = topk_count * 2
    values_bytes = data[values_start:values_start + values_size]
    values = torch.from_numpy(
        np.frombuffer(values_bytes, dtype=np.float16).copy()
    )

    # Move to device if specified
    if device is not None and device != "cpu":
        indices = indices.to(device)
        values = values.to(device)

    return indices, values, original_numel, original_dtype


def decompress_gradient(
    indices: torch.Tensor,
    values: torch.Tensor,
    original_numel: int,
    original_dtype: torch.dtype,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Decompress a sparse gradient back to dense.

    Args:
        indices: Tensor of indices.
        values: Tensor of values.
        original_numel: Original number of elements.
        original_dtype: Target dtype for the output.
        device: Target device.

    Returns:
        Dense gradient tensor.
    """
    # Create zero tensor
    if device:
        gradient = torch.zeros(original_numel, dtype=original_dtype, device=device)
    else:
        gradient = torch.zeros(original_numel, dtype=original_dtype)

    # Scatter values to indices
    gradient.scatter_(0, indices.long(), values.to(original_dtype))

    return gradient
