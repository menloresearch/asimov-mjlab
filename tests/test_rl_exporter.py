"""Tests for RL exporter utilities."""

import os
import tempfile

import onnx

from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  list_to_csv_str,
)


def test_list_to_csv_str():
  """Test CSV string conversion utility."""
  # Test with floats.
  result = list_to_csv_str([1.23456, 2.34567, 3.45678], decimals=3)
  assert result == "1.235,2.346,3.457"

  # Test with integers.
  result = list_to_csv_str([1, 2, 3], decimals=2)
  assert result == "1.00,2.00,3.00"

  # Test with mixed types.
  result = list_to_csv_str([1.5, "hello", 2.5], decimals=1)
  assert result == "1.5,hello,2.5"

  # Test custom delimiter.
  result = list_to_csv_str([1.0, 2.0, 3.0], decimals=1, delimiter=";")
  assert result == "1.0;2.0;3.0"


def test_attach_metadata_to_onnx():
  """Test that metadata can be attached to ONNX models."""
  # Create a dummy ONNX model.
  with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = os.path.join(tmpdir, "test_policy.onnx")

    # Create minimal ONNX model.
    input_tensor = onnx.helper.make_tensor_value_info(
      "input", onnx.TensorProto.FLOAT, [1, 2]
    )
    output_tensor = onnx.helper.make_tensor_value_info(
      "output", onnx.TensorProto.FLOAT, [1, 2]
    )
    node = onnx.helper.make_node("Identity", ["input"], ["output"])
    graph = onnx.helper.make_graph(
      [node], "test_graph", [input_tensor], [output_tensor]
    )
    model = onnx.helper.make_model(graph)
    onnx.save(model, onnx_path)

    # Attach metadata.
    metadata = {
      "run_path": "test/run/path",
      "joint_names": ["joint_a", "joint_b"],
      "joint_stiffness": [20.0, 10.0],
      "joint_damping": [1.0, 1.0],
      "extra_field": "extra_value",
    }
    attach_metadata_to_onnx(onnx_path, metadata)

    # Load and verify metadata was attached.
    loaded_model = onnx.load(onnx_path)
    metadata_props = {prop.key: prop.value for prop in loaded_model.metadata_props}

    # Check all metadata fields are present.
    assert "run_path" in metadata_props
    assert "joint_names" in metadata_props
    assert "joint_stiffness" in metadata_props
    assert "extra_field" in metadata_props

    # Check values are correct.
    assert metadata_props["run_path"] == "test/run/path"
    assert metadata_props["extra_field"] == "extra_value"

    # Check list was converted to CSV string.
    joint_names = metadata_props["joint_names"].split(",")
    assert len(joint_names) == 2
    assert "joint_a" in joint_names
    assert "joint_b" in joint_names

    # Check stiffness values are in natural joint order.
    stiffness_values = [float(x) for x in metadata_props["joint_stiffness"].split(",")]
    assert stiffness_values == [20.0, 10.0]  # Natural order: joint_a (20), joint_b (10)
