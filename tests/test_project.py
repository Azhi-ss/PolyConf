import sys
import unittest
from unittest.mock import patch, MagicMock
import torch

sys.path.insert(0, '/workspace')


class TestPyTorchCPUEnvironment(unittest.TestCase):
    """PyTorch CPU 环境检测测试"""

    def test_torch_import(self):
        self.assertIsNotNone(torch.__version__)
        print(f"[PASS] PyTorch: {torch.__version__}")

    def test_cpu_available(self):
        self.assertFalse(torch.cuda.is_available())
        print(f"[PASS] CUDA available: False (CPU only)")

    def test_tensor_creation_cpu(self):
        x = torch.randn(3, 3)
        self.assertEqual(x.device.type, 'cpu')
        print(f"[PASS] Tensor device: {x.device}")

    @patch('torch.cuda.is_available', return_value=False)
    def test_mock_cuda_detection(self, mock_cuda):
        self.assertFalse(torch.cuda.is_available())
        print("[PASS] Mock CUDA detection works")


class TestOpenFoldModules(unittest.TestCase):
    """openfold 核心模块测试"""

    def test_rigid_utils_import(self):
        from openfold.utils import rigid_utils
        self.assertIsNotNone(rigid_utils)
        print("[PASS] openfold.utils.rigid_utils")

    def test_tensor_utils_import(self):
        from openfold.utils import tensor_utils
        self.assertIsNotNone(tensor_utils)
        print("[PASS] openfold.utils.tensor_utils")

    def test_residue_constants_import(self):
        from openfold.np import residue_constants
        self.assertIsNotNone(residue_constants)
        print("[PASS] openfold.np.residue_constants")

    def test_rigid_rotation_creation(self):
        from openfold.utils.rigid_utils import Rotation
        rot = Rotation.identity(shape=(3,), device='cpu')
        self.assertIsNotNone(rot)
        self.assertEqual(rot.shape, (3,))
        print(f"[PASS] Rotation shape: {rot.shape}")

    def test_rigid_rigid_creation(self):
        from openfold.utils.rigid_utils import Rigid, Rotation
        rot = Rotation.identity(shape=(3,), device='cpu')
        trans = torch.zeros(3, 3)
        rigid = Rigid(rot, trans)
        self.assertIsNotNone(rigid)
        print("[PASS] Rigid created")


class TestDataModules(unittest.TestCase):
    """data 目录核心模块测试"""

    def test_chemical_import(self):
        from data import chemical
        self.assertIsNotNone(chemical)
        print("[PASS] data.chemical")

    def test_so3_utils_import(self):
        from data import so3_utils
        self.assertIsNotNone(so3_utils)
        print("[PASS] data.so3_utils")


class TestTorchFunctionality(unittest.TestCase):
    """PyTorch 功能性测试 (CPU)"""

    def test_matrix_multiplication(self):
        a = torch.randn(10, 20)
        b = torch.randn(20, 30)
        c = torch.matmul(a, b)
        self.assertEqual(c.shape, (10, 30))
        print(f"[PASS] MatMul: {a.shape} x {b.shape} = {c.shape}")

    def test_neural_network_forward(self):
        model = torch.nn.Linear(100, 10)
        x = torch.randn(32, 100)
        y = model(x)
        self.assertEqual(y.shape, (32, 10))
        print(f"[PASS] NN: {x.shape} -> {y.shape}")

    def test_gradient_computation(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = x.sum()
        y.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        print(f"[PASS] Gradient computation works")


if __name__ == '__main__':
    print("=" * 60)
    print("PolyConf ML Project - Unit Tests")
    print("Environment: PyTorch CPU Only")
    print("=" * 60)
    print()
    unittest.main(verbosity=2)
