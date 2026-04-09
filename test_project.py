import sys
import unittest
from unittest.mock import patch, MagicMock
import torch

sys.path.insert(0, '/workspace')


class TestPyTorchCPUEnvironment(unittest.TestCase):
    """PyTorch CPU 环境检测测试"""

    def test_torch_import(self):
        """测试 PyTorch 是否正确安装"""
        self.assertIsNotNone(torch.__version__)
        print(f"[PASS] PyTorch: {torch.__version__}")

    def test_cpu_available(self):
        """测试 CPU 是否可用"""
        self.assertFalse(torch.cuda.is_available())
        print(f"[PASS] CUDA available: False (CPU only)")

    def test_tensor_creation_cpu(self):
        """测试在 CPU 上创建张量"""
        x = torch.randn(3, 3)
        self.assertEqual(x.device.type, 'cpu')
        print(f"[PASS] Tensor device: {x.device}")

    @patch('torch.cuda.is_available', return_value=False)
    def test_mock_cuda_detection(self, mock_cuda):
        """Mock 测试 CUDA 不可用场景"""
        self.assertFalse(torch.cuda.is_available())
        print("[PASS] Mock CUDA detection works")


class TestOpenFoldModules(unittest.TestCase):
    """openfold 核心模块测试"""

    def test_rigid_utils_import(self):
        """测试 rigid_utils 导入"""
        from openfold.utils import rigid_utils
        self.assertIsNotNone(rigid_utils)
        print("[PASS] openfold.utils.rigid_utils")

    def test_tensor_utils_import(self):
        """测试 tensor_utils 导入"""
        from openfold.utils import tensor_utils
        self.assertIsNotNone(tensor_utils)
        print("[PASS] openfold.utils.tensor_utils")

    def test_residue_constants_import(self):
        """测试 residue_constants 导入"""
        from openfold.np import residue_constants
        self.assertIsNotNone(residue_constants)
        print("[PASS] openfold.np.residue_constants")

    def test_rigid_rotation_creation(self):
        """测试 Rotation 类创建"""
        from openfold.utils.rigid_utils import Rotation
        rot = Rotation.identity(shape=(3,), device='cpu')
        self.assertIsNotNone(rot)
        self.assertEqual(rot.shape, (3,))
        print(f"[PASS] Rotation shape: {rot.shape}")

    def test_rigid_rigid_creation(self):
        """测试 Rigid 类创建"""
        from openfold.utils.rigid_utils import Rigid, Rotation
        rot = Rotation.identity(shape=(3,), device='cpu')
        trans = torch.zeros(3, 3)  # shape[:-1] should be (3,)
        rigid = Rigid(rot, trans)
        self.assertIsNotNone(rigid)
        print("[PASS] Rigid created")


class TestDataModules(unittest.TestCase):
    """data 目录核心模块测试"""

    def test_chemical_import(self):
        """测试 chemical 模块导入"""
        from data import chemical
        self.assertIsNotNone(chemical)
        print("[PASS] data.chemical")

    def test_so3_utils_import(self):
        """测试 so3_utils 导入"""
        from data import so3_utils
        self.assertIsNotNone(so3_utils)
        print("[PASS] data.so3_utils")


class TestTorchFunctionality(unittest.TestCase):
    """PyTorch 功能性测试 (CPU)"""

    def test_matrix_multiplication(self):
        """测试矩阵乘法"""
        a = torch.randn(10, 20)
        b = torch.randn(20, 30)
        c = torch.matmul(a, b)
        self.assertEqual(c.shape, (10, 30))
        print(f"[PASS] MatMul: {a.shape} x {b.shape} = {c.shape}")

    def test_neural_network_forward(self):
        """测试简单神经网络前向传播"""
        model = torch.nn.Linear(100, 10)
        x = torch.randn(32, 100)
        y = model(x)
        self.assertEqual(y.shape, (32, 10))
        print(f"[PASS] NN: {x.shape} -> {y.shape}")

    def test_cpu_only_model(self):
        """测试 CPU 专用模型"""
        model = torch.nn.Linear(50, 50)
        x = torch.randn(8, 50)
        self.assertEqual(next(model.parameters()).device.type, 'cpu')
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape, (8, 50))
        print(f"[PASS] CPU model: {x.shape} -> {y.shape}")

    def test_gradient_computation(self):
        """测试梯度计算 (CPU)"""
        x = torch.randn(5, 5, requires_grad=True)
        y = x.sum()
        y.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        print(f"[PASS] Gradient computation works")


if __name__ == '__main__':
    print("=" * 60)
    print("PolyConf ML Project - Unit Tests & Dynamic Analysis")
    print("Environment: PyTorch CPU Only")
    print("=" * 60)
    print()
    
    unittest.main(verbosity=2)
