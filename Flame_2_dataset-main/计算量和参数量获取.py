from torchsummary import summary
import models

# 定义一个函数来打印模型的详细信息
def print_model_info(model, input_size):
    summary(model, input_size=input_size)

# 测试数据大小
input_size = (2, 3, 1024, 1024)  # 假设输入为2个样本，3个通道，254x254的图像

# 以下是模型实例化和信息打印的示例
# 请确保你已经定义了所有的模型类，并且它们都可以接收input_size作为输入

# 例如，对于Flame_one_stream模型：
flame_one_stream = models.Flame_one_stream()
print_model_info(flame_one_stream, input_size)

# 对于Flame_two_stream模型：
flame_two_stream = models.Flame_two_stream()
print_model_info(flame_two_stream, input_size)

# 以此类推，为其他模型实例化并打印信息
# ...

# 注意：你需要确保每个模型的forward函数都能接受input_size作为输入，并且正确处理数据。