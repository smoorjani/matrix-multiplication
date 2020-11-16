#include <torch/torch.h>

int main()
{
    torch::Tensor tensor = torch::zeros({2, 2});
    std::cout << tensor << std::endl;

    return 0;
}