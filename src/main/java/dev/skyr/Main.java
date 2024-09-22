package dev.skyr;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.functional.Functions;
import dev.skyr.core.nn.layers.Linear;
import dev.skyr.core.nn.layers.conv.Conv2D;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class Main {

    public static void main(String[] args) {
        INDArray tensor = Nd4j.rand(DataType.FLOAT, 32, 128,128,3);
        Tensor input = new Tensor(tensor, true);
        Conv2D conv2D = new Conv2D(3, 64, 3, 1, 1);
        Tensor output = conv2D.forward(input);
        output.backward(null,null);
        System.out.println(input.grad);

    }
}
