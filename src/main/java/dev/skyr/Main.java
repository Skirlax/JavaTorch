package dev.skyr;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.criterions.modules.MSE;
import dev.skyr.core.nn.functional.Functions;
import dev.skyr.core.nn.layers.Linear;
import dev.skyr.core.nn.layers.conv.Conv2D;
import dev.skyr.core.nn.optimizers.Adam;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class Main {

    public static void main(String[] args) {
        INDArray x = Nd4j.create(new float[]{1, 2,3,4}, new long[]{2, 2}, DataType.DOUBLE);
        INDArray y = Nd4j.create(new float[]{7, 8,3,4}, new long[]{2, 2}, DataType.DOUBLE);
        Tensor xT = new Tensor(x, true);
        Tensor yT = new Tensor(y, true);
        Tensor c = xT.matmul(yT).add(new Tensor(Nd4j.create(new double[]{3.0,3.0},new long[]{2,1},DataType.DOUBLE), true).broadcast(2,2));
        MSE mse = new MSE();
        Tensor loss = mse.forward(c.sum(1).view(2,1), yT.sum(1).view(2,1));
        loss.backward();
        System.out.println(loss.data);
        System.out.println(xT.grad);








    }
}
