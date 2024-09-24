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
        INDArray y = Nd4j.create(new float[]{7, 8}, new long[]{2, 1}, DataType.DOUBLE);
        Tensor xT = new Tensor(x, true);
        Tensor yT = new Tensor(y, true);
        SimpleLinear model = new SimpleLinear();
        Adam optimizer = new Adam(model.parameters(), 0.01);
        MSE criterion = new MSE();
        for (int i = 0; i < 1000; i++) {
            Tensor output = model.forward(xT);
            Tensor loss = criterion.forward(output, yT);
            System.out.println("Loss: " + loss.data);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }








    }
}
