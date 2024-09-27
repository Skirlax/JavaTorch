package dev.skyr;

import dev.skyr.core.SimpleConvNet;
import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.criterions.modules.CrossEntropy;
import dev.skyr.core.nn.optimizers.Adam;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class Main {

    public static void main(String[] args) {
        INDArray x = Nd4j.rand(DataType.DOUBLE, 32,3,10,10);
        INDArray y = Nd4j.zeros(DataType.DOUBLE, 32,10);
        for (int i = 0; i < 32; i++) {
            y.putScalar(i, i % 10, 1);
        }
        Tensor tensorX = new Tensor(x, true);
        Tensor tensorY = new Tensor(y, true);
        SimpleConvNet model = new SimpleConvNet(3);
        CrossEntropy lossFn = new CrossEntropy();
        double lr = 0.0005;
        Adam optimizer = new Adam(model.parameters(), lr);
        int epochs = 1000;
        for (int i = 0; i < epochs; i++) {
            Tensor pred = model.forward(tensorX);
            Tensor loss = lossFn.forward(pred, tensorY);
            System.out.println("Epoch: " + i + " Loss: " + loss.data);
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }
        System.out.println(model.forward(tensorX));










    }
}
