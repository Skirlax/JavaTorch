package dev.skyr.core.nn.criterions.modules;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.checks.StaticChecks;
import dev.skyr.core.nn.layers.Module;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.TransferQueue;

public class CrossEntropy extends Module {

    public Tensor forward(Tensor x, Tensor y) {
        if (!StaticChecks.isOneHotEncoded(y.data)) {
            throw new IllegalArgumentException("Expected y to be 2d and one-hot encoded (with one at the position of the correct class). You passed either non-2d array or it is not one hot encoded");
        }
        if (!Arrays.equals(x.data.shape(), y.data.shape())) {
            throw new IllegalArgumentException("Expected x and y to have the same shape. Got x: " + Arrays.toString(x.data.shape()) + " and y: " + Arrays.toString(y.data.shape()));
        }
        Tensor exp = x.exp();
        Tensor out = exp.truediv(exp.sum(1).broadcast(exp.data.shape()));
        Tensor loss = out.ln().neg();
        loss = loss.mul(y);
        return loss.sum(1).mean(0);


    }

    @Override
    public HashMap<String, Tensor> parameters() {
        return null;
    }
}
