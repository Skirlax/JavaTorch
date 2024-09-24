package dev.skyr.core.nn.criterions.modules;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.layers.Module;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.HashMap;

public class MSE extends Module {
    public Tensor forward(Tensor x, Tensor y) {
        Tensor se = x.sub(y);
        se = se.pow(new Tensor(Nd4j.create(new double[]{2.0}), false));
        return se.mean(0);

    }

    @Override
    public HashMap<String, Tensor> parameters() {
        return new HashMap<>();
    }
}
