package dev.skyr.core.nn.activations.modules;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.layers.Module;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.HashMap;

public class ReLU extends Module {
    public ReLU() {
    }

    public Tensor forward(Tensor x) {
        INDArray data = x.data.gt(0).castTo(DataType.DOUBLE).mul(x.data);
        Tensor dataT = new Tensor(data, true);
        x.createChildAndRegister(x,dataT, "relu_backward");
        return dataT;
    }

    @Override
    public HashMap<String, Tensor> parameters() {
        return new HashMap<>();
    }
}
