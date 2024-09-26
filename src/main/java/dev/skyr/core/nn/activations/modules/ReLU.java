package dev.skyr.core.nn.activations.modules;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.layers.Module;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;

public class ReLU extends Module {
    public ReLU() {
    }

    @Override
    public Tensor forward(Tensor x) {
        INDArray dataMask = x.data.gt(0).castTo(DataType.DOUBLE);
        INDArray data = x.data.mul(dataMask);
        Tensor dataT = new Tensor(data, true);
        x.createChildAndRegisterBackward(x,dataT, "relu_backward");
        return dataT;
    }

    @Override
    public HashMap<String, Tensor> parameters() {
        return new HashMap<>();
    }
}
