package dev.skyr.core.nn.criterions.modules;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.layers.Module;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.HashMap;

public class MSE extends Module {
    public Tensor forward(Tensor x, Tensor y) {
        Tensor se = x.sub(y).pow(new Tensor(2.0, true));
        return se.mul(new Tensor(1.0 / se.data.shape()[0], true));
    }

    @Override
    public HashMap<String, Serializable> parameters() {
        return null;
    }
}
