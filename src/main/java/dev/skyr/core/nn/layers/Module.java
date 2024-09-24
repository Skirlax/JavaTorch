package dev.skyr.core.nn.layers;

import dev.skyr.core.autograd.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.HashMap;

public abstract class Module {

    public Tensor forward(Tensor x){return null;};
    public Tensor forward(Tensor x, Tensor y){return null;};
    public abstract HashMap<String, Tensor> parameters();
}
