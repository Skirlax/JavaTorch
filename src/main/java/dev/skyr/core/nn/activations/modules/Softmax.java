package dev.skyr.core.nn.activations.modules;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.layers.Module;

import java.util.HashMap;

public class Softmax extends Module {

    public Tensor forward(Tensor x) {
        return x.softmax();
    }

    @Override
    public HashMap<String, Tensor> parameters() {
        return new HashMap<>();
    }
}
