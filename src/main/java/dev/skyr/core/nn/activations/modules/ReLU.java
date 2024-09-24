package dev.skyr.core.nn.activations.modules;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.layers.Module;

import java.util.HashMap;

public class ReLU extends Module {
    public ReLU() {
    }

    public Tensor forward(Tensor x) {
        return null;
    }

    @Override
    public HashMap<String, Tensor> parameters() {
        return new HashMap<>();
    }
}
