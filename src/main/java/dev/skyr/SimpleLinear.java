package dev.skyr;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.activations.modules.ReLU;
import dev.skyr.core.nn.layers.CustomModule;
import dev.skyr.core.nn.layers.Linear;

public class SimpleLinear extends CustomModule {
    private Linear linear1;
    private Linear linear2;
    private ReLU relu;

    public SimpleLinear() {
        this.linear1 = new Linear(2, 2);
        this.linear2 = new Linear(2, 1);
        this.relu = new ReLU();
    }

    public Tensor forward(Tensor x) {
        x = linear1.forward(x);
        x = relu.forward(x);
        x = linear2.forward(x);
        return x;
    }
}
