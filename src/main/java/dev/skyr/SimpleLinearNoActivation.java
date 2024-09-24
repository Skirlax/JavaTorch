package dev.skyr;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.layers.CustomModule;
import dev.skyr.core.nn.layers.Linear;

public class SimpleLinearNoActivation extends CustomModule {
    private Linear linear1;
    private Linear linear2;

    public SimpleLinearNoActivation() {
        this.linear1 = new Linear(3, 2);
        this.linear2 = new Linear(2, 1);
    }

    public Tensor forward(Tensor x) {
        x = linear1.forward(x);
        x = linear2.forward(x);
        return x;
    }
}
