package dev.skyr.core;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.activations.modules.ReLU;
import dev.skyr.core.nn.activations.modules.Softmax;
import dev.skyr.core.nn.layers.CustomModule;
import dev.skyr.core.nn.layers.Linear;
import dev.skyr.core.nn.layers.conv.Conv2D;

public class SimpleConvNet extends CustomModule {
    private Conv2D conv1;
    private ReLU relu;
    private Conv2D conv2;
    private Conv2D conv3;
    private Conv2D conv4;
    private Linear linear1;
    private Linear linear2;
    private Softmax softmax;


    public SimpleConvNet(int in_channels) {
        this.conv1 = new Conv2D(in_channels, 32, 3, 1, 1);
        this.relu = new ReLU();
        this.conv2 = new Conv2D(32, 64, 3, 1, 1);
        this.conv3 = new Conv2D(64, 128, 3, 1, 1);
        this.conv4 = new Conv2D(128, 32, 3, 1, 1);
        this.linear1 = new Linear(3200, 128);
        this.linear2 = new Linear(128, 10);
        this.softmax = new Softmax();
    }

    public Tensor forward(Tensor x) {
        x = relu.forward(conv1.forward(x));
        x = relu.forward(conv2.forward(x));
        x = relu.forward(conv3.forward(x));
        x = relu.forward(conv4.forward(x));
        x = x.view(x.size(0), -1);
        x = linear2.forward(relu.forward(linear1.forward(x)));
        x = softmax.forward(x);
        return x;
    }
}
