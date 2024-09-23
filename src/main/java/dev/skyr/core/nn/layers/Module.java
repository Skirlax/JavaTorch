package dev.skyr.core.nn.layers;

import dev.skyr.core.autograd.Tensor;

import java.io.Serializable;
import java.util.HashMap;

public abstract class Module {

//    public abstract Tensor forward(Tensor x);
    public abstract HashMap<String, Serializable> parameters();
}
