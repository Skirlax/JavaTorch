package dev.skyr.core.nn.layers;

import dev.skyr.core.autograd.Tensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;

// Checkout: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py

public class Linear extends Module {
    private int in_features;
    private int out_features;
    private Tensor weight;
    private Tensor bias;

    public Linear(int in_features, int out_features) {
        this.in_features = in_features;
        this.out_features = out_features;
        this.weight = init_weights(in_features, out_features);
        this.bias = init_bias(out_features);

    }

    // Pytorch style initialization. This helps to keep the parameters in a manageable range,
    //  so they don't grow or shrink too much when the layer size increases.

    private Tensor init_weights(int in_features, int out_features) {
        double upper_bound = 1 / Math.sqrt(in_features);
        INDArray weight = Nd4j.random.uniform(-upper_bound, upper_bound, DataType.DOUBLE, out_features, in_features);
        return new Tensor(weight, true);

    }
    private Tensor init_bias(int out_features) {
        double upper_bound = 1 / Math.sqrt(out_features);
        INDArray bias = Nd4j.random.uniform(-upper_bound, upper_bound, DataType.DOUBLE, out_features,1);
        return new Tensor(bias, true);
    }

    public Tensor forward(Tensor x) {
        Tensor partialOut = x.matmul(this.weight.transpose());
        return partialOut.add(this.bias.broadcast(partialOut.data.shape()));
    }

    @Override
    public HashMap<String, Tensor> parameters() {
        HashMap<String, Tensor> params = new HashMap<>();
        params.put("weights", this.weight);
        params.put("bias", this.bias);
        return params;
    }
}
