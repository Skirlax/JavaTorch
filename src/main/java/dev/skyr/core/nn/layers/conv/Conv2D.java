package dev.skyr.core.nn.layers.conv;

import dev.skyr.core.autograd.Tensor;
import dev.skyr.core.nn.layers.Module;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import dev.skyr.core.nn.functional.Functions;

import java.io.Serializable;
import java.util.HashMap;

public class Conv2D extends Module {
    private int inChannels;
    private int outChannels;
    private int kernelSize;
    private int stride;
    private int padding;
    private Tensor weights;
    private Tensor bias;

    public Conv2D(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.weights = makeWeights();
        this.bias = makeBias();
        this.weights.persistGrad();
        this.bias.persistGrad();
    }

    private Tensor makeWeights() {
        double upper_bound = Math.sqrt(1.0 / (inChannels * kernelSize * kernelSize));
        Tensor weight = new Tensor(Nd4j.random.uniform(-upper_bound, upper_bound, DataType.DOUBLE, outChannels, inChannels, kernelSize, kernelSize), true);
        return weight;

    }

    private Tensor makeBias() {
        double upper_bound = Math.sqrt(1.0 / inChannels * kernelSize * kernelSize);
        Tensor bias = new Tensor(Nd4j.random.uniform(-upper_bound, upper_bound, DataType.DOUBLE, outChannels), true);
        return bias;
    }

    @Override
    public Tensor forward(Tensor x) {
        long[] originalDataShape = x.data.shape();
        INDArray col = Functions.img2colOptimized(x.data, kernelSize, stride, padding);
        HashMap<String,Double> additionalInfo = new HashMap<>(){
            {
                put("width", (double) originalDataShape[2]);
                put("height", (double) originalDataShape[3]);
                put("channels", (double) originalDataShape[1]);
                put("kernelSize", (double) kernelSize);
                put("stride", (double) stride);
                put("padding", (double) padding);
            }
        };
        Tensor xCol = new Tensor(col,x.requiresGrad);
        x.createChildAndRegisterBackward(x,xCol,"reverse_col2img",additionalInfo);
        Tensor weight = weights.view(1, outChannels, -1);
        weight = weight.broadcast(xCol.data.shape()[0], weight.data.shape()[1], weight.data.shape()[2]);
        Tensor broadcastedBias = bias.view(1, -1, 1);
        broadcastedBias = broadcastedBias.broadcast(xCol.data.shape()[0], broadcastedBias.data.shape()[1], xCol.data.shape()[2]);
        Tensor output = weight.matmul(xCol).add(broadcastedBias);
        int outputWidth = Functions.getOutputSize((int) originalDataShape[2], kernelSize, stride, padding);
        int outputHeight = Functions.getOutputSize((int) originalDataShape[3], kernelSize, stride, padding);
        return output.view(output.data.shape()[0], outChannels,outputWidth, outputHeight);
    }

    @Override
    public HashMap<String, Tensor> parameters() {
        HashMap<String, Tensor> params = new HashMap<>();
        params.put("weights", this.weights);
        params.put("bias", this.bias);
        return params;
    }
}
