package dev.skyr.core.nn.layers.conv;

import dev.skyr.core.autograd.Tensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import dev.skyr.core.nn.functional.Functions;

public class Conv2D {
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
    }

    private Tensor makeWeights() {
        double upper_bound = Math.sqrt(1.0 / (inChannels * kernelSize * kernelSize));
        Tensor weight = new Tensor(Nd4j.random.uniform(-upper_bound, upper_bound, DataType.FLOAT, outChannels, inChannels, kernelSize, kernelSize), true);
        return weight;

    }

    private Tensor makeBias() {
        double upper_bound = Math.sqrt(1.0 / inChannels * kernelSize * kernelSize);
        Tensor bias = new Tensor(Nd4j.random.uniform(-upper_bound, upper_bound, DataType.FLOAT, outChannels), true);
        return bias;
    }

    public Tensor forward(Tensor x) {
        long[] originalDataShape = x.data.shape();
        INDArray col = Functions.img2col(x.data, kernelSize, stride, padding);
        x.rebuildWithNewData(col);
        Tensor weight = weights.view(1,outChannels,-1);
        weight = weight.broadcast(x.data.shape()[0],weight.data.shape()[1],weight.data.shape()[2]);
        Tensor broadcastedBias = bias.view(1,-1,1);
        broadcastedBias = broadcastedBias.broadcast(x.data.shape()[0],broadcastedBias.data.shape()[1],x.data.shape()[2]);
        Tensor output = weight.matmul(x).add(broadcastedBias);
        int outputWidth = Functions.getOutputSize((int) originalDataShape[1], kernelSize, stride, padding);
        int outputHeight = Functions.getOutputSize((int) originalDataShape[2], kernelSize, stride, padding);
        return output.view(output.data.shape()[0],outChannels,outputWidth,outputHeight);
    }
}
