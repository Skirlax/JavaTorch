package dev.skyr.core.nn.optimizers;

import dev.skyr.core.autograd.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.HashSet;

public class Adam {
    private HashMap<String, INDArray> m;
    private HashMap<String, INDArray> v;
    private double beta1;
    private double beta2;
    private double epsilon;
    private double learningRate;
    private double l2;
    private int t = 0;
    private HashMap<String,Tensor> parameters;


    public Adam(HashMap<String, Tensor> parameters, double learningRate, double beta1, double beta2, double epsilon, double l2) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.l2 = l2;
        this.parameters = parameters;
        this.m = initMV();
        this.v = initMV();
    }

    public Adam(HashMap<String, Tensor> parameters,double learningRate, double beta1, double beta2, double epsilon) {
        this(parameters,learningRate, beta1, beta2, epsilon, 0.0);
    }

    public Adam(HashMap<String, Tensor> parameters, double learningRate, double beta1, double beta2) {
        this(parameters,learningRate, beta1, beta2, 1e-7);
    }

    public Adam(HashMap<String, Tensor> parameters, double learningRate) {
        this(parameters,learningRate, 0.9, 0.999);
    }

    private HashMap<String, INDArray> initMV() {
        HashMap<String, INDArray> mv = new HashMap<>();
        for (String key : parameters.keySet()) {
            mv.put(key, Nd4j.zeros(parameters.get(key).data.shape()));
        }
        return mv;
    }

    public void step() {
        this.t += 1;
        for (String key : parameters.keySet()) {
            if (!parameters.get(key).requiresGrad) {
                continue;
            }
            Tensor p = parameters.get(key);
            if (this.l2 > 0) {
                p.grad.addi(p.data.mul(this.l2));
            }
            INDArray grad = p.grad;
            this.m.put(key, this.m.get(key).mul(this.beta1).add(grad.mul(1 - this.beta1)));
            this.v.put(key, this.v.get(key).mul(this.beta2).add(grad.mul(grad).mul(1 - this.beta2)));


            INDArray m_hat = this.m.get(key).div(1 - Math.pow(this.beta1,this.t));
            INDArray v_hat = this.v.get(key).div(1 - Math.pow(this.beta2,this.t));
            p.data = p.data.sub(m_hat.mul(this.learningRate).div(Transforms.sqrt(v_hat).add(this.epsilon)));
        }
    }

    public void zero_grad() {
        for (String key : parameters.keySet()) {
            parameters.get(key).grad = Nd4j.zeros(parameters.get(key).grad.shape());
        }
    }



}
