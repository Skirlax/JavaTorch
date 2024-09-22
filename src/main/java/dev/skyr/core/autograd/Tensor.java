package dev.skyr.core.autograd;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Tensor {
    public INDArray data;
    public INDArray grad;
    public boolean requiresGrad;
    public String backwardFn;
    public Tensor leftOperand;
    public int[] permute;
    public Tensor rightOperand;
    private Tensor child;
    public long[] originalShape;

    public Tensor(INDArray data, boolean requiresGrad) {
        this.data = data;
        this.requiresGrad = requiresGrad;
        this.grad = Nd4j.zeros(DataType.FLOAT, data.shape());
        this.backwardFn = null;
    }

    public Tensor(double data, boolean requiresGrad) {
        this.data = Nd4j.scalar(data);
        this.requiresGrad = requiresGrad;
        this.grad = Nd4j.zeros(DataType.FLOAT, this.data.shape());
        this.backwardFn = null;
    }

    public Tensor add(Tensor other, long[] broadcastShape, String broadcastType) {
        INDArray result = this.data.add(other.data);
        this.child = new Tensor(result,true);
        setOperands(other,child);
        this.child.backwardFn = "add_backward";
        return this.child;
    }

    public Tensor add(Tensor other) {
        return add(other,null,null);
    }

    public Tensor mul(Tensor other) {
        this.child = new Tensor(this.data.mul(other.data),true);
        setOperands(other,child);
        this.child.backwardFn = "mul_backward";
        return this.child;
    }

    public Tensor matmul(Tensor other,long[] broadcastShape,String broadcastType) {
        INDArray result = Nd4j.matmul(this.data,other.data);
        this.child = new Tensor(result,true);
        setOperands(other,child);
        this.child.backwardFn = "matmul_backward";
        return this.child;
    }

    public Tensor matmul(Tensor other) {
        return matmul(other,null,null);
    }

    public Tensor view(long... shape) {
        this.originalShape = this.data.shape();
        INDArray result = this.data.reshape(shape);
        this.child = new Tensor(result,true);
        this.child.leftOperand = this;
        this.child.backwardFn = "view_backward";
        return this.child;
    }

    public Tensor broadcast(long... shape) {
        this.originalShape = this.data.shape();
        INDArray result = this.data.broadcast(shape);
        this.child = new Tensor(result,true);
        this.child.leftOperand = this;
        this.child.backwardFn = "broadcast_backward";
        return this.child;
    }

    public Tensor neg() {
        INDArray result = this.data.neg();
        this.child = new Tensor(result,true);
        this.child.leftOperand = this;
        this.child.backwardFn = "neg_backward";
        return this.child;
    }

    public Tensor permute(int... permute) {
        this.originalShape = this.data.shape();
        this.permute = permute;
        INDArray result = this.data.permute(permute);
        this.child = new Tensor(result,true);
        this.child.leftOperand = this;
        this.child.backwardFn = "permute_backward";
        return this.child;
    }

    public Tensor sub(Tensor other) {
        INDArray result = this.data.add(other.neg().data);
        this.child = new Tensor(result,true);
        this.setOperands(other,child);
        this.child.backwardFn = "add_backward";
        return this.child;
    }
    public Tensor pow(Tensor power) {
        INDArray result = Transforms.pow(this.data,power.data);
        this.child = new Tensor(result,true);
        this.setOperands(power,child);
        this.child.backwardFn = "pow_backward";
        return this.child;
    }
    public Tensor truediv(Tensor other) {
        INDArray result = this.data.div(other.data);
        this.child = new Tensor(result,true);
        this.setOperands(other,child);
        this.child.backwardFn = "div_backward";
        return this.child;
    }

    public void backward(INDArray gradient,Tensor child_) {
        if (!this.requiresGrad) {
            System.out.println("No");
            return;
        }
        if (gradient == null) {
            gradient = Nd4j.ones(this.grad.dataType(),this.grad.shape());
        }
        if (child_ == null) {
            child_ = this;
        }
        this.grad = this.grad.add(gradient);
        if (!(this.backwardFn == null)) {
            BackwardFunctions.execute(this.backwardFn,this.grad,child_);
        }
    }

    private void setOperands(Tensor other,Tensor child) {
        child.leftOperand = this;
        child.rightOperand = other;
    }
    public void rebuildWithNewData(INDArray newData) {
        this.data = newData;
        this.grad = Nd4j.zeros(DataType.FLOAT, newData.shape());
    }

}
