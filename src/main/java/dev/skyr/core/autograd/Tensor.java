package dev.skyr.core.autograd;

import guru.nidi.graphviz.attribute.*;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;
import guru.nidi.graphviz.model.LinkSource;
import guru.nidi.graphviz.model.Node;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import java.util.HashMap;
import java.util.StringTokenizer;

import static guru.nidi.graphviz.model.Factory.graph;
import static guru.nidi.graphviz.model.Factory.node;

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
    private boolean grad_persists;
    HashMap<String, Double> additionalInfo;

    public Tensor(INDArray data, boolean requiresGrad) {
        this.data = data;
        this.requiresGrad = requiresGrad;
        if (this.requiresGrad) {
            this.grad = Nd4j.zeros(DataType.DOUBLE, data.shape());
        }
        this.backwardFn = null;
        this.grad_persists = false;
    }

    public Tensor(INDArray data) {
        this(data, false);
    }

    public Tensor(double data, boolean requiresGrad) {
        this.data = Nd4j.scalar(data);
        this.requiresGrad = requiresGrad;
        this.grad = Nd4j.zeros(DataType.DOUBLE, this.data.shape());
        this.backwardFn = null;
    }
    public Tensor(double data) {
        this(data, false);
    }


    public Tensor add(Tensor other) {
        INDArray result = this.data.add(other.data);
        this.child = new Tensor(result, shouldRequireGrad(this, other));
        setOperands(other, child);
        this.child.backwardFn = "add_backward";
        return this.child;
    }


    public Tensor mul(Tensor other) {
        this.child = new Tensor(this.data.mul(other.data), shouldRequireGrad(this, other));
        setOperands(other, child);
        this.child.backwardFn = "mul_backward";
        return this.child;
    }

    public Tensor matmul(Tensor other) {
        INDArray result = Nd4j.matmul(this.data, other.data);
        this.child = new Tensor(result, shouldRequireGrad(this, other));
        setOperands(other, child);
        this.child.backwardFn = "matmul_backward";
        return this.child;
    }


    public Tensor view(long... shape) {
        this.originalShape = this.data.shape();
        INDArray result = this.data.reshape(shape);
        this.child = new Tensor(result, this.requiresGrad);
        this.child.leftOperand = this;
        this.child.backwardFn = "view_backward";
        return this.child;
    }

    public Tensor broadcast(long... shape) {
        this.originalShape = this.data.shape();
        INDArray result = this.data.broadcast(shape);
        this.child = new Tensor(result, this.requiresGrad);
        this.child.leftOperand = this;
        this.child.backwardFn = "broadcast_backward";
        return this.child;
    }

    public Tensor neg() {
        INDArray result = this.data.neg();
        this.child = new Tensor(result, this.requiresGrad);
        this.child.leftOperand = this;
        this.child.backwardFn = "neg_backward";
        return this.child;
    }

    public Tensor permute(int... permute) {
        this.originalShape = this.data.shape();
        this.permute = permute;
        INDArray result = this.data.permute(permute);
        this.child = new Tensor(result, this.requiresGrad);
        this.child.leftOperand = this;
        this.child.backwardFn = "permute_backward";
        return this.child;
    }

    public Tensor sub(Tensor other) {
        INDArray result = this.data.add(other.neg().data);
        this.child = new Tensor(result, shouldRequireGrad(this, other));
        this.setOperands(other, child);
        this.child.backwardFn = "add_backward";
        return this.child;
    }

    public Tensor pow(Tensor power) {
        INDArray result = Transforms.pow(this.data, power.data);
        this.child = new Tensor(result, shouldRequireGrad(this, power));
        this.setOperands(power, child);
        this.child.backwardFn = "pow_backward";
        return this.child;
    }

    public Tensor truediv(Tensor other) {
        INDArray result = this.data.div(other.data);
        this.child = new Tensor(result, shouldRequireGrad(this, other));
        this.setOperands(other, child);
        this.child.backwardFn = "div_backward";
        return this.child;
    }

    public Tensor transpose() {
        if (this.data.rank() != 2) {
            throw new IllegalArgumentException("Only 2D tensors are supported");
        }
        INDArray result = this.data.transpose();
        this.child = new Tensor(result, this.requiresGrad);
        this.child.leftOperand = this;
        this.child.backwardFn = "transpose_backward";
        return this.child;
    }

    public Tensor max(int axis) {
        INDArray result = this.data.max(true, axis);
        this.child = new Tensor(result, this.requiresGrad);
        this.child.leftOperand = this;
        this.child.backwardFn = "max_backward";
        return this.child;
    }

    public Tensor sum(int axis) {
//        this.originalShape = this.data.shape();
        INDArray result = this.data.sum(true, axis);
        this.child = new Tensor(result, this.requiresGrad);
        this.child.leftOperand = this;
        this.child.backwardFn = "sum_backward";
        return this.child;
    }

    public Tensor mean(int axis) {
        long shapeAtAxis = this.data.shape()[axis];
        Tensor result = this.sum(axis);
        return result.truediv(new Tensor(shapeAtAxis, false));

    }

    public Tensor ln() {
        INDArray result = Transforms.log(this.data);
        this.child = new Tensor(result, this.requiresGrad);
        this.child.leftOperand = this;
        this.child.backwardFn = "ln_backward";
        return this.child;
    }

    public Tensor exp() {
        INDArray ees = Nd4j.ones(DataType.DOUBLE, this.data.shape()).mul(Math.E);
        return new Tensor(ees, false).pow(this);
    }

    public Tensor softmax() {
        Tensor exp = this.exp();
        Tensor sum = exp.detach(true).sum(1);
        return exp.truediv(sum.broadcast(exp.data.shape()));
    }

    public long size(int axis) {
        return this.data.size(axis);
    }


    public void backward(INDArray gradient, Tensor child_) {
        if (!this.requiresGrad) {
            if (this.child == null) {
                System.out.println("This tensor does not require grad, no backward pass performed.");
            }
            return;
        }
        if (gradient == null) {
            gradient = Nd4j.ones(this.grad.dataType(), this.grad.shape());
        }
        if (child_ == null) {
            child_ = this;
        }
        this.grad = this.grad.add(gradient);
        if (!(this.backwardFn == null)) {
            BackwardFunctions.execute(this.backwardFn, this.grad, child_);
        }
        if (!this.grad_persists) {
            this.grad = null;
        }

    }

    public void backward() {
        this.backward(null, null);
    }

    private void setOperands(Tensor other, Tensor child) {
        child.leftOperand = this;
        child.rightOperand = other;
    }


    public void createChildAndRegisterBackward(Tensor parent, Tensor child, String backwardFn, Tensor rightOperand, HashMap<String, Double> additionalNodeInfo) {
        parent.child = child;
        parent.child.leftOperand = parent;
        if (rightOperand != null) {
            parent.child.rightOperand = rightOperand;
        }
        if (additionalNodeInfo != null) {
            parent.child.additionalInfo = additionalNodeInfo;
        }
        if (child.requiresGrad) {
            parent.child.backwardFn = backwardFn;
        }
    }

    public void createChildAndRegisterBackward(Tensor parent, Tensor child, String backwardFn) {
        createChildAndRegisterBackward(parent, child, backwardFn, null, null);
    }

    public void createChildAndRegisterBackward(Tensor parent, Tensor child, String backwardFn, HashMap<String, Double> additionalNodeInfo) {
        createChildAndRegisterBackward(parent, child, backwardFn, null, additionalNodeInfo);
    }

    public void zero_() {
        this.grad = Nd4j.zeros(this.grad.dataType(), this.grad.shape());
    }

    public void persistGrad() {
        this.grad_persists = true;
    }

    public void deleteTree() {
        if (this.leftOperand != null) {
            this.leftOperand.deleteTree();
            this.leftOperand = null;
        }
        if (this.rightOperand != null) {
            this.rightOperand.deleteTree();
            this.rightOperand = null;
        }
        this.child = null;
    }

    private boolean shouldRequireGrad(Tensor a, Tensor b) {
        return a.requiresGrad || b.requiresGrad;
    }


    public Tensor detach(boolean track) {
        return new Tensor(this.data.dup(), track);
    }
    public Tensor detach() {
        return detach(false);
    }


    private Graph compGraphToFile(String filePath,int nodeBackwardIndex,Tensor start,Graph g) {
        if (start == null) {
            start = this;
        }
        if (g == null) {
            g = graph("comp_graph").directed().graphAttr().with(Rank.dir(Rank.RankDir.TOP_TO_BOTTOM)).nodeAttr().with(Font.name("arial"))
                    .linkAttr().with("class", "link-class");
        }
        String noBackwardValue = "No_backward_registered";
        ArrayList<Node> nodes = new ArrayList<>();
        nodes.add(node(String.valueOf(nodeBackwardIndex)));

        if (start.leftOperand != null) {
             nodes.add(nodes.get(0).link(node(String.valueOf(nodeBackwardIndex -1))));
        }
        if (start.rightOperand != null) {
            nodes.add(nodes.get(0).link(node(String.valueOf(nodeBackwardIndex -2))));
        }
        if (start.leftOperand != null && start.rightOperand != null) {
            nodes.get(1).link(nodes.get(2)).with(Attributes.attr("weight",5), Style.DASHED).with(Attributes.attr("label",noBackwardValue));
        }
        g.with(nodes);

        if (start.leftOperand != null) {
            return compGraphToFile(filePath, nodeBackwardIndex - 1, start.leftOperand, g);
        }
        if (start.rightOperand != null) {
            return compGraphToFile(filePath, nodeBackwardIndex - 2, start.rightOperand, g);
        }
        return g;


    }
    public void compGraphToFile(String filePath) {
        Graph graph = compGraphToFile(filePath,-1,null,null);
        try {
            Graphviz.fromGraph(graph).width(700).render(Format.PNG).toFile(new File(filePath));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
