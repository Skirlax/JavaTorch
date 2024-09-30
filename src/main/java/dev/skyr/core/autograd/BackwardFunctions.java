package dev.skyr.core.autograd;

import dev.skyr.core.nn.functional.Functions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.lang.reflect.Method;
import java.util.Arrays;

public class BackwardFunctions {
    public static void execute(String name, INDArray grad, Tensor tensor) {
        try {
            Method method = BackwardFunctions.class.getDeclaredMethod(name, INDArray.class, Tensor.class);
            method.invoke(null, grad, tensor);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static void add_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = grad.dup();
            child.leftOperand.backward(dl, child.leftOperand);
        }
        if (child.rightOperand.requiresGrad) {
            INDArray dr = grad.dup();
            child.rightOperand.backward(dr, child.rightOperand);
        }
    }

    public static void mul_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = grad.mul(child.rightOperand.data);
            child.leftOperand.backward(dl, child.leftOperand);
        }
        if (child.rightOperand.requiresGrad) {
            INDArray dr = grad.mul(child.leftOperand.data);
            child.rightOperand.backward(dr, child.rightOperand);
        }
    }

    public static void div_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = grad.div(child.rightOperand.data);
            child.leftOperand.backward(dl, child.leftOperand);
        }
        if (child.rightOperand.requiresGrad) {
            INDArray dr = grad.neg().mul(child.leftOperand.data).div(Transforms.pow(child.rightOperand.data, 2));
            child.rightOperand.backward(dr, child.rightOperand);
        }
    }

    public static void matmul_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            int rank = child.rightOperand.data.rank();
            INDArray swapped = child.rightOperand.data.swapAxes(rank - 1, rank - 2);
            INDArray dl = Nd4j.matmul(grad, swapped);
            child.leftOperand.backward(dl, child.leftOperand);
        }
        if (child.rightOperand.requiresGrad) {
            int rank = child.leftOperand.data.rank();
            INDArray swapped = child.leftOperand.data.swapAxes(rank - 1, rank - 2);
            INDArray dr = Nd4j.matmul(swapped, grad);
            child.rightOperand.backward(dr, child.rightOperand);
        }
    }

    public static void neg_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = grad.neg();
            child.leftOperand.backward(dl, child.leftOperand);
        }
    }

    public static void pow_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = grad.mul(Transforms.pow(child.leftOperand.data, child.rightOperand.data.sub(1)).mul(child.rightOperand.data));
            child.leftOperand.backward(dl, child.leftOperand);
        }
        if (child.rightOperand.requiresGrad) {
            INDArray dr = grad.mul(Transforms.pow(child.leftOperand.data, child.rightOperand.data).mul(Transforms.log(child.leftOperand.data)));
            child.rightOperand.backward(dr, child.rightOperand);
        }
    }

    public static void view_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = grad.dup().reshape(child.leftOperand.originalShape);
            child.leftOperand.backward(dl, child.leftOperand);
        }

    }

    public static void broadcast_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = reverse_broadcast(grad.dup(), child.leftOperand.originalShape);
            child.leftOperand.backward(dl, child.leftOperand);
        }
    }

    public static void permute_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = grad.dup().permute(child.leftOperand.permute);
            child.leftOperand.backward(dl, child.leftOperand);
        }
    }

    private static INDArray reverse_broadcast(INDArray broadcasted, long[] originalShape) {
        long[] broadcastedShape = broadcasted.shape();
        originalShape = alignDims(originalShape,broadcastedShape);
        long[] dimsToSum = new long[broadcastedShape.length];
        for (int i = 0; i < broadcastedShape.length; i++) {
            if (i >= originalShape.length) {
                dimsToSum[i] = i;
            } else if (broadcastedShape[i] != originalShape[i]) {
                dimsToSum[i] = i;
            } else {
                dimsToSum[i] = -10;
            }
        }
        for (int i = 0; i < dimsToSum.length; i++) {
            if (dimsToSum[i] == -10) {
                continue;
            }
            long[] newShape = broadcasted.shape();
            broadcasted = broadcasted.sum((int) dimsToSum[i]);

            newShape[i] = 1;
            broadcasted = broadcasted.reshape(newShape);
        }
        return broadcasted.reshape(originalShape);
    }

    public static void pass(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            child.leftOperand.backward(grad, child.leftOperand);
        }
    }

    public static void transpose_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = grad.dup().transpose();
            child.leftOperand.backward(dl, child.leftOperand);
        }
    }

    public static void sum_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = grad.dup().broadcast(child.leftOperand.data.shape());
            child.leftOperand.backward(dl, child.leftOperand);
        }
    }

    public static void max_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray mask = child.leftOperand.data.eq(child.data);
            INDArray dl = grad.mul(mask);
            child.leftOperand.backward(dl, child.leftOperand);
        }
    }

    public static void relu_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray mask = child.leftOperand.data.gt(0).castTo(DataType.DOUBLE);
            mask = mask.broadcast(grad.shape());
            INDArray dl = grad.mul(mask);
            child.leftOperand.backward(dl, child.leftOperand);
        }
    }

    public static void ln_backward(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = grad.div(child.leftOperand.data);
            child.leftOperand.backward(dl, child.leftOperand);
        }
    }

    public static void reverse_col2img(INDArray grad, Tensor child) {
        if (child.leftOperand.requiresGrad) {
            INDArray dl = Functions.col2img(grad.dup(), child.additionalInfo.get("stride").intValue(), child.additionalInfo.get("kernelSize").intValue(), child.additionalInfo.get("height").intValue(), child.additionalInfo.get("width").intValue(), child.additionalInfo.get("channels").intValue(),child.additionalInfo.get("padding").intValue());
            child.leftOperand.backward(dl, child.leftOperand);
        }
    }

    private static long[] alignDims(long[] inputShape, long... shape) {
        long[] newShape = new long[shape.length];
        for (int inputIndex = 0; inputIndex < inputShape.length; inputIndex++) {
            int lastIndexInTarget = getLastInTarget(shape,inputShape[inputIndex]);
            if (lastIndexInTarget != -1) {
                newShape[lastIndexInTarget] = inputShape[inputIndex];
            }
        }
        if (Arrays.stream(newShape).allMatch((s) -> s == 0)) {
            for (int i = 0; i < inputShape.length; i++) {
                newShape[i] = inputShape[i];
            }
        }
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] == 0) {
                newShape[i] = 1;
            }
        }
        return newShape;
    }

    private static int getLastInTarget(long[] shape, long element) {
        int lastIdx = -1;
        for (int currentIdx = 0; currentIdx < shape.length; currentIdx++) {
            if (shape[currentIdx] == element) {
                lastIdx = currentIdx;
            }
        }
        return lastIdx;
    }
}
