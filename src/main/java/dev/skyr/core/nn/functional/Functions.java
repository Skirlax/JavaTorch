package dev.skyr.core.nn.functional;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Functions {
    public static INDArray img2col(INDArray input, int kernelSize, int stride, int padding) {
        //TODO: Optimize this function.

        // https://oneflow2020.medium.com/adding-unfold-and-fold-ops-into-oneflow-a4ae5f0ca328,
        // specifically: https://miro.medium.com/v2/resize:fit:1400/format:webp/0*hVtZr9sBV_Za6Btn
        if (padding > 0) {
            input = Nd4j.pad(input, new int[][]{{0, 0}, {padding, padding}, {padding, padding}, {0, 0}});
        }
        long batch_size = input.size(0);
        long channels = input.size(3);
        long width = input.size(1);
        long height = input.size(2);
        long hor_slides = ((width - kernelSize) / stride) + 1;
        long ver_slides = ((height - kernelSize) / stride) + 1;
        long column_size = kernelSize * kernelSize * channels;
        INDArray col = Nd4j.zeros(batch_size, column_size, hor_slides * ver_slides);
        input = input.permute(0, 3, 1, 2);
        for (int vertical_index = 0; vertical_index < ver_slides; vertical_index++) {
            for (int horizontal_index = 0; horizontal_index < hor_slides; horizontal_index++) {
                INDArray patch = input.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.interval((long) vertical_index * stride, (long) vertical_index * stride + kernelSize),
                        NDArrayIndex.interval(horizontal_index * stride, horizontal_index * stride + kernelSize)
                );
                col.put(new INDArrayIndex[]{
                                NDArrayIndex.all(),
                                NDArrayIndex.all(),
                                NDArrayIndex.point(vertical_index * hor_slides + horizontal_index)
                        },
                        patch.reshape(-1));
            }
        }
        return col;
    }

    public static int getOutputSize(int inputSize, int kernelSize, int stride, int padding) {
        return ((inputSize - kernelSize + 2 * padding) / stride) + 1;
    }
}
