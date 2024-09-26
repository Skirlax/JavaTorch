package dev.skyr.core.nn.functional;

import org.nd4j.linalg.api.buffer.DataType;
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
            input = Nd4j.pad(input, new int[][]{{0, 0}, {0, 0}, {padding, padding}, {padding, padding}});
        }
        long batch_size = input.size(0);
        long channels = input.size(1);
        long width = input.size(2);
        long height = input.size(3);
        long hor_slides = ((width - kernelSize) / stride) + 1;
        long ver_slides = ((height - kernelSize) / stride) + 1;
        long column_size = kernelSize * kernelSize * channels;
        INDArray col = Nd4j.zeros(DataType.DOUBLE, batch_size, column_size, hor_slides * ver_slides);
        for (int vertical_index = 0; vertical_index < ver_slides; vertical_index += stride) {
            for (int horizontal_index = 0; horizontal_index < hor_slides; horizontal_index += stride) {
                INDArray patch = input.get(
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.interval((long) vertical_index, (long) vertical_index + kernelSize),
                        NDArrayIndex.interval(horizontal_index, horizontal_index + kernelSize)
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

    public static INDArray col2img(INDArray col, int stride, int kernelSize, int height, int width, int channels, int padding) {
        col = col.permute(0, 2, 1);
        INDArray img = Nd4j.zeros(DataType.DOUBLE, col.size(0), channels, height + 2L * padding, width + 2L * padding);
        int horizontal_slides = ((width + 2* padding - kernelSize) / stride) + 1;
        for (int i = 0; i < col.size(1); i += stride) {
            INDArray window = col.get(
                    NDArrayIndex.all(),
                    NDArrayIndex.point(i),
                    NDArrayIndex.all()
            ).reshape(col.size(0), channels, kernelSize, kernelSize);
            int vertical_index = i / horizontal_slides;
            int horizontal_index = i % horizontal_slides;
            img.get(
                    NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.interval(vertical_index, vertical_index + kernelSize),
                    NDArrayIndex.interval(horizontal_index, horizontal_index + kernelSize)).addi(window);
        }
        if (padding > 0) {
            return img.get(
                    NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.interval(padding,padding + height),
                    NDArrayIndex.interval(padding,padding + width)
            );
        }
        return img;
    }
}
