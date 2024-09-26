package dev.skyr.core.checks;

import org.nd4j.linalg.api.ndarray.INDArray;

public class StaticChecks {

    public static boolean isOneHotEncoded(INDArray array) {
        return array.rank() == 2 && array.minNumber().doubleValue() == 0.0 && array.sum(1).minNumber().doubleValue() == 1.0;
    }

}
