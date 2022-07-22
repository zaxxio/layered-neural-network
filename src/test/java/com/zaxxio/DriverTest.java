package com.zaxxio;

import com.zaxxio.network.MultiLayerNetwork;
import com.zaxxio.network.activation.ActivationFunction;
import com.zaxxio.network.config.MultiLayerConfiguration;
import com.zaxxio.network.config.ScoreListener;
import com.zaxxio.network.config.step.OptimizationAlgo;
import com.zaxxio.network.data.MLDataSet;
import com.zaxxio.network.model.DenseLayer;
import com.zaxxio.network.model.InputLayer;
import com.zaxxio.network.model.OutputLayer;
import com.zaxxio.network.weight.WeightInit;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

public class DriverTest {

    private static final double[][] XOR_INPUT = {
            {1, 1},
            {1, 0},
            {0, 1},
            {0, 0}
    };

    private static final double[][] XOR_IDEAL = {
            {0},
            {1},
            {1},
            {0},
    };

    @Test
    public void xorTest() {

        final MultiLayerConfiguration config = new MultiLayerConfiguration.Builder()
                .activation(ActivationFunction.LEAKY_RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgo.STOCHASTIC_GRADIENT_DESCENT)
                .momentum(0.5)
                .updater(0.001)
                .layer(0, new InputLayer.Builder().nIn(2)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(25)
                        .activation(ActivationFunction.LEAKY_RELU)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(15)
                        .activation(ActivationFunction.LEAKY_RELU)
                        .build())
                .layer(3, new DenseLayer.Builder().nIn(15)
                        .activation(ActivationFunction.LEAKY_RELU)
                        .build())
                .layer(4, new OutputLayer.Builder().nOut(1)
                        .activation(ActivationFunction.LEAKY_RELU)
                        .build())
                .list()
                .minError(0.01)
                .server(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        ScoreListener listener = new ScoreListener(1000);
        model.addScoreListener(listener);

        MLDataSet xor = new MLDataSet(XOR_INPUT, XOR_IDEAL);

        model.fit(xor);
        System.out.println("Training Completed");
        System.out.println(Arrays.toString(model.predict(1, 1)));
        System.out.println(Arrays.toString(model.predict(1, 0)));
        System.out.println(Arrays.toString(model.predict(0, 1)));
        System.out.println(Arrays.toString(model.predict(0, 0)));

        double[] p1 = model.predict(1, 1);
        double[] p2 = model.predict(1, 0);
        double[] p3 = model.predict(0, 1);
        double[] p4 = model.predict(0, 0);

        Assert.assertTrue(p1[0] < 0.1);
        Assert.assertTrue(p2[0] > 0.9);
        Assert.assertTrue(p3[0] > 0.9);
        Assert.assertTrue(p4[0] < 0.1);


    }

}
