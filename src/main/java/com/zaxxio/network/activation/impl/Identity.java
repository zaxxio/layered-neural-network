package com.zaxxio.network.activation.impl;

import com.zaxxio.network.activation.IActivationFunction;

public class Identity implements IActivationFunction {
    @Override
    public double output(double x) {
        return x;
    }

    @Override
    public double outputDerivative(double x) {
        return 1.0;
    }
}
