/*
 * MIT License
 *
 * Copyright (c) 2020 Partha Sutradhar.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.zaxxio.network.model;

import com.zaxxio.network.activation.ActivationFunction;
import com.zaxxio.network.config.step.ConfigurationBuilder;

import java.util.UUID;

public class DenseLayer extends Layer {

    protected int nIn;
    protected int nOut;
    protected ActivationFunction activationFunction;

    public DenseLayer(Builder builder) {
        this.layerId = UUID.randomUUID();
        this.nIn = builder.nIn;
        this.nOut = builder.nOut;
        this.activationFunction = builder.activationFunction;
    }

    public int getnIn() {
        return nIn;
    }

    public int getnOut() {
        return nOut;
    }

    public UUID getLayerId() {
        return layerId;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public static class Builder {

        private int nIn;
        private int nOut;
        private ActivationFunction activationFunction = ConfigurationBuilder.activationFunction;
        private double dropout;

        public Builder nIn(int fanIn) {
            this.nIn = fanIn;
            return this;
        }

        public Builder dropOut(double dropout) {
            this.dropout = dropout;
            return this;
        }

        public Builder nOut(int fanOut) {
            this.nOut = fanOut;
            return this;
        }

        public Builder activation(ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public DenseLayer build() {
            return new DenseLayer(this);
        }

    }

}