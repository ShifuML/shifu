/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.dtrain.dt;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import ml.shifu.guagua.io.Bytable;

/**
 * Predict wrappers classification and regression result in {@link #predict}, for classification {@link #prob} means the
 * probability, for regression no meaning for {@link #prob}
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class Predict implements Bytable {

    private double predict;

    private double prob;

    public Predict() {
    }

    public Predict(double predict) {
        this.predict = predict;
    }

    public Predict(double predict, double prob) {
        this.predict = predict;
        this.prob = prob;
    }

    /**
     * @return the predict
     */
    public double getPredict() {
        return predict;
    }

    /**
     * @return the prob
     */
    public double getProb() {
        return prob;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeDouble(predict);
        out.writeDouble(prob);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        predict = in.readDouble();
        prob = in.readDouble();
    }

    @Override
    public String toString() {
        return "Predict [predict=" + predict + ", prob=" + prob + "]";
    }

}
