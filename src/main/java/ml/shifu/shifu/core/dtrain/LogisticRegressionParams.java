/*
 * Copyright [2013-2014] eBay Software Foundation
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
package ml.shifu.shifu.core.dtrain;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import ml.shifu.guagua.io.HaltBytable;

/**
 * A model class to store logistic regression weight on first iteration by using {@link #parameters}, while in other
 * iterations {@link #parameters} is used to store gradients.
 * 
 * <p>
 * To make all workers started at the same model, master will compute a consistent model weights at the first iteration
 * and then send to all the workers. Workers will start computing from the second iteration.
 * 
 * <p>
 * Workers are responsible to compute local accumulated gradients and send to master while master accumulates all
 * gradients together to build a global model.
 */
public class LogisticRegressionParams extends HaltBytable {

    /**
     * Model weights in the first iteration, gradients in other iterations.
     */
    private double[] parameters;

    /**
     * Model error in one worker one iteration.
     */
    private double error;

    public LogisticRegressionParams() {
    }

    public LogisticRegressionParams(double[] parameters) {
        this.parameters = parameters;
    }

    public LogisticRegressionParams(double[] parameters, double error) {
        this.parameters = parameters;
        this.error = error;
    }

    public double[] getParameters() {
        return parameters;
    }

    public void setParameters(double[] parameters) {
        this.parameters = parameters;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {
        if(parameters != null) {
            out.writeInt(this.parameters.length);
            for(int i = 0; i < this.parameters.length; i++) {
                out.writeDouble(this.parameters[i]);
            }
        }
        out.writeDouble(this.error);
    }

    @Override
    public void doReadFields(DataInput in) throws IOException {
        int length = in.readInt();
        parameters = new double[length];
        for(int i = 0; i < length; i++) {
            parameters[i] = in.readDouble();
        }
        this.error = in.readDouble();
    }

}
