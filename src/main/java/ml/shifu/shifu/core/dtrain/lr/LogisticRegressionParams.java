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
package ml.shifu.shifu.core.dtrain.lr;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import ml.shifu.guagua.io.Combinable;
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
public class LogisticRegressionParams extends HaltBytable implements Combinable<LogisticRegressionParams>{

    /**
     * Model weights in the first iteration, gradients in other iterations.
     */
    private double[] parameters;

    /**
     * Current test error which can be sent to master
     */
    private double testError = 0;

    /**
     * Current train error which can be sent to master
     */
    private double trainError = 0;

    /**
     * Training record count in one worker
     */
    private long trainSize;

    /**
     * Testing record count in one worker
     */
    private long testSize;

    public LogisticRegressionParams() {
    }

    public LogisticRegressionParams(double[] parameters) {
        this.parameters = parameters;
    }

    public LogisticRegressionParams(double[] parameters, double trainError, double testError, long trainSize,
            long testSize) {
        this.parameters = parameters;
        this.trainError = trainError;
        this.testError = testError;
        this.trainSize = trainSize;
        this.testSize = testSize;
    }

    public double[] getParameters() {
        return parameters;
    }

    public void setParameters(double[] parameters) {
        this.parameters = parameters;
    }

    /**
     * @return the trainSize
     */
    public long getTrainSize() {
        return trainSize;
    }

    /**
     * @param trainSize
     *            the trainSize to set
     */
    public void setTrainSize(long trainSize) {
        this.trainSize = trainSize;
    }
    
    @Override
    public LogisticRegressionParams combine(LogisticRegressionParams from) {
        assert from != null;
        this.trainError += from.trainError;
        this.testError += from.testError;
        this.trainSize+=from.trainSize;
        this.testSize+=from.testSize;
        assert this.parameters != null && from.parameters != null;
        for(int i = 0; i < this.parameters.length; i++) {
            this.parameters[i] += from.parameters[i];
        }
        return this;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {
        if(parameters == null) {
            out.writeInt(0);
        } else {
            out.writeInt(this.parameters.length);
            for(int i = 0; i < this.parameters.length; i++) {
                out.writeDouble(this.parameters[i]);
            }
        }
        out.writeDouble(this.trainError);
        out.writeDouble(this.testError);
        out.writeLong(this.trainSize);
        out.writeLong(this.testSize);
    }

    @Override
    public void doReadFields(DataInput in) throws IOException {
        int length = in.readInt();
        this.parameters = new double[length];
        for(int i = 0; i < length; i++) {
            this.parameters[i] = in.readDouble();
        }
        this.trainError = in.readDouble();
        this.testError = in.readDouble();
        this.trainSize = in.readLong();
        this.testSize = in.readLong();
    }

    /**
     * @return the testError
     */
    public double getTestError() {
        return testError;
    }

    /**
     * @param testError
     *            the testError to set
     */
    public void setTestError(double testError) {
        this.testError = testError;
    }

    /**
     * @return the trainError
     */
    public double getTrainError() {
        return trainError;
    }

    /**
     * @param trainError
     *            the trainError to set
     */
    public void setTrainError(double trainError) {
        this.trainError = trainError;
    }

    /**
     * @return the testSize
     */
    public long getTestSize() {
        return testSize;
    }

    /**
     * @param testSize
     *            the testSize to set
     */
    public void setTestSize(long testSize) {
        this.testSize = testSize;
    }

}
