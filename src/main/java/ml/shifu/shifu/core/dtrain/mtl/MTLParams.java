/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.mtl;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import ml.shifu.guagua.io.Combinable;
import ml.shifu.guagua.io.HaltBytable;
import ml.shifu.shifu.core.dtrain.layer.SerializationType;

/**
 * {@link MTLParams} is message transferred between {@link MTLMaster} and {@link MTLWorker}s in multi-task learning
 * iterations.
 * 
 * <p>
 * {@link MTLWorker}s send out gradients through {@link MTLParams} to {@link MTLMaster} for master aggregation. Then
 * {@link MTLMaster} updates latest model weights based on aggregated gradients and sends back through new
 * {@link MTLParams} to all workers. This is typical one epoch (sync mode).
 * 
 * <p>
 * {@link #mtm} is a delegation for both weights and gradients depending on type of {@link #serializationType}.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class MTLParams extends HaltBytable implements Combinable<MTLParams> {

    /**
     * If {@link #mtm} field is null or not.
     */
    private static final boolean MTM_IS_NULL = true;

    /**
     * # of weighted training records of such worker.
     */
    private double trainSize;

    /**
     * # of weighted validation records of such worker.
     */
    private double validationSize;

    /**
     * # of training records of such worker.
     */
    private double trainCount;

    /**
     * # of weighted validation records of such worker.
     */
    private double validationCount;

    /**
     * Train error of such worker and such iteration.
     */
    private double trainError;

    /**
     * Validation error of such worker and such iteration.
     */
    private double validationError;

    /**
     * Serialization type. Default to MODEL_SPEC.
     */
    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    /**
     * Multi-task memory model: weights are used to transfer from Master to workers; gradients are used to transfer from
     * workers to master.
     */
    private MultiTaskModel mtm;

    public void update(MultiTaskModel mtm) {
        this.getMtm().updateWeights(mtm);
    }

    @Override
    public MTLParams combine(MTLParams from) {
        this.trainCount += from.trainCount;
        this.trainError += from.trainError;
        this.validationCount += from.validationCount;
        this.validationError += from.validationError;
        this.trainSize += from.trainSize;
        this.validationSize += from.validationSize;
        // In the first iteration, the worker may send a empty MTLParams without MultiTaskModel Init
        if(from.getMtm() != null) {
            this.mtm = this.mtm.combine(from.getMtm());
        }
        return this;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {
        if(this.mtm == null) {
            // for the first iteration, the wnd will be null
            out.writeBoolean(MTM_IS_NULL);
        } else {
            out.writeBoolean(!MTM_IS_NULL);
            this.mtm.setSerializationType(serializationType);
            this.mtm.write(out);
        }
        out.writeDouble(this.trainCount);
        out.writeDouble(this.validationCount);
        out.writeDouble(this.trainError);
        out.writeDouble(this.validationError);
        out.writeDouble(this.trainSize);
        out.writeDouble(this.validationSize);
        out.writeInt(this.serializationType.getValue());
    }

    @Override
    public void doReadFields(DataInput in) throws IOException {
        boolean mtmIsNull = in.readBoolean();
        if(!mtmIsNull) {
            if(this.mtm == null) {
                this.mtm = new MultiTaskModel();
            }
            this.mtm.readFields(in);
        }
        this.trainCount = in.readDouble();
        this.validationCount = in.readDouble();
        this.trainError = in.readDouble();
        this.validationError = in.readDouble();
        this.trainSize = in.readDouble();
        this.validationSize = in.readDouble();
        this.serializationType = SerializationType.getSerializationType(in.readInt());
    }

    /**
     * @return the trainCount
     */
    public double getTrainCount() {
        return trainCount;
    }

    /**
     * @param trainCount
     *            the trainCount to set
     */
    public void setTrainCount(double trainCount) {
        this.trainCount = trainCount;
    }

    /**
     * @return the validationCount
     */
    public double getValidationCount() {
        return validationCount;
    }

    /**
     * @param validationCount
     *            the validationCount to set
     */
    public void setValidationCount(double validationCount) {
        this.validationCount = validationCount;
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
     * @return the validationError
     */
    public double getValidationError() {
        return validationError;
    }

    /**
     * @param validationError
     *            the validationError to set
     */
    public void setValidationError(double validationError) {
        this.validationError = validationError;
    }

    /**
     * @return the serializationType
     */
    public SerializationType getSerializationType() {
        return serializationType;
    }

    /**
     * @param serializationType
     *            the serializationType to set
     */
    public void setSerializationType(SerializationType serializationType) {
        this.serializationType = serializationType;
    }

    /**
     * @return the trainSize
     */
    public double getTrainSize() {
        return trainSize;
    }

    /**
     * @param trainSize
     *            the trainSize to set
     */
    public void setTrainSize(double trainSize) {
        this.trainSize = trainSize;
    }

    /**
     * @return the validationSize
     */
    public double getValidationSize() {
        return validationSize;
    }

    /**
     * @param validationSize
     *            the validationSize to set
     */
    public void setValidationSize(double validationSize) {
        this.validationSize = validationSize;
    }

    /**
     * @return the mtm
     */
    public MultiTaskModel getMtm() {
        return mtm;
    }

    /**
     * @param mtm
     *            the mtm to set
     */
    public void setMtm(MultiTaskModel mtm) {
        this.mtm = mtm;
    }

}
