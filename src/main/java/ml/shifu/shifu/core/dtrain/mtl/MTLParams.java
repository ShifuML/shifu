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

import ml.shifu.guagua.io.Combinable;
import ml.shifu.guagua.io.HaltBytable;
import ml.shifu.shifu.core.dtrain.SerializationType;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * @author haillu
 */
public class MTLParams extends HaltBytable implements Combinable<MTLParams> {
    private static final boolean MTL_IS_NULL = true;

    private double trainSize;

    private double validationSize;

    /**
     * # of training records per such worker.(Now, we don't use it in MTL.)
     */
    private double trainCount;

    private double validationCount;

    private double[] trainErrors = {};

    private double[] validationErrors = {};

    private SerializationType serializationType = SerializationType.MODEL_SPEC;

    private MultiTaskLearning mtl;

    @Override
    public MTLParams combine(MTLParams from) {
        this.trainCount += from.trainCount;
        for(int i = 0; i < trainErrors.length; i++) {
            this.trainErrors[i] += from.trainErrors[i];
        }
        this.validationCount += from.validationCount;
        for(int i = 0; i < validationErrors.length; i++) {
            this.validationErrors[i] += from.validationErrors[i];
        }

        this.trainSize += from.trainSize;
        this.validationSize += from.validationSize;

        // In the first iteration, the worker may send a empty MTLParams without MultiTaskLearning Init.
        if(from.getMtl() != null) {
            this.mtl = this.mtl.combine(from.getMtl());
        }
        return this;
    }

    @Override
    public void doWrite(DataOutput out) throws IOException {
        if(this.mtl == null) {
            out.writeBoolean(MTL_IS_NULL);
        } else {
            out.writeBoolean(!MTL_IS_NULL);
            this.mtl.setSerializationType(serializationType);
            this.mtl.write(out);
        }
        out.writeDouble(this.trainCount);
        out.writeDouble(this.validationCount);
        for(int i = 0; i < trainErrors.length; i++) {
            out.writeDouble(trainErrors[i]);
        }
        for(int i = 0; i < validationErrors.length; i++) {
            out.writeDouble(validationErrors[i]);
        }
        out.writeInt(this.serializationType.getValue());

    }

    @Override
    public void doReadFields(DataInput in) throws IOException {
        boolean mtlIsNull = in.readBoolean();
        if(!mtlIsNull) {
            if(this.mtl == null) {
                this.mtl = new MultiTaskLearning();
            }
            this.mtl.readFields(in);
        }
        this.trainCount = in.readDouble();
        this.validationCount = in.readDouble();
        for(int i = 0; i < trainErrors.length; i++) {
            this.trainErrors[i] = in.readDouble();
        }
        for(int i = 0; i < validationErrors.length; i++) {
            this.validationErrors[i] = in.readDouble();
        }
        this.serializationType = SerializationType.getSerializationType(in.readInt());
    }

    public double getTrainCount() {
        return trainCount;
    }

    public void setTrainCount(double trainCount) {
        this.trainCount = trainCount;
    }

    public double getValidationCount() {
        return validationCount;
    }

    public void setValidationCount(double validationCount) {
        this.validationCount = validationCount;
    }

    public double getTrainSize() {
        return trainSize;
    }

    public void setTrainSize(double trainSize) {
        this.trainSize = trainSize;
    }

    public double getValidationSize() {
        return validationSize;
    }

    public void setValidationSize(double validationSize) {
        this.validationSize = validationSize;
    }

    public double[] getTrainErrors() {
        return trainErrors;
    }

    public void setTrainErrors(double[] trainError) {
        this.trainErrors = trainError;
    }

    public double[] getValidationErrors() {
        return validationErrors;
    }

    public void setValidationErrors(double[] validationErrors) {
        this.validationErrors = validationErrors;
    }

    public SerializationType getSerializationType() {
        return serializationType;
    }

    public void setSerializationType(SerializationType serializationType) {
        this.serializationType = serializationType;
    }

    public MultiTaskLearning getMtl() {
        return mtl;
    }

    public void setMtl(MultiTaskLearning mtl) {
        this.mtl = mtl;
    }
}
