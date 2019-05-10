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
package ml.shifu.shifu.core.dtrain.wdl;

import ml.shifu.shifu.core.dtrain.AssertUtils;
import ml.shifu.shifu.core.dtrain.wdl.optimization.Optimizer;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * {@link DenseInputLayer} denotes dense input number array. Forward computation is a wrapper with just current input
 * array. {@linkplain #backward(float[])} no need to be supported as it should not be called.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class DenseInputLayer extends AbstractLayer<float[], float[], float[], float[], DenseInputLayer> {

    /**
     * Output dimension.
     */
    private int out;

    public DenseInputLayer(int out) {
        this.out = out;
    }

    public DenseInputLayer() {
        this.out = 0;
    }

    @Override
    public int getOutDim() {
        return this.out;
    }

    @Override
    public float[] forward(float[] inputs) {
        AssertUtils.assertNotNull(inputs);
        AssertUtils.assertEquals(inputs.length, this.out);
        return inputs;
    }

    @Override
    public float[] backward(float[] backInputs) {
        throw new UnsupportedOperationException();
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#write(java.io.DataOutput)
     */
    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.out);
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#readFields(java.io.DataInput)
     */
    @Override
    public void readFields(DataInput in) throws IOException {
        this.out = in.readInt();
    }

    @Override
    public DenseInputLayer combine(DenseInputLayer from) {
        return this;
    }

    @Override
    public void update(DenseInputLayer gradLayer, Optimizer optimizer, String uniqueKey) {
        // no need to update output dimension
    }

}
