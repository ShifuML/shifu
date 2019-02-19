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
package ml.shifu.shifu.core.dtrain.wnd;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import ml.shifu.guagua.io.Bytable;

/**
 * {@link DenseInputLayer} denotes dense input number array. Forward computation is a wrapper with just current input
 * array. {@linkplain #backward(float[], float)} no need to be supported as it should not be called.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class DenseInputLayer implements Layer<float[], float[], float[], float[]>, Bytable {

    /**
     * Output dimension.
     */
    private final int out;

    public DenseInputLayer(int out) {
        this.out = out;
    }

    @Override
    public int getOutDim() {
        return this.out;
    }

    @Override
    public float[] forward(float[] inputs) {
        assert inputs.length == this.out;
        return inputs;
    }

    @Override
    public float[] backward(float[] backInputs, float sig) {
        throw new UnsupportedOperationException();
    }

    /* (non-Javadoc)
     * @see ml.shifu.guagua.io.Bytable#write(java.io.DataOutput)
     */
    @Override
    public void write(DataOutput out) throws IOException {
        // TODO Auto-generated method stub
        
    }

    /* (non-Javadoc)
     * @see ml.shifu.guagua.io.Bytable#readFields(java.io.DataInput)
     */
    @Override
    public void readFields(DataInput in) throws IOException {
        // TODO Auto-generated method stub
        
    }

}
