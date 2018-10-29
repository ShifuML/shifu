/*
 * Copyright [2012-2018] PayPal Software Foundation
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
package ml.shifu.shifu.core;

import org.encog.ml.BasicML;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

import java.util.Map;
import java.util.HashMap;

public class GenericModel extends BasicML implements MLRegression {

    private static final long serialVersionUID = 2L;

    private Computable model = null;

    private Map<String, Object> properties = new HashMap<String, Object>();

    public GenericModel(Map<String, Object> properties) {
        this.properties.putAll(properties);
    }

    public GenericModel(Computable model, Map<String, Object> properties) {
        this.model = model;
        this.properties.putAll(properties);
    }

    @Override
    public final MLData compute(final MLData input) {
        MLData result = new BasicMLData(1);
        if(model != null) {
            double score = model.compute(input);
            result.setData(0, score);
        }
        return result;
    }

    @Override
    public String toString() {
        return GenericModel.class.getCanonicalName() + ":"  + model.getClass().getCanonicalName();
    }

    @Override
    public int getOutputCount() {
        return 1;
    }

    @Override
    public void updateProperties() {
    }

    @Override 
    public int getInputCount() {
        return 1;
    }
    
    public Computable getModel() {
        return this.model;
    }

    public Map<String, Object> getGMProperties() {
        return this.properties;
    }
}
