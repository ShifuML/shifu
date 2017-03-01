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
package ml.shifu.shifu.container.obj;

import org.apache.commons.collections.ListUtils;

import java.util.List;

/**
 * Created by zhanhu on 11/28/16.
 */
public class VarTrainConf {

    private List<String> variables;
    private ModelTrainConf modelTrainConf;

    public List<String> getVariables() {
        return variables;
    }

    public void setVariables(List<String> variables) {
        this.variables = variables;
    }

    public ModelTrainConf getModelTrainConf() {
        return modelTrainConf;
    }

    public void setModelTrainConf(ModelTrainConf modelTrainConf) {
        this.modelTrainConf = modelTrainConf;
    }

    @Override
    public boolean equals(Object obj) {
        if ( obj == null || !(obj instanceof VarTrainConf) ) {
            return false;
        }

        VarTrainConf other = (VarTrainConf) obj;
        if ( this == other ) {
            return true;
        }

        return ListUtils.isEqualList(this.variables, other.getVariables())
                && modelTrainConf.equals(other.getModelTrainConf());
    }
}
