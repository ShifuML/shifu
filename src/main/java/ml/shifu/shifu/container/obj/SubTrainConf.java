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

/**
 * Created by zhanhu on 11/28/16.
 */
public class SubTrainConf {

    private String modelName;
    private String dataFilterExpr;
    private ModelStatsConf modelStatsConf;
    private ModelNormalizeConf modelNormalizeConf;
    private ModelVarSelectConf modelVarSelectConf;
    private ModelTrainConf modelTrainConf;

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getDataFilterExpr() {
        return dataFilterExpr;
    }

    public void setDataFilterExpr(String dataFilterExpr) {
        this.dataFilterExpr = dataFilterExpr;
    }

    public ModelStatsConf getModelStatsConf() {
        return modelStatsConf;
    }

    public void setModelStatsConf(ModelStatsConf modelStatsConf) {
        this.modelStatsConf = modelStatsConf;
    }

    public ModelNormalizeConf getModelNormalizeConf() {
        return modelNormalizeConf;
    }

    public void setModelNormalizeConf(ModelNormalizeConf modelNormalizeConf) {
        this.modelNormalizeConf = modelNormalizeConf;
    }

    public ModelVarSelectConf getModelVarSelectConf() {
        return modelVarSelectConf;
    }

    public void setModelVarSelectConf(ModelVarSelectConf modelVarSelectConf) {
        this.modelVarSelectConf = modelVarSelectConf;
    }

    public ModelTrainConf getModelTrainConf() {
        return modelTrainConf;
    }

    public void setModelTrainConf(ModelTrainConf modelTrainConf) {
        this.modelTrainConf = modelTrainConf;
    }

    @Override
    public int hashCode() {
        int hash = 1;
        if ( modelStatsConf != null ) {
            hash *= modelStatsConf.hashCode();
        }

        if ( modelNormalizeConf != null ) {
            hash *= modelNormalizeConf.hashCode();
        }

        if ( modelVarSelectConf != null ) {
            hash *= modelVarSelectConf.hashCode();
        }

        if ( modelTrainConf != null ) {
            hash *= modelTrainConf.hashCode();
        }

        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if ( obj == null || !(obj instanceof SubTrainConf) ) {
            return false;
        }

        SubTrainConf other = (SubTrainConf) obj;
        if ( this == other ) {
            return true;
        }

        return modelStatsConf.equals(other.getModelStatsConf())
                && modelNormalizeConf.equals(other.getModelNormalizeConf())
                && modelVarSelectConf.equals(other.getModelVarSelectConf())
                && modelTrainConf.equals(other.getModelTrainConf());
    }
}
